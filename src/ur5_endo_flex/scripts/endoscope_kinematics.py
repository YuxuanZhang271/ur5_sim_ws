#!/usr/bin/env python3
"""
KDL kinematics for the UR5 + endoscope + endo_flex chain.

Provides:
  - Chain setup from a URDF string (both base_link -> endoscope_link
    and base_link -> flex_base_link)
  - Forward kinematics (points / axes / frames in the base frame)
  - Automatic horizon-leveling twist solver for endo_flex so that
    the y-axis of endo_flex (and end_cylinder) is always kept on the
    horizontal plane
  - Bird-eye view bending angle solver (30 deg downward angle between
    end_cylinder and the horizon plane)
  - Inverse kinematics:
      * 5-DOF wrist_1-constrained axial/line IK (ik_along_axis)
        that simultaneously enforces:
          1) p_tip(q) == p_tip_target
          2) tube_axis(q) == axis_target
          3) shoulder_lift + elbow + wrist_1 == -270 deg
        eliminating post-IK wrist_1 override drift.
      * Position-only IK (ik_position_only)
      * RCM-constrained IK (ik_with_rcm)
  - Tube-axis geometry (two points on the endoscope mesh)
  - Auto-solve of wrist_1 so the wrist joint points out from the base
"""

import math
import numpy as np

try:
    import PyKDL
    from urdf_parser_py.urdf import URDF
    from kdl_parser_py.urdf import treeFromUrdfModel
    HAS_KDL = True
except Exception as _e:  # noqa
    print("KDL imports failed:", _e)
    HAS_KDL = False


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Default GUI pose (degrees); nothing is sent to the robot on startup.
HOME_DEG = [0.0, -120.0, -120.0, -30.0, 90.0, 45.0]

# Fallback transform from endoscope_link to flex_base_link (from ur5_endo_flex.urdf.xacro)
# origin xyz="0 0 -0.398" rpy="3.141593 0 0.616096"
ENDO_TO_FLEX_XYZ = np.array([0.0, 0.0, -0.398], dtype=float)
ENDO_TO_FLEX_RPY = (3.141593, 0.0, 0.616096)

# Straight-configuration offset from flex_base_link to end_cylinder center:
# CYL_LEN/2 (0.0025) + 8 * SEG_LEN (0.064) = 0.0665 m along +Z of flex_base_link
# (which is z = -0.398 - 0.0665 = -0.4645 m in endoscope_link frame).
FLEX_STRAIGHT_LEN_TO_END_CYL = 0.0665
END_CYLINDER_IN_ENDO_LINK = np.array(
    [0.0, 0.0, -0.398 - FLEX_STRAIGHT_LEN_TO_END_CYL], dtype=float
)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], dtype=float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[  c,  -s, 0.0],
                     [  s,   c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def _rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _wrap_pi(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


class EndoscopeKinematics:
    """KDL chain + FK/IK helpers for the UR5 + endoscope + endo_flex."""

    def __init__(self, base_link, tip_link,
                 rcm_point_local, tip_local, logger=None):
        self.base_link = base_link
        self.tip_link = tip_link
        self.rcm_point_local = np.asarray(rcm_point_local, dtype=float)
        self.tip_local = np.asarray(tip_local, dtype=float)
        self.tube_length = float(
            np.linalg.norm(self.tip_local - self.rcm_point_local))
        self.logger = logger

        self.chain = None
        self.fk_solver = None
        self.ik_solver = None

        self.flex_base_chain = None
        self.flex_base_fk_solver = None
        self._R_endo_to_flex = _rpy_to_mat(*ENDO_TO_FLEX_RPY)

        xy_gap = self.tip_local[:2] - self.rcm_point_local[:2]
        if np.linalg.norm(xy_gap) > 1e-4 and logger is not None:
            logger.warn(
                'rcm_point_in_tip_frame and endoscope_tip_in_link_frame '
                'do NOT share x,y in the link frame. The tube axis derived '
                'from them will NOT be parallel to the URDF collision '
                'cylinder. Check endoscope.xacro / launch parameters.')

    # ------------------------------------------------------------------
    @property
    def ready(self):
        return self.chain is not None

    def setup_from_urdf(self, urdf_str):
        if not HAS_KDL:
            if self.logger:
                self.logger.warn('KDL not available; Cartesian/RCM disabled.')
            return False
        try:
            robot = URDF.from_xml_string(urdf_str)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                raise RuntimeError('treeFromUrdfModel failed')
            self.chain = tree.getChain(self.base_link, self.tip_link)
            if self.chain.getNrOfJoints() == 0:
                raise RuntimeError('empty chain base_link -> tip_link')
            self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
            self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.chain)

            # Optional chain to flex_base_link (same 6 UR5 joints, includes
            # endoscope_to_flex_joint fixed transform from the URDF).
            try:
                fb_chain = tree.getChain(self.base_link, 'flex_base_link')
                if fb_chain.getNrOfJoints() == self.chain.getNrOfJoints():
                    self.flex_base_chain = fb_chain
                    self.flex_base_fk_solver = (
                        PyKDL.ChainFkSolverPos_recursive(fb_chain)
                    )
            except Exception:
                self.flex_base_chain = None
                self.flex_base_fk_solver = None

            if self.logger:
                self.logger.info(
                    f'KDL chain built: {self.chain.getNrOfJoints()} joints')
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f'KDL setup failed: {e}')
            self.chain = None
            return False

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------
    def fk(self, q):
        n = self.chain.getNrOfJoints()
        q_kdl = PyKDL.JntArray(n)
        for i in range(n):
            q_kdl[i] = float(q[i])
        frame = PyKDL.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)
        return frame

    def fk_matrix(self, q):
        """Return (R_3x3, p_3) of tip_link (endoscope_link) in base_link frame."""
        f = self.fk(q)
        R = np.array([[f.M[0, 0], f.M[0, 1], f.M[0, 2]],
                      [f.M[1, 0], f.M[1, 1], f.M[1, 2]],
                      [f.M[2, 0], f.M[2, 1], f.M[2, 2]]], dtype=float)
        p = np.array([f.p.x(), f.p.y(), f.p.z()], dtype=float)
        return R, p

    def flex_base_frame_in_base(self, q):
        """Return (R_3x3, p_3) of flex_base_link (start_cylinder) in base_link frame."""
        if self.flex_base_fk_solver is not None:
            n = self.flex_base_chain.getNrOfJoints()
            q_kdl = PyKDL.JntArray(n)
            for i in range(n):
                q_kdl[i] = float(q[i])
            f = PyKDL.Frame()
            self.flex_base_fk_solver.JntToCart(q_kdl, f)
            R = np.array([[f.M[0, 0], f.M[0, 1], f.M[0, 2]],
                          [f.M[1, 0], f.M[1, 1], f.M[1, 2]],
                          [f.M[2, 0], f.M[2, 1], f.M[2, 2]]], dtype=float)
            p = np.array([f.p.x(), f.p.y(), f.p.z()], dtype=float)
            return R, p

        R_endo, p_endo = self.fk_matrix(q)
        R_flex = R_endo @ self._R_endo_to_flex
        p_flex = p_endo + R_endo @ ENDO_TO_FLEX_XYZ
        return R_flex, p_flex

    def point_in_base(self, q, p_local):
        f = self.fk(q)
        p = f * PyKDL.Vector(float(p_local[0]),
                             float(p_local[1]),
                             float(p_local[2]))
        return np.array([p.x(), p.y(), p.z()])

    def end_cylinder_in_base(self, q):
        """Return 3D position of end_cylinder in base_link frame when endo_flex is straight."""
        R_flex, p_flex = self.flex_base_frame_in_base(q)
        return p_flex + R_flex[:, 2] * FLEX_STRAIGHT_LEN_TO_END_CYL

    def tool_axis_in_base(self, q):
        f = self.fk(q)
        z = f.M.UnitZ()
        return np.array([z.x(), z.y(), z.z()])

    def tube_axis_in_base(self, q):
        """Return (unit_axis, p_rcm_world, p_tip_world)."""
        f = self.fk(q)
        pr = f * PyKDL.Vector(float(self.rcm_point_local[0]),
                              float(self.rcm_point_local[1]),
                              float(self.rcm_point_local[2]))
        pt = f * PyKDL.Vector(float(self.tip_local[0]),
                              float(self.tip_local[1]),
                              float(self.tip_local[2]))
        p_rcm = np.array([pr.x(), pr.y(), pr.z()])
        p_tip = np.array([pt.x(), pt.y(), pt.z()])
        v = p_tip - p_rcm
        n = np.linalg.norm(v)
        if n < 1e-9:
            return self.tool_axis_in_base(q), p_rcm, p_tip
        return v / n, p_rcm, p_tip

    # ------------------------------------------------------------------
    # Endo-flex Automatic Horizon-Keeping Twist & Bird-Eye View Solvers
    # ------------------------------------------------------------------
    def solve_horizon_flex_yaw(self, q, steer_yaw=0.0, theta_total=0.0,
                               flex_mode='fixed'):
        """
        Automatically compute (base_yaw, ext_yaw) so that the y-axis of
        endo_flex (and end_cylinder) is always kept horizontal (parallel
        to the base_link / world XY horizon plane), with +x pointing
        downward in the vertical plane when unsteered.

        Parameters
        ----------
        q : array-like (6,)
            Current UR5 joint angles in radians.
        steer_yaw : float
            User steering angle offset (rad) from the downward vertical plane
            (adjusted via A/D keys).
        theta_total : float
            Total bending angle (rad) across the continuum segments.
        flex_mode : str
            'fixed' keeps end_cylinder's y-axis strictly on the horizon for
            any steer_yaw and theta_total by coordinating base_yaw & ext_yaw;
            'free' keeps yaw_base's y-axis horizon-compensated via base_yaw
            with ext_yaw = 0.
        """
        R_0, _ = self.flex_base_frame_in_base(q)

        # 1. Base twist angle psi_h around start_cylinder Z so yaw_base has:
        #      y-axis horizontal: (R_0 @ rot_z(psi_h))[2, 1] == 0
        #      +x-axis downward : (R_0 @ rot_z(psi_h))[2, 0] <= 0
        if math.hypot(R_0[2, 0], R_0[2, 1]) < 1e-6:
            psi_h = 0.0
        else:
            psi_h = math.atan2(-R_0[2, 1], -R_0[2, 0])

        if flex_mode == 'free':
            base_yaw = _wrap_pi(psi_h + steer_yaw)
            ext_yaw = 0.0
            return base_yaw, ext_yaw

        # 2. In 'fixed' mode (or general horizon-locked mode), the bending
        #    plane in start_cylinder frame is alpha = psi_h + steer_yaw, and
        #    R_end = R_0 @ rot_z(alpha) @ rot_y(theta_total) @ rot_z(-ext_yaw).
        #    Choose gamma = -ext_yaw so end_cylinder's y-axis is strictly
        #    horizontal (R_end[2, 1] == 0) and oriented consistently with
        #    the horizon-leveled base y-axis (no 180-deg flip past vertical).
        R_horiz_base = R_0 @ _rot_z(psi_h)
        y_horiz_ref = R_horiz_base[:, 1]

        alpha = psi_h + steer_yaw
        R_b = R_0 @ _rot_z(alpha) @ _rot_y(theta_total)

        if math.hypot(R_b[2, 0], R_b[2, 1]) < 1e-6:
            gamma = -steer_yaw
        else:
            gamma = math.atan2(-R_b[2, 1], -R_b[2, 0])
            y_end = -math.sin(gamma) * R_b[:, 0] + math.cos(gamma) * R_b[:, 1]
            if float(np.dot(y_end, y_horiz_ref)) < 0.0:
                gamma = math.atan2(R_b[2, 1], R_b[2, 0])

        ext_yaw = _wrap_pi(-gamma)
        base_yaw = _wrap_pi(alpha + gamma)
        return base_yaw, ext_yaw

    def solve_bird_eye_bend_angle(self, q, target_horizon_deg=30.0,
                                  max_bend_rad=math.pi / 2.0):
        """
        Compute the downward total bending angle theta_total (rad) required
        so that end_cylinder's viewing axis (+Z) makes a `target_horizon_deg`
        (30 deg) downward angle with the horizontal plane.
        """
        R_0, _ = self.flex_base_frame_in_base(q)
        # Current downward pitch angle of start_cylinder +Z below horizon:
        beta_0 = math.asin(float(np.clip(-R_0[2, 2], -1.0, 1.0)))
        theta_target = math.radians(float(target_horizon_deg)) - beta_0
        return float(np.clip(theta_target, 0.0, max_bend_rad))

    # ------------------------------------------------------------------
    @staticmethod
    def rotate_about(v, k, theta):
        """Rodrigues' rotation: rotate v about unit axis k by theta."""
        k = k / np.linalg.norm(k)
        return (v * np.cos(theta)
                + np.cross(k, v) * np.sin(theta)
                + k * np.dot(k, v) * (1.0 - np.cos(theta)))

    # ------------------------------------------------------------------
    # Solve wrist_1 so shoulder_lift + elbow + wrist_1 = -270 deg.
    # ------------------------------------------------------------------
    def solve_wrist1_for_out(self, q_in):
        """
        Return a copy of q_in (length-6 rad) with joint index 3 (wrist_1)
        adjusted so that:

            shoulder_lift + elbow + wrist_1 = -270 deg
        """
        q = np.array(q_in, dtype=float)
        target_sum = np.deg2rad(-270.0)
        q[3] = target_sum - q[1] - q[2]
        return q

    # ------------------------------------------------------------------
    # 5-DOF Wrist_1-Constrained Axial IK
    # ------------------------------------------------------------------
    def ik_along_axis(self, p_tip_target, axis_target, q_seed,
                      max_iter=15, tol=1e-7):
        """
        Solve for 6 joint angles q such that:
          1) p_tip(q) == p_tip_target (3D tip position on the target point)
          2) tube_axis(q) == axis_target (2D tube direction strictly aligned)
          3) q[1] + q[2] + q[3] == -270 deg (wrist_1 constraint satisfied inside IK)

        Because the wrist_1 constraint removes 1 DOF (leaving 5 free joints:
        q0, q1, q2, q4, q5) and a 3D tip position + 2D unit axis direction
        requires exactly 5 DOFs (leaving tube roll free), this system is
        exact and eliminates the lateral/angular drift caused by overwriting
        q[3] after 6-DOF IK.
        """
        p_tip_target = np.asarray(p_tip_target, dtype=float)
        axis_u = np.asarray(axis_target, dtype=float)
        nrm = np.linalg.norm(axis_u)
        if nrm < 1e-9:
            return None
        axis_u = axis_u / nrm

        p_rcm_target = p_tip_target - self.tube_length * axis_u
        target_sum = np.deg2rad(-270.0)

        # If target is far from seed, warm-start with KDL LMA
        q_init = np.asarray(q_seed, dtype=float).copy()
        _, _, p_tip_seed = self.tube_axis_in_base(q_init)
        if np.linalg.norm(p_tip_target - p_tip_seed) > 0.025:
            q_lma = self._kdl_ik_with_axis(p_tip_target, axis_u, q_init)
            if q_lma is not None:
                q_init = self.solve_wrist1_for_out(q_lma)
            max_iter = max(max_iter, 25)

        u = np.array([q_init[0], q_init[1], q_init[2],
                      q_init[4], q_init[5]], dtype=float)

        def u_to_q(u_vec):
            q1 = u_vec[1]
            q2 = u_vec[2]
            return np.array([
                u_vec[0],
                q1,
                q2,
                target_sum - q1 - q2,
                u_vec[3],
                u_vec[4],
            ], dtype=float)

        def residual(u_vec):
            q = u_to_q(u_vec)
            _, p_rcm, p_tip = self.tube_axis_in_base(q)
            return np.concatenate([
                p_tip - p_tip_target,
                p_rcm - p_rcm_target,
            ])

        eps = 1e-6
        eye5 = np.eye(5)
        for _ in range(max_iter):
            r = residual(u)
            err = float(np.linalg.norm(r))
            if err < tol:
                break
            J = np.zeros((6, 5), dtype=float)
            for j in range(5):
                du_j = np.zeros(5, dtype=float)
                du_j[j] = eps
                J[:, j] = (residual(u + du_j) - r) / eps
            # Damped least-squares (Levenberg-Marquardt step)
            JTJ = J.T @ J + 1e-8 * eye5
            du = -np.linalg.solve(JTJ, J.T @ r)
            max_step = float(np.max(np.abs(du)))
            if max_step > 0.25:
                du *= (0.25 / max_step)
            u = u + du

        r_final = residual(u)
        if float(np.linalg.norm(r_final)) > 1e-3:
            return None
        return u_to_q(u)

    def _kdl_ik_with_axis(self, p_tip_target, axis_new, q_seed):
        """Internal 6-DOF KDL LMA helper for warm-starting large steps."""
        f_cur = self.fk(q_seed)
        x_cur = np.array([f_cur.M.UnitX().x(),
                          f_cur.M.UnitX().y(),
                          f_cur.M.UnitX().z()])
        x_new = x_cur - np.dot(x_cur, axis_new) * axis_new
        if np.linalg.norm(x_new) < 1e-6:
            tmp = np.array([1.0, 0.0, 0.0])
            x_new = tmp - np.dot(tmp, axis_new) * axis_new
        x_new /= np.linalg.norm(x_new)
        y_new = np.cross(axis_new, x_new)

        R = PyKDL.Rotation(
            x_new[0], y_new[0], axis_new[0],
            x_new[1], y_new[1], axis_new[1],
            x_new[2], y_new[2], axis_new[2])
        R_np = np.array([[R[0, 0], R[0, 1], R[0, 2]],
                         [R[1, 0], R[1, 1], R[1, 2]],
                         [R[2, 0], R[2, 1], R[2, 2]]])

        link_origin = p_tip_target - R_np @ self.tip_local
        f_target = PyKDL.Frame(R, PyKDL.Vector(*link_origin))

        n = self.chain.getNrOfJoints()
        q_init = PyKDL.JntArray(n)
        for i in range(n):
            q_init[i] = float(q_seed[i])
        q_out = PyKDL.JntArray(n)

        if self.ik_solver.CartToJnt(q_init, f_target, q_out) < 0:
            return None
        return np.array([q_out[i] for i in range(n)])

    # ------------------------------------------------------------------
    # IK (Position-only and RCM-constrained, both using 5-DOF exact solver)
    # ------------------------------------------------------------------
    def ik_position_only(self, p_tip_target, q_seed, axis_target=None):
        """Translate p_tip to p_tip_target while keeping the tube axis parallel
        and satisfying shoulder_lift + elbow + wrist_1 = -270 deg."""
        if axis_target is None:
            axis_target, _, _ = self.tube_axis_in_base(q_seed)
        return self.ik_along_axis(p_tip_target, axis_target, q_seed)

    def ik_with_rcm(self, p_tip_target, rcm_world, q_seed):
        """Move p_tip to p_tip_target with the shaft passing strictly through
        rcm_world and satisfying shoulder_lift + elbow + wrist_1 = -270 deg."""
        v = np.asarray(p_tip_target, dtype=float) - np.asarray(rcm_world, dtype=float)
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            return None
        axis_new = v / d
        return self.ik_along_axis(p_tip_target, axis_new, q_seed)