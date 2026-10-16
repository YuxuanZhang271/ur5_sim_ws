#!/usr/bin/env python3
"""
KDL kinematics for the UR5 + endoscope chain.

Provides:
  - Chain setup from a URDF string
  - Forward kinematics (points / axes in the base frame)
  - Inverse kinematics: position-only, and RCM-constrained
  - Tube-axis geometry (two points on the endoscope mesh)
  - Auto-solve of wrist_1 so the wrist joint points out from the base
"""

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


class EndoscopeKinematics:
    """KDL chain + FK/IK helpers for the UR5 + endoscope."""

    def __init__(self, base_link, tip_link,
                 rcm_point_local, tip_local, logger=None):
        self.base_link = base_link
        self.tip_link = tip_link
        self.rcm_point_local = np.asarray(rcm_point_local, dtype=float)
        self.tip_local = np.asarray(tip_local, dtype=float)
        self.logger = logger

        self.chain = None
        self.fk_solver = None
        self.ik_solver = None

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

    def point_in_base(self, q, p_local):
        f = self.fk(q)
        p = f * PyKDL.Vector(float(p_local[0]),
                             float(p_local[1]),
                             float(p_local[2]))
        return np.array([p.x(), p.y(), p.z()])

    def tool_axis_in_base(self, q):
        f = self.fk(q)
        z = f.M.UnitZ()
        return np.array([z.x(), z.y(), z.z()])

    def tube_axis_in_base(self, q):
        """Return (unit_axis, p_rcm_world, p_tip_world)."""
        p_rcm = self.point_in_base(q, self.rcm_point_local)
        p_tip = self.point_in_base(q, self.tip_local)
        v = p_tip - p_rcm
        n = np.linalg.norm(v)
        if n < 1e-9:
            return self.tool_axis_in_base(q), p_rcm, p_tip
        return v / n, p_rcm, p_tip

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
    #
    # shoulder_lift, elbow and wrist_1 all rotate around parallel axes
    # (world Y of the base).  Summing them to -270 deg makes the wrist_1
    # joint axis point "out" of the supporter/base in a consistent way,
    # regardless of the other joints.
    # ------------------------------------------------------------------
    def solve_wrist1_for_out(self, q_in):
        """
        Return a copy of q_in (length-6 rad) with joint index 3 (wrist_1)
        adjusted so that:

            shoulder_lift + elbow + wrist_1 = -270 deg

        This keeps the wrist_1 joint pointing out from the supporter
        and base (they all rotate around the same axis).
        """
        q = np.array(q_in, dtype=float)
        target_sum = np.deg2rad(-270.0)
        q[3] = target_sum - q[1] - q[2]
        return q

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------
    def ik_position_only(self, p_tip_target, q_seed):
        n = self.chain.getNrOfJoints()
        q_init = PyKDL.JntArray(n)
        for i in range(n):
            q_init[i] = float(q_seed[i])
        q_out = PyKDL.JntArray(n)

        f_cur = self.fk(q_seed)
        tip_offset_world = f_cur.M * PyKDL.Vector(*self.tip_local)
        link_origin = PyKDL.Vector(*p_tip_target) - tip_offset_world
        f_target = PyKDL.Frame(f_cur.M, link_origin)

        if self.ik_solver.CartToJnt(q_init, f_target, q_out) < 0:
            return None
        return np.array([q_out[i] for i in range(n)])

    def ik_with_rcm(self, p_tip_target, rcm_world, q_seed):
        v = p_tip_target - rcm_world
        d = np.linalg.norm(v)
        if d < 1e-6:
            return None
        axis_new = v / d

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
