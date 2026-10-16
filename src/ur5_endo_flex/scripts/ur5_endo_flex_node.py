#!/usr/bin/env python3
"""
ur5_endo_flex_node.py — unified ROS 2 control node for the integrated
UR5 + endoscope + endo-flex continuum system.

Arm (ur5_endoscope side):
    Same KDL kinematics as the upstream ur5_endoscope node (RCM, FK,
    IK position-only and IK-with-RCM).  Always-on wrist_1 auto-solve
    so shoulder_lift + elbow + wrist_1 = -270 deg (AGENT constraint 1).

Flex (endo_flex_control side):
    Continuum parameterised by base_yaw, arc_len, ext_yaw, curvature κ.
    - Automatic horizon twist compensation: at every control tick,
      base_yaw and ext_yaw are automatically adjusted from the current
      UR5 FK so that the y-axis of endo_flex (and end_cylinder) is
      always kept horizontal (parallel to the world horizon plane).
    - JointState for the 7 segment groups is published on /joint_states
      (UR joint names do not overlap, so robot_state_publisher merges).
    - Float64 position commands are simultaneously published on
      /flex/base_yaw, /flex/ext_yaw, /flex/ext_pitch, and
      /flex/ext_yaw_back (bridged via ros_gz_bridge to Gazebo's
      JointPositionController plugins) so that the physical simulation
      and distal camera in Gazebo rotate and bend synchronously with
      RViz.

Modes:
    arm_mode ∈ {linear, rcm}     default = linear
    flex_mode ∈ {free, fixed}    default = fixed
    Press-and-hold keyboard / GUI button control drives a 20 Hz tick
    that applies velocity-limited commands per held key.

Semi-auto:
    ctrl+H   HOME       arm → HOME_DEG; flex → straight, horizon-leveled
    ctrl+R   RESET      HOME + arm_mode=linear, flex_mode=fixed
    ctrl+B   BIRD-EYE   flex bends downward in the vertical plane so
                        end_cylinder has a 30 deg angle with the horizon
    ctrl+P   PULL OUT   (RCM mode only) two-stage pull-out:
                        (a) fully straighten endo_flex and align the
                            endoscope direction to the fixed central
                            RCM axis (keeping current insertion depth),
                        (b) continuously retract along the fixed central
                            axis until the endo_flex end (end_cylinder)
                            is 10 cm behind the fixed RCM point.

Topics
------
    Sub : /joint_states          (UR joint positions from ros2_control)
    Pub : <controller_topic>     JointTrajectory for arm
    Pub : /joint_states          JointState for the flex joints (RViz TF)
    Pub : /flex/base_yaw         Float64 for Gazebo base_yaw joint
    Pub : /flex/ext_yaw          Float64 for Gazebo ext_yaw_1..7 joints
    Pub : /flex/ext_pitch        Float64 for Gazebo ext_pitch_1..7 joints
    Pub : /flex/ext_yaw_back     Float64 for Gazebo ext_yaw_back_1..7 joints
    Pub : <marker_topic>         MarkerArray (visualization)
"""

import math
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, ColorRGBA, Float64
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

try:
    from pynput import keyboard as pynput_keyboard
    HAS_PYNPUT = True
except Exception:
    HAS_PYNPUT = False

from endoscope_kinematics import (
    EndoscopeKinematics, JOINT_NAMES, HOME_DEG,
)


# ----------------------------- Flex constants -----------------------------
FLEX_N = 8
FLEX_SEG_LEN = 0.008
FLEX_ARC_MIN = 0.030
FLEX_ARC_MAX = 0.050
FLEX_MAX_BEND = math.pi / 2.0
FLEX_TENDON_R = 0.004
FLEX_TENDONS = [(0, 'R'), (90, 'G'), (180, 'B'), (270, 'Y')]
BIRD_EYE_HORIZON_DEG = 30.0
PULL_OUT_BEHIND_RCM_M = 0.10


class Ur5EndoFlexNode(Node):
    """Unified UR5 + endoscope + endo-flex control node."""

    def __init__(self):
        super().__init__('ur5_endo_flex_node')

        # ---------- Parameters ----------
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'endoscope_link')
        self.declare_parameter(
            'controller_topic',
            '/scaled_joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('step_size_mm', 5.0)
        self.declare_parameter('tilt_step_deg', 5.0)
        self.declare_parameter('traj_duration_s', 1.0)

        self.declare_parameter('rcm_point_in_tip_frame',
                               [0.0, 0.0, -0.30])
        self.declare_parameter('endoscope_tip_in_link_frame',
                               [0.0, 0.0, 0.20195])

        self.declare_parameter('marker_topic', '/endoscope_markers')
        self.declare_parameter('marker_publish_period', 0.1)
        self.declare_parameter('markers_enabled', True)
        self.declare_parameter('rcm_cone_half_angle_deg', 20.0)
        self.declare_parameter('rcm_cone_length', 0.20)
        self.declare_parameter('rcm_sphere_diameter', 0.03)
        self.declare_parameter('endoscope_line_width', 0.006)
        self.declare_parameter('endoscope_line_overshoot', 0.05)

        # Flex parameters (read-only here; used by the formulas)
        self.declare_parameter('flex_N', FLEX_N)
        self.declare_parameter('flex_SEG_LEN', FLEX_SEG_LEN)
        self.declare_parameter('flex_ARC_MIN', FLEX_ARC_MIN)
        self.declare_parameter('flex_ARC_MAX', FLEX_ARC_MAX)
        self.declare_parameter('flex_TENDON_R', FLEX_TENDON_R)

        # Continuous-control velocity limits
        self.declare_parameter('arm_linear_vel_m_s', 0.050)
        self.declare_parameter('arm_axis_vel_m_s', 0.050)
        self.declare_parameter('arm_tilt_vel_deg_s', 30.0)
        self.declare_parameter('flex_yaw_vel_rad_s', 0.5)
        self.declare_parameter('flex_curv_vel_1pm_s', 5.0)
        self.declare_parameter('control_tick_hz', 20.0)

        # Optional: disable the pynput keyboard listener. Used when the
        # node is launched headless alongside the simulation so that the
        # GUI launch's instance remains the only keyboard consumer.
        self.declare_parameter('enable_keyboard', True)

        # Cache commonly used values
        self.base_link = self.get_parameter('base_link').value
        self.tip_link = self.get_parameter('tip_link').value
        self.controller_topic = self.get_parameter('controller_topic').value
        self.traj_duration = float(self.get_parameter('traj_duration_s').value)

        self.rcm_point_local = np.array(
            self.get_parameter('rcm_point_in_tip_frame').value, dtype=float)
        self.tip_local = np.array(
            self.get_parameter('endoscope_tip_in_link_frame').value,
            dtype=float)

        self.marker_topic = self.get_parameter('marker_topic').value
        self.markers_enabled = bool(
            self.get_parameter('markers_enabled').value)
        self.rcm_cone_half_angle = np.deg2rad(
            float(self.get_parameter('rcm_cone_half_angle_deg').value))
        self.rcm_cone_length = float(
            self.get_parameter('rcm_cone_length').value)
        self.rcm_sphere_diameter = float(
            self.get_parameter('rcm_sphere_diameter').value)
        self.line_width = float(
            self.get_parameter('endoscope_line_width').value)
        self.line_overshoot = float(
            self.get_parameter('endoscope_line_overshoot').value)

        self.arm_linear_vel = float(
            self.get_parameter('arm_linear_vel_m_s').value)
        self.arm_axis_vel = float(
            self.get_parameter('arm_axis_vel_m_s').value)
        self.arm_tilt_vel = np.deg2rad(
            float(self.get_parameter('arm_tilt_vel_deg_s').value))
        self.flex_yaw_vel = float(
            self.get_parameter('flex_yaw_vel_rad_s').value)
        self.flex_curv_vel = float(
            self.get_parameter('flex_curv_vel_1pm_s').value)
        self.tick_hz = float(
            self.get_parameter('control_tick_hz').value)
        self.dt = 1.0 / self.tick_hz

        # ---------- Kinematics ----------
        self.kin = EndoscopeKinematics(
            base_link=self.base_link,
            tip_link=self.tip_link,
            rcm_point_local=self.rcm_point_local,
            tip_local=self.tip_local,
            logger=self.get_logger(),
        )

        # ---------- State ----------
        self._state_lock = threading.Lock()

        self.joint_positions = np.zeros(6)
        self.joint_received = False

        self.rcm_enabled = False
        self.rcm_world_point = None
        self.rcm_ref_axis = None
        self.rcm_ref_depth = None

        self.arm_mode = 'linear'    # 'linear' | 'rcm'
        self.flex_mode = 'fixed'    # 'free'   | 'fixed'

        # Flex state — start straight (curvature=0) per spec.
        # flex_steer_yaw is the user-commanded steering offset (A/D) relative
        # to the downward vertical plane; flex_base_yaw and flex_ext_yaw are
        # automatically updated to keep the y-axis horizontal.
        self.flex_steer_yaw = 0.0
        self.flex_base_yaw = 0.0
        self.flex_arc_len = FLEX_ARC_MAX
        self.flex_ext_yaw = 0.0
        self.flex_curvature = 0.0

        # Active two-stage PULL OUT state machine (None when idle).
        self._pull_out_state = None

        # Continuous-control key state (keyboard + GUI buttons share this)
        self._key_lock = threading.Lock()
        self._key_state = {
            'arm_up': False, 'arm_down': False,
            'arm_left': False, 'arm_right': False,
            'arm_fwd': False, 'arm_bwd': False,
            'flex_w': False, 'flex_s': False,
            'flex_a': False, 'flex_d': False,
        }
        # Ctrl modifier tracking (for ctrl+letter combos).
        self._ctrl_held = False

        # ---------- ROS I/O ----------
        pub_qos = QoSProfile(
            depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.traj_pub = self.create_publisher(
            JointTrajectory, self.controller_topic, pub_qos)

        self.flex_js_pub = self.create_publisher(
            JointState, '/joint_states', pub_qos)

        # Gazebo JointPositionController command publishers (bridged via ros_gz_bridge)
        self.gz_base_yaw_pub = self.create_publisher(
            Float64, '/flex/base_yaw', pub_qos)
        self.gz_ext_yaw_pub = self.create_publisher(
            Float64, '/flex/ext_yaw', pub_qos)
        self.gz_ext_pitch_pub = self.create_publisher(
            Float64, '/flex/ext_pitch', pub_qos)
        self.gz_ext_yaw_back_pub = self.create_publisher(
            Float64, '/flex/ext_yaw_back', pub_qos)

        js_sub_qos = QoSProfile(
            depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, js_sub_qos)

        urdf_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(
            String, '/robot_description', self._robot_description_cb, urdf_qos)

        marker_qos = QoSProfile(
            depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.marker_topic, marker_qos)

        marker_period = float(
            self.get_parameter('marker_publish_period').value)
        self.create_timer(marker_period, self._publish_markers)

        # Continuous-control tick timer.
        self.create_timer(self.dt, self._tick)

        # Publish the initial (straight) flex state.
        self._publish_flex_joint_states()

        # ---------- Keyboard listener (optional) ----------
        self._key_listener = None
        enable_keyboard = bool(self.get_parameter('enable_keyboard').value)
        if not enable_keyboard:
            self.get_logger().info(
                'enable_keyboard=false; pynput listener disabled.')
        elif HAS_PYNPUT:
            try:
                self._key_listener = pynput_keyboard.Listener(
                    on_press=self._on_press,
                    on_release=self._on_release,
                )
                self._key_listener.daemon = True
                self._key_listener.start()
                self.get_logger().info('pynput keyboard listener started.')
            except Exception as e:
                self.get_logger().warn(
                    f'pynput keyboard listener failed to start: {e}. '
                    f'GUI buttons still work.')
        else:
            self.get_logger().warn(
                'pynput not available; keyboard control disabled. '
                'Use the GUI buttons.')

        self.get_logger().info(
            f'Control node ready. Modes: arm={self.arm_mode}, '
            f'flex={self.flex_mode}. Trajectory on {self.controller_topic}.')

    # ==================================================================
    # ROS callbacks
    # ==================================================================
    def _joint_state_cb(self, msg: JointState):
        """Update UR joint positions from /joint_states (flex joints skipped)."""
        with self._state_lock:
            for i, name in enumerate(JOINT_NAMES):
                if name in msg.name:
                    self.joint_positions[i] = msg.position[
                        msg.name.index(name)]
            self.joint_received = True

    def _robot_description_cb(self, msg: String):
        if self.kin.chain is None and msg.data:
            self.kin.setup_from_urdf(msg.data)

    # ==================================================================
    # Continuous control tick (called by timer at ~control_tick_hz)
    # ==================================================================
    def _tick(self):
        """Apply velocity-based commands for any currently-held keys/buttons
        or active semi-automatic sequences, and auto-level endo_flex twist."""
        with self._key_lock:
            ks = dict(self._key_state)

        any_manual_key = any(ks.values())
        with self._state_lock:
            pull_out_active = self._pull_out_state is not None
            if pull_out_active and any_manual_key:
                self._pull_out_state = None
                pull_out_active = False
                self.get_logger().info(
                    'PULL OUT cancelled by manual key input.')

        if pull_out_active:
            self._step_pull_out()
            self._publish_flex_joint_states()
            return

        # Aggregate per-DoF increments for this tick.
        # In linear mode: arrows move in (vertical, horiz perp tube, none).
        #        — [/] moves along tube axis.
        # In RCM mode: arrows tilt around RCM in world frame.
        #        — [/] moves along tube axis (in/out).
        arm_dx = 0.0    # tube axis (forward/backward)
        arm_dy = 0.0    # horizontal perpendicular to tube (left/right)
        arm_dz = 0.0    # world vertical (up/down)
        arm_tilt_right = 0.0
        arm_tilt_up = 0.0

        with self._state_lock:
            mode = self.arm_mode

        if mode == 'rcm':
            # Arrow keys → tilt around RCM (world frame)
            if ks['arm_up']:    arm_tilt_up    += self.arm_tilt_vel * self.dt
            if ks['arm_down']:  arm_tilt_up    -= self.arm_tilt_vel * self.dt
            if ks['arm_right']: arm_tilt_right += self.arm_tilt_vel * self.dt
            if ks['arm_left']:  arm_tilt_right -= self.arm_tilt_vel * self.dt
        else:
            # Linear mode → arrows do Cartesian position deltas.
            if ks['arm_up']:    arm_dz += self.arm_linear_vel * self.dt
            if ks['arm_down']:  arm_dz -= self.arm_linear_vel * self.dt
            if ks['arm_right']: arm_dy += self.arm_linear_vel * self.dt
            if ks['arm_left']:  arm_dy -= self.arm_linear_vel * self.dt

        # [/] always means along the tube axis (forward/backward).
        #   [  = fwd key  → along -axis (toward distal tip)
        #   ]  = bwd key  → along +axis (toward proximal mount / out)
        if ks['arm_fwd']: arm_dx -= self.arm_axis_vel * self.dt
        if ks['arm_bwd']: arm_dx += self.arm_axis_vel * self.dt

        if arm_dx or arm_dy or arm_dz:
            self._arm_linear_step(arm_dx, arm_dy, arm_dz,
                                  traj_duration=self.dt)
        if arm_tilt_up or arm_tilt_right:
            self._arm_rcm_tilt(np.rad2deg(arm_tilt_right),
                               np.rad2deg(arm_tilt_up),
                               traj_duration=self.dt)

        # Flex commands
        flex_updated = False
        if ks['flex_w']:
            self._flex_curvature_step(-self.flex_curv_vel * self.dt)
            flex_updated = True
        if ks['flex_s']:
            self._flex_curvature_step(+self.flex_curv_vel * self.dt)
            flex_updated = True
        if ks['flex_a']:
            self._flex_yaw_step(+self.flex_yaw_vel * self.dt)
            flex_updated = True
        if ks['flex_d']:
            self._flex_yaw_step(-self.flex_yaw_vel * self.dt)
            flex_updated = True

        # Always run _publish_flex_joint_states() on every tick so that
        # automatic horizon twist compensation tracks any UR5 arm motion
        # and keeps RViz + Gazebo JointPositionControllers synchronized.
        if not flex_updated:
            self._publish_flex_joint_states()

    # ==================================================================
    # Arm commands
    # ==================================================================
    def _arm_linear_step(self, dx_m, dy_m, dz_m, traj_duration=None):
        """Move tool0 by (dx, dy, dz) in the linear-mode direction frame."""
        if not self._arm_ready():
            return
        if traj_duration is None:
            traj_duration = self.dt
        with self._state_lock:
            q_now = self.joint_positions.copy()
            rcm_on = self.rcm_enabled
            rcm_pt = (None if self.rcm_world_point is None
                      else self.rcm_world_point.copy())
        try:
            axis, _, p_tip = self.kin.tube_axis_in_base(q_now)
        except Exception as e:
            self.get_logger().warn(f'FK failed in linear step: {e}')
            return

        world_z = np.array([0.0, 0.0, 1.0])
        tube_axis = axis / max(np.linalg.norm(axis), 1e-9)
        fwd = tube_axis
        up = world_z

        # Horizontal projection of tube axis (used for left/right).
        proj = tube_axis - np.dot(tube_axis, world_z) * world_z
        pn = np.linalg.norm(proj)
        if pn < 1e-6:
            left = np.array([0.0, 1.0, 0.0])
        else:
            proj = proj / pn
            left = np.cross(world_z, proj)
            ln = np.linalg.norm(left)
            if ln < 1e-9:
                left = np.array([0.0, 1.0, 0.0])
            else:
                left = left / ln

        delta = up * dz_m + left * dy_m + fwd * dx_m
        if np.linalg.norm(delta) < 1e-9:
            return
        p_target = p_tip + delta

        if rcm_on and rcm_pt is not None:
            q_target = self.kin.ik_with_rcm(p_target, rcm_pt, q_now)
        else:
            q_target = self.kin.ik_position_only(p_target, q_now)
        if q_target is None:
            return
        q_target = self._auto_solve_wrist1(q_target)
        self._send_trajectory(q_target, traj_duration=traj_duration)

    def _arm_rcm_tilt(self, angle_right_deg, angle_up_deg,
                      traj_duration=None):
        """Tilt the tube around the RCM point (RCM-mode arrow keys)."""
        if not self._arm_ready():
            return
        if traj_duration is None:
            traj_duration = self.dt
        with self._state_lock:
            rcm_on = self.rcm_enabled
            rcm_pt = (None if self.rcm_world_point is None
                      else self.rcm_world_point.copy())
            ref_axis = (None if self.rcm_ref_axis is None
                        else self.rcm_ref_axis.copy())
            q_now = self.joint_positions.copy()
        if not rcm_on or rcm_pt is None or ref_axis is None:
            return

        try:
            axis, _, p_tip = self.kin.tube_axis_in_base(q_now)
        except Exception:
            return
        distance = float(np.linalg.norm(p_tip - rcm_pt))
        if distance < 1e-6:
            return

        z_world = np.array([0.0, 0.0, 1.0])
        new_axis = axis.copy()

        if angle_up_deg != 0.0:
            theta = np.deg2rad(float(angle_up_deg))
            a_h = np.array([new_axis[0], new_axis[1], 0.0])
            n = np.linalg.norm(a_h)
            if n < 1e-6:
                pitch_axis = np.array([1.0, 0.0, 0.0])
            else:
                a_h = a_h / n
                pitch_axis = np.cross(a_h, z_world)
                pitch_axis = pitch_axis / max(np.linalg.norm(pitch_axis), 1e-9)
            new_axis = self.kin.rotate_about(new_axis, pitch_axis, -theta)
            new_axis = new_axis / max(np.linalg.norm(new_axis), 1e-9)

        if angle_right_deg != 0.0:
            theta = np.deg2rad(float(angle_right_deg))
            new_axis = self.kin.rotate_about(new_axis, z_world, -theta)
            new_axis = new_axis / max(np.linalg.norm(new_axis), 1e-9)

        if ref_axis is not None:
            cos_a = float(np.clip(np.dot(new_axis, ref_axis), -1.0, 1.0))
            new_deg = float(np.rad2deg(np.arccos(cos_a)))
            limit_deg = float(np.rad2deg(self.rcm_cone_half_angle))
            if new_deg > limit_deg:
                return

        p_target = rcm_pt + distance * new_axis
        q_target = self.kin.ik_with_rcm(p_target, rcm_pt, q_now)
        if q_target is None:
            return
        q_target = self._auto_solve_wrist1(q_target)
        self._send_trajectory(q_target, traj_duration=traj_duration)

    def _auto_solve_wrist1(self, q):
        """Override wrist_1 so shoulder_lift + elbow + wrist_1 = -270°."""
        q = np.array(q, dtype=float).copy()
        target_sum = np.deg2rad(-270.0)
        q[3] = target_sum - q[1] - q[2]
        return q

    def _arm_ready(self):
        with self._state_lock:
            ready = self.kin.ready and self.joint_received
        if not ready:
            return False
        return True

    # ==================================================================
    # Flex commands & automatic horizon-keeping twist
    # ==================================================================
    def _flex_curvature_step(self, dkappa):
        with self._state_lock:
            self.flex_curvature = max(0.0, self.flex_curvature + dkappa)
            eff = max(1e-6, self.flex_arc_len - FLEX_SEG_LEN)
            kappa_max = FLEX_MAX_BEND / eff
            self.flex_curvature = min(kappa_max, self.flex_curvature)
        self._publish_flex_joint_states()

    def _flex_yaw_step(self, dyaw_rad):
        with self._state_lock:
            self.flex_steer_yaw = (
                (self.flex_steer_yaw + dyaw_rad + math.pi)
                % (2.0 * math.pi) - math.pi
            )
        self._publish_flex_joint_states()

    def _update_auto_horizon_twist_unlocked(self, theta_total):
        """Recompute (flex_base_yaw, flex_ext_yaw) so endo_flex's y-axis is
        always kept on the horizontal plane. Must be called with _state_lock held."""
        if self.kin.ready and self.joint_received:
            try:
                base_yaw, ext_yaw = self.kin.solve_horizon_flex_yaw(
                    q=self.joint_positions,
                    steer_yaw=self.flex_steer_yaw,
                    theta_total=theta_total,
                    flex_mode=self.flex_mode,
                )
                self.flex_base_yaw = float(base_yaw)
                self.flex_ext_yaw = float(ext_yaw)
                return
            except Exception as e:
                self.get_logger().warn(
                    f'Auto horizon twist computation failed: {e}')

        # Fallback before KDL/joint_states are ready:
        if self.flex_mode == 'fixed':
            self.flex_base_yaw = float(self.flex_steer_yaw)
            self.flex_ext_yaw = float(-self.flex_steer_yaw)
        else:
            self.flex_base_yaw = float(self.flex_steer_yaw)
            self.flex_ext_yaw = 0.0

    def _publish_flex_joint_states(self):
        with self._state_lock:
            eff = max(1e-6, self.flex_arc_len - FLEX_SEG_LEN)
            kappa = max(0.0, self.flex_curvature)
            theta_total = min(FLEX_MAX_BEND, eff * kappa)
            self._update_auto_horizon_twist_unlocked(theta_total)
            base_yaw = float(self.flex_base_yaw)
            ext_yaw = float(self.flex_ext_yaw)

        theta_seg = float(theta_total / (FLEX_N - 1))
        d_seg = float(eff / (FLEX_N - 1))

        # 1. Publish JointState to /joint_states for robot_state_publisher (RViz TF).
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = ['base_yaw']
        positions = [base_yaw]
        for i in range(1, FLEX_N):
            names.append(f'ext_yaw_{i}')
            positions.append(ext_yaw)
            names.append(f'ext_pitch_{i}')
            positions.append(theta_seg)
            names.append(f'ext_yaw_back_{i}')
            positions.append(-ext_yaw)
            names.append(f'extension_{i}')
            positions.append(d_seg)
        msg.name = names
        msg.position = [float(x) for x in positions]
        self.flex_js_pub.publish(msg)

        # 2. Publish Float64 commands to Gazebo JointPositionControllers so the
        #    physical simulation and distal camera move identically to RViz.
        self.gz_base_yaw_pub.publish(Float64(data=base_yaw))
        self.gz_ext_yaw_pub.publish(Float64(data=ext_yaw))
        self.gz_ext_pitch_pub.publish(Float64(data=theta_seg))
        self.gz_ext_yaw_back_pub.publish(Float64(data=-ext_yaw))

    # ==================================================================
    # Mode toggles
    # ==================================================================
    def toggle_arm_mode(self):
        with self._state_lock:
            self._pull_out_state = None
            if self.arm_mode == 'linear':
                self.arm_mode = 'rcm'
                self._capture_rcm_unlocked()
                self.get_logger().info('Arm mode -> RCM (RCM captured).')
            else:
                self.arm_mode = 'linear'
                self.rcm_enabled = False
                self.rcm_ref_axis = None
                self.rcm_ref_depth = None
                self.get_logger().info('Arm mode -> linear (RCM disabled).')

    def _capture_rcm_unlocked(self):
        """Enable RCM (must be called with _state_lock held)."""
        if not (self.kin.ready and self.joint_received):
            self.rcm_enabled = False
            return
        q = self.joint_positions.copy()
        axis, p_rcm, p_tip = self.kin.tube_axis_in_base(q)
        self.rcm_world_point = p_rcm
        self.rcm_ref_axis = axis
        self.rcm_ref_depth = float(np.dot(p_tip - p_rcm, axis))
        self.rcm_enabled = True

    def toggle_flex_mode(self):
        with self._state_lock:
            self.flex_mode = 'free' if self.flex_mode == 'fixed' else 'fixed'
            mode = self.flex_mode
        self._publish_flex_joint_states()
        self.get_logger().info(f'Flex mode -> {mode}.')

    # ==================================================================
    # Semi-automatic commands
    # ==================================================================
    def cmd_home(self):
        """HOME: arm to HOME_DEG (with wrist_1 auto-solved); flex straight."""
        with self._state_lock:
            self._pull_out_state = None
            self.flex_steer_yaw = 0.0
            self.flex_arc_len = FLEX_ARC_MAX
            self.flex_curvature = 0.0
        self._publish_flex_joint_states()

        q_target = np.deg2rad(np.array(HOME_DEG, dtype=float))
        q_target = self._auto_solve_wrist1(q_target)
        self._send_trajectory(q_target, traj_duration=3.0)

    def cmd_reset(self):
        """RESET: HOME + arm_mode=linear, flex_mode=fixed; clear RCM."""
        with self._state_lock:
            self._pull_out_state = None
            self.arm_mode = 'linear'
            self.flex_mode = 'fixed'
            self.rcm_enabled = False
            self.rcm_ref_axis = None
            self.rcm_ref_depth = None
        self.cmd_home()

    def cmd_bird_eye_view(self):
        """BIRD-EYE VIEW: bend endo_flex downward in the vertical plane so
        that end_cylinder has a 30 deg downward angle with the horizon plane,
        while keeping the y-axis horizontal."""
        if not (self.kin.ready and self.joint_received):
            self.get_logger().warn(
                'BIRD-EYE VIEW waiting for kinematics and /joint_states.')
            return

        with self._state_lock:
            self._pull_out_state = None
            q_now = self.joint_positions.copy()
            eff = max(1e-6, self.flex_arc_len - FLEX_SEG_LEN)
            theta_target = self.kin.solve_bird_eye_bend_angle(
                q=q_now,
                target_horizon_deg=BIRD_EYE_HORIZON_DEG,
                max_bend_rad=FLEX_MAX_BEND,
            )
            self.flex_steer_yaw = 0.0
            self.flex_curvature = float(theta_target / eff)

        self._publish_flex_joint_states()
        self.get_logger().info(
            f'BIRD-EYE VIEW: bent downward to {math.degrees(theta_target):.1f} deg '
            f'(30.0 deg angle with horizon plane).')

    def cmd_pull_out(self):
        """PULL OUT (RCM mode only):
        Stage (a): Fully straighten endo_flex and align the endoscope shaft
                   direction to the captured RCM fixed central axis (keeping
                   current insertion depth).
        Stage (b): Continuously retract outward along the fixed central axis
                   until the endo_flex end (end_cylinder) is 10 cm behind
                   the fixed RCM point.
        """
        with self._state_lock:
            rcm_on = (self.arm_mode == 'rcm' and self.rcm_enabled)
            rcm_pt = (None if self.rcm_world_point is None
                      else self.rcm_world_point.copy())
            ref_axis = (None if self.rcm_ref_axis is None
                        else self.rcm_ref_axis.copy())
            q_now = self.joint_positions.copy()

        if not (rcm_on and rcm_pt is not None and ref_axis is not None):
            self.get_logger().warn(
                'PULL OUT can only be triggered when RCM mode is ON.')
            return
        if not (self.kin.ready and self.joint_received):
            return

        try:
            axis_cur, _, p_tip_cur = self.kin.tube_axis_in_base(q_now)
            p_end_cur = self.kin.end_cylinder_in_base(q_now)
        except Exception as e:
            self.get_logger().warn(f'PULL OUT FK failed: {e}')
            return

        # Distance from rcm_pt to p_tip along current tube axis (preserves
        # current insertion depth during Stage a direction alignment).
        d_tip_align = float(np.linalg.norm(p_tip_cur - rcm_pt))
        if d_tip_align < 0.05:
            d_tip_align = float(self.kin.tube_length)

        # Constant distance along the shaft between p_tip (z=+0.20195) and
        # straight end_cylinder (z=-0.4645): ~0.66645 m.
        tip_from_end = float(np.linalg.norm(p_tip_cur - p_end_cur))

        with self._state_lock:
            # Step (a.1): Immediately straighten endo_flex.
            self.flex_steer_yaw = 0.0
            self.flex_curvature = 0.0
            self._pull_out_state = {
                'stage': 'align',
                'axis_cmd': axis_cur.copy(),
                'd_tip_align': d_tip_align,
                'tip_from_end': tip_from_end,
                's_end_cmd': None,
            }

        self._publish_flex_joint_states()
        self.get_logger().info(
            'PULL OUT started: Stage (a) straightening flex & aligning '
            'endoscope direction to fixed RCM central axis.')

    def _step_pull_out(self):
        """Execute one 20 Hz tick of the two-stage PULL OUT sequence."""
        if not self._arm_ready():
            return

        with self._state_lock:
            st = self._pull_out_state
            if st is None:
                return
            if not (self.arm_mode == 'rcm' and self.rcm_enabled
                    and self.rcm_world_point is not None
                    and self.rcm_ref_axis is not None):
                self._pull_out_state = None
                return
            rcm_pt = self.rcm_world_point.copy()
            ref_axis = self.rcm_ref_axis.copy()
            q_now = self.joint_positions.copy()
            # Keep flex straight throughout pull-out.
            self.flex_steer_yaw = 0.0
            self.flex_curvature = 0.0

        try:
            axis_cur, _, p_tip_cur = self.kin.tube_axis_in_base(q_now)
            p_end_cur = self.kin.end_cylinder_in_base(q_now)
        except Exception as e:
            self.get_logger().warn(f'PULL OUT FK failed: {e}')
            with self._state_lock:
                self._pull_out_state = None
            return

        # ---------------- Stage (a): Align direction to ref_axis ----------------
        if st['stage'] == 'align':
            axis_cmd = st['axis_cmd']
            cos_cmd = float(np.clip(np.dot(axis_cmd, ref_axis), -1.0, 1.0))
            err_cmd = math.acos(cos_cmd)
            step_rad = self.arm_tilt_vel * self.dt

            if err_cmd <= step_rad or err_cmd < 1e-3:
                axis_next = ref_axis.copy()
            else:
                rot_k = np.cross(axis_cmd, ref_axis)
                kn = float(np.linalg.norm(rot_k))
                if kn < 1e-9:
                    axis_next = ref_axis.copy()
                else:
                    axis_next = self.kin.rotate_about(
                        axis_cmd, rot_k / kn, step_rad)
                    axis_next /= max(float(np.linalg.norm(axis_next)), 1e-9)

            st['axis_cmd'] = axis_next
            p_tip_target = rcm_pt + st['d_tip_align'] * axis_next
            q_target = self.kin.ik_along_axis(p_tip_target, axis_next, q_now)
            if q_target is not None:
                q_target = self._auto_solve_wrist1(q_target)
                self._send_trajectory(q_target, traj_duration=self.dt)

            # Check actual alignment with ref_axis to transition to Stage (b)
            cos_actual = float(np.clip(np.dot(axis_cur, ref_axis), -1.0, 1.0))
            err_actual_deg = math.degrees(math.acos(cos_actual))
            if np.allclose(axis_next, ref_axis, atol=1e-3) and err_actual_deg < 1.0:
                s_end_cur = float(np.dot(p_end_cur - rcm_pt, ref_axis))
                st['stage'] = 'retract'
                st['s_end_cmd'] = s_end_cur
                self.get_logger().info(
                    f'PULL OUT Stage (a) aligned (err={err_actual_deg:.2f} deg). '
                    f'Stage (b) retracting along fixed central axis to 10 cm behind RCM.')
            return

        # ---------------- Stage (b): Retract along ref_axis ----------------
        if st['stage'] == 'retract':
            s_end_cur = float(np.dot(p_end_cur - rcm_pt, ref_axis))
            if st['s_end_cmd'] is None:
                st['s_end_cmd'] = s_end_cur

            s_end_cmd = float(st['s_end_cmd'])
            step_m = self.arm_axis_vel * self.dt
            if s_end_cmd + step_m >= PULL_OUT_BEHIND_RCM_M:
                s_end_next = PULL_OUT_BEHIND_RCM_M
            else:
                s_end_next = s_end_cmd + step_m
            st['s_end_cmd'] = s_end_next

            p_tip_target = (
                rcm_pt + (st['tip_from_end'] + s_end_next) * ref_axis
            )
            q_target = self.kin.ik_along_axis(p_tip_target, ref_axis, q_now)
            if q_target is not None:
                q_target = self._auto_solve_wrist1(q_target)
                self._send_trajectory(q_target, traj_duration=self.dt)

            if (s_end_next >= PULL_OUT_BEHIND_RCM_M - 1e-4
                    and s_end_cur >= PULL_OUT_BEHIND_RCM_M - 0.003):
                with self._state_lock:
                    self._pull_out_state = None
                self.get_logger().info(
                    f'PULL OUT complete: endo_flex end is '
                    f'{s_end_cur * 100.0:.1f} cm behind fixed RCM point.')

    # ==================================================================
    # Key state (used by keyboard listener AND GUI button bindings)
    # ==================================================================
    def set_key(self, key, down):
        with self._key_lock:
            if key in self._key_state:
                self._key_state[key] = bool(down)

    # ==================================================================
    # Keyboard callbacks (pynput)
    # ==================================================================
    def _on_press(self, key):
        # Track ctrl state.
        if key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
            with self._key_lock:
                self._ctrl_held = True
            return

        ch = getattr(key, 'char', None)
        with self._key_lock:
            ctrl_held = self._ctrl_held

        # Ctrl + letter → semi-auto commands.
        if ctrl_held and ch is not None:
            ch_l = ch.lower()
            if ch_l == 'm': self.toggle_arm_mode();   return
            if ch_l == 'n': self.toggle_flex_mode();  return
            if ch_l == 'h': self.cmd_home();          return
            if ch_l == 'r': self.cmd_reset();         return
            if ch_l == 'b': self.cmd_bird_eye_view(); return
            if ch_l == 'p': self.cmd_pull_out();      return

        # Arrow keys → arm.
        m = {
            pynput_keyboard.Key.up:    'arm_up',
            pynput_keyboard.Key.down:  'arm_down',
            pynput_keyboard.Key.left:  'arm_left',
            pynput_keyboard.Key.right: 'arm_right',
        }
        if key in m:
            self.set_key(m[key], True)
            return

        # Letter keys.
        if ch is not None:
            if ch == '[': self.set_key('arm_fwd', True); return
            if ch == ']': self.set_key('arm_bwd', True); return
            if ch in ('w', 'W'): self.set_key('flex_w', True); return
            if ch in ('s', 'S'): self.set_key('flex_s', True); return
            if ch in ('a', 'A'): self.set_key('flex_a', True); return
            if ch in ('d', 'D'): self.set_key('flex_d', True); return

    def _on_release(self, key):
        if key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
            with self._key_lock:
                self._ctrl_held = False
            return

        m = {
            pynput_keyboard.Key.up:    'arm_up',
            pynput_keyboard.Key.down:  'arm_down',
            pynput_keyboard.Key.left:  'arm_left',
            pynput_keyboard.Key.right: 'arm_right',
        }
        if key in m:
            self.set_key(m[key], False)
            return

        ch = getattr(key, 'char', None)
        if ch is not None:
            if ch == '[': self.set_key('arm_fwd', False); return
            if ch == ']': self.set_key('arm_bwd', False); return
            if ch in ('w', 'W'): self.set_key('flex_w', False); return
            if ch in ('s', 'S'): self.set_key('flex_s', False); return
            if ch in ('a', 'A'): self.set_key('flex_a', False); return
            if ch in ('d', 'D'): self.set_key('flex_d', False); return

    # ==================================================================
    # Trajectory publication
    # ==================================================================
    def _send_trajectory(self, q_target, traj_duration=None):
        if traj_duration is None:
            traj_duration = self.traj_duration
        with self._state_lock:
            q_now = (self.joint_positions.copy() if self.joint_received
                     else np.asarray(q_target, dtype=float))

        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        pt0 = JointTrajectoryPoint()
        pt0.positions = [float(x) for x in q_now]
        pt0.velocities = [0.0] * len(q_now)
        pt0.accelerations = [0.0] * len(q_now)
        pt0.time_from_start = Duration(sec=0, nanosec=0)
        msg.points.append(pt0)

        pt1 = JointTrajectoryPoint()
        pt1.positions = [float(x) for x in q_target]
        pt1.velocities = [0.0] * len(q_target)
        pt1.accelerations = [0.0] * len(q_target)
        pt1.time_from_start = Duration(
            sec=int(traj_duration),
            nanosec=int((traj_duration % 1.0) * 1e9))
        msg.points.append(pt1)

        self.traj_pub.publish(msg)

    # ==================================================================
    # Visualization
    # ==================================================================
    def _make_delete_marker(self, ns: str, mid: int, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = self.base_link
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.action = Marker.DELETE
        return m

    def _publish_markers(self):
        if not self.markers_enabled:
            return
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()
        ready = self.kin.ready and self.joint_received
        if not ready:
            t = Marker()
            t.header.frame_id = self.base_link
            t.header.stamp = stamp
            t.ns = 'status'
            t.id = 100
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.z = 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.05
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            msg = []
            if not self.kin.ready:
                msg.append('waiting for /robot_description')
            if not self.joint_received:
                msg.append('waiting for /joint_states')
            t.text = 'markers idle: ' + '; '.join(msg)
            ma.markers.append(t)
            self.marker_pub.publish(ma)
            return

        # Clear startup status marker if it was previously shown.
        ma.markers.append(self._make_delete_marker('status', 100, stamp))

        with self._state_lock:
            q = self.joint_positions.copy()
            rcm_on = self.rcm_enabled
            rcm_stored = (None if self.rcm_world_point is None
                          else self.rcm_world_point.copy())
            ref_axis = (None if self.rcm_ref_axis is None
                        else self.rcm_ref_axis.copy())
            ref_depth = self.rcm_ref_depth

        try:
            axis, p_rcm_fk, p_tip_world = self.kin.tube_axis_in_base(q)
        except Exception as e:
            self.get_logger().error(f'FK failed in marker publish: {e}')
            return

        # 1. Central axis that ALWAYS follows the endoscope movement (always showing).
        cur_depth = float(np.dot(p_tip_world - p_rcm_fk, axis))
        cur_front = max(cur_depth, 0.05) + self.line_overshoot
        cur_back_pt = p_rcm_fk - self.line_overshoot * axis
        cur_front_pt = p_rcm_fk + cur_front * axis

        cur_line = Marker()
        cur_line.header.frame_id = self.base_link
        cur_line.header.stamp = stamp
        cur_line.ns = 'endoscope_current_axis'
        cur_line.id = 4
        cur_line.type = Marker.LINE_STRIP
        cur_line.action = Marker.ADD
        cur_line.scale.x = self.line_width
        cur_line.color = ColorRGBA(r=0.0, g=0.85, b=1.0, a=0.90)
        cur_line.pose.orientation.w = 1.0
        cur_line.points = [
            Point(x=float(cur_back_pt[0]),
                  y=float(cur_back_pt[1]),
                  z=float(cur_back_pt[2])),
            Point(x=float(cur_front_pt[0]),
                  y=float(cur_front_pt[1]),
                  z=float(cur_front_pt[2])),
        ]
        ma.markers.append(cur_line)

        # 2. RCM markers (fixed world central axis, RCM sphere, and cone area)
        #    ONLY appear when RCM mode is ON.
        use_fixed = (rcm_on and rcm_stored is not None
                     and ref_axis is not None)
        if use_fixed:
            cos_a = float(np.clip(np.dot(axis, ref_axis), -1.0, 1.0))
            drift_deg = float(np.rad2deg(np.arccos(cos_a)))
            drift_exceeded = drift_deg > np.rad2deg(self.rcm_cone_half_angle)

            fixed_depth = (float(ref_depth) if ref_depth is not None
                           else cur_depth)
            fixed_front = max(fixed_depth, 0.05) + self.line_overshoot
            line_back = rcm_stored - self.line_overshoot * ref_axis
            line_front = rcm_stored + fixed_front * ref_axis

            # Fixed central axis in world/base_link coordinates
            fixed_line = Marker()
            fixed_line.header.frame_id = self.base_link
            fixed_line.header.stamp = stamp
            fixed_line.ns = 'endoscope_line'
            fixed_line.id = 0
            fixed_line.type = Marker.LINE_STRIP
            fixed_line.action = Marker.ADD
            fixed_line.scale.x = self.line_width
            fixed_line.color = (ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
                                if drift_exceeded
                                else ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0))
            fixed_line.pose.orientation.w = 1.0
            fixed_line.points = [
                Point(x=float(line_back[0]),
                      y=float(line_back[1]),
                      z=float(line_back[2])),
                Point(x=float(line_front[0]),
                      y=float(line_front[1]),
                      z=float(line_front[2])),
            ]
            ma.markers.append(fixed_line)

            # RCM sphere
            sphere = Marker()
            sphere.header.frame_id = self.base_link
            sphere.header.stamp = stamp
            sphere.ns = 'rcm_point'
            sphere.id = 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(rcm_stored[0])
            sphere.pose.position.y = float(rcm_stored[1])
            sphere.pose.position.z = float(rcm_stored[2])
            sphere.pose.orientation.w = 1.0
            d = self.rcm_sphere_diameter
            sphere.scale.x = d
            sphere.scale.y = d
            sphere.scale.z = d
            sphere.color = (ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.55)
                            if drift_exceeded
                            else ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.45))
            ma.markers.append(sphere)

            # RCM cones (+ref_axis and -ref_axis)
            cone_color = (ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.28)
                          if drift_exceeded
                          else ColorRGBA(r=0.10, g=0.80, b=0.20, a=0.28))
            cone_fwd = self._build_cone_marker(
                apex=rcm_stored, direction=+ref_axis,
                length=self.rcm_cone_length,
                half_angle=self.rcm_cone_half_angle,
                color=cone_color, stamp=stamp, mid=2)
            cone_bwd = self._build_cone_marker(
                apex=rcm_stored, direction=-ref_axis,
                length=self.rcm_cone_length,
                half_angle=self.rcm_cone_half_angle,
                color=cone_color, stamp=stamp, mid=3)
            if cone_fwd is not None:
                ma.markers.append(cone_fwd)
            if cone_bwd is not None:
                ma.markers.append(cone_bwd)
        else:
            # RCM mode is OFF: hide fixed axis, RCM point sphere, and RCM cones.
            ma.markers.append(self._make_delete_marker('endoscope_line', 0, stamp))
            ma.markers.append(self._make_delete_marker('rcm_point', 1, stamp))
            ma.markers.append(self._make_delete_marker('rcm_cone', 2, stamp))
            ma.markers.append(self._make_delete_marker('rcm_cone', 3, stamp))

        self.marker_pub.publish(ma)

    def _build_cone_marker(self, apex, direction, length, half_angle,
                           color, stamp, mid):
        d = np.asarray(direction, dtype=float)
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            return None
        d = d / nrm
        if abs(d[2]) < 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        else:
            tmp = np.array([1.0, 0.0, 0.0])
        u = np.cross(d, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        base_center = apex + length * d
        base_radius = length * np.tan(half_angle)
        m = Marker()
        m.header.frame_id = self.base_link
        m.header.stamp = stamp
        m.ns = 'rcm_cone'
        m.id = mid
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color = color
        n = 32
        for i in range(n):
            th0 = 2.0 * np.pi * i / n
            th1 = 2.0 * np.pi * (i + 1) / n
            p0 = base_center + base_radius * (
                np.cos(th0) * u + np.sin(th0) * v)
            p1 = base_center + base_radius * (
                np.cos(th1) * u + np.sin(th1) * v)
            m.points.append(Point(x=float(apex[0]),
                                  y=float(apex[1]),
                                  z=float(apex[2])))
            m.points.append(Point(x=float(p0[0]),
                                  y=float(p0[1]),
                                  z=float(p0[2])))
            m.points.append(Point(x=float(p1[0]),
                                  y=float(p1[1]),
                                  z=float(p1[2])))
            m.points.append(Point(x=float(base_center[0]),
                                  y=float(base_center[1]),
                                  z=float(base_center[2])))
            m.points.append(Point(x=float(p1[0]),
                                  y=float(p1[1]),
                                  z=float(p1[2])))
            m.points.append(Point(x=float(p0[0]),
                                  y=float(p0[1]),
                                  z=float(p0[2])))
        return m

    # ==================================================================
    # Getters for GUI
    # ==================================================================
    def get_state_dict(self):
        """Snapshot of state for the GUI's periodic refresh."""
        with self._state_lock:
            q = self.joint_positions.copy() if self.joint_received else None
            eff = max(1e-6, self.flex_arc_len - FLEX_SEG_LEN)
            theta_total = eff * max(0.0, self.flex_curvature)
            if theta_total > FLEX_MAX_BEND:
                theta_total = FLEX_MAX_BEND

            state = {
                'arm_mode': self.arm_mode,
                'flex_mode': self.flex_mode,
                'rcm_enabled': self.rcm_enabled,
                'rcm_point': (self.rcm_world_point.copy()
                              if self.rcm_world_point is not None else None),
                'joints_deg': (np.rad2deg(q).tolist() if q is not None
                               else None),
                'flex_base_yaw': math.degrees(self.flex_base_yaw),
                'flex_arc_len': self.flex_arc_len,
                'flex_ext_yaw': math.degrees(self.flex_ext_yaw),
                'flex_curvature': self.flex_curvature,
                'flex_theta_total_deg': math.degrees(theta_total),
                'flex_theta_seg_deg': math.degrees(
                    theta_total / (FLEX_N - 1)),
            }
            alpha = self.flex_base_yaw + self.flex_ext_yaw
            tendons = {}
            for phi_deg, name in FLEX_TENDONS:
                phi = math.radians(phi_deg)
                dL = -FLEX_TENDON_R * math.cos(phi - alpha) * theta_total
                tendons[name] = (self.flex_arc_len + dL, dL)
            state['tendons'] = {
                n: (L * 1000, dL * 1000) for n, (L, dL) in tendons.items()
            }
        return state


def main(args=None):
    rclpy.init(args=args)
    node = Ur5EndoFlexNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._key_listener is not None:
            try:
                node._key_listener.stop()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()