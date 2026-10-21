#!/usr/bin/env python3
"""
robot_node.py — unified ROS 2 control node for the integrated
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

Tracking:
    Per-target GUI buttons engage a closed-loop hybrid RCM servo that keeps
    the distal camera looking at the selected scene target while holding
    tracking_standoff_m (default 10.5 cm) from it, following dynamic targets
    continuously.  Tracking is only allowed in RCM mode (arm_mode == 'rcm',
    RCM point captured) with flex_mode == 'fixed' (no-roll constraint);
    otherwise the request is refused with a GUI-visible reason.
    The servo is flex-primary: the endo_flex curvature takes all lateral /
    aiming motion while the flex bend is below 50 deg, the UR5 RCM tilt/yaw
    is smoothly recruited between 50-80 deg and takes over above 80 deg
    (with an outward-bend lock).  Axial insertion depth always routes to the
    arm.  The aim point is p_cam + standoff * R_end[:, 2] (the camera views
    along the flex-end +z axis, i.e. -shaft while straight).
    Any manual key cancels tracking; the STOP button disengages tracking and
    holds the current state.  Disabling RCM mid-track auto-cancels tracking.

Topics
------
    Sub : /joint_states          (UR joint positions from ros2_control)
    Sub : <targets_topic>        String JSON scene target state
    Pub : <controller_topic>     JointTrajectory for arm
    Pub : /joint_states          JointState for the flex joints (RViz TF)
    Pub : /flex/base_yaw         Float64 for Gazebo base_yaw joint
    Pub : /flex/ext_yaw          Float64 for Gazebo ext_yaw_1..7 joints
    Pub : /flex/ext_pitch        Float64 for Gazebo ext_pitch_1..7 joints
    Pub : /flex/ext_yaw_back     Float64 for Gazebo ext_yaw_back_1..7 joints
    Pub : <marker_topic>         MarkerArray (visualization)
"""

import json
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

from kinematics import (
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
# Curvature tolerance (1/m) below which the flex is considered straight for
# sequencing gates such as PULL OUT retraction.
FLEX_STRAIGHT_KAPPA_TOL = 1e-3
BIRD_EYE_HORIZON_DEG = 30.0
PULL_OUT_BEHIND_RCM_M = 0.10

# Hybrid RCM target-tracking servo geometry (see kinematics).
# Effective bending length of the continuum: total arc minus one segment.
TRACK_L_EFF = FLEX_ARC_MAX - FLEX_SEG_LEN          # 0.042 m
# Shaft distance from endoscope tip (z=+0.20195) to flex_base_link (z=-0.398)
# in the endoscope frame: 0.59995 m.
TRACK_TIP_TO_FLEX_BASE = 0.59995
# Absolute DLS damping scale (m^2). J is in [m] per [1/m, rad, m], so J^T J is
# of order 1e-6..1e-2; the zone weights below are relative to this scale.
TRACK_W_SCALE = 1e-6
# Over-bend recovery: a tracking task that starts with the flex already bent
# beyond TRACK_RECOVER_BEND_DEG first unwinds the flex (arm held still) down
# to TRACK_RECOVER_DONE_DEG before resuming normal hybrid tracking.  These
# thresholds straddle the zone schedule (50/80 deg) and the null-space
# recentering threshold (55 deg -> 45 deg).
TRACK_RECOVER_BEND_DEG = 55.0
TRACK_RECOVER_DONE_DEG = 50.0

# GUI target-tracking buttons: (key, label, rgba color)
TRACK_TARGETS = [
    ('LN_No3',  'LN3 Red', (1.00, 0.15, 0.15, 0.96)),
    ('LN_No4d', 'LN4d Grn', (0.12, 0.92, 0.25, 0.96)),
    ('LN_No6',  'LN6 Blu', (0.15, 0.48, 1.00, 0.96)),
    ('LGA_No7', 'LGA7 Dyn', (1.00, 0.86, 0.05, 0.98)),
]


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

        # Target tracking
        self.declare_parameter('tracking_standoff_m', 0.105)
        self.declare_parameter('targets_topic', '/gastrectomy/targets_state')

        # Flex parameters (read-only here; used by the formulas)
        self.declare_parameter('flex_N', FLEX_N)
        self.declare_parameter('flex_SEG_LEN', FLEX_SEG_LEN)
        self.declare_parameter('flex_ARC_MIN', FLEX_ARC_MIN)
        self.declare_parameter('flex_ARC_MAX', FLEX_ARC_MAX)
        self.declare_parameter('flex_TENDON_R', FLEX_TENDON_R)

        # Continuous-control velocity limits
        self.declare_parameter('arm_linear_vel_m_s', 0.025)
        self.declare_parameter('arm_axis_vel_m_s', 0.025)
        self.declare_parameter('arm_tilt_vel_deg_s', 4.0)
        self.declare_parameter('flex_yaw_vel_rad_s', 1.0)
        self.declare_parameter('flex_curv_vel_1pm_s', 10.0)
        # Arc-length slew rate for semi-automatic flex moves (m/s).
        self.declare_parameter('flex_arc_vel_m_s', 0.020)
        self.declare_parameter('control_tick_hz', 30.0)

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

        self.tracking_standoff = float(
            self.get_parameter('tracking_standoff_m').value)
        self.targets_topic = self.get_parameter('targets_topic').value

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
        self.flex_arc_vel = float(
            self.get_parameter('flex_arc_vel_m_s').value)
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

        # Active speed-limited flex slew target (None when idle).  Set by
        # semi-automatic commands (HOME, BIRD-EYE VIEW, PULL OUT) so the flex
        # ramps toward its goal instead of snapping in a single tick.
        self._flex_slew_target = None

        # Target tracking.  All fields below are protected by _state_lock.
        self._track_target = None        # active target key or None
        self._targets_state = {}         # key -> np.array(3,) pos_base
        self._track_warned = False       # missing-target warn already issued
        self._track_locked = False       # servo converged and holding
        self._track_dist_m = None        # camera->target distance (m)
        self._track_aim_deg = None       # camera-axis aim error (deg)
        self._track_zone = None          # last WDLS zone (1/2/3) or None
        self._track_recover = False      # over-bend recovery phase active
        self._track_status_msg = None    # transient refusal reason for GUI
        self._targets_logged = False     # first targets-state receipt logged

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

        targets_qos = QoSProfile(
            depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(
            String, self.targets_topic, self._targets_state_cb, targets_qos)

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

    def _targets_state_cb(self, msg: String):
        """Update known scene-target positions from the JSON target state."""
        try:
            data = json.loads(msg.data)
            targets = data.get('targets', {})
            parsed = {}
            for key, _, _ in TRACK_TARGETS:
                t = targets.get(key)
                if t is None:
                    continue
                parsed[key] = np.array(t['pos_base'], dtype=float)
        except Exception as e:
            self.get_logger().warn(
                f'Failed to parse targets state: {e}',
                throttle_duration_sec=2.0)
            return

        with self._state_lock:
            self._targets_state.update(parsed)
            log_first = bool(parsed) and not self._targets_logged
            if log_first:
                self._targets_logged = True
        if log_first:
            self.get_logger().info('targets state received: 4 targets')

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
                self._flex_slew_target = None
                pull_out_active = False
                self.get_logger().info(
                    'PULL OUT cancelled by manual key input.')
            tracking_active = self._track_target is not None
            if tracking_active and any_manual_key:
                self._clear_track_state_unlocked()
                self._flex_slew_target = None
                tracking_active = False
                self.get_logger().info(
                    'TRACKING cancelled by manual key input.')
            # Manual press-and-hold input always wins over a semi-automatic
            # flex slew that is still ramping.
            if any_manual_key and self._flex_slew_target is not None:
                self._flex_slew_target = None
                self.get_logger().info(
                    'FLEX slew cancelled by manual key input.')
            # The tracking servo owns the flex state while engaged.
            if tracking_active and self._flex_slew_target is not None:
                self._flex_slew_target = None

        # Ramp any active semi-automatic flex slew (HOME / BIRD-EYE VIEW /
        # PULL OUT straightening) toward its target, once per control tick.
        self._step_flex_slew()

        if pull_out_active:
            self._step_pull_out()
            self._publish_flex_joint_states()
            return

        if tracking_active:
            self._step_hybrid_rcm_tracking()
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

    def _set_flex_slew_target(self, curvature=None, steer_yaw=None,
                              arc_len=None):
        """Record the desired flex end state for speed-limited slewing.

        Unspecified components (None) hold their current value so a slew
        request cannot make an unrelated flex DoF drift.  Must be called with
        _state_lock held.
        """
        self._flex_slew_target = {
            'curvature': (self.flex_curvature
                          if curvature is None else float(curvature)),
            'steer_yaw': (self.flex_steer_yaw
                          if steer_yaw is None else float(steer_yaw)),
            'arc_len': (self.flex_arc_len
                        if arc_len is None else float(arc_len)),
        }

    def _step_flex_slew(self):
        """Ramp flex state toward the active slew target at velocity limits.

        Moves flex_curvature, flex_steer_yaw and flex_arc_len by at most
        flex_curv_vel*dt, flex_yaw_vel*dt and flex_arc_vel*dt respectively,
        clamping without overshoot, and clears the target once all three
        components have converged.  No-op when no slew is active.
        """
        with self._state_lock:
            target = self._flex_slew_target
            if target is None:
                return

            # Arc length (m) first: the curvature clamp below depends on it.
            a = self.flex_arc_len
            a_tgt = min(max(FLEX_ARC_MIN, target['arc_len']), FLEX_ARC_MAX)
            step_a = self.flex_arc_vel * self.dt
            if abs(a_tgt - a) <= step_a:
                a_new = a_tgt
            else:
                a_new = a + math.copysign(step_a, a_tgt - a)
            self.flex_arc_len = min(max(FLEX_ARC_MIN, a_new), FLEX_ARC_MAX)

            # Curvature (1/m), clamped to the current geometric bend limit.
            c = self.flex_curvature
            eff = max(1e-6, self.flex_arc_len - FLEX_SEG_LEN)
            kappa_max = FLEX_MAX_BEND / eff
            c_tgt = min(max(0.0, target['curvature']), kappa_max)
            step_c = self.flex_curv_vel * self.dt
            if abs(c_tgt - c) <= step_c:
                c_new = c_tgt
            else:
                c_new = c + math.copysign(step_c, c_tgt - c)
            self.flex_curvature = min(max(0.0, c_new), kappa_max)

            # Steering yaw (rad) along the shortest angular path.
            y = self.flex_steer_yaw
            y_tgt = target['steer_yaw']
            d_yaw = ((y_tgt - y + math.pi) % (2.0 * math.pi)) - math.pi
            step_y = self.flex_yaw_vel * self.dt
            if abs(d_yaw) <= step_y:
                y_new = y_tgt
            else:
                y_new = y + math.copysign(step_y, d_yaw)
            self.flex_steer_yaw = (
                (y_new + math.pi) % (2.0 * math.pi) - math.pi)

            # Clear the target once every component has reached its goal.
            y_err = ((self.flex_steer_yaw - y_tgt + math.pi)
                     % (2.0 * math.pi)) - math.pi
            converged = (
                abs(self.flex_curvature - c_tgt) <= 1e-6
                and abs(y_err) <= 1e-6
                and abs(self.flex_arc_len - a_tgt) <= 1e-6)
            if converged:
                self._flex_slew_target = None

    def _flex_is_straight(self):
        """True when flex curvature is within the straight tolerance (1/m)."""
        with self._state_lock:
            return abs(self.flex_curvature) <= FLEX_STRAIGHT_KAPPA_TOL

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
            self._flex_slew_target = None
            tracking_was_active = self._track_target is not None
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
                if tracking_was_active:
                    self._clear_track_state_unlocked()
                    self.get_logger().warn(
                        'TRACKING auto-cancelled: RCM mode disabled.')

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
            if mode != 'fixed' and self._track_target is not None:
                self._clear_track_state_unlocked()
                self.get_logger().warn(
                    'TRACKING auto-cancelled: flex mode left fixed.')
        self._publish_flex_joint_states()
        self.get_logger().info(f'Flex mode -> {mode}.')

    # ==================================================================
    # Semi-automatic commands
    # ==================================================================
    def cmd_home(self):
        """HOME: arm to HOME_DEG (with wrist_1 auto-solved); flex straight."""
        with self._state_lock:
            self._pull_out_state = None
            # Ramp the flex straight instead of snapping in a single tick.
            self._set_flex_slew_target(
                curvature=0.0, steer_yaw=0.0, arc_len=FLEX_ARC_MAX)
        self._publish_flex_joint_states()

        q_target = np.deg2rad(np.array(HOME_DEG, dtype=float))
        q_target = self._auto_solve_wrist1(q_target)
        self._send_trajectory(q_target, traj_duration=3.0)

    def cmd_reset(self):
        """RESET: HOME + arm_mode=linear, flex_mode=fixed; clear RCM."""
        with self._state_lock:
            self._pull_out_state = None
            self._flex_slew_target = None
            tracking_was_active = self._track_target is not None
            self._clear_track_state_unlocked()
            self.arm_mode = 'linear'
            self.flex_mode = 'fixed'
            self.rcm_enabled = False
            self.rcm_ref_axis = None
            self.rcm_ref_depth = None
        if tracking_was_active:
            self.get_logger().warn(
                'TRACKING auto-cancelled: RESET disabled RCM mode.')
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
            # Ramp to the bird-eye bend instead of snapping in one tick.
            self._set_flex_slew_target(
                curvature=float(theta_target / eff), steer_yaw=0.0)

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
            self._clear_track_state_unlocked()

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
            # Step (a.1): Ramp endo_flex straight via the slew mechanism so
            # the flex does not snap in a single tick.
            self._set_flex_slew_target(curvature=0.0, steer_yaw=0.0)
            self._pull_out_state = {
                'stage': 'align',
                'axis_cmd': axis_cur.copy(),
                'd_tip_align': d_tip_align,
                'tip_from_end': tip_from_end,
                's_end_cmd': None,
                'waiting_logged': False,
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
                self._flex_slew_target = None
                return
            rcm_pt = self.rcm_world_point.copy()
            ref_axis = self.rcm_ref_axis.copy()
            q_now = self.joint_positions.copy()
            # Keep flex straight throughout pull-out (speed-limited slew).
            self._set_flex_slew_target(curvature=0.0, steer_yaw=0.0)

        try:
            axis_cur, _, p_tip_cur = self.kin.tube_axis_in_base(q_now)
            p_end_cur = self.kin.end_cylinder_in_base(q_now)
        except Exception as e:
            self.get_logger().warn(f'PULL OUT FK failed: {e}')
            with self._state_lock:
                self._pull_out_state = None
                self._flex_slew_target = None
            return

        # ---------------- Stage (a): Align direction to ref_axis ----------------
        if st['stage'] == 'align':
            # Rotation/alignment and flex straightening run concurrently.
            # Always perform this tick's alignment work first, then decide
            # whether the transition to retraction is allowed.
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

            # Check actual alignment with ref_axis to transition to Stage (b).
            cos_actual = float(np.clip(np.dot(axis_cur, ref_axis), -1.0, 1.0))
            err_actual_deg = math.degrees(math.acos(cos_actual))
            aligned = (np.allclose(axis_next, ref_axis, atol=1e-3)
                       and err_actual_deg < 1.0)
            if not aligned:
                return

            # Gate only the backward retraction: the arm must not start
            # pulling out until the flex is fully straightened.  Alignment
            # keeps running above while the flex slew converges.
            if not self._flex_is_straight():
                if not st.get('waiting_logged', False):
                    self.get_logger().info(
                        'PULL OUT Stage (a): direction aligned; waiting for '
                        'flex to finish straightening before retraction.')
                    st['waiting_logged'] = True
                return

            # Gate passed: re-arm the one-shot wait log for any later gate.
            st['waiting_logged'] = False
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
                    self._flex_slew_target = None
                self.get_logger().info(
                    f'PULL OUT complete: endo_flex end is '
                    f'{s_end_cur * 100.0:.1f} cm behind fixed RCM point.')

    # ==================================================================
    # Target tracking (hybrid flex-primary RCM servo, one tick per call)
    # ==================================================================
    def _clear_track_state_unlocked(self):
        """Reset all per-task tracking state (must hold _state_lock).

        Clears every field a new task must not inherit from a stopped or
        auto-cancelled task, so stale flags cannot block the next task.
        """
        self._track_target = None
        self._track_zone = None
        self._track_warned = False
        self._track_locked = False
        self._track_recover = False
        self._track_dist_m = None
        self._track_aim_deg = None
        self._track_status_msg = None

    def start_tracking(self, key):
        """Engage closed-loop hybrid RCM tracking of a scene target by key.

        Refused (with a GUI-visible reason) unless the arm is in RCM mode
        with a captured RCM point and the flex is in 'fixed' mode (the
        no-roll constraint of the continuum).
        """
        if key not in {k for k, _, _ in TRACK_TARGETS}:
            self.get_logger().warn(f'Unknown tracking target: {key}')
            return

        refusal = None
        recovering = False
        start_theta_deg = 0.0
        with self._state_lock:
            rcm_ok = (self.arm_mode == 'rcm' and self.rcm_enabled
                      and self.rcm_world_point is not None)
            if not rcm_ok or self.flex_mode != 'fixed':
                refusal = 'requires RCM mode (Ctrl+M) + fixed flex mode'
                self._track_status_msg = refusal
            else:
                self._pull_out_state = None
                self._flex_slew_target = None
                # Drop every flag from any previous task before installing
                # this task, so a cancelled task cannot leave stale state.
                self._clear_track_state_unlocked()
                self._track_target = key
                start_theta_deg = math.degrees(min(
                    FLEX_MAX_BEND, TRACK_L_EFF * self.flex_curvature))
                self._track_recover = (
                    start_theta_deg > TRACK_RECOVER_BEND_DEG)
                recovering = self._track_recover
                has_state = bool(self._targets_state)

        if refusal is not None:
            self.get_logger().warn(
                f'TRACKING refused for {key}: {refusal}.')
            return

        if recovering:
            self.get_logger().info(
                f'TRACKING recovery: flex bend {start_theta_deg:.1f} deg > '
                f'{TRACK_RECOVER_BEND_DEG:.0f} deg; unbending flex before '
                f'servo engagement (arm held still).')
        self.get_logger().info(
            f'TRACKING started: {key} '
            f'(standoff {self.tracking_standoff * 100.0:.1f} cm)')
        if not has_state:
            self.get_logger().info(
                f'TRACKING will engage when {self.targets_topic} arrives.')

    def stop_tracking(self):
        """Stop tracking and hold the current arm/flex state."""
        with self._state_lock:
            if self._track_target is None:
                return
            self._clear_track_state_unlocked()
        self.get_logger().info('TRACKING stopped; holding current state.')

    @staticmethod
    def _wdls_solve(J, e_vec, w_rel):
        """Weighted damped least squares step (None when singular)."""
        W = TRACK_W_SCALE * np.diag(np.asarray(w_rel, dtype=float))
        A = J.T @ J + W
        try:
            return np.linalg.solve(A, J.T @ e_vec)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _step_hybrid_rcm_tracking(self):
        """Execute one 20 Hz tick of the hybrid RCM tracking servo.

        Control variables u = (kappa_x, kappa_y, theta_tilt, theta_yaw,
        d_ins).  The task is the virtual aim point p_aim = p_cam +
        standoff * v_hat with v_hat = R_end[:, 2]; driving p_aim onto the
        target achieves look-at and the standoff distance simultaneously.
        Zone weights keep endo_flex primary and recruit the UR5 RCM arm
        only as the bend approaches saturation.
        """
        if not self._arm_ready():
            return

        with self._state_lock:
            key = self._track_target
            if key is None:
                return
            p_target = (None if key not in self._targets_state
                        else self._targets_state[key].copy())
            arm_mode = self.arm_mode
            rcm_on = self.rcm_enabled
            rcm_pt = (None if self.rcm_world_point is None
                      else self.rcm_world_point.copy())
            ref_axis = (None if self.rcm_ref_axis is None
                        else self.rcm_ref_axis.copy())
            q_now = self.joint_positions.copy()
            kappa = float(self.flex_curvature)
            steer_yaw = float(self.flex_steer_yaw)
            arc_len = float(self.flex_arc_len)
            flex_mode = self.flex_mode
            standoff = float(self.tracking_standoff)
            already_warned = self._track_warned
            recovering = bool(self._track_recover)
            if p_target is None and not already_warned:
                self._track_warned = True

        # RCM (and fixed flex) are prerequisites: mid-track loss auto-cancels.
        if not (arm_mode == 'rcm' and rcm_on and rcm_pt is not None
                and ref_axis is not None):
            with self._state_lock:
                self._clear_track_state_unlocked()
            self.get_logger().warn(
                'TRACKING auto-cancelled: RCM mode is no longer active.')
            return
        if flex_mode != 'fixed':
            with self._state_lock:
                self._clear_track_state_unlocked()
            self.get_logger().warn(
                'TRACKING auto-cancelled: flex mode is no longer fixed.')
            return

        if p_target is None:
            if not already_warned:
                self.get_logger().warn(
                    f"TRACKING waiting for target '{key}' on "
                    f'{self.targets_topic}.')
            return

        # ---- Control-space FK (no arm IK) ----
        try:
            R_shaft, _ = self.kin.fk_matrix(q_now)
            _, _, p_tip = self.kin.tube_axis_in_base(q_now)
            d_ins = float(np.linalg.norm(p_tip - rcm_pt))
            p_cam, v_hat = self.kin.camera_pose_on_shaft(
                rcm_pt, R_shaft, d_ins, kappa, steer_yaw,
                arc_len=arc_len, flex_mode=flex_mode,
                tip_to_flex_base=TRACK_TIP_TO_FLEX_BASE)
        except Exception as e:
            self.get_logger().warn(
                f'TRACKING FK failed: {e}', throttle_duration_sec=2.0)
            return

        p_aim = p_cam + standoff * v_hat
        e_vec = p_target - p_aim
        err_norm = float(np.linalg.norm(e_vec))

        aim_vec = p_target - p_cam
        aim_norm = float(np.linalg.norm(aim_vec))
        if aim_norm > 1e-9:
            cos_aim = float(np.clip(
                np.dot(v_hat, aim_vec / aim_norm), -1.0, 1.0))
            aim_deg = math.degrees(math.acos(cos_aim))
        else:
            aim_deg = 0.0
        dist_m = float(np.linalg.norm(p_cam - p_target))

        # ---- Zone schedule on the current flex bend ----
        theta_total = min(FLEX_MAX_BEND, TRACK_L_EFF * kappa)
        theta_deg = math.degrees(theta_total)
        # Over-bend recovery ends once the flex is back inside the
        # flex-primary range (50 deg), giving 5 deg of hysteresis.
        if recovering and theta_deg <= TRACK_RECOVER_DONE_DEG:
            with self._state_lock:
                self._track_recover = False
            recovering = False
            self.get_logger().info(
                f'TRACKING recovery complete: flex bend {theta_deg:.1f} deg '
                f'<= {TRACK_RECOVER_DONE_DEG:.0f} deg; resuming hybrid servo.')
        if theta_deg <= 50.0:
            zone = 1
            w_rel = np.array([1.0, 1.0, 1e6, 1e6, 1.0])
        elif theta_deg < 80.0:
            zone = 2
            s = (theta_deg - 50.0) / 30.0
            w_flex = 1.0 + 20.0 * s * s
            w_tilt = 1.0 + (1e6 - 1.0) * (1.0 - s) ** 2
            w_rel = np.array([w_flex, w_flex, w_tilt, w_tilt, 1.0])
        else:
            zone = 3
            w_rel = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        max_dkappa = self.flex_curv_vel * self.dt
        max_dpsi = self.flex_yaw_vel * self.dt
        max_dtilt = self.arm_tilt_vel * self.dt
        max_dins = self.arm_axis_vel * self.dt

        # ---- Flex/arm step: over-bend recovery or normal hybrid WDLS ----
        if recovering:
            # Over-bend recovery: unwind the flex at its full rate with the
            # arm held still.  This guarantees inward (reducing) curvature
            # regardless of the zone weights or the zone-3 outward lock, and
            # avoids a compensating arm lunge while the bend recovers.
            d_kappa = -max_dkappa
            d_psi = 0.0
            d_tilt = 0.0
            d_yaw = 0.0
            d_ins_step = 0.0
        else:
            # ---- WDLS primary solve ----
            if err_norm < 1e-12:
                du = np.zeros(5, dtype=float)
            else:
                try:
                    J = self.kin.hybrid_rcm_jacobian(
                        rcm_pt, R_shaft, d_ins, kappa, steer_yaw, standoff,
                        arc_len=arc_len, flex_mode=flex_mode,
                        tip_to_flex_base=TRACK_TIP_TO_FLEX_BASE)
                except Exception as e:
                    self.get_logger().warn(
                        f'TRACKING Jacobian failed: {e}',
                        throttle_duration_sec=2.0)
                    return
                du = self._wdls_solve(J, e_vec, w_rel)
                if du is None:
                    self.get_logger().warn(
                        'TRACKING WDLS singular; skipping tick.',
                        throttle_duration_sec=2.0)
                    return

                # Zone 3: lock outward bending.  This re-solve is only
                # entered when the primary step would increase curvature, so
                # inward (reducing) curvature is never blocked.
                if zone == 3:
                    kx0 = kappa * math.cos(steer_yaw)
                    ky0 = kappa * math.sin(steer_yaw)
                    kappa_solved = math.hypot(kx0 + du[0], ky0 + du[1])
                    if kappa_solved > kappa + 1e-12:
                        w_lock = w_rel.copy()
                        w_lock[0] = 1e4
                        w_lock[1] = 1e4
                        du_lock = self._wdls_solve(J, e_vec, w_lock)
                        if du_lock is None:
                            self.get_logger().warn(
                                'TRACKING WDLS singular (outward lock); '
                                'skipping tick.', throttle_duration_sec=2.0)
                            return
                        du = du_lock

                # ---- Null-space recentering above 55 deg (low gain) ----
                if theta_total > math.radians(55.0) and kappa > 1e-9:
                    J_pinv = np.linalg.pinv(J)
                    N_null = np.eye(5) - J_pinv @ J
                    g = np.zeros(5, dtype=float)
                    g_flex = -TRACK_L_EFF * (theta_total - math.radians(45.0))
                    g[0] = g_flex * math.cos(steer_yaw)
                    g[1] = g_flex * math.sin(steer_yaw)
                    du = du - 0.1 * (N_null @ g)

            # ---- Hard clamps ----
            kx0 = kappa * math.cos(steer_yaw)
            ky0 = kappa * math.sin(steer_yaw)
            kappa_u = math.hypot(kx0 + du[0], ky0 + du[1])
            if kappa_u > 1e-12:
                psi_u = math.atan2(ky0 + du[1], kx0 + du[0])
            else:
                psi_u = steer_yaw
            d_kappa = float(np.clip(
                kappa_u - kappa, -max_dkappa, max_dkappa))
            d_psi = float(np.clip(
                (psi_u - steer_yaw + math.pi) % (2.0 * math.pi) - math.pi,
                -max_dpsi, max_dpsi))
            d_tilt = float(np.clip(du[2], -max_dtilt, max_dtilt))
            d_yaw = float(np.clip(du[3], -max_dtilt, max_dtilt))
            d_ins_step = float(np.clip(du[4], -max_dins, max_dins))

        # ---- Common flex target (also enforces the global bend clamp) ----
        kappa_new = float(np.clip(
            kappa + d_kappa, 0.0, FLEX_MAX_BEND / TRACK_L_EFF))
        if kappa_new < 1e-6:
            psi_new = steer_yaw
        else:
            psi_new = (steer_yaw + d_psi + math.pi) % (2.0 * math.pi) - math.pi

        # ---- RCM cone guard: freeze the arm, keep the flex moving ----
        R_shaft_new = self.kin.rotate_shaft_frame(
            R_shaft, d_tilt=d_tilt, d_yaw=d_yaw)
        u_new = R_shaft_new[:, 2]
        cos_cone = float(np.clip(np.dot(u_new, ref_axis), -1.0, 1.0))
        cone_deg = math.degrees(math.acos(cos_cone))
        if cone_deg > math.degrees(self.rcm_cone_half_angle):
            if abs(d_tilt) + abs(d_yaw) > 1e-12:
                self.get_logger().warn(
                    f'TRACKING RCM cone limit ({cone_deg:.1f} deg > '
                    f'{math.degrees(self.rcm_cone_half_angle):.1f} deg); '
                    f'arm frozen for this tick.',
                    throttle_duration_sec=2.0)
            d_tilt = 0.0
            d_yaw = 0.0
            u_new = R_shaft[:, 2].copy()

        # ---- Apply flex ----
        with self._state_lock:
            self.flex_curvature = kappa_new
            if kappa_new >= 1e-6:
                self.flex_steer_yaw = float(psi_new)

        # ---- Apply arm (only when the RCM coordinates changed) ----
        if abs(d_tilt) + abs(d_yaw) + abs(d_ins_step) > 1e-9:
            d_new = d_ins + d_ins_step
            p_tip_target = rcm_pt + d_new * u_new
            q_target = self.kin.ik_with_rcm(p_tip_target, rcm_pt, q_now)
            if q_target is None:
                self.get_logger().warn(
                    'TRACKING IK failed; arm skipped this tick '
                    '(flex still applied).', throttle_duration_sec=2.0)
            else:
                q_target = self._auto_solve_wrist1(q_target)
                self._send_trajectory(q_target, traj_duration=self.dt)

        # ---- GUI / marker telemetry ----
        # Do not report "locked" while the flex is still in over-bend
        # recovery; the arm is intentionally held and the aim is transient.
        locked = (not recovering and aim_deg < 0.5
                  and abs(dist_m - standoff) < 0.0015)
        with self._state_lock:
            self._track_dist_m = dist_m
            self._track_aim_deg = aim_deg
            self._track_zone = zone
            newly_locked = locked and not self._track_locked
            if locked:
                self._track_locked = True
        if newly_locked:
            self.get_logger().info(
                f'TRACKING locked on {key}: d={dist_m * 100.0:.1f} cm, '
                f'aim={aim_deg:.2f} deg.')

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
            track_key = self._track_target
            track_pos = (None if track_key is None
                         or track_key not in self._targets_state
                         else self._targets_state[track_key].copy())
            flex_curvature = self.flex_curvature
            flex_steer_yaw = self.flex_steer_yaw
            flex_arc_len = self.flex_arc_len
            flex_mode = self.flex_mode

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

        # 3. Target tracking: sight-line camera -> target and status text.
        track_rgba = None
        if track_key is not None:
            for t_key, _t_label, t_rgba in TRACK_TARGETS:
                if t_key == track_key:
                    track_rgba = t_rgba
                    break

        p_cam = None
        if (track_key is not None and track_pos is not None
                and track_rgba is not None):
            try:
                _, p_cam, _ = self.kin.flex_fk_in_base(
                    q, curvature=flex_curvature, steer_yaw=flex_steer_yaw,
                    arc_len=flex_arc_len, flex_mode=flex_mode)
            except Exception as e:
                self.get_logger().warn(
                    f'TRACKING FK failed in marker publish: {e}',
                    throttle_duration_sec=2.0)
                p_cam = None

        if p_cam is not None:
            sight = Marker()
            sight.header.frame_id = self.base_link
            sight.header.stamp = stamp
            sight.ns = 'tracking_sightline'
            sight.id = 5
            sight.type = Marker.LINE_STRIP
            sight.action = Marker.ADD
            sight.scale.x = self.line_width * 1.5
            sight.color = ColorRGBA(
                r=float(track_rgba[0]), g=float(track_rgba[1]),
                b=float(track_rgba[2]), a=float(track_rgba[3]))
            sight.pose.orientation.w = 1.0
            sight.points = [
                Point(x=float(p_cam[0]), y=float(p_cam[1]),
                      z=float(p_cam[2])),
                Point(x=float(track_pos[0]), y=float(track_pos[1]),
                      z=float(track_pos[2])),
            ]
            ma.markers.append(sight)

            d_track = float(np.linalg.norm(p_cam - track_pos))
            track_status = Marker()
            track_status.header.frame_id = self.base_link
            track_status.header.stamp = stamp
            track_status.ns = 'status'
            track_status.id = 101
            track_status.type = Marker.TEXT_VIEW_FACING
            track_status.action = Marker.ADD
            track_status.pose.position.x = float(p_cam[0])
            track_status.pose.position.y = float(p_cam[1])
            track_status.pose.position.z = float(p_cam[2] + 0.05)
            track_status.pose.orientation.w = 1.0
            track_status.scale.z = 0.02
            track_status.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            track_status.text = (
                f'TRACKING {track_key}: d={d_track * 100.0:.1f} cm')
            ma.markers.append(track_status)
        else:
            # Not tracking (or target not received yet): hide tracking markers.
            ma.markers.append(
                self._make_delete_marker('tracking_sightline', 5, stamp))
            ma.markers.append(
                self._make_delete_marker('status', 101, stamp))

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
                'tracking': self._track_target,
                'tracking_dist_m': self._track_dist_m,
                'tracking_aim_deg': self._track_aim_deg,
                'tracking_zone': self._track_zone,
                'tracking_recovering': self._track_recover,
                'tracking_locked': self._track_locked,
                'tracking_refusal': self._track_status_msg,
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