#!/usr/bin/env python3
"""
ROS 2 node for the UR5 + endoscope control panel.

RCM behaviour (fixed-line / fixed-cone mode)
--------------------------------------------
When RCM is enabled the node captures:
    * the RCM point in the base frame        (stays fixed)
    * the current tube axis                  (reference direction)

The yellow line and the green ±20° cone are then drawn from that
fixed RCM point along that fixed reference axis.  The endoscope
should stay inside the cone; if the actual tube axis (from FK)
deviates from the reference axis by more than the cone half-angle,
the line / sphere / cone turn red and further tilt commands are
rejected (see `tilt_with_rcm`).

Tilt directions
---------------
"UP"/"DOWN"/"LEFT"/"RIGHT" are interpreted in the **world** frame:
  * UP    = tip tilts toward world +Z
  * DOWN  = tip tilts toward world -Z
  * LEFT  = tip tilts toward world +Y   (UR base: left)
  * RIGHT = tip tilts toward world -Y   (UR base: right)
"""

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, ColorRGBA
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from endoscope_kinematics import EndoscopeKinematics, JOINT_NAMES


class ControlPanel(Node):

    def __init__(self):
        super().__init__('endoscope_control_panel')

        # ---------------- Parameters --------------------------------
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'endoscope_link')
        self.declare_parameter(
            'controller_topic',
            '/scaled_joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('step_size_mm', 5.0)
        self.declare_parameter('tilt_step_deg', 5.0)
        self.declare_parameter('traj_duration_s', 1.0)

        self.declare_parameter('rcm_point_in_tip_frame',
                               [-0.0193, -0.01615, -0.12955])
        self.declare_parameter('endoscope_tip_in_link_frame',
                               [-0.0193, -0.01615, 0.20195])

        self.declare_parameter('marker_topic', '/endoscope_markers')
        self.declare_parameter('marker_publish_period', 0.1)
        self.declare_parameter('markers_enabled', True)
        self.declare_parameter('rcm_cone_half_angle_deg', 20.0)
        self.declare_parameter('rcm_cone_length', 0.20)
        self.declare_parameter('rcm_sphere_diameter', 0.03)
        self.declare_parameter('endoscope_line_width', 0.006)
        self.declare_parameter('endoscope_line_overshoot', 0.05)

        self.base_link = self.get_parameter('base_link').value
        self.tip_link = self.get_parameter('tip_link').value
        self.controller_topic = self.get_parameter('controller_topic').value
        self.traj_duration = float(
            self.get_parameter('traj_duration_s').value)

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

        # ---------------- Kinematics --------------------------------
        self.kin = EndoscopeKinematics(
            base_link=self.base_link,
            tip_link=self.tip_link,
            rcm_point_local=self.rcm_point_local,
            tip_local=self.tip_local,
            logger=self.get_logger())

        # ---------------- State -------------------------------------
        self.joint_positions = np.zeros(6)
        self.joint_received = False
        self.joint_lock = threading.Lock()

        self.rcm_enabled = False
        self.rcm_world_point = None    # fixed world RCM
        self.rcm_ref_axis = None       # fixed reference tube axis
        self.rcm_lock = threading.Lock()

        self._marker_first_published_logged = False
        self._drift_was_exceeded = False

        # ---------------- ROS I/O -----------------------------------
        traj_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.traj_pub = self.create_publisher(
            JointTrajectory, self.controller_topic, traj_qos)

        js_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, js_qos)

        urdf_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(
            String, '/robot_description',
            self._robot_description_cb, urdf_qos)

        marker_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.marker_topic, marker_qos)

        marker_period = float(
            self.get_parameter('marker_publish_period').value)
        self.create_timer(marker_period, self._publish_markers)

        self.get_logger().info(
            f'Control panel ready. Publishing on {self.controller_topic}')
        self.get_logger().info(
            f'Visualization markers on {self.marker_topic}')

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState):
        with self.joint_lock:
            for i, name in enumerate(JOINT_NAMES):
                if name in msg.name:
                    self.joint_positions[i] = msg.position[
                        msg.name.index(name)]
            self.joint_received = True

    def _robot_description_cb(self, msg: String):
        if self.kin.chain is None and msg.data:
            self.kin.setup_from_urdf(msg.data)

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------
    def send_trajectory(self, q_target):
        """Publish a 2-point trajectory: current state at t=0, target at t=T.

        ur_controllers/ScaledJointTrajectoryController checks the first
        trajectory point against the current joint state and refuses the
        command if the mismatch exceeds its state tolerance.  A single
        target-only point therefore always aborts; seeding the trajectory
        with the current state fixes it.
        """
        with self.joint_lock:
            if self.joint_received:
                q_now = self.joint_positions.copy()
            else:
                q_now = np.asarray(q_target, dtype=float)

        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        # Point 0: current state, t=0.
        pt0 = JointTrajectoryPoint()
        pt0.positions     = [float(x) for x in q_now]
        pt0.velocities    = [0.0] * len(q_now)
        pt0.accelerations = [0.0] * len(q_now)
        pt0.time_from_start = Duration(sec=0, nanosec=0)
        msg.points.append(pt0)

        # Point 1: target, t=traj_duration.
        pt1 = JointTrajectoryPoint()
        pt1.positions     = [float(x) for x in q_target]
        pt1.velocities    = [0.0] * len(q_target)
        pt1.accelerations = [0.0] * len(q_target)
        pt1.time_from_start = Duration(
            sec=int(self.traj_duration),
            nanosec=int((self.traj_duration % 1.0) * 1e9))
        msg.points.append(pt1)

        self.traj_pub.publish(msg)

    def set_joint_positions(self, q):
        self.send_trajectory(np.asarray(q, dtype=float))

    def step_along_axis(self, distance_m):
        if not self.kin.ready:
            self.get_logger().warn('No KDL chain; Cartesian step ignored.')
            return False
        if not self.joint_received:
            self.get_logger().warn('No /joint_states yet; command ignored.')
            return False

        with self.joint_lock:
            q_now = self.joint_positions.copy()

        axis, _, p_tip = self.kin.tube_axis_in_base(q_now)
        p_target = p_tip + distance_m * axis

        with self.rcm_lock:
            rcm_on = self.rcm_enabled
            rcm_pt = None if self.rcm_world_point is None \
                else self.rcm_world_point.copy()

        if rcm_on and rcm_pt is not None:
            q_target = self.kin.ik_with_rcm(p_target, rcm_pt, q_now)
        else:
            q_target = self.kin.ik_position_only(p_target, q_now)

        if q_target is None:
            self.get_logger().warn('IK failed; command ignored.')
            return False

        self.send_trajectory(q_target)
        return True

    def tilt_with_rcm(self, angle_right_deg, angle_up_deg):
        """
        Pivot the tube about the fixed RCM point.

        World-frame tilt axes:
            angle_up_deg    > 0 -> tip tilts toward world +Z (up)
            angle_up_deg    < 0 -> tip tilts toward world -Z (down)
            angle_right_deg > 0 -> tip tilts toward world -Y (right)
            angle_right_deg < 0 -> tip tilts toward world +Y (left)

        Rejected if the new axis would drift more than the cone
        half-angle away from the reference axis captured when RCM was
        enabled.
        """
        if not self.kin.ready:
            self.get_logger().warn('No KDL chain; tilt ignored.')
            return False
        if not self.joint_received:
            self.get_logger().warn('No /joint_states yet; tilt ignored.')
            return False

        with self.rcm_lock:
            rcm_on = self.rcm_enabled
            rcm_pt = None if self.rcm_world_point is None \
                else self.rcm_world_point.copy()
            ref_axis = None if self.rcm_ref_axis is None \
                else self.rcm_ref_axis.copy()
        if not rcm_on or rcm_pt is None:
            self.get_logger().warn(
                'RCM is not enabled; arrow-key tilt ignored.')
            return False

        with self.joint_lock:
            q_now = self.joint_positions.copy()

        axis, _, p_tip = self.kin.tube_axis_in_base(q_now)
        distance = float(np.linalg.norm(p_tip - rcm_pt))
        if distance < 1e-6:
            return False

        z_world = np.array([0.0, 0.0, 1.0])
        new_axis = axis.copy()

        # --- Pitch (up/down): rotate in the vertical plane ----------
        if angle_up_deg != 0.0:
            theta = np.deg2rad(float(angle_up_deg))
            a_h = np.array([new_axis[0], new_axis[1], 0.0])
            n = np.linalg.norm(a_h)
            if n < 1e-6:
                pitch_axis = np.array([1.0, 0.0, 0.0])
            else:
                a_h = a_h / n
                pitch_axis = np.cross(a_h, z_world)
                pitch_axis = pitch_axis / np.linalg.norm(pitch_axis)
            new_axis = self.kin.rotate_about(new_axis, pitch_axis, -theta)
            new_axis = new_axis / np.linalg.norm(new_axis)

        # --- Yaw (left/right): rotate about world +Z ----------------
        if angle_right_deg != 0.0:
            theta = np.deg2rad(float(angle_right_deg))
            new_axis = self.kin.rotate_about(new_axis, z_world, -theta)
            new_axis = new_axis / np.linalg.norm(new_axis)

        # Enforce the fixed cone limit against the RCM reference axis.
        if ref_axis is not None:
            cos_a = float(np.clip(np.dot(new_axis, ref_axis), -1.0, 1.0))
            new_deg = float(np.rad2deg(np.arccos(cos_a)))
            limit_deg = float(np.rad2deg(self.rcm_cone_half_angle))
            if new_deg > limit_deg:
                self.get_logger().warn(
                    f'Tilt rejected: would put the tube {new_deg:.1f} deg '
                    f'from the RCM reference axis '
                    f'(limit {limit_deg:.1f} deg).')
                return False

        p_target = rcm_pt + distance * new_axis
        q_target = self.kin.ik_with_rcm(p_target, rcm_pt, q_now)
        if q_target is None:
            self.get_logger().warn('IK failed for tilt; command ignored.')
            return False
        self.send_trajectory(q_target)
        return True

    # ------------------------------------------------------------------
    # RCM management
    # ------------------------------------------------------------------
    def enable_rcm(self):
        if not self.kin.ready:
            return False
        if not self.joint_received:
            self.get_logger().warn('No /joint_states yet; cannot set RCM.')
            return False
        with self.joint_lock:
            q = self.joint_positions.copy()
        p = self.kin.point_in_base(q, self.kin.rcm_point_local)
        axis, _, _ = self.kin.tube_axis_in_base(q)
        with self.rcm_lock:
            self.rcm_world_point = p
            self.rcm_ref_axis = axis
            self.rcm_enabled = True
        self.get_logger().info(
            f'RCM ENABLED  point=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  '
            f'ref_axis=({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})')
        return True

    def disable_rcm(self):
        with self.rcm_lock:
            self.rcm_enabled = False
            self.rcm_ref_axis = None
        self.get_logger().info('RCM DISABLED')

    # ==================================================================
    # Visualization
    # ==================================================================
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

        with self.joint_lock:
            q = self.joint_positions.copy()

        try:
            axis, p_rcm_fk, p_tip_world = self.kin.tube_axis_in_base(q)
        except Exception as e:
            self.get_logger().error(f'FK failed in marker publish: {e}')
            return

        with self.rcm_lock:
            rcm_on = self.rcm_enabled
            rcm_stored = (None if self.rcm_world_point is None
                          else self.rcm_world_point.copy())
            ref_axis = (None if self.rcm_ref_axis is None
                        else self.rcm_ref_axis.copy())

        use_fixed = (rcm_on and rcm_stored is not None
                     and ref_axis is not None)

        if use_fixed:
            anchor = rcm_stored
            display_axis = ref_axis

            cos_a = float(np.clip(np.dot(axis, ref_axis), -1.0, 1.0))
            drift_deg = float(np.rad2deg(np.arccos(cos_a)))
            drift_exceeded = (
                drift_deg > np.rad2deg(self.rcm_cone_half_angle))
            if drift_exceeded and not self._drift_was_exceeded:
                self.get_logger().warn(
                    f'Endoscope deviated {drift_deg:.1f} deg from the '
                    f'RCM reference axis (cone half-angle = '
                    f'{np.rad2deg(self.rcm_cone_half_angle):.1f} deg).')
            self._drift_was_exceeded = drift_exceeded
        else:
            anchor = p_rcm_fk
            display_axis = axis
            drift_deg = 0.0
            drift_exceeded = False
            self._drift_was_exceeded = False

        if not self._marker_first_published_logged:
            self._marker_first_published_logged = True
            self.get_logger().info(
                'Tube geometry (base frame): '
                f'RCM=({p_rcm_fk[0]:+.3f},{p_rcm_fk[1]:+.3f},'
                f'{p_rcm_fk[2]:+.3f}) '
                f'Tip=({p_tip_world[0]:+.3f},{p_tip_world[1]:+.3f},'
                f'{p_tip_world[2]:+.3f}) '
                f'axis=({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f})')

        # 1. Yellow line along the tube.
        depth = float(np.dot(p_tip_world - anchor, display_axis))
        front = max(depth, 0.05) + self.line_overshoot
        line_back = anchor - self.line_overshoot * display_axis
        line_front = anchor + front * display_axis

        line = Marker()
        line.header.frame_id = self.base_link
        line.header.stamp = stamp
        line.ns = 'endoscope_line'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = self.line_width
        if drift_exceeded:
            line.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
        else:
            line.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        line.pose.orientation.w = 1.0
        line.points = [
            Point(x=float(line_back[0]),
                  y=float(line_back[1]),
                  z=float(line_back[2])),
            Point(x=float(line_front[0]),
                  y=float(line_front[1]),
                  z=float(line_front[2])),
        ]
        ma.markers.append(line)

        # 2. RCM sphere.
        sphere = Marker()
        sphere.header.frame_id = self.base_link
        sphere.header.stamp = stamp
        sphere.ns = 'rcm_point'
        sphere.id = 1
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(anchor[0])
        sphere.pose.position.y = float(anchor[1])
        sphere.pose.position.z = float(anchor[2])
        sphere.pose.orientation.w = 1.0
        d = self.rcm_sphere_diameter
        sphere.scale.x = d
        sphere.scale.y = d
        sphere.scale.z = d
        if rcm_on:
            if drift_exceeded:
                sphere.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.55)
            else:
                sphere.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.45)
        else:
            sphere.color = ColorRGBA(r=0.55, g=0.55, b=0.55, a=0.35)
        ma.markers.append(sphere)

        # 3. Double cone.
        cone_color = (ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.28)
                      if drift_exceeded
                      else ColorRGBA(r=0.10, g=0.80, b=0.20, a=0.28))
        cone_fwd = self._build_cone_marker(
            apex=anchor, direction=+display_axis,
            length=self.rcm_cone_length,
            half_angle=self.rcm_cone_half_angle,
            color=cone_color, stamp=stamp, mid=2)
        cone_bwd = self._build_cone_marker(
            apex=anchor, direction=-display_axis,
            length=self.rcm_cone_length,
            half_angle=self.rcm_cone_half_angle,
            color=cone_color, stamp=stamp, mid=3)
        if cone_fwd is not None:
            ma.markers.append(cone_fwd)
        if cone_bwd is not None:
            ma.markers.append(cone_bwd)

        # 4. Cyan line = actual current tube axis.
        if use_fixed:
            cur_line = Marker()
            cur_line.header.frame_id = self.base_link
            cur_line.header.stamp = stamp
            cur_line.ns = 'endoscope_current_axis'
            cur_line.id = 4
            cur_line.type = Marker.LINE_STRIP
            cur_line.action = Marker.ADD
            cur_line.scale.x = self.line_width * 0.6
            cur_line.color = ColorRGBA(r=0.0, g=0.85, b=1.0, a=0.85)
            cur_line.pose.orientation.w = 1.0
            cur_line.points = [
                Point(x=float(p_rcm_fk[0]),
                      y=float(p_rcm_fk[1]),
                      z=float(p_rcm_fk[2])),
                Point(x=float(p_tip_world[0]),
                      y=float(p_tip_world[1]),
                      z=float(p_tip_world[2])),
            ]
            ma.markers.append(cur_line)

        self.marker_pub.publish(ma)

        if not hasattr(self, '_markers_pub_logged'):
            self._markers_pub_logged = True
            self.get_logger().info(
                f'Publishing {len(ma.markers)} markers on '
                f'{self.marker_topic}.')

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