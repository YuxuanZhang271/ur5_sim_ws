#!/usr/bin/env python3
"""
gastrectomy_env_setup.py — Launch-time Laparoscopic Gastrectomy Environment
Generator & Synchronizer for Gazebo Sim (`gz-sim`) and RViz.

Features:
  1. Automatically computes the exact HOME-pose distal camera (`end_cylinder`)
     position and horizon-leveled viewing frame from `/robot_description` via
     `EndoscopeKinematics` (with deterministic fallback if URDF is delayed).
  2. Procedurally generates a randomized **Laparoscopic Gastrectomy Surgical
     Scene** at realistic human upper-abdominal scale (~22 x 18 x 12 cm
     pneumoperitoneum cavity, 9-20 cm in front of `end_cylinder`):
       - Surrounding gastrectomy anatomy (randomly jittered in pose & scale):
           * `stomach_corpus`    (gastric body ellipsoid, serosa coral-pink)
           * `stomach_antrum`    (pyloric antrum capsule, warm mucosa pink)
           * `liver_left_lobe`   (superior hepatic lobe dome, dark red-brown)
           * `pancreas_bed`      (posterior pancreas body, lobular tan-yellow)
           * `spleen_pole`       (left lateral splenic pole, deep maroon)
       - 4 Trackable Surgical / Lymph-Node Targets arranged across the 3 zones
         of the Flex-Primary / Arm-Auxiliary RCM controller:
           * `LN_No3`  (Red)    : Lesser Curvature Station No. 3 -> Zone 1 (Pure Flex)
           * `LN_No4d` (Green)  : Greater Curvature Station No. 4d -> Zone 2 (Coop Blend)
           * `LN_No6`  (Blue)   : Infrapyloric Station No. 6 -> Zone 3 (Arm Assist + Depth)
           * `LGA_No7` (Yellow) : Left Gastric Artery Station No. 7 -> Dynamic Breathing
                                  & Pulsation Target (+/-2.2 cm at 0.22 Hz)
  3. Spawns the complete multi-link SDF model (`gastrectomy_phantom`) into
     Gazebo (`ros_gz_sim create`) so the distal endoscope camera
     (`/endoscope_camera/image_raw`) renders all organs and targets in 3D, AND
     drives the dynamic target joint via `/gastrectomy/lga_cmd`.
  4. Publishes synchronized RViz `MarkerArray` on `/endoscope_markers` and live
     target 3D coordinates on `/gastrectomy/targets_state` at 20 Hz.
  5. Subscribes to `/gastrectomy/randomize` (`std_msgs/Empty`) to re-randomize
     the surgical scene on demand from the GUI or CLI.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time

# Ensure sibling scripts in lib/ur5_endo_flex or src/.../scripts are on sys.path
for _d in (
    os.path.dirname(os.path.abspath(__file__)),
    os.path.dirname(os.path.realpath(__file__)),
):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,
)
from std_msgs.msg import String, Empty, ColorRGBA, Float64
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from endoscope_kinematics import EndoscopeKinematics, HOME_DEG


# Canonical target keys and display metadata
TARGET_ORDER = ['LN_No3', 'LN_No4d', 'LN_No6', 'LGA_No7']

TARGET_SPECS = {
    'LN_No3': {
        'label': 'LN_No3 (Lesser Curv - Zone 1)',
        'short': 'LN3 Red',
        'color_rgba': (1.00, 0.15, 0.15, 0.96),
        'radius_m': 0.0085,
        'dynamic': False,
    },
    'LN_No4d': {
        'label': 'LN_No4d (Greater Curv - Zone 2)',
        'short': 'LN4d Grn',
        'color_rgba': (0.12, 0.92, 0.25, 0.96),
        'radius_m': 0.0090,
        'dynamic': False,
    },
    'LN_No6': {
        'label': 'LN_No6 (Infrapyloric - Zone 3)',
        'short': 'LN6 Blu',
        'color_rgba': (0.15, 0.48, 1.00, 0.96),
        'radius_m': 0.0095,
        'dynamic': False,
    },
    'LGA_No7': {
        'label': 'LGA_No7 (Left Gastric Art - Dynamic)',
        'short': 'LGA7 Dyn',
        'color_rgba': (1.00, 0.86, 0.05, 0.98),
        'radius_m': 0.0095,
        'dynamic': True,
    },
}


def _rot_To_quat_xyzw(R: np.ndarray):
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    nrm = math.sqrt(x * x + y * y + z * z + w * w)
    if nrm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / nrm, y / nrm, z / nrm, w / nrm)


def _rot_to_rpy(R: np.ndarray):
    """Convert a 3x3 rotation matrix to (roll, pitch, yaw) in radians."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return (roll, pitch, yaw)


class GastrectomyEnvSetupNode(Node):
    """Spawns and synchronizes a randomized gastrectomy scene in Gazebo & RViz."""

    def __init__(self):
        super().__init__('gastrectomy_env_setup')

        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'endoscope_link')
        self.declare_parameter('rcm_point_in_tip_frame', [0.0, 0.0, -0.30])
        self.declare_parameter(
            'endoscope_tip_in_link_frame', [0.0, 0.0, 0.20195]
        )
        self.declare_parameter('marker_topic', '/endoscope_markers')
        self.declare_parameter('targets_topic', '/gastrectomy/targets_state')
        self.declare_parameter('lga_cmd_topic', '/gastrectomy/lga_cmd')
        self.declare_parameter('random_seed', -1)
        self.declare_parameter('spawn_in_gazebo', True)
        self.declare_parameter('breathing_freq_hz', 0.22)
        self.declare_parameter('breathing_amp_m', 0.022)

        self.base_link = self.get_parameter('base_link').value
        self.tip_link = self.get_parameter('tip_link').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.targets_topic = self.get_parameter('targets_topic').value
        self.lga_cmd_topic = self.get_parameter('lga_cmd_topic').value
        self.spawn_in_gazebo = bool(
            self.get_parameter('spawn_in_gazebo').value
        )
        self.breathing_freq = float(
            self.get_parameter('breathing_freq_hz').value
        )
        self.breathing_amp = float(
            self.get_parameter('breathing_amp_m').value
        )

        seed_param = int(self.get_parameter('random_seed').value)
        if seed_param >= 0:
            self._rng = random.Random(seed_param)
        else:
            self._rng = random.Random()

        self.kin = EndoscopeKinematics(
            base_link=self.base_link,
            tip_link=self.tip_link,
            rcm_point_local=self.get_parameter('rcm_point_in_tip_frame').value,
            tip_local=self.get_parameter('endoscope_tip_in_link_frame').value,
            logger=self.get_logger(),
        )

        self._lock = threading.Lock()
        self._scene_initialized = False
        self._startup_time = time.monotonic()
        self._sim_t0 = time.monotonic()

        self._p_cam_home = np.array([-0.145, 0.085, 0.165], dtype=float)
        self._R_cam_home = np.eye(3, dtype=float)
        self._base_in_world = np.array([0.0, 0.0, 0.80], dtype=float)

        self._organs = []
        self._targets = {}
        self._breathing_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        self._model_generation = 0

        pub_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, self.marker_topic, pub_qos
        )
        self.targets_pub = self.create_publisher(
            String, self.targets_topic, pub_qos
        )
        self.lga_cmd_pub = self.create_publisher(
            Float64, self.lga_cmd_topic, pub_qos
        )

        urdf_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            String, '/robot_description', self._urdf_cb, urdf_qos
        )
        self.create_subscription(
            Empty, '/gastrectomy/randomize', self._randomize_cb, pub_qos
        )

        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            'GastrectomyEnvSetupNode started; waiting for /robot_description '
            'to anchor gastrectomy surgical cavity at HOME distal camera pose.'
        )

    def _urdf_cb(self, msg: String):
        if not msg.data:
            return
        if not self.kin.ready:
            if self.kin.setup_from_urdf(msg.data):
                self._compute_home_camera_anchor()
                self.randomize_environment()

    def _randomize_cb(self, _msg: Empty):
        self.get_logger().info(
            'Received /gastrectomy/randomize — generating new randomized '
            'gastrectomy anatomy & target layout.'
        )
        self.randomize_environment()

    def _compute_home_camera_anchor(self):
        q_home = np.deg2rad(np.array(HOME_DEG, dtype=float))
        q_home = self.kin.solve_wrist1_for_out(q_home)
        R_end, p_end, _ = self.kin.flex_fk_in_base(
            q_home, curvature=0.0, steer_yaw=0.0, arc_len=0.050, flex_mode='fixed'
        )
        with self._lock:
            self._p_cam_home = p_end.copy()
            self._R_cam_home = R_end.copy()
            self._base_in_world = self.kin.base_in_world_xyz.copy()

        self.get_logger().info(
            f'Anchored gastrectomy workspace to HOME end_cylinder at '
            f'base_link=({p_end[0]:+.3f}, {p_end[1]:+.3f}, {p_end[2]:+.3f}) m.'
        )

    def _cam_local_to_base(self, x_down: float, y_horiz: float, z_fwd: float) -> np.ndarray:
        v_cam = np.array([x_down, y_horiz, z_fwd], dtype=float)
        return self._p_cam_home + self._R_cam_home @ v_cam

    def _sample_spherical_in_cam(self, theta_deg_range, phi_deg_range, dist_m_range):
        theta = math.radians(self._rng.uniform(*theta_deg_range))
        phi = math.radians(self._rng.uniform(*phi_deg_range))
        dist = self._rng.uniform(*dist_m_range)
        x_down = dist * math.sin(theta) * math.cos(phi)
        y_horiz = dist * math.sin(theta) * math.sin(phi)
        z_fwd = dist * math.cos(theta)
        return self._cam_local_to_base(x_down, y_horiz, z_fwd)

    def randomize_environment(self):
        with self._lock:
            R_c = self._R_cam_home.copy()
            rng = self._rng

            sampled_targets = {}

            # Zone 1 target: LN_No3 (Lesser Curvature, 16-36 deg off-axis, 10.5-13.5 cm)
            p_ln3 = self._sample_spherical_in_cam(
                theta_deg_range=(16.0, 36.0),
                phi_deg_range=(-45.0, 45.0),
                dist_m_range=(0.105, 0.135),
            )
            sampled_targets['LN_No3'] = p_ln3

            # Zone 2 target: LN_No4d (Greater Curvature, 52-66 deg off-axis, 11.5-15.0 cm)
            for _ in range(20):
                p_ln4d = self._sample_spherical_in_cam(
                    theta_deg_range=(52.0, 66.0),
                    phi_deg_range=(35.0, 115.0) if rng.random() < 0.5 else (-115.0, -35.0),
                    dist_m_range=(0.115, 0.150),
                )
                if np.linalg.norm(p_ln4d - p_ln3) > 0.040:
                    break
            sampled_targets['LN_No4d'] = p_ln4d

            # Zone 3 target: LN_No6 (Infrapyloric / Extreme Lateral, 68-80 deg off-axis, 14.5-18.0 cm)
            for _ in range(20):
                p_ln6 = self._sample_spherical_in_cam(
                    theta_deg_range=(68.0, 80.0),
                    phi_deg_range=(-120.0, 120.0),
                    dist_m_range=(0.145, 0.180),
                )
                if (np.linalg.norm(p_ln6 - p_ln3) > 0.045
                        and np.linalg.norm(p_ln6 - p_ln4d) > 0.045):
                    break
            sampled_targets['LN_No6'] = p_ln6

            # Dynamic target: LGA_No7 (Left Gastric Artery, 24-48 deg off-axis, 11.0-14.0 cm)
            for _ in range(20):
                p_lga = self._sample_spherical_in_cam(
                    theta_deg_range=(24.0, 48.0),
                    phi_deg_range=(-80.0, 80.0),
                    dist_m_range=(0.110, 0.140),
                )
                if all(np.linalg.norm(p_lga - pt) > 0.038
                       for pt in sampled_targets.values()):
                    break
            sampled_targets['LGA_No7'] = p_lga

            angle_b = rng.uniform(-0.6, 0.6)
            b_cam = np.array([
                math.sin(angle_b),
                math.cos(angle_b),
                0.25 * math.sin(angle_b),
            ], dtype=float)
            b_cam /= np.linalg.norm(b_cam)
            self._breathing_axis = R_c @ b_cam

            self._targets = {}
            for key in TARGET_ORDER:
                spec = TARGET_SPECS[key]
                p_base = sampled_targets[key]
                self._targets[key] = {
                    'name': key,
                    'label': spec['label'],
                    'short': spec['short'],
                    'p_origin_base': p_base.copy(),
                    'p_current_base': p_base.copy(),
                    'radius_m': spec['radius_m'],
                    'color_rgba': spec['color_rgba'],
                    'dynamic': spec['dynamic'],
                }

            jx = lambda s=0.012: rng.uniform(-s, s)
            ja = lambda deg=12.0: math.radians(rng.uniform(-deg, deg))

            def make_organ(name, geom_type, cam_xyz, cam_rpy, scale_xyz, rgba):
                p_b = self._cam_local_to_base(*cam_xyz)
                R_local = (
                    EndoscopeKinematics._rodrigues_matrix(np.array([0, 0, 1.0]), cam_rpy[2])
                    @ EndoscopeKinematics._rodrigues_matrix(np.array([0, 1.0, 0]), cam_rpy[1])
                    @ EndoscopeKinematics._rodrigues_matrix(np.array([1.0, 0, 0]), cam_rpy[0])
                )
                R_b = R_c @ R_local
                return {
                    'name': name,
                    'type': geom_type,
                    'p_base': p_b,
                    'R_base': R_b,
                    'scale_xyz': scale_xyz,
                    'color_rgba': rgba,
                }

            self._organs = [
                make_organ(
                    name='stomach_corpus',
                    geom_type='ellipsoid',
                    cam_xyz=(0.055 + jx(0.010), 0.015 + jx(0.015), 0.175 + jx(0.010)),
                    cam_rpy=(0.25 + ja(10), -0.35 + ja(10), 0.40 + ja(15)),
                    scale_xyz=(0.048, 0.105, 0.042),
                    rgba=(0.88, 0.48, 0.44, 0.85),
                ),
                make_organ(
                    name='stomach_antrum',
                    geom_type='ellipsoid',
                    cam_xyz=(0.075 + jx(0.010), -0.045 + jx(0.012), 0.170 + jx(0.010)),
                    cam_rpy=(-0.20 + ja(10), 0.30 + ja(10), -0.50 + ja(15)),
                    scale_xyz=(0.036, 0.068, 0.034),
                    rgba=(0.84, 0.43, 0.40, 0.85),
                ),
                make_organ(
                    name='liver_left_lobe',
                    geom_type='ellipsoid',
                    cam_xyz=(-0.055 + jx(0.008), -0.015 + jx(0.015), 0.165 + jx(0.010)),
                    cam_rpy=(0.10 + ja(8), 0.15 + ja(8), 0.15 + ja(12)),
                    scale_xyz=(0.036, 0.115, 0.072),
                    rgba=(0.56, 0.19, 0.16, 0.85),
                ),
                make_organ(
                    name='pancreas_bed',
                    geom_type='ellipsoid',
                    cam_xyz=(0.020 + jx(0.008), 0.005 + jx(0.012), 0.195 + jx(0.008)),
                    cam_rpy=(0.05 + ja(6), 0.10 + ja(6), 0.20 + ja(10)),
                    scale_xyz=(0.028, 0.098, 0.030),
                    rgba=(0.91, 0.76, 0.45, 0.85),
                ),
                make_organ(
                    name='spleen_pole',
                    geom_type='ellipsoid',
                    cam_xyz=(0.015 + jx(0.010), 0.085 + jx(0.012), 0.170 + jx(0.010)),
                    cam_rpy=(-0.15 + ja(10), 0.20 + ja(10), -0.30 + ja(12)),
                    scale_xyz=(0.042, 0.046, 0.068),
                    rgba=(0.46, 0.16, 0.36, 0.85),
                ),
            ]

            self._scene_initialized = True
            self._model_generation += 1
            gen_id = self._model_generation

        if self.spawn_in_gazebo:
            threading.Thread(
                target=self._spawn_sdf_in_gazebo,
                args=(gen_id,),
                daemon=True,
            ).start()

    def _build_sdf_xml(self) -> str:
        with self._lock:
            offset_w = self._base_in_world.copy()
            organs = list(self._organs)
            targets = {k: dict(v) for k, v in self._targets.items()}
            b_axis = self._breathing_axis.copy()

        lines = [
            '<?xml version="1.0" ?>',
            '<sdf version="1.9">',
            '  <model name="gastrectomy_phantom">',
            '    <static>false</static>',
            '    <link name="phantom_root">',
            '      <pose>0 0 0 0 0 0</pose>',
            '      <inertial>',
            '        <mass>10.0</mass>',
            '        <inertia><ixx>0.1</ixx><ixy>0</ixy><ixz>0</ixz>'
            '<iyy>0.1</iyy><iyz>0</iyz><izz>0.1</izz></inertia>',
            '      </inertial>',
            '      <gravity>false</gravity>',
            '      <kinematic>true</kinematic>',
        ]

        for org in organs:
            p_w = org['p_base'] + offset_w
            r, p, y = _rot_to_rpy(org['R_base'])
            sx, sy, sz = org['scale_xyz']
            cr, cg, cb, ca = org['color_rgba']
            lines.extend([
                f'      <visual name="vis_{org["name"]}">',
                f'        <pose>{p_w[0]:.4f} {p_w[1]:.4f} {p_w[2]:.4f} '
                f'{r:.4f} {p:.4f} {y:.4f}</pose>',
                '        <geometry>',
                '          <ellipsoid>',
                f'            <radii>{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}</radii>',
                '          </ellipsoid>',
                '        </geometry>',
                '        <material>',
                f'          <ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1.0</ambient>',
                f'          <diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1.0</diffuse>',
                '          <specular>0.35 0.35 0.35 1.0</specular>',
                '        </material>',
                '      </visual>',
            ])

        for key in TARGET_ORDER:
            t = targets[key]
            if t['dynamic']:
                continue
            p_w = t['p_origin_base'] + offset_w
            rad = t['radius_m']
            cr, cg, cb, _ = t['color_rgba']
            lines.extend([
                f'      <visual name="vis_target_{key}">',
                f'        <pose>{p_w[0]:.4f} {p_w[1]:.4f} {p_w[2]:.4f} 0 0 0</pose>',
                '        <geometry>',
                f'          <sphere><radius>{rad:.4f}</radius></sphere>',
                '        </geometry>',
                '        <material>',
                f'          <ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1.0</ambient>',
                f'          <diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1.0</diffuse>',
                '          <specular>0.6 0.6 0.6 1.0</specular>',
                '        </material>',
                '      </visual>',
            ])

        lines.append('    </link>')
        lines.extend([
            '    <joint name="world_to_phantom_root" type="fixed">',
            '      <parent>world</parent>',
            '      <child>phantom_root</child>',
            '    </joint>',
        ])

        t_dyn = targets['LGA_No7']
        p_dyn_w = t_dyn['p_origin_base'] + offset_w
        rad_dyn = t_dyn['radius_m']
        cr, cg, cb, _ = t_dyn['color_rgba']
        lines.extend([
            '    <link name="lga_dynamic_link">',
            f'      <pose>{p_dyn_w[0]:.4f} {p_dyn_w[1]:.4f} {p_dyn_w[2]:.4f} 0 0 0</pose>',
            '      <gravity>false</gravity>',
            '      <inertial>',
            '        <mass>0.005</mass>',
            '        <inertia><ixx>1e-6</ixx><ixy>0</ixy><ixz>0</ixz>'
            '<iyy>1e-6</iyy><iyz>0</iyz><izz>1e-6</izz></inertia>',
            '      </inertial>',
            '      <visual name="vis_target_LGA_No7">',
            '        <geometry>',
            f'          <sphere><radius>{rad_dyn:.4f}</radius></sphere>',
            '        </geometry>',
            '        <material>',
            f'          <ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1.0</ambient>',
            f'          <diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1.0</diffuse>',
            '          <specular>0.8 0.8 0.8 1.0</specular>',
            '        </material>',
            '      </visual>',
            '    </link>',
            '    <joint name="lga_breathing_joint" type="prismatic">',
            '      <parent>phantom_root</parent>',
            '      <child>lga_dynamic_link</child>',
            '      <axis>',
            f'        <xyz>{b_axis[0]:.4f} {b_axis[1]:.4f} {b_axis[2]:.4f}</xyz>',
            '        <limit><lower>-0.08</lower><upper>0.08</upper>'
            '<effort>50</effort><velocity>2.0</velocity></limit>',
            '        <dynamics><damping>0.2</damping></dynamics>',
            '      </axis>',
            '    </joint>',
            '    <plugin filename="gz-sim-joint-position-controller-system"',
            '            name="gz::sim::systems::JointPositionController">',
            '      <joint_name>lga_breathing_joint</joint_name>',
            f'      <topic>{self.lga_cmd_topic}</topic>',
            '      <use_velocity_commands>true</use_velocity_commands>',
            '      <p_gain>20.0</p_gain>',
            '      <i_gain>0.0</i_gain>',
            '      <d_gain>0.0</d_gain>',
            '      <cmd_max>2.0</cmd_max>',
            '      <cmd_min>-2.0</cmd_min>',
            '      <initial_position>0.0</initial_position>',
            '    </plugin>',
            '  </model>',
            '</sdf>',
        ])
        return '\n'.join(lines)

    def _spawn_sdf_in_gazebo(self, gen_id: int):
        sdf_xml = self._build_sdf_xml()
        sdf_path = os.path.join(
            tempfile.gettempdir(), 'gastrectomy_phantom.sdf'
        )
        try:
            with open(sdf_path, 'w', encoding='utf-8') as f:
                f.write(sdf_xml)
        except Exception as e:
            self.get_logger().warn(f'Failed to write SDF file: {e}')
            return

        if gen_id > 1:
            try:
                subprocess.run(
                    [
                        'gz', 'service', '-s', '/world/default/remove',
                        '--reqtype', 'gz.msgs.Entity',
                        '--reptype', 'gz.msgs.Boolean',
                        '--timeout', '1500',
                        '--req', 'name: "gastrectomy_phantom", type: MODEL',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.5,
                )
            except Exception:
                pass

        try:
            res = subprocess.run(
                [
                    'ros2', 'run', 'ros_gz_sim', 'create',
                    '-name', 'gastrectomy_phantom',
                    '-file', sdf_path,
                ],
                capture_output=True,
                text=True,
                timeout=8.0,
            )
            if res.returncode == 0:
                self.get_logger().info(
                    f'Spawned gastrectomy_phantom (gen #{gen_id}) in Gazebo.'
                )
            else:
                self.get_logger().debug(
                    f'ros_gz_sim create exited with {res.returncode}: {res.stderr}'
                )
        except Exception as e:
            self.get_logger().debug(f'Gazebo spawn skipped/failed: {e}')

    def _tick(self):
        if not self._scene_initialized:
            if (time.monotonic() - self._startup_time) > 2.5:
                self.randomize_environment()
            else:
                return

        t_elapsed = time.monotonic() - self._sim_t0
        lga_disp = self.breathing_amp * math.sin(
            2.0 * math.pi * self.breathing_freq * t_elapsed
        )

        self.lga_cmd_pub.publish(Float64(data=float(lga_disp)))

        with self._lock:
            if 'LGA_No7' in self._targets:
                p0 = self._targets['LGA_No7']['p_origin_base']
                self._targets['LGA_No7']['p_current_base'] = (
                    p0 + lga_disp * self._breathing_axis
                )

            payload = {
                'targets': {
                    k: {
                        'name': v['name'],
                        'label': v['label'],
                        'short': v['short'],
                        'pos_base': [
                            float(v['p_current_base'][0]),
                            float(v['p_current_base'][1]),
                            float(v['p_current_base'][2]),
                        ],
                        'radius_m': float(v['radius_m']),
                        'color_rgba': list(v['color_rgba']),
                        'dynamic': bool(v['dynamic']),
                    }
                    for k, v in self._targets.items()
                }
            }
            organs_snap = list(self._organs)
            targets_snap = {k: dict(v) for k, v in self._targets.items()}

        self.targets_pub.publish(String(data=json.dumps(payload)))

        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

        for idx, org in enumerate(organs_snap):
            m = Marker()
            m.header.frame_id = self.base_link
            m.header.stamp = stamp
            m.ns = 'gastrectomy_anatomy'
            m.id = 200 + idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            p_b = org['p_base']
            qx, qy, qz, qw = _rot_To_quat_xyzw(org['R_base'])
            m.pose.position.x = float(p_b[0])
            m.pose.position.y = float(p_b[1])
            m.pose.position.z = float(p_b[2])
            m.pose.orientation.x = float(qx)
            m.pose.orientation.y = float(qy)
            m.pose.orientation.z = float(qz)
            m.pose.orientation.w = float(qw)
            sx, sy, sz = org['scale_xyz']
            m.scale.x = float(sx)
            m.scale.y = float(sy)
            m.scale.z = float(sz)
            cr, cg, cb, ca = org['color_rgba']
            m.color = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=float(ca))
            ma.markers.append(m)

        for idx, key in enumerate(TARGET_ORDER):
            if key not in targets_snap:
                continue
            t = targets_snap[key]
            p_b = t['p_current_base']
            rad = float(t['radius_m'])
            cr, cg, cb, ca = t['color_rgba']

            ms = Marker()
            ms.header.frame_id = self.base_link
            ms.header.stamp = stamp
            ms.ns = 'gastrectomy_targets'
            ms.id = 210 + idx
            ms.type = Marker.SPHERE
            ms.action = Marker.ADD
            ms.pose.position.x = float(p_b[0])
            ms.pose.position.y = float(p_b[1])
            ms.pose.position.z = float(p_b[2])
            ms.pose.orientation.w = 1.0
            ms.scale.x = 2.0 * rad
            ms.scale.y = 2.0 * rad
            ms.scale.z = 2.0 * rad
            ms.color = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=float(ca))
            ma.markers.append(ms)

            ml = Marker()
            ml.header.frame_id = self.base_link
            ml.header.stamp = stamp
            ml.ns = 'gastrectomy_labels'
            ml.id = 220 + idx
            ml.type = Marker.TEXT_VIEW_FACING
            ml.action = Marker.ADD
            ml.pose.position.x = float(p_b[0])
            ml.pose.position.y = float(p_b[1])
            ml.pose.position.z = float(p_b[2] + rad + 0.012)
            ml.pose.orientation.w = 1.0
            ml.scale.z = 0.014
            ml.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            ml.text = key
            ma.markers.append(ml)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = GastrectomyEnvSetupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()