from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur5_endoscope',
            executable='control_panel.py',
            name='endoscope_control_panel',
            output='screen',
            parameters=[{
                # Required so the node's clock matches Gazebo.
                'use_sim_time': True,

                'base_link': 'base_link',
                'tip_link': 'endoscope_link',
                'controller_topic':
                    '/scaled_joint_trajectory_controller/joint_trajectory',
                'step_size_mm': 5.0,
                'traj_duration_s': 1.0,
                # Point on the STL Z-axis (link frame == STL frame):
                'rcm_point_in_tip_frame': [0.0, 0.0, -0.30],
                # Same axis, forward end:
                'endoscope_tip_in_link_frame': [0.0, 0.0, 0.20195],

                # ---------------- Visualization ----------------
                'marker_topic': '/endoscope_markers',
                'marker_publish_period': 0.1,
                'markers_enabled': True,
                'rcm_cone_half_angle_deg': 20.0,   # ±20°
                'rcm_cone_length': 0.15,           # 15 cm long cone
                'rcm_sphere_diameter': 0.02,       # 2 cm translucent ball
                'endoscope_line_width': 0.003,     # 3 mm thick yellow line
            }],
        ),
    ])
