from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur5_endo_flex',
            executable='control_panel_gui.py',
            name='ur5_endo_flex_control_panel',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'tip_link': 'endoscope_link',
                'controller_topic':
                    '/scaled_joint_trajectory_controller/joint_trajectory',
                'step_size_mm': 5.0,
                'tilt_step_deg': 5.0,
                'traj_duration_s': 1.0,
                'rcm_point_in_tip_frame': [0.0, 0.0, -0.30],
                'endoscope_tip_in_link_frame': [0.0, 0.0, 0.20195],
                'marker_topic': '/endoscope_markers',
                'marker_publish_period': 0.1,
                'markers_enabled': True,
                'rcm_cone_half_angle_deg': 20.0,
                'rcm_cone_length': 0.15,
                'rcm_sphere_diameter': 0.02,
                'endoscope_line_width': 0.003,
                # Flex kinematics (kept in this node for self-containment)
                'flex_N': 8,
                'flex_SEG_LEN': 0.008,
                'flex_ARC_MIN': 0.030,
                'flex_ARC_MAX': 0.050,
                'flex_TENDON_R': 0.004,
                # Continuous-control velocity limits
                'arm_linear_vel_m_s': 0.050,          # 50 mm/s
                'arm_axis_vel_m_s':  0.050,          # 50 mm/s along tube
                'arm_tilt_vel_deg_s': 30.0,
                'flex_yaw_vel_rad_s': 0.5,
                'flex_curv_vel_1pm_s': 5.0,
                'control_tick_hz': 20.0,
            }],
        ),
    ])
