# ur5_sim.launch.py
#
# Gazebo simulation + ros2_control for the UR5 + endoscope.
# No MoveIt, no motion planning plugin.
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ur_sim_gz_share = get_package_share_directory('ur_simulation_gz')
    ur5_endo_share  = get_package_share_directory('ur5_endoscope')

    description_file = os.path.join(
        ur5_endo_share, 'urdf', 'ur5_with_endoscope.urdf.xacro')

    # RViz config shipped in this package (see config/view.rviz).
    rviz_config = os.path.join(ur5_endo_share, 'config', 'view.rviz')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_sim_gz_share, 'launch',
                         'ur_sim_control.launch.py')),
        launch_arguments={
            'ur_type':            'ur5',
            'description_file':   description_file,
            'use_sim_time':       'true',
            'launch_rviz':        'true',
            'rviz_config_file':   rviz_config,
            'safety_limits':      'false',
            'safety_pos_margin':  '0.15',
            'safety_k_position':  '20',
        }.items(),
    )

    # Move the simulated robot to a defined home pose so /joint_states
    # (and therefore RViz) reflects the home configuration on first load.
    initial_pose = Node(
        package='ur5_endoscope',
        executable='initial_pose_publisher.py',
        name='initial_pose_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'home_positions_deg':     [0.0, -120.0, -120.0, -30.0, 90.0, 45.0],
            'trajectory_duration_s':  3.0,
            'wait_for_joint_states':  True,
        }],
    )

    return LaunchDescription([sim, initial_pose])