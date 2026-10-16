#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('endo_flex_control')

    xacro_file = os.path.join(pkg_share, 'urdf', 'endo_flex.xacro')
    rviz_file  = os.path.join(pkg_share, 'rviz', 'endo_flex.rviz')

    # ---- Launch args ----
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Whether to start RViz'
    )

    # ---- robot_description from xacro ----
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # ---- Optional joint_state_publisher_gui (useful for debugging without teleop) ----
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # ---- Teleop node (publishes /joint_states) ----
    teleop_node = Node(
        package='endo_flex_control',
        executable='endoflex_teleop',
        name='endoscope_teleop',
        output='screen'
    )

    # ---- RViz ----
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        use_rviz_arg,
        robot_state_publisher,
        # joint_state_publisher_gui,   # <-- uncomment if you want manual sliders
        teleop_node,
        rviz_node,
    ])