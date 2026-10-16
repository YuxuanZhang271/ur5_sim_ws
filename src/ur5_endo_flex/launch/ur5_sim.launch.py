# ur5_sim.launch.py
#
# Unified Gazebo simulation + ros2_control + RViz + Tkinter control panel
# + ros_gz_bridge for the distal endoscope camera at end_cylinder and
# the endo_flex Gazebo joint position controllers.
# No MoveIt, no motion planning plugin.
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ur_sim_gz_share = get_package_share_directory('ur_simulation_gz')
    ur5_endo_flex_share = get_package_share_directory('ur5_endo_flex')

    # gz-sim resolves model:// URIs by walking GZ_SIM_RESOURCE_PATH for a
    # directory named after the "package" that contains a model.config shim.
    # Register both mesh packages; include each share dir and its parent so
    # the entry works for both isolated and merged colcon install layouts.
    ur_description_share = get_package_share_directory('ur_description')
    ur5_endoscope_share = get_package_share_directory('ur5_endoscope')
    gz_resource_path = ':'.join(filter(None, [
        os.path.dirname(ur_description_share),
        ur_description_share,
        os.path.dirname(ur5_endoscope_share),
        ur5_endoscope_share,
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ]))

    description_file = os.path.join(
        ur5_endo_flex_share, 'urdf', 'ur5_endo_flex.urdf.xacro')

    # RViz config shipped in this package (see config/view.rviz).
    rviz_config = os.path.join(ur5_endo_flex_share, 'config', 'view.rviz')

    launch_gui_arg = DeclareLaunchArgument(
        'launch_gui',
        default_value='true',
        description='Launch the Tkinter control panel GUI (if false, runs headless control node).',
    )
    launch_gui = LaunchConfiguration('launch_gui')

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

    # Bridge Gazebo's endoscope camera topics (/endoscope_camera/image_raw and
    # /endoscope_camera/camera_info) to ROS 2 sensor_msgs, AND bridge the 4
    # endo_flex JointPositionController command topics from ROS 2 to Gazebo so
    # Gazebo physically rotates and bends the continuum joints with end_cylinder.
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='endoscope_camera_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '/endoscope_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/endoscope_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/flex/base_yaw@std_msgs/msg/Float64]gz.msgs.Double',
            '/flex/ext_yaw@std_msgs/msg/Float64]gz.msgs.Double',
            '/flex/ext_pitch@std_msgs/msg/Float64]gz.msgs.Double',
            '/flex/ext_yaw_back@std_msgs/msg/Float64]gz.msgs.Double',
        ],
    )

    # Move the simulated robot to a defined home pose so /joint_states
    # (and therefore RViz) reflects the home configuration on first load.
    initial_pose = Node(
        package='ur5_endo_flex',
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

    shared_node_params = {
        'use_sim_time': True,
        'base_link': 'base_link',
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
        'rcm_cone_length': 0.20,
        'rcm_sphere_diameter': 0.03,
        'endoscope_line_width': 0.006,
        'endoscope_line_overshoot': 0.05,
        # Flex kinematics
        'flex_N': 8,
        'flex_SEG_LEN': 0.008,
        'flex_ARC_MIN': 0.030,
        'flex_ARC_MAX': 0.050,
        'flex_TENDON_R': 0.004,
        # Continuous-control velocity limits
        'arm_linear_vel_m_s': 0.050,          # 50 mm/s
        'arm_axis_vel_m_s':  0.050,           # 50 mm/s along tube
        'arm_tilt_vel_deg_s': 30.0,
        'flex_yaw_vel_rad_s': 0.5,
        'flex_curv_vel_1pm_s': 5.0,
        'control_tick_hz': 20.0,
    }

    # Integrated GUI control panel (default). Runs a single Ur5EndoFlexNode
    # instance so /endoscope_markers and /joint_states have no duplicate publisher.
    control_panel_gui = Node(
        package='ur5_endo_flex',
        executable='control_panel_gui.py',
        name='ur5_endo_flex_control_panel',
        output='screen',
        parameters=[shared_node_params],
        condition=IfCondition(launch_gui),
    )

    # Fallback headless node when launch_gui:=false.
    flex_state_publisher_headless = Node(
        package='ur5_endo_flex',
        executable='ur5_endo_flex_node.py',
        name='ur5_endo_flex_node_headless',
        output='screen',
        parameters=[{
            **shared_node_params,
            'enable_keyboard': False,
        }],
        condition=UnlessCondition(launch_gui),
    )

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=gz_resource_path),
        launch_gui_arg,
        sim,
        camera_bridge,
        initial_pose,
        control_panel_gui,
        flex_state_publisher_headless,
    ])