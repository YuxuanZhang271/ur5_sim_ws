cmake_minimum_required(VERSION 3.8)
project(ur5_endo_flex)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf launch config meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/initial_pose_publisher.py
           scripts/endoscope_kinematics.py
           scripts/gastrectomy_env_setup.py
           scripts/ur5_endo_flex_level1_io.py
           scripts/ur5_endo_flex_level2_cmd.py
           scripts/ur5_endo_flex_level3_intelligent.py
           scripts/ur5_endo_flex_node.py
           scripts/control_panel_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
