cmake_minimum_required(VERSION 3.8)
project(ur5_endo_flex)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf launch config meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/initial_pose_publisher.py
           scripts/ur5_endo_flex_node.py
           scripts/control_panel_gui.py
           scripts/endoscope_kinematics.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
