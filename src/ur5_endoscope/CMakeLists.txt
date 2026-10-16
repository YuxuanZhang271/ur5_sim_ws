cmake_minimum_required(VERSION 3.8)
project(ur5_endoscope)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY meshes urdf config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/control_panel.py
         scripts/initial_pose_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()