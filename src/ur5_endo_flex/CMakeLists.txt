cmake_minimum_required(VERSION 3.8)
project(ur5_endo_flex)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf launch config meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/home_pose_publisher.py
           scripts/kinematics.py
           scripts/scene_setup.py
           scripts/robot_node.py
           scripts/gui_panel.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
