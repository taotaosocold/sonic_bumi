cmake_minimum_required(VERSION 3.8)
project(noetix_gazebo)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(noetix_msgs REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(noetix_gazebo_backend SHARED
  src/noetix_gazebo_backend.cpp
)
target_compile_features(noetix_gazebo_backend PUBLIC cxx_std_17)
target_include_directories(noetix_gazebo_backend PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
)
target_link_libraries(noetix_gazebo_backend
  ${GAZEBO_LIBRARIES}
)
ament_target_dependencies(noetix_gazebo_backend
  rclcpp
  builtin_interfaces
  sensor_msgs
  std_msgs
  noetix_msgs
  gazebo_ros
)

install(TARGETS noetix_gazebo_backend
  LIBRARY DESTINATION lib
)

install(DIRECTORY
  launch
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sim_startup_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
