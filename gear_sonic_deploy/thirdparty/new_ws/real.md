# 编译指令
colcon build --packages-skip noetix_gazebo

# 指定编译包
colcon build --packages-select noetix_policy 

# 启动指令
source install/setup.bash

- 真机启动 robot_name:=机器型号名称，目前支持 bumi2, bumi3, bumi4340, n3
ros2 launch noetix_bringup real.launch.py robot_name:=bumi3

- 真机启动后可以切换模式，需要切换策略启动下面策略
ros2 launch noetix_policy policy.launch.py robot_name:=bumi3