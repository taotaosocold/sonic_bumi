from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("noetix_policy"), "config", "policy_manager.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("manager_config", default_value=default_config),
        DeclareLaunchArgument("robot_name", default_value="bumi4340"),
        DeclareLaunchArgument("inference_provider", default_value=""),
        DeclareLaunchArgument("gpu_device_id", default_value="-1"),
        DeclareLaunchArgument("policy_hz", default_value="-1.0"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("trajectory_use_ros_time", default_value="false"),
        Node(
            package="noetix_policy",
            executable="policy_node",
            name="noetix_policy_node",
            output="screen",
            parameters=[{
                "manager_config": LaunchConfiguration("manager_config"),
                "robot_name": LaunchConfiguration("robot_name"),
                "inference_provider": LaunchConfiguration("inference_provider"),
                "gpu_device_id": ParameterValue(
                    LaunchConfiguration("gpu_device_id"), value_type=int),
                "policy_hz_override": ParameterValue(
                    LaunchConfiguration("policy_hz"), value_type=float),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
                "trajectory_use_ros_time": ParameterValue(
                    LaunchConfiguration("trajectory_use_ros_time"), value_type=bool),
            }],
        ),
    ])
