from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from noetix_bringup.robot_resources import load_robot_resource


def _launch_robot(context):
    robot_name = LaunchConfiguration("robot_name").perform(context)
    resource = load_robot_resource(robot_name)
    

    control_frequency_text = LaunchConfiguration("control_frequency_hz").perform(context).strip()
    control_frequency_hz = (
        float(control_frequency_text)
        if control_frequency_text
        else 1000.0
    )

    description_file = resource.real_description

    if description_file.suffix == ".xacro":
        description_value = Command([
            FindExecutable(name="xacro"),
            " ",
            str(description_file),
            " robot_name:=", robot_name,
            " robot_root:=", str(resource.robot_root),
            " config_root:=", str(resource.robot_root),
        ])
        robot_description = ParameterValue(
            description_value, value_type=str)
    elif description_file.suffix == ".urdf":
        robot_description = description_file.read_text(
            encoding="utf-8")
    else:
        raise RuntimeError(
            f"Unsupported real robot description: "
            f"{description_file}"
        )

    network_text = ""
    if robot_name.startswith("bumi"):
        network_file = resource.robot_root.parent / "common" / "bumi" / "network.yaml"
        network_text = f"  Bumi network:    {network_file}\n"

    print(
        "\n[Noetix Real Resource]\n"
        f"  robot_name:      {resource.name}\n"
        f"  robot_root:      {resource.robot_root}\n"
        f"  robot.yaml:      {resource.robot_yaml}\n"
        f"  real_description:{description_file}\n"
        f"  hardware.yaml:   {resource.hardware_yaml}\n"
        f"{network_text}"
        f"  dof:             {resource.dof}\n"
        f"  control rate:    {control_frequency_hz:.1f} Hz\n"
    )

    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
            }],
        ),
        Node(
            package="noetix_bridge",
            executable="noetix_bridge_node",
            name="noetix_bridge_node",
            output="screen",
            parameters=[{
                "robot_name": robot_name,
                "control_frequency_hz": control_frequency_hz,
                "rt_realtime": ParameterValue(
                    LaunchConfiguration("rt_realtime"), value_type=bool),
                "rt_lock_memory": ParameterValue(
                    LaunchConfiguration("rt_lock_memory"), value_type=bool),
                "rt_strict": ParameterValue(
                    LaunchConfiguration("rt_strict"), value_type=bool),
                "rt_priority": ParameterValue(
                    LaunchConfiguration("rt_priority"), value_type=int),
                "rt_cpu": ParameterValue(
                    LaunchConfiguration("rt_cpu"), value_type=int),
            }],
        ),
        Node(
            package="noetix_bridge",
            executable="noetix_pose_mode_node",
            name="noetix_pose_mode_node",
            output="screen",
            parameters=[{
                "robot_name": robot_name,
            }],
        ),
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            output="screen",
            parameters=[{
                "device_id":
                    LaunchConfiguration(
                        "joy_device_id"),
                "deadzone":
                    LaunchConfiguration(
                        "joy_deadzone"),
                "autorepeat_rate":
                    20.0,
            }],
            condition=IfCondition(
                LaunchConfiguration(
                    "use_gamepad")
            ),
        ),
        Node(
            package="noetix_bridge",
            executable="noetix_gamepad_mode_node",
            name="noetix_gamepad_mode_node",
            output="screen",
            parameters=[{
                "robot_name": robot_name,
            }],
            condition=IfCondition(
                LaunchConfiguration("use_gamepad")
            ),
        ),
    ]

    if resource.real_rviz is not None:
        actions.insert(
            1,
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", str(resource.real_rviz)],
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("use_rviz")
                ),
            ),
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_name",
            default_value="n3",
            description=(
                "Robot directory name under "
                "share/noetix_robots/"
            ),
        ),
        DeclareLaunchArgument(
            "control_frequency_hz",
            default_value="",
            description=(
                "Low-level control rate. Empty selects 500 Hz for Bumi and "
                "1000 Hz for N3."
            ),
        ),
        DeclareLaunchArgument(
            "rt_realtime",
            default_value="true",
            description="Request SCHED_FIFO for the hardware control loop",
        ),
        DeclareLaunchArgument(
            "rt_lock_memory",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "rt_strict",
            default_value="false",
            description="Fail startup if RT scheduling or mlock setup fails",
        ),
        DeclareLaunchArgument(
            "rt_priority",
            default_value="80",
        ),
        DeclareLaunchArgument(
            "rt_cpu",
            default_value="-1",
            description="CPU core for the control thread; -1 leaves affinity unchanged",
        ),
        DeclareLaunchArgument(
            "use_gamepad",
            default_value="true",
            description="Start joystick mode-only adapter",
        ),
        DeclareLaunchArgument(
            "joy_device_id",
            default_value="0",
        ),
        DeclareLaunchArgument(
            "joy_deadzone",
            default_value="0.001",
            description="Raw joystick deadzone; policy command deadzone is applied separately",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz when a config is available",
        ),
        OpaqueFunction(function=_launch_robot),
    ])
