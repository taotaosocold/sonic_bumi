#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _vec(text: str | None, n: int, default: float = 0.0) -> list[float]:
    if not text:
        return [default] * n
    values = [float(x) for x in text.split()]
    if len(values) != n:
        raise RuntimeError(f"Expected {n} numbers, got: {text}")
    return values


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mm(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def _xyz_rpy(xyz, rpy):
    x, y, z = xyz
    r, p, yaw = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # URDF fixed-axis RPY: Rz(yaw) * Ry(pitch) * Rx(roll)
    rot = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    out = _identity()
    for i in range(3):
        for j in range(3):
            out[i][j] = rot[i][j]
    out[0][3], out[1][3], out[2][3] = x, y, z
    return out


def _axis_angle(axis, angle):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1.0e-12 or abs(angle) < 1.0e-15:
        return _identity()
    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    out = _identity()
    out[0][0] = x * x * v + c
    out[0][1] = x * y * v - z * s
    out[0][2] = x * z * v + y * s
    out[1][0] = y * x * v + z * s
    out[1][1] = y * y * v + c
    out[1][2] = y * z * v - x * s
    out[2][0] = z * x * v - y * s
    out[2][1] = z * y * v + x * s
    out[2][2] = z * z * v + c
    return out


def _point(t, p):
    x, y, z = p
    return [
        t[0][0] * x + t[0][1] * y + t[0][2] * z + t[0][3],
        t[1][0] * x + t[1][1] * y + t[1][2] * z + t[1][3],
        t[2][0] * x + t[2][1] * y + t[2][2] * z + t[2][3],
    ]


def _pose_map(robot_yaml: dict, key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in robot_yaml.get(key, []) or []:
        result[str(item["name"])] = float(item.get("pos", 0.0))
    return result


def _link_transforms(
    urdf_root: ET.Element,
    base_link: str,
    joint_positions: dict[str, float],
) -> dict[str, list[list[float]]]:
    children: dict[str, list[tuple[str, str, list[float], list[float], list[float], str]]] = {}

    for joint in urdf_root.findall("joint"):
        name = str(joint.attrib["name"])
        joint_type = str(joint.attrib.get("type", "fixed"))
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = str(parent.attrib["link"])
        child_name = str(child.attrib["link"])
        origin = joint.find("origin")
        xyz = _vec(origin.attrib.get("xyz") if origin is not None else None, 3)
        rpy = _vec(origin.attrib.get("rpy") if origin is not None else None, 3)
        axis_element = joint.find("axis")
        axis = _vec(
            axis_element.attrib.get("xyz") if axis_element is not None else "1 0 0",
            3,
        )
        children.setdefault(parent_name, []).append(
            (name, child_name, xyz, rpy, axis, joint_type)
        )

    transforms = {base_link: _identity()}
    queue = [base_link]
    while queue:
        parent_name = queue.pop(0)
        parent_t = transforms[parent_name]
        for name, child_name, xyz, rpy, axis, joint_type in children.get(parent_name, []):
            t = _mm(parent_t, _xyz_rpy(xyz, rpy))
            if joint_type in ("revolute", "continuous"):
                t = _mm(t, _axis_angle(axis, float(joint_positions.get(name, 0.0))))
            transforms[child_name] = t
            queue.append(child_name)
    return transforms


def _standing_base_height(
    urdf_root: ET.Element,
    base_link: str,
    stand_positions: dict[str, float],
    foot_links: list[str],
) -> float:
    transforms = _link_transforms(urdf_root, base_link, stand_positions)
    link_by_name = {str(link.attrib["name"]): link for link in urdf_root.findall("link")}

    minimum_z = None
    for foot_link in foot_links:
        if foot_link not in transforms or foot_link not in link_by_name:
            raise RuntimeError(f"Foot link not found in URDF: {foot_link}")
        link_t = transforms[foot_link]
        collisions = link_by_name[foot_link].findall("collision")
        for collision in collisions:
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            sphere = geometry.find("sphere")
            if sphere is None:
                # Current Bumi2 foot contact geometry intentionally uses
                # support spheres.  Fail loudly if that contract changes.
                continue
            radius = float(sphere.attrib["radius"])
            origin = collision.find("origin")
            xyz = _vec(origin.attrib.get("xyz") if origin is not None else None, 3)
            center = _point(link_t, xyz)
            bottom = center[2] - radius
            minimum_z = bottom if minimum_z is None else min(minimum_z, bottom)

    if minimum_z is None:
        raise RuntimeError(
            "No spherical foot collision points found; cannot compute spawn height"
        )
    return -float(minimum_z)


def _joint_effort_limits(
    urdf_root: ET.Element,
    joint_order: list[str],
) -> dict[str, float]:
    joint_by_name = {str(j.attrib["name"]): j for j in urdf_root.findall("joint")}
    result: dict[str, float] = {}
    for name in joint_order:
        joint = joint_by_name.get(name)
        if joint is None:
            raise RuntimeError(f"Configured joint missing from URDF: {name}")
        limit = joint.find("limit")
        if limit is None or "effort" not in limit.attrib:
            raise RuntimeError(f"URDF joint has no effort limit: {name}")
        effort = float(limit.attrib["effort"])
        if not math.isfinite(effort) or effort <= 0.0:
            raise RuntimeError(f"Invalid effort limit for {name}: {effort}")
        result[name] = effort
    return result


def _inject_gazebo(
    urdf_text: str,
    *,
    joint_order: list[str],
    effort_limits: dict[str, float],
    initial_positions: dict[str, float],
    base_link: str,
    imu_link: str,
    state_publish_hz: float,
    hold_on_start: bool,
    viscous_damping: float,
    coulomb_friction: float,
    foot_links: list[str],
    foot_contact: dict,
) -> str:
    root = ET.fromstring(urdf_text)

    for foot_link in foot_links:
        gazebo = ET.SubElement(root, "gazebo", {"reference": foot_link})
        for tag, key in (
            ("mu1", "mu1"),
            ("mu2", "mu2"),
            ("kp", "kp"),
            ("kd", "kd"),
            ("minDepth", "min_depth"),
            ("maxVel", "max_vel"),
        ):
            value = ET.SubElement(gazebo, tag)
            value.text = str(float(foot_contact[key]))

    gazebo = ET.SubElement(root, "gazebo")
    plugin = ET.SubElement(
        gazebo,
        "plugin",
        {"name": "noetix_gazebo_backend", "filename": "libnoetix_gazebo_backend.so"},
    )

    def add(name: str, value) -> None:
        element = ET.SubElement(plugin, name)
        element.text = str(value)

    add("base_link", base_link)
    add("imu_link", imu_link)
    add("state_publish_hz", state_publish_hz)
    add("hold_on_start", "true" if hold_on_start else "false")
    add("joint_viscous_damping", viscous_damping)
    add("joint_coulomb_friction", coulomb_friction)

    ros = ET.SubElement(plugin, "ros")
    ns = ET.SubElement(ros, "namespace")
    ns.text = "/"

    for name in joint_order:
        joint = ET.SubElement(plugin, "controlled_joint")
        e = ET.SubElement(joint, "name")
        e.text = name
        e = ET.SubElement(joint, "effort_limit")
        e.text = str(effort_limits[name])
        e = ET.SubElement(joint, "initial_position")
        e.text = str(float(initial_positions.get(name, 0.0)))

    return ET.tostring(root, encoding="unicode")


def _launch_robot(context):
    robot_name = LaunchConfiguration("robot_name").perform(context).strip()
    gui = LaunchConfiguration("gui")

    robots_share = Path(get_package_share_directory("noetix_robots")).resolve()
    robot_root = robots_share / robot_name
    robot_yaml_path = robot_root / "config" / "robot.yaml"
    if not robot_yaml_path.is_file():
        raise RuntimeError(f"robot.yaml not found: {robot_yaml_path}")

    with robot_yaml_path.open("r", encoding="utf-8") as f:
        robot_yaml = yaml.safe_load(f) or {}

    robot = robot_yaml.get("robot") or {}
    joint_order = [str(x) for x in robot.get("joint_order", [])]
    if not joint_order:
        raise RuntimeError("robot.joint_order is empty")

    frames = robot.get("frames") or {}
    base_link = str(frames.get("base", "base_link"))
    imu_link = str(frames.get("imu", "base_imu"))

    description = robot.get("description") or {}
    description_rel = description.get("real")
    if not description_rel:
        raise RuntimeError("robot.description.real is required")
    description_path = (robot_root / str(description_rel)).resolve()
    if description_path.suffix != ".urdf":
        raise RuntimeError(
            "This Gazebo Classic launch currently expects the installed real "
            f"description to be URDF; got: {description_path}"
        )
    urdf_text = description_path.read_text(encoding="utf-8")
    urdf_root = ET.fromstring(urdf_text)

    gazebo_share = Path(get_package_share_directory("noetix_gazebo")).resolve()
    sim_cfg_path = gazebo_share / "config" / f"{robot_name}.yaml"
    if not sim_cfg_path.is_file():
        raise RuntimeError(
            f"Gazebo config not found for robot '{robot_name}': {sim_cfg_path}"
        )
    with sim_cfg_path.open("r", encoding="utf-8") as f:
        sim_cfg = (yaml.safe_load(f) or {}).get("simulation") or {}

    physics_hz = 1000.0
    state_publish_hz = float(sim_cfg.get("state_publish_hz", 200.0))
    clearance = float(sim_cfg.get("spawn_clearance_m", 0.03))
    hold_on_start = bool(sim_cfg.get("hold_on_start", True))
    auto_stand = bool(sim_cfg.get("auto_stand", True))
    stand_settle_s = float(sim_cfg.get("stand_settle_s", 0.5))
    initial_pose_name = str(sim_cfg.get("initial_pose", "lie")).lower()
    foot_links = [str(x) for x in sim_cfg.get(
        "foot_links", ["l_ankle_roll_link", "r_ankle_roll_link"])]
    foot_contact = sim_cfg.get("foot_contact") or {}
    viscous_damping = float(sim_cfg.get("joint_viscous_damping", 0.0))
    coulomb_friction = float(sim_cfg.get("joint_coulomb_friction", 0.0))

    if physics_hz <= 0.0 or state_publish_hz <= 0.0 or clearance < 0.0:
        raise RuntimeError("Invalid Gazebo frequency/clearance configuration")

    stand_positions = _pose_map(robot_yaml, "stand_joint_angles")
    lie_positions = _pose_map(robot_yaml, "lie_joint_angles")
    if initial_pose_name == "lie":
        initial_positions = lie_positions
    elif initial_pose_name == "stand":
        initial_positions = stand_positions
    elif initial_pose_name == "zero":
        initial_positions = {}
    else:
        raise RuntimeError(
            f"simulation.initial_pose must be lie/stand/zero, got: {initial_pose_name}"
        )

    stand_control = robot_yaml.get("stand_control") or {}
    stand_duration_s = float(stand_control.get("duration_s", 2.0))

    base_contact_z = _standing_base_height(
        urdf_root,
        base_link,
        stand_positions,
        foot_links,
    )
    spawn_z = base_contact_z + clearance
    effort_limits = _joint_effort_limits(urdf_root, joint_order)

    sim_description = _inject_gazebo(
        urdf_text,
        joint_order=joint_order,
        effort_limits=effort_limits,
        initial_positions=initial_positions,
        base_link=base_link,
        imu_link=imu_link,
        state_publish_hz=state_publish_hz,
        hold_on_start=hold_on_start,
        viscous_damping=viscous_damping,
        coulomb_friction=coulomb_friction,
        foot_links=foot_links,
        foot_contact=foot_contact,
    )

    print(
        "\n[Noetix Gazebo Classic Resource]\n"
        f"  robot_name:          {robot_name}\n"
        f"  robot.yaml:          {robot_yaml_path}\n"
        f"  shared URDF:         {description_path}\n"
        f"  dof:                 {len(joint_order)}\n"
        f"  physics:             {physics_hz:.1f} Hz\n"
        f"  LowState publish:    {state_publish_hz:.1f} Hz\n"
        f"  stand contact baseZ: {base_contact_z:.6f} m\n"
        f"  free-fall clearance: {clearance:.3f} m\n"
        f"  spawn/hold baseZ:    {spawn_z:.6f} m\n"
        f"  auto STAND:          {auto_stand}\n"
        f"  release behaviour:   INSTANT (gravity free-fall)\n"
    )

    gazebo_ros_share = Path(get_package_share_directory("gazebo_ros"))
    world_path = gazebo_share / "worlds" / "flat.world"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(gazebo_ros_share / "launch" / "gazebo.launch.py")
        ),
        launch_arguments={
            "world": str(world_path),
            "gui": gui,
            "verbose": "false",
        }.items(),
    )

    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": sim_description,
            "use_sim_time": True,
        }],
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_bumi",
        output="screen",
        arguments=[
            "-entity", robot_name,
            "-topic", "robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", f"{spawn_z:.9f}",
            "-R", "0.0",
            "-P", "0.0",
            "-Y", "0.0",
        ],
    )

    pose_mode = Node(
        package="noetix_bridge",
        executable="noetix_pose_mode_node",
        name="noetix_pose_mode_node",
        output="screen",
        parameters=[{
            "robot_name": robot_name,
            "use_sim_time": True,
            "trajectory_use_ros_time": True,
        }],
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
        parameters=[{
            "device_id": LaunchConfiguration("joy_device_id"),
            "deadzone": LaunchConfiguration("joy_deadzone"),
            "autorepeat_rate": 20.0,
            "use_sim_time": True,
        }],
        condition=IfCondition(LaunchConfiguration("use_gamepad")),
    )

    gamepad = Node(
        package="noetix_bridge",
        executable="noetix_gamepad_mode_node",
        name="noetix_gamepad_mode_node",
        output="screen",
        parameters=[{
            "robot_name": robot_name,
            "use_sim_time": True,
        }],
        condition=IfCondition(LaunchConfiguration("use_gamepad")),
    )

    startup = Node(
        package="noetix_gazebo",
        executable="sim_startup_manager.py",
        name="noetix_sim_startup_manager",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "auto_stand": auto_stand and hold_on_start,
            "stand_duration_s": stand_duration_s,
            "stand_settle_s": stand_settle_s,
        }],
    )

    return [
        gazebo,
        state_publisher,
        spawn,
        pose_mode,
        joy_node,
        gamepad,
        startup,
    ]


def generate_launch_description():
    prefix = Path(get_package_prefix("noetix_gazebo")).resolve()
    plugin_dir = str(prefix / "lib")
    old_plugin_path = os.environ.get("GAZEBO_PLUGIN_PATH", "")
    plugin_path = plugin_dir + (os.pathsep + old_plugin_path if old_plugin_path else "")

    return LaunchDescription([
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", plugin_path),
        DeclareLaunchArgument(
            "robot_name",
            default_value="bumi2",
            description="Robot resource under share/noetix_robots (bumi2 configured)",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start Gazebo Classic GUI (gzclient)",
        ),
        DeclareLaunchArgument(
            "use_gamepad",
            default_value="true",
            description="Start the same joystick mode adapter used on real hardware",
        ),
        DeclareLaunchArgument("joy_device_id", default_value="0"),
        DeclareLaunchArgument("joy_deadzone", default_value="0.001"),
        OpaqueFunction(function=_launch_robot),
    ])
