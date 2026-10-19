"""Portable BUMI3 settings shared by offline conversion, launch and simulation."""

from __future__ import annotations

from pathlib import Path
import json
import zlib

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/bumi3_21dof_sonic_model12.yaml"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_deploy_config(path: str | Path) -> dict:
    result = yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("version") != 1:
        raise ValueError("BUMI3 deployment config requires version: 1")
    for section in ("ros2", "teleop", "hardware", "policies"):
        if not isinstance(result.get(section), dict):
            raise ValueError(f"deployment config requires {section}")
    for section, key in (("ros2", "state_timeout"), ("ros2", "command_timeout"),
                         ("teleop", "startup_timeout"), ("teleop", "stream_timeout"),
                         ("hardware", "entry_duration")):
        value = result[section].get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{key} must be finite and positive")
    domain = result["ros2"].get("domain_id")
    if not isinstance(domain, int) or isinstance(domain, bool) or not 0 <= domain <= 232:
        raise ValueError("ros2.domain_id must be an integer in [0, 232]")
    for key in ("state_topic", "command_topic", "request_topic", "sim_reset_topic",
                "sim_status_topic", "sim_reference_topic"):
        topic = result["ros2"].get(key)
        if not isinstance(topic, str) or not topic.startswith("/"):
            raise ValueError(f"ros2.{key} must be an absolute ROS topic name")
    rotation = np.asarray(result["hardware"].get("imu_to_base_wxyz"), dtype=float)
    if rotation.shape != (4,) or not np.isfinite(rotation).all() or not np.isclose(np.linalg.norm(rotation), 1):
        raise ValueError("hardware.imu_to_base_wxyz must be a unit quaternion")
    return result


def joint_mappings(config: dict, contract) -> list[dict]:
    """URDF joints -> named SDK logical actuators, in policy order.

    Hardware calibration and parallel ankle conversion stay in Noetix SDK.
    """
    resource = repo_path(config["hardware"]["robot_root"]) / "config/robot.yaml"
    robot = yaml.safe_load(resource.read_text(encoding="utf-8"))
    mappings = robot["joints"]
    if len(mappings) != 21 or len({item["name"] for item in mappings}) != 21:
        raise ValueError("robot.yaml requires 21 unique joint mappings")
    by_name = {item["name"]: item for item in mappings}
    result = [dict(by_name[name]) for name in contract.policy_joint_names]
    for item in result:
        if item.get("direction") not in (-1, 1) or not np.isfinite(item.get("offset", float("nan"))):
            raise ValueError(f"invalid joint mapping: {item['name']}")
    if len({item["actuator"] for item in result}) != 21:
        raise ValueError("robot.yaml requires 21 unique logical actuators")
    return result


def finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite [{size}] vector")
    return vector


def contract_fields(contract) -> dict:
    return {
        "policy_joint_names": list(contract.policy_joint_names),
        "control_dt": contract.control_dt,
        "default_policy": contract.default_policy.tolist(),
        "action_scale_policy": contract.action_scale_policy.tolist(),
        "stiffness_policy": contract.stiffness_mujoco[contract.mujoco_to_policy].tolist(),
        "damping_policy": contract.damping_mujoco[contract.mujoco_to_policy].tolist(),
        "effort_policy": contract.effort_mujoco[contract.mujoco_to_policy].tolist(),
        "action_clip": contract.action_clip,
    }


def contract_id(contract, config: dict | None = None) -> int:
    fields = contract_fields(contract)
    fields.update(sim_dt=contract.sim_dt, decimation=contract.decimation,
                  history_length=contract.history_length, future_frames=contract.num_future_frames,
                  future_stride=contract.future_frame_stride, armature=contract.armature_mujoco.tolist())
    if config is not None:
        fields.update(joint_mappings=joint_mappings(config, contract),
                      imu_to_base_wxyz=config["hardware"]["imu_to_base_wxyz"])
    return zlib.crc32(json.dumps(fields, sort_keys=True, allow_nan=False).encode())


