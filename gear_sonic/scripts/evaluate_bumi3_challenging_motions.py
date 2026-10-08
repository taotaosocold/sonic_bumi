# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""用 BUMI3 SONIC ONNX 策略在 MuJoCo 中筛选具有挑战性的参考动作。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import joblib
import json
import math
from pathlib import Path
import re
from typing import Iterator, Literal

import mujoco
import numpy as np
import tyro
import yaml

from gear_sonic.utils.mujoco_sim.bumi3_motion_dataset import load_smpl_reference_motion
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    OnnxRobotPolicy,
    ReferenceMotion,
    load_reference_motion,
    quaternion_to_matrix,
)


DEFAULT_TRAINING_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config/exp/manager/universal_token/all_modes/sonic_bumi3.yaml"
)
DEFAULT_COM_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config/manager_env/events/terms/base_com.yaml"
)
DEFAULT_EVENTS_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config/manager_env/events/tracking/level0_4.yaml"
)


@dataclass
class Args:
    """MuJoCo 困难动作筛选参数。"""

    policy: Path | None = None
    """策略 ONNX：Robot 编码器用 *_g1.onnx，SMPL 编码器用 *_smpl.onnx。"""

    encoder: Literal["robot", "smpl"] = "robot"
    """与策略匹配的输入编码器。"""

    motion: Path | None = None
    """单条参考动作文件；可选配对 robot_motion。"""

    robot_motion: Path | None = None
    """SMPL 单条动作的可选配对 Robot 参考。"""

    robot_motion_key: str | None = None
    """配对 Robot 容器中的动作键。"""

    dataset: Path | None = None
    """动作目录或 version: 1 的 JSON/YAML 清单；目录会递归扫描支持的动作文件。"""

    limit: int | None = None
    """最多评测多少条源动作；不指定时评测全部。按清单顺序或路径字典序选择。"""

    output_dir: Path | None = None
    """评测结果目录；会写入逐次结果、汇总和困难动作片段。"""

    config: Path = DEFAULT_BUMI3_SIM2SIM_CONFIG
    """MuJoCo sim2sim 契约配置。"""

    motion_key: str | None = None
    """单条动作文件容器中的动作键。"""

    joint_order: Literal["auto", "policy", "isaaclab", "mujoco"] = "auto"
    """Robot 参考动作关节顺序。"""

    quaternion_order: Literal["auto", "wxyz", "xyzw"] = "auto"
    """Robot 参考 root quaternion 顺序。"""

    provider: Literal["cpu", "cuda"] = "cpu"
    """ONNX Runtime 推理设备。"""

    repeats: int = 5
    """每个 10 秒片段的 rollout 次数。"""

    mastered_success_rate: float = 0.8
    """成功率达到该值即记为 mastered；低于该值记为 challenging。"""

    height_error_threshold: float = 0.2
    """实际根高与参考根高的最大允许误差，单位米。"""

    clip_seconds: float = 10.0
    """动作切片长度；论文第一阶段采用 10 秒片段。"""

    seed: int = 1
    """可复现的基础随机种子。"""

    domain_randomization: bool = True
    """启用训练配置中的质量、质心和定时速度扰动。"""

    self_test: bool = False
    """只运行脚本内的切片、分类和种子逻辑自检。"""


@dataclass(frozen=True)
class MotionSource:
    """一条已加载动作及其来源信息。"""

    motion: ReferenceMotion
    path: Path
    motion_key: str | None
    robot_path: Path | None = None
    robot_motion_key: str | None = None


@dataclass(frozen=True)
class RandomizationConfig:
    """从现有 BUMI3 训练 YAML 读取的扰动范围。"""

    body_name: str
    mass_scale: tuple[float, float]
    com_offset: dict[str, tuple[float, float]]
    push_interval: tuple[float, float] | None
    push_velocity: dict[str, tuple[float, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "body_name": self.body_name,
            "mass_scale": list(self.mass_scale),
            "com_offset_m": {key: list(value) for key, value in self.com_offset.items()},
            "push_interval_s": (
                None if self.push_interval is None else list(self.push_interval)
            ),
            "push_velocity_ranges": {
                key: list(value) for key, value in self.push_velocity.items()
            },
            "sources": {
                "training": str(DEFAULT_TRAINING_CONFIG),
                "com": str(DEFAULT_COM_CONFIG),
                "events": str(DEFAULT_EVENTS_CONFIG),
            },
            "friction_randomization": (
                "未应用：Isaac Lab 的静/动摩擦与 MuJoCo 单一滑动摩擦系数不一一对应"
            ),
        }


def _pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} 必须是 [min, max]")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{label} 范围无效: {value}")
    return low, high


def _load_randomization_config() -> RandomizationConfig:
    """读取现有 BUMI3 配置，避免复制一份容易过期的随机化常量。"""

    training = yaml.safe_load(DEFAULT_TRAINING_CONFIG.read_text(encoding="utf-8"))
    com_source = yaml.safe_load(DEFAULT_COM_CONFIG.read_text(encoding="utf-8"))
    events = yaml.safe_load(DEFAULT_EVENTS_CONFIG.read_text(encoding="utf-8"))
    configured_events = training["manager_env"]["events"]

    body_names = configured_events["base_com"]["params"]["asset_cfg"]["body_names"]
    if not isinstance(body_names, str):
        raise ValueError("BUMI3 base_com 目前要求恰好指定一个 body 名称")
    mass_cfg = configured_events["randomize_rigid_body_mass"]["params"]
    mass_body = mass_cfg["asset_cfg"]["body_names"]
    if mass_body != body_names or mass_cfg.get("operation") != "scale":
        raise ValueError(
            "质量与质心随机化必须作用于同一 body，且质量操作必须为 scale"
        )
    mass_scale = _pair(mass_cfg["mass_distribution_params"], "mass_scale")

    com_ranges_raw = com_source["base_com"]["params"]["com_range"]
    com_offset = {
        axis: _pair(com_ranges_raw.get(axis, [0.0, 0.0]), f"COM {axis}")
        for axis in ("x", "y", "z")
    }
    push = events.get("push_robot")
    if push is None:
        push_interval = None
        push_velocity: dict[str, tuple[float, float]] = {}
    else:
        push_interval = _pair(push["interval_range_s"], "push interval")
        push_velocity = {
            axis: _pair(bounds, f"push velocity {axis}")
            for axis, bounds in push["params"]["velocity_range"].items()
        }
        if set(push_velocity) != {"x", "y", "z", "roll", "pitch", "yaw"}:
            raise ValueError("BUMI3 push_robot 必须包含 xyz 平移和 roll/pitch/yaw 角速度")
    return RandomizationConfig(body_names, mass_scale, com_offset, push_interval, push_velocity)


def _resolve_manifest_path(manifest: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空路径")
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = manifest.parent / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} 不存在: {resolved}")
    return resolved


def _read_dataset_entries(
    path: Path, encoder: Literal["robot", "smpl"], limit: int | None
) -> list[dict[str, object]]:
    """读取并校验清单，只保留 limit 指定的前 N 条动作。"""

    manifest = path.expanduser().resolve()
    content = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or content.get("version") != 1:
        raise ValueError("动作清单必须是含 version: 1 的 JSON/YAML 对象")
    if content.get("robot_type") != "bumi3":
        raise ValueError("动作清单 robot_type 必须为 bumi3")
    entries = content.get("motions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("动作清单 motions 必须是非空列表")
    selected = entries if limit is None else entries[:limit]
    names: set[str] = set()
    normalized: list[dict[str, object]] = []
    primary_field = "robot" if encoder == "robot" else "smpl"
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise ValueError(f"motions[{index}] 必须是对象")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"motions[{index}] 的 name 必须非空且唯一")
        names.add(name)
        primary_path = _resolve_manifest_path(
            manifest, item.get(primary_field), f"轨迹 {name} 的 {primary_field}"
        )
        robot_path = None
        if encoder == "smpl" and item.get("robot") is not None:
            robot_path = _resolve_manifest_path(
                manifest, item["robot"], f"轨迹 {name} 的 robot"
            )
        motion_key = (
            item.get("motion_key")
            if encoder == "robot"
            else item.get("smpl_motion_key", item.get("motion_key"))
        )
        normalized.append(
            {
                "name": name,
                "path": primary_path,
                "motion_key": motion_key,
                "robot_path": robot_path,
                "robot_motion_key": (
                    item.get("robot_motion_key", item.get("motion_key"))
                    if encoder == "smpl"
                    else None
                ),
                "joint_order": item.get("joint_order", "auto"),
                "quaternion_order": item.get("quaternion_order", "auto"),
            }
        )
    return normalized


def _scan_motion_paths(
    directory: Path, encoder: Literal["robot", "smpl"], limit: int | None
) -> list[Path]:
    """递归扫描动作目录并按相对路径排序，确保 limit 结果可复现。"""

    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"动作目录不存在或不是目录: {root}")
    if encoder == "robot":
        suffixes = {".pkl", ".joblib", ".npz"}
    else:
        suffixes = {".pkl", ".npz"}
    paths = {
        item.resolve()
        for item in root.rglob("*")
        if item.is_file()
        and item.suffix.lower() in suffixes
        and not any(part.startswith(".") for part in item.relative_to(root).parts)
    }
    if encoder == "robot":
        paths.update(
            item.parent.resolve()
            for item in root.rglob("joint_pos.csv")
            if not any(part.startswith(".") for part in item.relative_to(root).parts)
        )
    ordered = sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    if limit is not None:
        ordered = ordered[:limit]
    if not ordered:
        raise ValueError(f"目录没有可读取的 {encoder} 动作: {root}")
    return ordered


def _iter_motion_sources(args: Args, contract: Bumi3Contract) -> Iterator[MotionSource]:
    if args.motion is not None:
        path = args.motion.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"动作文件不存在: {path}")
        specs = [
            {
                "path": path,
                "name": None,
                "motion_key": args.motion_key,
                "robot_path": (
                    None if args.robot_motion is None else args.robot_motion.expanduser().resolve()
                ),
                "robot_motion_key": args.robot_motion_key,
                "joint_order": args.joint_order,
                "quaternion_order": args.quaternion_order,
            }
        ]
    else:
        assert args.dataset is not None
        source_path = args.dataset.expanduser().resolve()
        if source_path.is_dir():
            paths = _scan_motion_paths(source_path, args.encoder, args.limit)
            robot_paths_by_name: dict[str, list[Path]] = {}
            if args.encoder == "smpl":
                robot_root = source_path.parent / "robot"
                if robot_root.is_dir():
                    for candidate in _scan_motion_paths(robot_root, "robot", None):
                        robot_paths_by_name.setdefault(candidate.name, []).append(candidate)
            specs = []
            for path in paths:
                relative = path.relative_to(source_path)
                name = relative.with_suffix("").as_posix().replace("/", "::")
                robot_path = None
                if args.encoder == "smpl":
                    candidate = (source_path.parent / "robot" / relative).resolve()
                    if candidate.exists():
                        robot_path = candidate
                    else:
                        same_name = robot_paths_by_name.get(path.name, [])
                        if len(same_name) == 1:
                            robot_path = same_name[0]
                specs.append(
                    {
                        "path": path,
                        "name": name,
                        "motion_key": None,
                        "robot_path": robot_path,
                        "robot_motion_key": None,
                        "joint_order": "auto",
                        "quaternion_order": "auto",
                    }
                )
        else:
            specs = _read_dataset_entries(source_path, args.encoder, args.limit)

    for spec in specs:
        path = spec["path"]
        assert isinstance(path, Path)
        motion_key = spec["motion_key"]
        motion_key = None if motion_key is None else str(motion_key)
        robot_path = spec["robot_path"]
        robot_path = None if robot_path is None else Path(robot_path)
        robot_key = spec["robot_motion_key"]
        robot_key = None if robot_key is None else str(robot_key)
        if args.encoder == "robot":
            motion = load_reference_motion(
                path,
                contract,
                motion_key=motion_key,
                joint_order=str(spec["joint_order"]),
                quaternion_order=str(spec["quaternion_order"]),
            )
        else:
            motion = load_smpl_reference_motion(
                path,
                contract,
                motion_key=motion_key,
                robot_path=robot_path,
                robot_motion_key=robot_key,
                joint_order=str(spec["joint_order"]),
                quaternion_order=str(spec["quaternion_order"]),
            )
        name = spec["name"]
        if name is not None:
            motion = replace(motion, name=str(name))
        yield MotionSource(motion, path, motion_key, robot_path, robot_key)


def _clip_ranges(num_frames: int, frames_per_clip: int) -> Iterator[tuple[int, int]]:
    if num_frames <= 0 or frames_per_clip <= 0:
        raise ValueError("动作帧数和切片帧数必须大于零")
    for start in range(0, num_frames, frames_per_clip):
        yield start, min(start + frames_per_clip, num_frames)


def _slice_motion(motion: ReferenceMotion, start: int, end: int, name: str) -> ReferenceMotion:
    """复制一个半开区间片段，并把片段起点平移到水平原点。"""

    if not 0 <= start < end <= motion.num_frames:
        raise ValueError(f"动作切片越界: [{start}, {end})/{motion.num_frames}")
    root_position = None
    if motion.root_position_world is not None:
        root_position = motion.root_position_world[start:end].copy()
        root_position[:, :2] -= root_position[0, :2]
    return replace(
        motion,
        joint_pos_policy=motion.joint_pos_policy[start:end].copy(),
        joint_vel_policy=motion.joint_vel_policy[start:end].copy(),
        root_position_world=root_position,
        root_quat_wxyz=motion.root_quat_wxyz[start:end].copy(),
        root_lin_vel_world=(
            None
            if motion.root_lin_vel_world is None
            else motion.root_lin_vel_world[start:end].copy()
        ),
        root_ang_vel_world=(
            None
            if motion.root_ang_vel_world is None
            else motion.root_ang_vel_world[start:end].copy()
        ),
        smpl_reference=(
            None
            if motion.smpl_reference is None
            else replace(
                motion.smpl_reference,
                local_joints=motion.smpl_reference.local_joints[start:end].copy(),
                root_quat_wxyz=motion.smpl_reference.root_quat_wxyz[start:end].copy(),
                name=name,
            )
        ),
        has_robot_reference=motion.has_robot_reference,
        name=name,
    )


def _stable_trial_seed(base_seed: int, clip_name: str, trial_index: int) -> int:
    digest = hashlib.sha256(clip_name.encode("utf-8")).digest()
    clip_hash = int.from_bytes(digest[:4], "little")
    return (int(base_seed) ^ clip_hash ^ (int(trial_index) * 0x9E3779B1)) & 0x7FFFFFFF


def _is_mastered(success_count: int, repeats: int, threshold: float) -> bool:
    if repeats <= 0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("repeats 必须为正数，成功率阈值必须在 [0,1]")
    return success_count / repeats >= threshold


def _safe_stem(name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") or "motion"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}_{digest}"


def _self_test() -> None:
    assert list(_clip_ranges(1, 500)) == [(0, 1)]
    assert list(_clip_ranges(500, 500)) == [(0, 500)]
    assert list(_clip_ranges(1001, 500)) == [(0, 500), (500, 1000), (1000, 1001)]
    assert _is_mastered(4, 5, 0.8)
    assert not _is_mastered(3, 5, 0.8)
    assert _stable_trial_seed(17, "clip-a", 2) == _stable_trial_seed(17, "clip-a", 2)
    assert _stable_trial_seed(17, "clip-a", 2) != _stable_trial_seed(17, "clip-a", 3)
    print("BUMI3_CHALLENGE_SELF_TEST=PASS")


def _apply_trial_randomization(
    runner: Bumi3SonicSim2Sim,
    rng: np.random.Generator,
    config: RandomizationConfig,
    nominal: dict[str, np.ndarray],
    enabled: bool,
) -> dict[str, object]:
    model = runner.model
    model.body_mass[:] = nominal["body_mass"]
    model.body_inertia[:] = nominal["body_inertia"]
    model.body_ipos[:] = nominal["body_ipos"]
    details: dict[str, object] = {"enabled": enabled, "pushes": []}
    if enabled:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo 模型缺少随机化 body: {config.body_name}")
        mass_scale = float(rng.uniform(*config.mass_scale))
        model.body_mass[body_id] = nominal["body_mass"][body_id] * mass_scale
        model.body_inertia[body_id] = nominal["body_inertia"][body_id] * mass_scale
        com_offset = np.asarray(
            [rng.uniform(*config.com_offset[axis]) for axis in ("x", "y", "z")],
            dtype=np.float64,
        )
        model.body_ipos[body_id] = nominal["body_ipos"][body_id] + com_offset
        details.update(
            {
                "body_name": config.body_name,
                "mass_scale": mass_scale,
                "com_offset_m": com_offset.tolist(),
            }
        )
    mujoco.mj_setConst(model, runner.data)
    return details


def _sample_push(
    rng: np.random.Generator, config: RandomizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    ranges = config.push_velocity
    linear_world = np.asarray(
        [rng.uniform(*ranges[axis]) for axis in ("x", "y", "z")], dtype=np.float64
    )
    angular_world = np.asarray(
        [rng.uniform(*ranges[axis]) for axis in ("roll", "pitch", "yaw")], dtype=np.float64
    )
    return linear_world, angular_world


def _run_trial(
    runner: Bumi3SonicSim2Sim,
    clip: ReferenceMotion,
    *,
    source: MotionSource,
    clip_start: int,
    clip_end: int,
    trial_index: int,
    trial_seed: int,
    args: Args,
    randomization: RandomizationConfig,
    nominal_model: dict[str, np.ndarray],
) -> dict[str, object]:
    rng = np.random.default_rng(trial_seed)
    if runner.motion is not clip:
        runner.motion = clip
        runner.reference_anchor_quat_wxyz = runner._compute_reference_anchor_quaternions()
    perturbation = _apply_trial_randomization(
        runner, rng, randomization, nominal_model, args.domain_randomization
    )
    runner.playing = True
    runner.reset(start_frame=0)

    max_height_error = 0.0
    completed_steps = 0
    failure_reason: str | None = None
    failure_frame: int | None = None
    push_events: list[dict[str, object]] = []
    next_push_time: float | None = None
    if args.domain_randomization and randomization.push_interval is not None:
        next_push_time = float(rng.uniform(*randomization.push_interval))
    control_dt = runner.contract.control_dt
    root = runner.root_qpos_address

    while runner.playing:
        elapsed = completed_steps * control_dt
        if next_push_time is not None and elapsed >= next_push_time:
            linear_world, angular_world = _sample_push(rng, randomization)
            root_quat = runner.data.qpos[root + 3 : root + 7]
            angular_local = quaternion_to_matrix(root_quat).T @ angular_world
            runner.data.qvel[:3] = linear_world
            runner.data.qvel[3:6] = angular_local
            mujoco.mj_forward(runner.model, runner.data)
            push_events.append(
                {
                    "time_s": elapsed,
                    "linear_velocity_world_m_s": linear_world.tolist(),
                    "angular_velocity_world_rad_s": angular_world.tolist(),
                }
            )
            if randomization.push_interval is None:
                next_push_time = None
            else:
                next_push_time += float(rng.uniform(*randomization.push_interval))

        try:
            runner.step_control()
        except FloatingPointError:
            failure_reason = "nonfinite_state"
            failure_frame = int(runner.motion_frame)
            break
        completed_steps += 1
        frame = min(runner.motion_frame, clip.num_frames - 1)
        actual_height = float(runner.data.qpos[root + 2])
        reference_height = (
            float(clip.root_position_world[frame, 2])
            if clip.root_position_world is not None
            else float(runner.contract.initial_root_position[2])
        )
        height_error = abs(actual_height - reference_height)
        if not math.isfinite(height_error):
            failure_reason = "nonfinite_root_height_error"
            failure_frame = frame
            break
        max_height_error = max(max_height_error, height_error)
        if height_error > args.height_error_threshold:
            failure_reason = "root_height_error"
            failure_frame = frame
            break

    success = failure_reason is None and not runner.playing
    perturbation["pushes"] = push_events
    return {
        "source_motion": str(source.path),
        "source_motion_key": source.motion_key,
        "source_name": source.motion.name,
        "clip_name": clip.name,
        "clip_start_frame": clip_start,
        "clip_end_frame_exclusive": clip_end,
        "clip_frames": clip.num_frames,
        "fps": clip.fps,
        "trial": trial_index,
        "seed": trial_seed,
        "success": success,
        "termination_reason": failure_reason or "completed",
        "failure_frame_in_clip": failure_frame,
        "completed_control_steps": completed_steps,
        "progress": completed_steps / max(1, clip.num_frames - 1),
        "max_abs_root_height_error_m": max_height_error,
        "height_error_threshold_m": args.height_error_threshold,
        "domain_randomization": perturbation,
    }


def _save_motion_npz(path: Path, motion: ReferenceMotion) -> None:
    values: dict[str, object] = {
        "joint_pos": motion.joint_pos_policy,
        "joint_vel": motion.joint_vel_policy,
        "root_quat": motion.root_quat_wxyz,
        "fps": np.asarray(motion.fps, dtype=np.float32),
        "joint_order": np.asarray("policy"),
        "quaternion_order": np.asarray("wxyz"),
    }
    if motion.root_position_world is not None:
        values["root_position"] = motion.root_position_world
    if motion.root_lin_vel_world is not None:
        values["root_lin_vel_world"] = motion.root_lin_vel_world
    if motion.root_ang_vel_world is not None:
        values["root_ang_vel_world"] = motion.root_ang_vel_world
    np.savez_compressed(path, **values)


def _raw_smpl_mapping(source: MotionSource) -> dict[str, object]:
    """读取 SMPL 原始字段，导出片段时保持训练加载器需要的数据格式。"""

    if source.path.suffix.lower() == ".npz":
        with np.load(source.path, allow_pickle=False) as archive:
            content: object = {key: archive[key] for key in archive.files}
    else:
        content = joblib.load(source.path)
    if not isinstance(content, dict):
        raise ValueError(f"SMPL 文件顶层必须为 dict: {source.path}")
    if "pose_aa" not in content:
        key = source.motion_key
        if key is None:
            candidates = [name for name, value in content.items() if isinstance(value, dict)]
            if len(candidates) != 1:
                raise ValueError(f"SMPL 容器需指定唯一动作键: {source.path}")
            key = candidates[0]
        if key not in content or not isinstance(content[key], dict):
            raise ValueError(f"SMPL 容器不存在动作 {key!r}: {source.path}")
        content = content[key]
    if not isinstance(content, dict) or not {"pose_aa", "smpl_joints", "fps"} <= content.keys():
        raise ValueError(f"SMPL 文件缺少 pose_aa/smpl_joints/fps: {source.path}")
    return content


def _save_smpl_motion_npz(
    path: Path, source: MotionSource, start: int, end: int
) -> None:
    raw = _raw_smpl_mapping(source)
    np.savez_compressed(
        path,
        pose_aa=np.asarray(raw["pose_aa"])[start:end],
        smpl_joints=np.asarray(raw["smpl_joints"])[start:end],
        fps=np.asarray(raw["fps"]).reshape(-1)[0],
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(args: Args) -> None:
    if args.self_test:
        _self_test()
        return
    if (args.motion is None) == (args.dataset is None):
        raise ValueError("必须且只能指定 --motion 或 --dataset")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于零；不指定时评测全部动作")
    if args.motion_key is not None and args.motion is None:
        raise ValueError("--motion-key 仅能与 --motion 同时使用")
    if args.robot_motion is not None and (
        args.encoder != "smpl" or args.motion is None
    ):
        raise ValueError("--robot-motion 仅用于 --encoder smpl --motion")
    if args.robot_motion_key is not None and args.robot_motion is None:
        raise ValueError("--robot-motion-key 必须与 --robot-motion 同时使用")
    if args.dataset is not None and (
        args.motion_key is not None
        or args.robot_motion is not None
        or args.joint_order != "auto"
        or args.quaternion_order != "auto"
    ):
        raise ValueError("目录/清单模式请在清单中声明动作键和 Robot 顺序参数")
    if args.policy is not None:
        args.policy = args.policy.expanduser()
    if args.policy is None or not args.policy.is_file():
        suffix = "*_g1.onnx" if args.encoder == "robot" else "*_smpl.onnx"
        raise FileNotFoundError(f"请用 --policy 指定存在的 BUMI3 {args.encoder} ONNX（如 {suffix}）")
    if args.output_dir is None:
        raise ValueError("请用 --output-dir 指定评测结果目录")
    if args.repeats <= 0:
        raise ValueError("--repeats 必须大于零")
    if not 0.0 <= args.mastered_success_rate <= 1.0:
        raise ValueError("--mastered-success-rate 必须位于 [0,1]")
    if args.height_error_threshold <= 0.0 or args.clip_seconds <= 0.0:
        raise ValueError("根高阈值和切片秒数必须大于零")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"评测输出目录非空，为避免覆盖已有结果而停止: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    challenging_dir = output_dir / "challenging_clips"
    challenging_dir.mkdir(exist_ok=True)

    contract = Bumi3Contract.from_yaml(args.config)
    if not np.isclose(contract.target_fps, 50.0):
        raise ValueError(f"论文筛选协议要求 50 FPS，当前配置为 {contract.target_fps}")
    frames_per_clip = max(1, round(args.clip_seconds * contract.target_fps))
    randomization = _load_randomization_config()
    sources = iter(_iter_motion_sources(args, contract))
    first_source = next(sources, None)
    if first_source is None:
        raise ValueError("没有可评测动作")
    first_start, first_end = next(_clip_ranges(first_source.motion.num_frames, frames_per_clip))
    first_name = f"{first_source.motion.name}::frames_{first_start:06d}_{first_end:06d}"
    first_clip = _slice_motion(first_source.motion, first_start, first_end, first_name)

    policy = OnnxRobotPolicy(
        args.policy, contract, provider=args.provider, encoder=args.encoder
    )
    runner = Bumi3SonicSim2Sim(
        contract,
        first_clip,
        policy,
        loop_motion=False,
        start_frame=0,
        align_reference_heading=contract.align_reference_heading,
        start_paused=False,
        encoder=args.encoder,
    )
    nominal_model = {
        key: getattr(runner.model, key).copy()
        for key in ("body_mass", "body_inertia", "body_ipos")
    }

    summary: list[dict[str, object]] = []
    challenging_manifest: list[dict[str, object]] = []
    records_path = output_dir / "rollouts.jsonl"
    records_path.write_text("", encoding="utf-8")

    def evaluate_clip(
        source: MotionSource, start: int, end: int, clip: ReferenceMotion
    ) -> None:
        clip_trials: list[dict[str, object]] = []
        for trial in range(args.repeats):
            trial_seed = _stable_trial_seed(args.seed, clip.name, trial)
            record = _run_trial(
                runner,
                clip,
                source=source,
                clip_start=start,
                clip_end=end,
                trial_index=trial,
                trial_seed=trial_seed,
                args=args,
                randomization=randomization,
                nominal_model=nominal_model,
            )
            clip_trials.append(record)
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        success_count = sum(bool(record["success"]) for record in clip_trials)
        mastered = _is_mastered(success_count, args.repeats, args.mastered_success_rate)
        clip_summary: dict[str, object] = {
            "source_motion": str(source.path),
            "source_motion_key": source.motion_key,
            "source_name": source.motion.name,
            "clip_name": clip.name,
            "clip_start_frame": start,
            "clip_end_frame_exclusive": end,
            "clip_frames": clip.num_frames,
            "fps": clip.fps,
            "successful_trials": success_count,
            "total_trials": args.repeats,
            "success_rate": success_count / args.repeats,
            "classification": "mastered" if mastered else "challenging",
        }
        summary.append(clip_summary)
        if not mastered:
            file_stem = _safe_stem(clip.name)
            entry: dict[str, object] = {
                "name": clip.name,
                "source_motion": str(source.path),
                "source_motion_key": source.motion_key,
                "clip_start_frame": start,
                "clip_end_frame_exclusive": end,
                "success_rate": success_count / args.repeats,
            }
            if args.encoder == "robot":
                file_name = file_stem + ".npz"
                _save_motion_npz(challenging_dir / file_name, clip)
                entry["robot"] = Path("challenging_clips", file_name).as_posix()
            else:
                smpl_name = file_stem + "_smpl.npz"
                _save_smpl_motion_npz(
                    challenging_dir / smpl_name, source, start, end
                )
                entry["smpl"] = Path("challenging_clips", smpl_name).as_posix()
                if clip.has_robot_reference:
                    robot_name = file_stem + "_robot.npz"
                    _save_motion_npz(challenging_dir / robot_name, clip)
                    entry["robot"] = Path("challenging_clips", robot_name).as_posix()
            challenging_manifest.append(entry)
        print(
            f"BUMI3_MOTION_RESULT={json.dumps(clip_summary, ensure_ascii=False)}",
            flush=True,
        )

    evaluate_clip(first_source, first_start, first_end, first_clip)
    for source in sources:
        for start, end in _clip_ranges(source.motion.num_frames, frames_per_clip):
            name = f"{source.motion.name}::frames_{start:06d}_{end:06d}"
            clip = _slice_motion(source.motion, start, end, name)
            evaluate_clip(source, start, end, clip)

    output_summary = {
        "protocol": {
            "simulator": "MuJoCo",
            "encoder": args.encoder,
            "policy": str(args.policy.expanduser().resolve()),
            "motion_limit": args.limit,
            "config": str(args.config.expanduser().resolve()),
            "clip_seconds": args.clip_seconds,
            "frames_per_clip": frames_per_clip,
            "rollouts_per_clip": args.repeats,
            "mastered_success_rate": args.mastered_success_rate,
            "align_reference_heading": contract.align_reference_heading,
            "root_height_error_threshold_m": args.height_error_threshold,
            "domain_randomization": (
                randomization.as_dict() if args.domain_randomization else {"enabled": False}
            ),
            "seed": args.seed,
            "classification_rule": "成功率达到阈值为 mastered，否则为 challenging",
        },
        "source_motion_count": len(
            {(item["source_motion"], item["source_motion_key"]) for item in summary}
        ),
        "clip_count": len(summary),
        "mastered_count": sum(item["classification"] == "mastered" for item in summary),
        "challenging_count": len(challenging_manifest),
        "clips": summary,
    }
    _write_json(output_dir / "summary.json", output_summary)
    manifest_path = output_dir / "challenging_motions.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {"version": 1, "robot_type": "bumi3", "motions": challenging_manifest},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(
        "BUMI3_SOURCE_MOTION_COUNT="
        + str(output_summary["source_motion_count"])
        + (" (limit=" + str(args.limit) + ")" if args.limit is not None else " (all)")
    )
    print(f"BUMI3_CHALLENGE_SUMMARY={output_dir / 'summary.json'}")
    print(f"BUMI3_CHALLENGING_MANIFEST={manifest_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
