# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""用 BUMI3 SONIC ONNX 策略在 MuJoCo 中初筛完整运动序列。

每条序列从第一帧参考状态初始化，仅回放一次，不切片、不添加观测噪声或
域随机化。全程根高误差不超过阈值且状态有限、到达最后一帧即成功；
失败序列的名称、失败原因和帧号会打印到终端，并写入 challenging_names.txt。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import chain
import json
import math
from pathlib import Path
from typing import Iterator, Literal

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
    """评测结果目录；保存完整序列结果、失败名称/原因/帧号和源动作清单。"""

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

    height_error_threshold: float = 0.12
    """实际根高与参考根高的最大允许误差，单位米；默认与 BUMI3 普通姿态训练阈值一致。"""


@dataclass(frozen=True)
class MotionSource:
    """一条已加载动作及其来源信息。"""

    motion: ReferenceMotion
    path: Path
    motion_key: str | None
    robot_path: Path | None = None
    robot_motion_key: str | None = None
    joint_order: str = "auto"
    quaternion_order: str = "auto"


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
        yield MotionSource(
            motion=motion,
            path=path,
            motion_key=motion_key,
            robot_path=robot_path,
            robot_motion_key=robot_key,
            joint_order=str(spec["joint_order"]),
            quaternion_order=str(spec["quaternion_order"]),
        )


def _evaluate_motion(
    runner: Bumi3SonicSim2Sim,
    source: MotionSource,
    height_error_threshold: float,
) -> dict[str, object]:
    """从第一帧初始化，单次评测完整序列；首帧和末帧均检查根高误差。"""

    motion = source.motion
    if runner.motion is not motion:
        runner.motion = motion
        runner.reference_anchor_quat_wxyz = runner._compute_reference_anchor_quaternions()
    runner.playing = True
    runner.reset(start_frame=0)

    max_height_error = 0.0
    completed_steps = 0
    failure_reason: str | None = None
    failure_frame: int | None = None
    root = runner.root_qpos_address

    while True:
        frame = int(runner.motion_frame)
        actual_height = float(runner.data.qpos[root + 2])
        reference_height = (
            float(motion.root_position_world[frame, 2])
            if motion.root_position_world is not None
            else float(runner.contract.initial_root_position[2])
        )
        height_error = abs(actual_height - reference_height)
        if not math.isfinite(height_error):
            failure_reason = "nonfinite_root_height_error"
            failure_frame = frame
            break
        max_height_error = max(max_height_error, height_error)
        if height_error > height_error_threshold:
            failure_reason = "root_height_error"
            failure_frame = frame
            break
        if frame == motion.num_frames - 1:
            runner.playing = False
            break

        try:
            runner.step_control()
        except FloatingPointError:
            failure_reason = "nonfinite_state"
            failure_frame = int(runner.motion_frame)
            break
        completed_steps += 1

    success = failure_reason is None and runner.motion_frame == motion.num_frames - 1
    runner.playing = False
    return {
        "source_motion": str(source.path),
        "source_motion_key": source.motion_key,
        "source_name": motion.name,
        "motion_frames": motion.num_frames,
        "fps": motion.fps,
        "success": success,
        "classification": "success" if success else "challenging",
        "termination_reason": failure_reason or "completed",
        "failure_frame": failure_frame,
        "completed_control_steps": completed_steps,
        "progress": 1.0 if success else completed_steps / max(1, motion.num_frames - 1),
        "max_abs_root_height_error_m": max_height_error,
        "height_error_threshold_m": height_error_threshold,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(args: Args) -> None:
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
    if not math.isfinite(args.height_error_threshold) or args.height_error_threshold <= 0.0:
        raise ValueError("根高误差阈值必须为有限正数")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"评测输出目录非空，为避免覆盖已有结果而停止: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = Bumi3Contract.from_yaml(args.config)
    if not np.isclose(contract.target_fps, 50.0):
        raise ValueError(f"BUMI3 动作评测要求 50 FPS，当前配置为 {contract.target_fps}")
    sources = iter(_iter_motion_sources(args, contract))
    first_source = next(sources, None)
    if first_source is None:
        raise ValueError("没有可评测动作")

    policy = OnnxRobotPolicy(
        args.policy, contract, provider=args.provider, encoder=args.encoder
    )
    runner = Bumi3SonicSim2Sim(
        contract,
        first_source.motion,
        policy,
        loop_motion=False,
        start_frame=0,
        align_reference_heading=contract.align_reference_heading,
        start_paused=False,
        encoder=args.encoder,
    )
    results: list[dict[str, object]] = []
    challenging_manifest: list[dict[str, object]] = []
    records_path = output_dir / "rollouts.jsonl"
    names_path = output_dir / "challenging_names.txt"

    with records_path.open("w", encoding="utf-8") as records, names_path.open(
        "w", encoding="utf-8"
    ) as names:
        for source in chain((first_source,), sources):
            record = _evaluate_motion(runner, source, args.height_error_threshold)
            results.append(record)
            records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            records.flush()
            if record["success"]:
                continue

            failure_line = (
                f"{source.motion.name} "
                f"reason={record['termination_reason']} "
                f"failure_frame={record['failure_frame']}"
            )
            print(f"BUMI3_CHALLENGING_MOTION={failure_line}", flush=True)
            names.write(failure_line + "\n")
            names.flush()
            entry: dict[str, object] = {
                "name": source.motion.name,
                "source_motion": str(source.path),
                "joint_order": source.joint_order,
                "quaternion_order": source.quaternion_order,
            }
            if args.encoder == "robot":
                entry["robot"] = str(source.path)
                if source.motion_key is not None:
                    entry["motion_key"] = source.motion_key
            else:
                entry["smpl"] = str(source.path)
                if source.motion_key is not None:
                    entry["smpl_motion_key"] = source.motion_key
                if source.robot_path is not None:
                    entry["robot"] = str(source.robot_path)
                if source.robot_motion_key is not None:
                    entry["robot_motion_key"] = source.robot_motion_key
            challenging_manifest.append(entry)

    success_count = sum(bool(item["success"]) for item in results)
    output_summary = {
        "protocol": {
            "simulator": "MuJoCo",
            "encoder": args.encoder,
            "policy": str(args.policy.resolve()),
            "motion_limit": args.limit,
            "config": str(args.config.expanduser().resolve()),
            "evaluation_unit": "full_motion",
            "rollouts_per_motion": 1,
            "start_frame": 0,
            "align_reference_heading": contract.align_reference_heading,
            "root_height_error_threshold_m": args.height_error_threshold,
            "domain_randomization": {"enabled": False},
            "observation_noise": {"enabled": False},
            "classification_rule": "完整序列通过根高误差及有限值检查为成功，否则为 challenging",
        },
        "source_motion_count": len(results),
        "success_count": success_count,
        "challenging_count": len(challenging_manifest),
        "motions": results,
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
        + str(len(results))
        + (" (limit=" + str(args.limit) + ")" if args.limit is not None else " (all)")
    )
    print(f"BUMI3_SUCCESS_COUNT={success_count}")
    print(f"BUMI3_CHALLENGING_COUNT={len(challenging_manifest)}")
    print(f"BUMI3_CHALLENGE_SUMMARY={output_dir / 'summary.json'}")
    print(f"BUMI3_CHALLENGING_NAMES={names_path}")
    print(f"BUMI3_CHALLENGING_MANIFEST={manifest_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
