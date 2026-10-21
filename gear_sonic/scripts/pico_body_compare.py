#!/usr/bin/env python3
"""Record raw PICO [24,7] poses and compare skeleton dimensions (no SONIC conversion).

Requires numpy; recording additionally requires the existing xrobotoolkit_sdk.
See docs/source/tutorials/pico_height_check.md for the experiment procedure.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数值")
    return number


def validate_poses(poses):
    if poses.ndim != 3 or poses.shape[1:] != (24, 7) or len(poses) == 0:
        raise ValueError(f"需要非空 [N,24,7] 数组，实际为 {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("位姿含 NaN/Inf")
    norms = np.linalg.norm(poses[..., 3:], axis=-1)
    if np.any((norms < 0.5) | (norms > 1.5)):
        raise ValueError("部分关节四元数无效，可能未开启完整 24 点全身追踪")
    lengths = np.linalg.norm(poses[:, 1:, :3] - poses[:, PARENTS[1:], :3], axis=-1)
    if np.any(lengths < 1e-5):
        raise ValueError("存在重合的父子关节，可能收到未初始化或不完整的骨架")


def metric_series(poses, up_axis):
    """All metrics are in metres; bone lengths are invariant to rigid transforms."""
    p = poses[..., :3]
    up = p[..., "xyz".index(up_axis)]
    metrics = {}
    for child in range(1, 24):
        parent = PARENTS[child]
        metrics[f"bone/{JOINTS[parent]}->{JOINTS[child]}"] = np.linalg.norm(
            p[:, child] - p[:, parent], axis=-1
        )
    metrics["width/shoulders"] = np.linalg.norm(p[:, 16] - p[:, 17], axis=-1)
    metrics["width/hips"] = np.linalg.norm(p[:, 1] - p[:, 2], axis=-1)
    metrics["height/head_minus_mean_ankles"] = up[:, 15] - up[:, [7, 8]].mean(axis=1)
    metrics["height/head_minus_mean_feet"] = up[:, 15] - up[:, [10, 11]].mean(axis=1)
    metrics["height/all_joint_vertical_span"] = np.ptp(up, axis=1)
    metrics["world_up/head"] = up[:, 15]
    metrics["world_up/pelvis"] = up[:, 0]
    metrics["world_up/mean_ankles"] = up[:, [7, 8]].mean(axis=1)
    return metrics


def timestamp(xrt, source=None):
    # Some Unity clients omit Body.timeStampNs; prefer per-joint timestamps.
    sources = [source] if source else ["body_joints", "body", "stream"]
    for name in sources:
        getter = getattr(xrt, {
            "body_joints": "get_body_joints_timestamp",
            "body": "get_body_timestamp_ns",
            "stream": "get_time_stamp_ns",
        }[name], None)
        if getter is None:
            continue
        values = np.asarray(getter(), dtype=np.int64).reshape(-1)
        if np.any(values > 0):
            return name, tuple(int(v) for v in values)
    return source, None


def record(args):
    if args.out.suffix != ".npz":
        raise ValueError("采集输出文件必须以 .npz 结尾")
    if args.out.exists():
        raise ValueError(f"文件已存在，拒绝覆盖：{args.out}")
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError("请使用 sonic_bumi 已安装 xrobotoolkit_sdk 的 teleop Python 环境") from exc

    print(f"记录设置身高：{args.height:g} cm（仅记录标签，不会替你修改 PICO 设置）", flush=True)
    print("请先在 PICO 修改身高、重新校准，在 XRoboToolkit 开启 Full-body 和 Send。", flush=True)
    print(f"保持直立、双脚平放、姿态一致；{args.countdown:g} 秒后等待新数据。", flush=True)
    time.sleep(args.countdown)
    frames, stamps, host_times = [], [], []
    source, previous = None, None
    start = None
    last_fresh = time.monotonic()
    skipped = 0
    last_report = -1
    interrupted = False
    xrt.init()
    try:
        # Establish a baseline: cached data from an old session must not start a recording.
        source, previous = timestamp(xrt)
        while start is None or time.monotonic() - start < args.seconds:
            now = time.monotonic()
            if now - last_fresh > args.timeout:
                raise RuntimeError("未收到有效的新骨架帧：请检查 PC Service、Full-body、Send 和网络连接")
            if not xrt.is_body_data_available():
                time.sleep(0.005)
                continue
            current_source, current = timestamp(xrt, source)
            if current is None or (source == current_source and current == previous):
                time.sleep(0.002)
                continue
            source, previous = current_source, current
            pose = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
            try:
                validate_poses(pose[None])
            except ValueError:
                skipped += 1
                time.sleep(0.005)
                continue
            last_fresh = now
            if start is None:
                start = now
                print(f"收到新数据，开始采集；时间戳来源：{source}", flush=True)
                if source == "stream":
                    print("注意：SDK 仅提供总消息时间戳，无法严格验证骨架自身的新鲜度。", flush=True)
            frames.append(pose.copy())
            stamps.append(current)
            host_times.append(time.time_ns())
            elapsed = int(now - start)
            if elapsed != last_report:
                print(f"  {now - start:.1f}/{args.seconds:g} s，{len(frames)} 帧", flush=True)
                last_report = elapsed
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        interrupted = True
        print("采集被中断，将保存已获取的数据。")
    finally:
        xrt.close()
    if not frames:
        raise RuntimeError("没有采集到数据")
    metadata = {
        "setting_height_cm": args.height,
        "up_axis": args.up_axis,
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "timestamp_source": source,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seconds": args.seconds,
        "sample_rate_limit_hz": args.hz,
        "interrupted": interrupted,
        "skipped_invalid_frames": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        np.savez_compressed(
            output, poses=np.stack(frames), sdk_timestamps=np.asarray(stamps, dtype=np.int64),
            host_time_ns=np.asarray(host_times, dtype=np.int64),
            joint_names=np.asarray(JOINTS), parents=np.asarray(PARENTS),
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    print(f"已保存 {len(frames)} 帧：{args.out.resolve()}")


def load_record(path):
    with np.load(path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=np.float64)
        metadata = json.loads(str(data["metadata"].item()))
        if tuple(data["joint_names"].tolist()) != JOINTS or tuple(data["parents"].tolist()) != PARENTS:
            raise ValueError(f"关节顺序或骨架定义不匹配：{path}")
    validate_poses(poses)
    if metadata["position_unit"] != "m" or metadata["up_axis"] not in ("x", "y", "z"):
        raise ValueError(f"不支持的单位或竖直轴：{path}")
    return poses, metadata


def compare(args):
    a, ma = load_record(args.a)
    b, mb = load_record(args.b)
    if ma["up_axis"] != mb["up_axis"]:
        raise ValueError("两批数据的 up_axis 不同，不能直接比较高度")
    if args.out and (args.out in (args.a, args.b) or args.out.exists()):
        raise ValueError(f"比较结果文件已存在或与输入冲突：{args.out}")
    am = metric_series(a, ma["up_axis"])
    bm = metric_series(b, mb["up_axis"])
    print(f"A: {args.a}，设置 {ma['setting_height_cm']:g} cm，{len(a)} 帧")
    print(f"B: {args.b}，设置 {mb['setting_height_cm']:g} cm，{len(b)} 帧")
    print(f"位置单位 m；竖直轴 {ma['up_axis']}；下面全部转换为 cm；Δ = B - A。")
    print("高度是骨架关节点间的竖直距离，不是头顶到脚底的真实身高。")
    print(f"{'metric':<44} {'A mean±std':>17} {'B mean±std':>17} {'Δ cm':>9} {'Δ %':>8}")
    rows = []
    for name, values in am.items():
        av, bv = float(values.mean()), float(bm[name].mean())
        ast, bst = float(values.std()), float(bm[name].std())
        delta = bv - av
        # Relative changes of world coordinates depend on origin; omit those percentages.
        percent = 100 * delta / av if abs(av) > 1e-9 and not name.startswith("world_up/") else None
        percent_text = f"{percent:+.2f}" if percent is not None else "--"
        print(f"{name:<44} {100*av:8.3f}±{100*ast:6.3f} "
              f"{100*bv:8.3f}±{100*bst:6.3f} {100*delta:+9.3f} {percent_text:>8}")
        rows.append({"metric": name, "a_mean_cm": 100 * av, "a_std_cm": 100 * ast,
                     "b_mean_cm": 100 * bv, "b_std_cm": 100 * bst,
                     "delta_cm": 100 * delta, "delta_percent": percent})
    print("\n判断：骨长随设置变化→尺寸变化；骨长不变且 world_up 同幅变化→整体高度偏移。")
    print("无明显变化不能单独证明固定骨架；还需确认校准生效、重复采样和姿态一致。")
    print("std 反映批内波动，不是显著性检验；比较差值时请同时参考波动幅度。")
    for label, poses, meta in (("A", a, ma), ("B", b, mb)):
        identical = np.all(poses[1:] == poses[:-1], axis=(1, 2))
        if identical.size and identical.mean() > 0.1:
            print(f"注意：{label} 有 {identical.mean():.1%} 相邻帧完全相同，请检查骨架流是否停更。")
        if meta.get("interrupted"):
            print(f"注意：{label} 为中断后保存的部分采集。")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"已保存比较表：{args.out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="从现有 XRoboToolkit 接口采集原始 PICO 骨架")
    rec.add_argument("--height", type=positive, required=True, help="PICO 界面填写的身高，cm；仅作标签")
    rec.add_argument("--out", type=Path, required=True, help="输出 .npz 文件，不覆盖已有文件")
    rec.add_argument("--seconds", type=positive, default=10, help="采集秒数，默认 10")
    rec.add_argument("--hz", type=positive, default=50, help="读取频率上限，默认 50 Hz")
    rec.add_argument("--countdown", type=positive, default=3, help="采集前等待秒数，默认 3")
    rec.add_argument("--timeout", type=positive, default=10, help="等待有效新数据的超时秒数，默认 10")
    rec.add_argument("--up-axis", choices=("x", "y", "z"), default="y", help="原始 PICO 默认 Y-up")
    comp = sub.add_parser("compare", help="离线比较两批采集，只需要 numpy")
    comp.add_argument("a", type=Path)
    comp.add_argument("b", type=Path)
    comp.add_argument("--out", type=Path, help="可选：保存 CSV 比较表，不覆盖已有文件")
    args = parser.parse_args()
    try:
        record(args) if args.command == "record" else compare(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
