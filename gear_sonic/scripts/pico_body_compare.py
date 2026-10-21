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
import subprocess
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
PC_SERVICE_SCRIPT = Path("/opt/apps/roboticsservice/runService.sh")


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数值")
    return number


def nonnegative(value):
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须是大于或等于 0 的有限数值")
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


def describe_record_timeout(stats):
    if stats["body_available"] == 0:
        return "等待期间 SDK 一直未报告 Body 数据可用；请检查 PC Service、Full-body 和 Send。"
    if stats["new_timestamp"] == 0:
        return "Body 标志可用，但 get_time_stamp_ns() 没有更新；请检查 PICO 数据流是否停住。"
    if stats["invalid"]:
        return (
            f"读取到 {stats['new_timestamp']} 个新时间戳，但 {stats['invalid']} 帧数据形状或数值无效"
            f"（最近原因：{stats['last_invalid']}）。"
        )
    return "等待有效新帧超时。请检查 PC Service、Full-body、Send 和网络连接。"


def start_pc_service(skip_start):
    """Match pico_manager_thread_server.py's XRoboToolkit startup sequence."""
    if skip_start:
        print("不启动 PC Service：使用已经运行的 XRoboToolkit 服务。", flush=True)
        return
    if not PC_SERVICE_SCRIPT.is_file():
        raise RuntimeError(
            f"找不到 XRoboToolkit PC Service 启动脚本：{PC_SERVICE_SCRIPT}；"
            "请用 --no-start-pc-service 并自行启动服务。"
        )
    subprocess.Popen(["bash", str(PC_SERVICE_SCRIPT)])
    print("已按 pico_manager_thread_server.py 的方式启动 XRoboToolkit PC Service。", flush=True)


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
    frames, stamps, host_times = [], [], []
    previous_stamp = None
    start = None
    last_fresh = None
    skipped = 0
    stats = {
        "body_available": 0,
        "new_timestamp": 0,
        "invalid": 0,
        "last_invalid": "无",
    }
    last_report = -1
    interrupted = False
    start_pc_service(args.no_start_pc_service)
    xrt.init()
    try:
        # Initialize the SDK before the countdown, so the PC Service and headset can connect.
        print(f"保持直立、双脚平放、姿态一致；{args.countdown:g} 秒后等待新数据。", flush=True)
        time.sleep(args.countdown)
        last_fresh = time.monotonic()
        print("Waiting for body tracking data...", flush=True)
        last_wait_message = 0.0
        # Match pico_manager_thread_server._init_input_source(): do not read poses
        # until the SDK has received its first Body message.
        while not xrt.is_body_data_available():
            if args.timeout and time.monotonic() - last_fresh > args.timeout:
                raise RuntimeError(describe_record_timeout(stats))
            now = time.monotonic()
            if now - last_wait_message >= 1.0:
                print("waiting for body data...", flush=True)
                last_wait_message = now
            time.sleep(0.01)
        print("Body data available! Capturing raw PICO [24,7] poses...", flush=True)
        while start is None or time.monotonic() - start < args.seconds:
            now = time.monotonic()
            if args.timeout and now - last_fresh > args.timeout:
                raise RuntimeError(describe_record_timeout(stats))
            if not xrt.is_body_data_available():
                time.sleep(0.005)
                continue
            stats["body_available"] += 1
            # PicoReader uses this same stream timestamp to detect each new frame.
            stamp_ns = int(xrt.get_time_stamp_ns())
            if previous_stamp is not None and stamp_ns == previous_stamp:
                time.sleep(0.000001)
                continue
            previous_stamp = stamp_ns
            stats["new_timestamp"] += 1
            pose = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
            try:
                if pose.shape != (24, 7):
                    raise ValueError(f"需要 [24,7]，实际为 {pose.shape}")
                if not np.isfinite(pose).all():
                    raise ValueError("位姿含 NaN/Inf")
            except ValueError as exc:
                skipped += 1
                stats["invalid"] += 1
                stats["last_invalid"] = str(exc)
                time.sleep(0.005)
                continue
            last_fresh = now
            if start is None:
                start = now
                print("收到新数据，开始采集；时间戳来源：get_time_stamp_ns()（与 PicoReader 相同）", flush=True)
            frames.append(pose.copy())
            stamps.append(stamp_ns)
            host_times.append(time.time_ns())
            elapsed = int(now - start)
            if elapsed != last_report:
                print(f"  {now - start:.1f}/{args.seconds:g} s，{len(frames)} 帧", flush=True)
                last_report = elapsed
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        interrupted = True
        print("采集被中断，将保存已获取的数据。", flush=True)
    finally:
        try:
            if frames:
                metadata = {
                    "setting_height_cm": args.height,
                    "up_axis": args.up_axis,
                    "position_unit": "m",
                    "quaternion_order": "xyzw",
                    "timestamp_source": "get_time_stamp_ns",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_seconds": args.seconds,
                    "sample_rate_limit_hz": args.hz,
                    "interrupted": interrupted,
                    "skipped_invalid_frames": skipped,
                }
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("xb") as output:
                    np.savez_compressed(
                        output,
                        poses=np.stack(frames),
                        sdk_timestamps=np.asarray(stamps, dtype=np.int64),
                        host_time_ns=np.asarray(host_times, dtype=np.int64),
                        joint_names=np.asarray(JOINTS),
                        parents=np.asarray(PARENTS),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                    )
                print(f"已保存 {len(frames)} 帧：{args.out.resolve()}")
        finally:
            # Save before SDK teardown: the vendor SDK may abort during PXREADeinit.
            xrt.close()
    if not frames:
        raise RuntimeError("没有采集到数据")


def probe(args):
    """Inspect the raw XRoboToolkit body stream without creating a recording."""
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError("请使用 sonic_bumi 已安装 xrobotoolkit_sdk 的 teleop Python 环境") from exc

    print(f"探测 {args.seconds:g} 秒：只读取 XRoboToolkit Body 流，不启动 SONIC 或机器人。", flush=True)
    start_pc_service(args.no_start_pc_service)
    xrt.init()
    total = available = new_messages = valid = invalid = 0
    last_timestamp = None
    last_invalid = "无"
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            total += 1
            body_available = bool(xrt.is_body_data_available())
            if body_available:
                available += 1
                stamp_ns = int(xrt.get_time_stamp_ns())
            else:
                stamp_ns = None
            if body_available and (last_timestamp is None or stamp_ns != last_timestamp):
                last_timestamp = stamp_ns
                new_messages += 1
                try:
                    pose = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
                    if pose.shape != (24, 7):
                        raise ValueError(f"需要 [24,7]，实际为 {pose.shape}")
                    if not np.isfinite(pose).all():
                        raise ValueError("位姿含 NaN/Inf")
                    valid += 1
                except ValueError as exc:
                    invalid += 1
                    last_invalid = str(exc)
            time.sleep(1.0 / args.hz)
    finally:
        xrt.close()

    print(f"Body 可用轮次：{available}/{total}")
    print(f"时间戳更新：{new_messages}；有效 [24,7] 骨架帧：{valid}；无效帧：{invalid}")
    if invalid:
        print(f"最近无效原因：{last_invalid}")
    if available == 0:
        print("结论：PC Service 没有把 Body 流交给 SDK。检查 WORKING、Full-body 和 Send。")
    elif new_messages == 0:
        print("结论：Body 标志存在但数据流停住。关闭再开启 Send，并重连 PC Service 后重试。")
    elif valid == 0:
        print("结论：收到 Body 消息但骨架未初始化或不完整。重新进行 Motion Tracker 校准。")
    else:
        print("结论：原始 [24,7] Body 流正常，可以执行 record。")


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
    rec.add_argument(
        "--timeout", type=nonnegative, default=0,
        help="等待 Body/新帧的超时秒数；默认 0 表示一直等待，与 manager 一致",
    )
    rec.add_argument("--up-axis", choices=("x", "y", "z"), default="y", help="原始 PICO 默认 Y-up")
    rec.add_argument("--no-start-pc-service", action="store_true", help="不执行 runService.sh；仅在服务已由外部启动时使用")
    probe_parser = sub.add_parser("probe", help="诊断 XRoboToolkit 是否正在提供有效的 PICO Body 流")
    probe_parser.add_argument("--seconds", type=positive, default=5, help="探测秒数，默认 5")
    probe_parser.add_argument("--hz", type=positive, default=20, help="读取频率上限，默认 20 Hz")
    probe_parser.add_argument("--no-start-pc-service", action="store_true", help="不执行 runService.sh；仅在服务已由外部启动时使用")
    comp = sub.add_parser("compare", help="离线比较两批采集，只需要 numpy")
    comp.add_argument("a", type=Path)
    comp.add_argument("b", type=Path)
    comp.add_argument("--out", type=Path, help="可选：保存 CSV 比较表，不覆盖已有文件")
    args = parser.parse_args()
    try:
        if args.command == "record":
            record(args)
        elif args.command == "probe":
            probe(args)
        else:
            compare(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
