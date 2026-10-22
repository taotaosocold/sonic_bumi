"""把身体追踪直接重采样为五点参考窗口，复用现有 PICO 读取器与消息协议。"""

from collections import deque
from pathlib import Path
import time

import numpy as np

from gear_sonic.utils.teleop.smpl_five_point import (
    FIVE_POINT_SCHEMA,
    five_point_pose_features,
    pico_body_to_five_point,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


class FivePointPoseStreamer:
    """直接选择骨架五点，不创建 SMPL 身体模型或 G1 逆运动学处理器。

    输入读取器提供带设备时间戳的 [24,7] 世界姿态；位置线性插值、四元数按
    最短弧球面插值后，以固定五十赫兹保存窗口。十帧依次为旧到新，跨度
    0.18 秒；实时端没有未来数据，所以使用完整的最近窗口，不伪造预测帧。
    无效追踪、长时间断流或时间戳重置时清空窗口，重新积累十帧后才恢复发送。
    """

    def __init__(self, socket, reader, num_frames_to_send=10, target_fps=50,
                 record_dir="", record_format="npz", log_prefix="FivePoint"):
        if target_fps != 50 or num_frames_to_send < 10:
            raise ValueError("五点模式要求 50 Hz 和至少十帧窗口")
        if record_format not in ("npz", "bin"):
            raise ValueError("五点录制格式仅支持 npz 或 bin")
        self.socket, self.reader = socket, reader
        self.frames = deque(maxlen=num_frames_to_send)
        self.num_frames = num_frames_to_send
        self.step_ns = 20_000_000
        self.previous = None
        self.next_target_ns = None
        self.frame_index = 0
        self.record_dir = Path(record_dir) if record_dir else None
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
        self.record_format, self.record_index = record_format, 0
        self.log_prefix, self.last_warning = log_prefix, 0.0
        self.ready_reported = False

    @staticmethod
    def _interpolate_quaternions(previous, current, alpha):
        """对 wxyz 四元数沿最短弧插值；近重合时用归一化线性插值避免除零。"""
        previous = np.asarray(previous, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        previous = previous / np.linalg.norm(previous, axis=-1, keepdims=True)
        current = current / np.linalg.norm(current, axis=-1, keepdims=True)
        dot = np.sum(previous * current, axis=-1, keepdims=True)
        current = np.where(dot < 0.0, -current, current)
        dot = np.clip(np.abs(dot), 0.0, 1.0)
        angle = np.arccos(dot)
        denominator = np.maximum(np.sin(angle), 1e-8)
        spherical = (np.sin((1.0 - alpha) * angle) * previous + np.sin(alpha * angle) * current) / denominator
        linear = (1.0 - alpha) * previous + alpha * current
        result = np.where(dot > 0.9995, linear, spherical)
        return result / np.linalg.norm(result, axis=-1, keepdims=True)

    def _reset_window(self):
        """仅丢弃参考缓存；保留递增帧号，使接收端能够识别断流后的新窗口。"""
        self.frames.clear()
        self.previous = None
        self.next_target_ns = None
        self.ready_reported = False

    def run_once(self):
        """处理一个新设备样本；重复时间戳不会生成新帧或刷新控制端看门狗。"""
        sample = self.reader.get_latest()
        if sample is None:
            time.sleep(0.001)
            return
        stamp = int(sample.get("timestamp_ns", 0))
        age = time.monotonic() - sample.get("timestamp_monotonic", 0.0)
        if age > 0.25 or stamp <= 0:
            self._reset_window()
            time.sleep(0.001)
            return
        if self.previous is not None and stamp == self.previous[0]:
            time.sleep(0.001)
            return
        try:
            points, root = pico_body_to_five_point(sample["body_poses_np"])
        except ValueError as error:
            self._reset_window()
            if time.monotonic() - self.last_warning > 2.0:
                print(f"[{self.log_prefix}] 五点追踪无效，等待完整新窗口：{error}")
                self.last_warning = time.monotonic()
            time.sleep(0.001)
            return
        if self.previous is not None and not 0 < stamp - self.previous[0] <= 100_000_000:
            self._reset_window()
        if self.previous is None:
            self.previous = (stamp, points, root)
            self.next_target_ns = stamp + self.step_ns
            self._append(points, root)
            return

        previous_stamp, previous_points, previous_root = self.previous
        added = False
        while self.next_target_ns <= stamp:
            alpha = (self.next_target_ns - previous_stamp) / (stamp - previous_stamp)
            positions = (1.0 - alpha) * previous_points[:, :3] + alpha * points[:, :3]
            orientations = self._interpolate_quaternions(previous_points[:, 3:], points[:, 3:], alpha)
            use_points = np.concatenate((positions, orientations), axis=-1)
            use_root = self._interpolate_quaternions(previous_root, root, alpha)
            self._append(use_points, use_root)
            self.next_target_ns += self.step_ns
            added = True
        self.previous = (stamp, points, root)
        if added and len(self.frames) == self.num_frames:
            self._send(sample)

    def _append(self, points, root):
        """旋转完成插值后才转换成六维表示，避免对旋转矩阵直接插值。"""
        self.frames.append((five_point_pose_features(points), np.asarray(root, dtype=np.float32), self.frame_index))
        self.frame_index += 1

    def _send(self, sample):
        """按带版本号的固定布局发布完整窗口；可选录制保存实际发送的内容。"""
        data = {
            "smpl_five_point_pose": np.stack([frame[0] for frame in self.frames]),
            "body_quat_w": np.stack([frame[1] for frame in self.frames]),
            "frame_index": np.asarray([frame[2] for frame in self.frames], dtype=np.int64),
            "smpl_five_point_schema": np.asarray([FIVE_POINT_SCHEMA], dtype=np.int32),
            "timestamp_monotonic": np.asarray([sample["timestamp_monotonic"]], dtype=np.float64),
        }
        packet = pack_pose_message(data, topic="pose")
        self.socket.send(packet)
        if not self.ready_reported:
            print(f"BUMI3_FIVE_POINT_STREAM_READY: 50Hz frames={self.num_frames} schema={FIVE_POINT_SCHEMA}")
            self.ready_reported = True
        if self.record_dir:
            path = self.record_dir / f"five_point_{self.record_index:06d}.{self.record_format}"
            if self.record_format == "npz":
                np.savez_compressed(path, **data)
            else:
                path.write_bytes(packet)
            self.record_index += 1
