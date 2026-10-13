from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .runtime import (
    EntryPoseSpec,
    PolicyRuntime,
    PolicySpec,
    RuntimeSpec,
    _inference_spec,
    _load_yaml,
    _merge_mapping,
    _resolve_model,
    _resolve_policy_actuators,
    _vector,
    load_policy,
)


@dataclass(frozen=True)
class TrackingSpec:
    command_input: str
    command_size: int
    policy_input: str
    policy_size: int
    future_steps: List[int]
    history_steps: List[int]
    prev_action_steps: int
    reference_path: Path
    reference_fps: float
    loop: bool
    root_body_name: str


@dataclass
class ReferenceMotion:
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    fps: float

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])


def _resolve_data_file(cfg: Mapping[str, Any], policy_yaml: Path) -> Path:
    file_name = str(cfg.get("file", "")).strip()
    if not file_name:
        raise ValueError(f"tracking.reference.file missing in {policy_yaml}")
    candidate = Path(file_name).expanduser()
    if candidate.is_absolute():
        return candidate
    package = str(cfg.get("package", "")).strip()
    if package:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(package)) / candidate
    return policy_yaml.parent / candidate


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    result: List[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("reference motion contains zero quaternion")
    return q / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float32, copy=True)
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float32), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float32), -1, 0)
    return np.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), axis=-1).astype(np.float32, copy=False)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(np.asarray(q, dtype=np.float32))
    v = np.asarray(v, dtype=np.float32)
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_rotate(_quat_conjugate(q), v)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(np.asarray(q, dtype=np.float32))
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3)).astype(np.float32, copy=False)


def _projected_yaw_quat(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(np.asarray(q, dtype=np.float32).reshape(4))
    x_axis = _quat_rotate(q, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    z_axis = _quat_rotate(q, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    if float(np.linalg.norm(x_axis[:2])) > 0.1:
        heading = x_axis[:2]
    else:
        heading = z_axis[:2] if float(x_axis[2]) < 0.0 else -z_axis[:2]
    yaw = float(np.arctan2(heading[1], heading[0]))
    half = 0.5 * yaw
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _pick_npz(data: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    for name in names:
        if name in data:
            return data[name]
    return None


def _load_reference_motion(path: Path, joints: Sequence[str], root_body_name: str, fps: float) -> ReferenceMotion:
    if not path.is_file():
        raise FileNotFoundError(
            f"TaiChi reference motion not found: {path}. "
            "The ONNX is a reference-conditioned MimicLite tracker and needs the score1 trajectory."
        )
    if path.suffix.lower() != ".npz":
        raise ValueError(f"tracking reference must be .npz: {path}")

    with np.load(path, allow_pickle=False) as data:
        joint_pos_raw = _pick_npz(data, ("joint_pos", "joint_positions", "q"))
        if joint_pos_raw is None:
            raise ValueError("reference npz needs joint_pos [T, 21]")
        joint_pos = np.asarray(joint_pos_raw, dtype=np.float32)
        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be [T, J], got {joint_pos.shape}")

        source_joint_names = _as_string_list(_pick_npz(data, ("joint_names", "joints")))
        if source_joint_names:
            missing = [name for name in joints if name not in source_joint_names]
            if missing:
                raise ValueError(f"reference joint_names missing policy joints: {missing}")
            indices = [source_joint_names.index(name) for name in joints]
            joint_pos = joint_pos[:, indices]
        elif joint_pos.shape[1] != len(joints):
            raise ValueError(
                f"joint_pos has {joint_pos.shape[1]} joints but policy needs {len(joints)}; "
                "include joint_names in the npz so it can be reordered safely"
            )

        root_pos_raw = _pick_npz(data, ("root_pos", "root_position", "base_pos"))
        root_quat_raw = _pick_npz(data, ("root_quat", "root_quat_wxyz", "base_quat"))

        if root_pos_raw is None or root_quat_raw is None:
            body_pos_raw = _pick_npz(data, ("body_pos_w", "body_pos"))
            body_quat_raw = _pick_npz(data, ("body_quat_w", "body_quat"))
            body_names = _as_string_list(_pick_npz(data, ("body_names", "bodies")))
            if body_pos_raw is None or body_quat_raw is None or not body_names:
                raise ValueError(
                    "reference npz needs root_pos/root_quat, or body_pos_w/body_quat_w plus body_names"
                )
            if root_body_name not in body_names:
                raise ValueError(f"reference body_names does not contain {root_body_name}")
            body_idx = body_names.index(root_body_name)
            root_pos_raw = np.asarray(body_pos_raw)[:, body_idx]
            root_quat_raw = np.asarray(body_quat_raw)[:, body_idx]

        root_pos = np.asarray(root_pos_raw, dtype=np.float32)
        root_quat = _normalize_quat_wxyz(np.asarray(root_quat_raw, dtype=np.float32))
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must be [T, 3], got {root_pos.shape}")
        if root_quat.ndim != 2 or root_quat.shape[1] != 4:
            raise ValueError(f"root_quat must be [T, 4] in wxyz order, got {root_quat.shape}")

        frames = joint_pos.shape[0]
        if frames < 2 or root_pos.shape[0] != frames or root_quat.shape[0] != frames:
            raise ValueError(
                f"reference frame counts disagree: joint={frames}, root_pos={root_pos.shape[0]}, "
                f"root_quat={root_quat.shape[0]}"
            )
        if not (np.all(np.isfinite(joint_pos)) and np.all(np.isfinite(root_pos)) and np.all(np.isfinite(root_quat))):
            raise ValueError("reference motion contains NaN/Inf")

        file_fps = _pick_npz(data, ("fps", "frame_rate"))
        motion_fps = float(np.asarray(file_fps).reshape(-1)[0]) if file_fps is not None else float(fps)

    if abs(motion_fps - fps) > 1.0e-3:
        raise ValueError(
            f"reference fps={motion_fps} does not match policy fps={fps}; resample score1 to {fps:g} Hz first"
        )
    return ReferenceMotion(root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=motion_fps)


def load_policy_runtime(policy_yaml: Path, runtime_cfg: Mapping[str, Any]) -> PolicyRuntime:
    root = _load_yaml(policy_yaml)
    policy_type = str(root.get("policy_type", "locomotion")).strip().lower()
    if policy_type not in ("mimic_tracking", "tracking"):
        spec, runtime = load_policy(policy_yaml, runtime_cfg)
        return PolicyRuntime(spec, runtime)

    if not bool(root.get("enabled", True)):
        raise ValueError(f"policy disabled: {policy_yaml}")

    policy_runtime_cfg = root.get("runtime", {}) or {}
    if not isinstance(policy_runtime_cfg, Mapping):
        raise ValueError(f"runtime must be a mapping: {policy_yaml}")
    effective_runtime_cfg = _merge_mapping(policy_runtime_cfg, runtime_cfg)

    name = str(root.get("name", policy_yaml.stem))
    robot = str(root.get("robot", "")).strip()
    joints_cfg = root.get("joints", {}) or {}
    joints = [str(x) for x in joints_cfg.get("names", [])]
    if not joints:
        raise ValueError(f"joints.names is empty: {policy_yaml}")
    model_joints = [str(x) for x in joints_cfg.get("model_names", joints)]
    if len(model_joints) != len(joints):
        raise ValueError("joints.model_names length mismatch")
    actuators = _resolve_policy_actuators(robot, joints, policy_yaml)

    model_path = _resolve_model(root.get("model", {}), policy_yaml)
    inputs_cfg = root.get("inputs", {}) or {}
    if not isinstance(inputs_cfg, Mapping):
        raise ValueError("mimic_tracking inputs must be a mapping with command and policy")
    command_cfg = inputs_cfg.get("command", {}) or {}
    policy_cfg = inputs_cfg.get("policy", {}) or {}
    command_input = str(command_cfg.get("name", "command"))
    command_size = int(command_cfg.get("size", 240))
    policy_input = str(policy_cfg.get("name", "policy"))
    policy_size = int(policy_cfg.get("size", 378))

    action = root.get("action", {}) or {}
    action_dim = int(action.get("dimension", len(joints)))
    if action_dim != len(joints):
        raise ValueError(f"action.dimension={action_dim}, joints={len(joints)}")
    if str(action.get("type", "")) != "joint_position_delta":
        raise ValueError("only action.type=joint_position_delta is supported")

    safety = root.get("safety", {}) or {}
    spec = PolicySpec(
        name=name,
        robot=robot,
        model_path=model_path,
        model_input=policy_input,
        model_output=str(action.get("output_name", "action")),
        frame_size=policy_size,
        stack_size=1,
        model_input_size=policy_size,
        observation_clip=float(policy_cfg.get("clip", 1.0e9)),
        action_dimension=action_dim,
        action_clip=float(action.get("clip", 18.0)),
        joints=joints,
        model_joints=model_joints,
        control_model_names=list(model_joints),
        control_model_indices=list(range(len(model_joints))),
        actuators=actuators,
        virtual_joint_mode="fixed_default",
        terms=[],
        fall_protection_enabled=bool(safety.get("fall_protection_enabled", True)),
        projected_gravity_z_max=float(safety.get("projected_gravity_z_max", -0.3)),
    )

    cmd = effective_runtime_cfg.get("command", {}) or {}
    entry_pose_cfg = effective_runtime_cfg.get("entry_pose", {}) or {}
    entry_pose = None
    if bool(entry_pose_cfg.get("enabled", False)):
        duration_s = float(entry_pose_cfg.get("duration_s", 2.0))
        entry_pose = EntryPoseSpec(
            enabled=True,
            duration_s=duration_s,
            position=_vector(entry_pose_cfg.get("position", 0.0), joints, "entry_pose.position"),
            kp=_vector(entry_pose_cfg.get("kp", 10.0), joints, "entry_pose.kp"),
            kd=_vector(entry_pose_cfg.get("kd", 0.5), joints, "entry_pose.kd"),
        )

    runtime = RuntimeSpec(
        button=int(effective_runtime_cfg["button"]),
        action_scale=_vector(effective_runtime_cfg.get("action_scale", 0.25), joints, "action_scale"),
        default_position=_vector(effective_runtime_cfg.get("default_position", 0.0), joints, "default_position"),
        kp=_vector(effective_runtime_cfg.get("kp", 30.0), joints, "kp"),
        kd=_vector(effective_runtime_cfg.get("kd", 2.0), joints, "kd"),
        max_position_delta=(
            None if effective_runtime_cfg.get("max_position_delta", None) is None
            else _vector(effective_runtime_cfg.get("max_position_delta"), joints, "max_position_delta")
        ),
        history_init="repeat",
        command_axes={"x": int(cmd.get("axes", {}).get("x", 1)), "y": int(cmd.get("axes", {}).get("y", 0)), "yaw": int(cmd.get("axes", {}).get("yaw", 3))},
        command_signs={"x": 1.0, "y": 1.0, "yaw": 1.0},
        command_limits={"x": 0.0, "y": 0.0, "yaw": 0.0},
        inference=_inference_spec(effective_runtime_cfg),
        entry_pose=entry_pose,
    )

    tracking_cfg = root.get("tracking", {}) or {}
    reference_cfg = tracking_cfg.get("reference", {}) or {}
    future_steps = [int(x) for x in tracking_cfg.get("future_steps", [-8, -4, -2, 0, 1, 4, 8, 16])]
    history_steps = [int(x) for x in tracking_cfg.get("history_steps", [0, 1, 2, 3, 4, 8, 16])]
    prev_action_steps = int(tracking_cfg.get("prev_action_steps", 2))
    reference_fps = float(reference_cfg.get("fps", 50.0))
    tracking = TrackingSpec(
        command_input=command_input,
        command_size=command_size,
        policy_input=policy_input,
        policy_size=policy_size,
        future_steps=future_steps,
        history_steps=history_steps,
        prev_action_steps=prev_action_steps,
        reference_path=_resolve_data_file(reference_cfg, policy_yaml),
        reference_fps=reference_fps,
        loop=bool(reference_cfg.get("loop", False)),
        root_body_name=str(tracking_cfg.get("root_body_name", "base_link")),
    )

    expected_policy_size = (
        len(history_steps) * 3
        + len(history_steps) * 3
        + len(history_steps) * len(joints)
        + len(history_steps) * len(joints)
        + prev_action_steps * len(joints)
    )
    if expected_policy_size != policy_size:
        raise ValueError(
            f"tracking policy input layout computes {expected_policy_size}, configured {policy_size}; "
            "check history_steps/prev_action_steps"
        )
    expected_command_size = len(future_steps) * (3 + 6 + len(joints))
    if expected_command_size != command_size:
        raise ValueError(
            f"tracking command input layout computes {expected_command_size}, configured {command_size}"
        )

    return MimicTrackingRuntime(spec, runtime, tracking)


class MimicTrackingRuntime(PolicyRuntime):
    is_mimic_tracking = True

    def __init__(self, spec: PolicySpec, runtime: RuntimeSpec, tracking: TrackingSpec):
        super().__init__(spec, runtime)
        self.tracking = tracking
        self._tracking_command_input = np.zeros((1, tracking.command_size), dtype=np.float32)
        self._tracking_command_bound = False
        max_hist = max(tracking.history_steps)
        n = spec.action_dimension
        self._gyro_history = np.zeros((max_hist + 1, 3), dtype=np.float32)
        self._gravity_history = np.zeros((max_hist + 1, 3), dtype=np.float32)
        self._joint_pos_history = np.zeros((max_hist + 1, n), dtype=np.float32)
        self._joint_vel_history = np.zeros((max_hist + 1, n), dtype=np.float32)
        self._prev_action_history = np.zeros((tracking.prev_action_steps, n), dtype=np.float32)
        self._tracking_history_valid = False
        self._policy_obs = np.zeros(tracking.policy_size, dtype=np.float32)
        self._command_obs = np.zeros(tracking.command_size, dtype=np.float32)
        self._reference: ReferenceMotion | None = None
        self._reference_index = 0
        self._reference_complete = False

    def load_model(self) -> None:
        super().load_model()
        assert self._session is not None
        input_meta = {x.name: x for x in self._session.get_inputs()}
        if self.tracking.command_input not in input_meta:
            raise ValueError(
                f"model input '{self.tracking.command_input}' not found; available={sorted(input_meta)}"
            )
        command_meta = input_meta[self.tracking.command_input]
        if command_meta.shape:
            last = command_meta.shape[-1]
            if isinstance(last, int) and last > 0 and last != self.tracking.command_size:
                raise ValueError(
                    f"model command input size {last} != configured {self.tracking.command_size}"
                )

    def _run_session(self) -> None:
        assert self._session is not None
        if self._io_binding is not None:
            if not self._tracking_command_bound:
                self._io_binding.bind_cpu_input(
                    self.tracking.command_input, self._tracking_command_input)
                self._tracking_command_bound = True
            self._session.run_with_iobinding(self._io_binding)
            self._io_binding.synchronize_outputs()
        else:
            result = self._session.run(
                [self.spec.model_output],
                {
                    self.tracking.command_input: self._tracking_command_input,
                    self.spec.model_input: self._model_input,
                },
            )[0]
            np.copyto(self._model_output, np.asarray(result, dtype=np.float32).reshape(1, -1))

    def reset(self) -> None:
        super().reset()
        self._gyro_history.fill(0.0)
        self._gravity_history.fill(0.0)
        self._joint_pos_history.fill(0.0)
        self._joint_vel_history.fill(0.0)
        self._prev_action_history.fill(0.0)
        self._tracking_history_valid = False
        self._reference_index = 0
        self._reference_complete = False

    def prepare_activation(self) -> None:
        if self._reference is None:
            self._reference = _load_reference_motion(
                self.tracking.reference_path,
                self.spec.joints,
                self.tracking.root_body_name,
                self.tracking.reference_fps,
            )
        self._reference_index = 0
        self._reference_complete = False

    @property
    def reference_status(self) -> str:
        if self._reference is None:
            return f"unloaded:{self.tracking.reference_path}"
        return f"{self._reference_index}/{self._reference.frames - 1}"

    @property
    def entry_pose_target(self) -> np.ndarray:
        """Use reference frame 0 as the hand-off pose for a tracking policy."""
        if self._reference is None:
            raise RuntimeError("tracking reference is not loaded")
        return self._reference.joint_pos[0]

    @property
    def reference_complete(self) -> bool:
        return self._reference_complete

    def _push_history(self, buf: np.ndarray, value: np.ndarray) -> None:
        if not self._tracking_history_valid:
            buf[:] = value
        else:
            buf[1:] = buf[:-1]
            buf[0] = value

    def _sample_history(self, buf: np.ndarray) -> np.ndarray:
        return buf[np.asarray(self.tracking.history_steps, dtype=np.int64)].reshape(-1)

    def _reference_indices(self) -> np.ndarray:
        assert self._reference is not None
        idx = self._reference_index + np.asarray(self.tracking.future_steps, dtype=np.int64)
        if self.tracking.loop:
            return np.mod(idx, self._reference.frames)
        return np.clip(idx, 0, self._reference.frames - 1)

    def _build_reference_command(self, robot_root_quat_wxyz: np.ndarray) -> np.ndarray:
        assert self._reference is not None
        indices = self._reference_indices()
        ref_root_pos = self._reference.root_pos[indices]
        ref_root_quat = self._reference.root_quat[indices]
        ref_joint_pos = self._reference.joint_pos[indices]

        current_idx = int(np.clip(self._reference_index, 0, self._reference.frames - 1))
        anchor_pos = np.array(self._reference.root_pos[current_idx], dtype=np.float32, copy=True)
        anchor_pos[2] = 0.0
        anchor_yaw = _projected_yaw_quat(self._reference.root_quat[current_idx])
        root_pos_local = _quat_rotate_inverse(anchor_yaw, ref_root_pos - anchor_pos)

        robot_q = _normalize_quat_wxyz(np.asarray(robot_root_quat_wxyz, dtype=np.float32).reshape(4))
        robot_inv = _quat_conjugate(robot_q)
        rel_quat = _quat_mul(np.broadcast_to(robot_inv, ref_root_quat.shape), ref_root_quat)
        rel_mat = _quat_to_matrix(rel_quat)
        root_ori_6d = rel_mat[:, :2, :].reshape(-1)

        cursor = 0
        a = root_pos_local.reshape(-1)
        self._command_obs[cursor:cursor + a.size] = a
        cursor += a.size
        self._command_obs[cursor:cursor + root_ori_6d.size] = root_ori_6d
        cursor += root_ori_6d.size
        a = ref_joint_pos.reshape(-1)
        self._command_obs[cursor:cursor + a.size] = a
        cursor += a.size
        if cursor != self.tracking.command_size:
            raise RuntimeError(
                f"tracking command built {cursor} values, expected {self.tracking.command_size}"
            )
        return self._command_obs

    def build_tracking_inputs(self, low_state: Any, projected_gravity: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self._reference is None:
            self.prepare_activation()

        indices = self._resolve_joint_indices(low_state)
        for i, index in enumerate(indices):
            state = low_state.motor_state[index]
            self._q[i] = float(state.q)
            self._dq[i] = float(state.dq)

        gravity = projected_gravity
        if gravity is None:
            gravity = self.projected_gravity(low_state)
        else:
            np.copyto(self._gravity, np.asarray(gravity, dtype=np.float32).reshape(3))
            gravity = self._gravity

        gyro = np.asarray(low_state.imu.gyroscope, dtype=np.float32).reshape(3)
        self._push_history(self._gyro_history, gyro)
        self._push_history(self._gravity_history, np.asarray(gravity, dtype=np.float32).reshape(3))
        self._push_history(self._joint_pos_history, self._q)
        self._push_history(self._joint_vel_history, self._dq)
        self._tracking_history_valid = True

        parts = (
            self._sample_history(self._gyro_history),
            self._sample_history(self._gravity_history),
            self._sample_history(self._joint_pos_history),
            self._sample_history(self._joint_vel_history),
            self._prev_action_history.reshape(-1),
        )
        cursor = 0
        for part in parts:
            end = cursor + part.size
            self._policy_obs[cursor:end] = part
            cursor = end
        if cursor != self.tracking.policy_size:
            raise RuntimeError(
                f"tracking policy observation built {cursor}, expected {self.tracking.policy_size}"
            )

        command = self._build_reference_command(low_state.imu.quaternion)
        return command, self._policy_obs

    def infer_tracking(self, command: np.ndarray, policy_observation: np.ndarray) -> np.ndarray:
        self.load_model()
        command_arr = np.asarray(command, dtype=np.float32).reshape(-1)
        policy_arr = np.asarray(policy_observation, dtype=np.float32).reshape(-1)
        if command_arr.size != self.tracking.command_size:
            raise ValueError(
                f"command size {command_arr.size} != configured {self.tracking.command_size}"
            )
        if policy_arr.size != self.tracking.policy_size:
            raise ValueError(
                f"policy observation size {policy_arr.size} != configured {self.tracking.policy_size}"
            )
        np.copyto(self._tracking_command_input[0], command_arr)
        np.copyto(self._model_input[0], policy_arr)

        import time
        start_ns = time.perf_counter_ns()
        self._run_session()
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._infer_count += 1
        self._infer_total_ns += elapsed_ns
        self._infer_last_ns = elapsed_ns
        self._infer_max_ns = max(self._infer_max_ns, elapsed_ns)

        result = self._model_output.reshape(-1)
        if result.size != self.spec.action_dimension:
            raise ValueError(f"model action size {result.size} != {self.spec.action_dimension}")
        np.clip(result, -self.spec.action_clip, self.spec.action_clip, out=self._action)
        np.copyto(self.last_action, self._action)

        if self.tracking.prev_action_steps > 0:
            self._prev_action_history[1:] = self._prev_action_history[:-1]
            self._prev_action_history[0] = self._action

        assert self._reference is not None
        if self.tracking.loop:
            self._reference_index = (self._reference_index + 1) % self._reference.frames
        else:
            if self._reference_index < self._reference.frames - 1:
                self._reference_index += 1
            else:
                self._reference_complete = True
        return self._action
