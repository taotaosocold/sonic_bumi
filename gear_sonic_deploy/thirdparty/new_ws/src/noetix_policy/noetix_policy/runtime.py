from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass(frozen=True)
class PolicySpec:
    """策略静态描述：从 policy YAML 解析出来的、不随运行环境变化的信息。

    命名空间说明：
      - joints / actuators 描述的是机器人物理层的 LowState / LowCmd 控制空间。
      - model_joints 描述的是 ONNX 模型的观测/动作关节空间。
      - 两者通常一致，但策略可能包含机器人上不存在的“虚拟关节”。
    """

    name: str                       # 策略名
    robot: str                      # 机器人名（用于查找 robot.yaml）
    model_path: Path                # ONNX 模型文件路径
    model_input: str                # ONNX 输入张量名
    model_output: str               # ONNX 输出张量名
    frame_size: int                 # 单帧观测的维度
    stack_size: int                 # 观测历史堆叠帧数
    model_input_size: int           # ONNX 实际输入维度 = frame_size * stack_size
    observation_clip: float         # 观测裁剪阈值
    action_dimension: int           # 动作维度（模型输出维度）
    action_clip: float              # 动作裁剪阈值
    # joints/actuators 描述物理 LowState/LowCmd 控制空间
    # model_joints 描述 ONNX 动作/观测关节空间
    joints: List[str]               # URDF 关节名列表（物理控制顺序）
    model_joints: List[str]         # 模型侧关节名列表（模型动作顺序）
    control_model_names: List[str]  # 与 joints 一一对应，指向 model_joints 中的名字
    control_model_indices: List[int]# control_model_names 在 model_joints 中的下标
    actuators: List[str]            # robot.yaml 中解析出的执行器名（旧接口）
    virtual_joint_mode: str         # 虚拟关节策略：fixed_default / commanded_position
    terms: List[Dict[str, Any]]     # 观测项列表
    fall_protection_enabled: bool   # 是否启用摔倒保护
    projected_gravity_z_max: float  # 摔倒保护：投影重力 z 上限


@dataclass(frozen=True)
class InferenceSpec:
    """推理相关配置:ONNX Runtime / TensorRT / CUDA / CPU。"""

    device: str                     # gpu / auto / cpu
    provider: str                   # auto / tensorrt / cuda / cpu
    device_id: int                  # GPU 编号
    allow_cpu_fallback: bool        # 是否允许回退到 CPU
    io_binding: bool                # 是否启用 io_binding（减少拷贝）
    warmup_runs: int                # 预热次数
    cuda: Dict[str, Any]            # CUDA provider 参数
    tensorrt: Dict[str, Any]        # TensorRT provider 参数


@dataclass(frozen=True)
class EntryPoseSpec:
    """策略启动时缓慢平滑到目标姿态的配置。"""

    enabled: bool
    duration_s: float               # 过渡时间
    position: np.ndarray            # 目标关节位置
    kp: np.ndarray                  # 过渡期间使用的刚度
    kd: np.ndarray                  # 过渡期间使用的阻尼


@dataclass(frozen=True)
class RuntimeSpec:
    """运行时配置:包含按钮、动作缩放、默认姿态、PD 增益、命令映射等。"""

    button: int                             # 触发该策略的手柄按钮编号
    action_scale: np.ndarray                # 动作缩放（乘到模型输出上）
    default_position: np.ndarray            # 默认关节位置
    kp: np.ndarray                          # 关节刚度
    kd: np.ndarray                          # 关节阻尼
    max_position_delta: np.ndarray | None   # 目标位置相对默认位置的最大偏移（可空）
    history_init: str                       # 观测历史初始化方式：repeat / zero
    command_axes: Dict[str, int]            # 手柄轴到 x/y/yaw 的映射
    command_signs: Dict[str, float]         # 各方向符号
    command_limits: Dict[str, float]        # 各方向速度上限
    inference: InferenceSpec                # 推理配置
    entry_pose: EntryPoseSpec | None        # 入场过渡姿态（可空）


# =============================================================================
# 通用工具函数
# =============================================================================

def _load_yaml(path: Path) -> Mapping[str, Any]:
    """加载 YAML 文件并确保其根节点是 mapping。"""
    with path.open("r", encoding="utf-8") as f:
        root = yaml.safe_load(f)
    if not isinstance(root, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return root


def _resolve_policy_actuators(robot: str, joints: Sequence[str], policy_yaml: Path) -> List[str]:
    """把策略里的关节名映射到 LowState/LowCmd 实际暴露的执行器名。

    背景：
      - 策略和 ONNX 元数据使用 URDF/模型关节名。
      - Noetix 桥接层在 MotorState.name / MotorCmd.name 中暴露的是执行器名。
      - 所以需要把两个命名空间通过 robot.yaml 关联起来。

    兼容：
      - 正常策略使用 URDF 关节名。
      - 老 Bumi 控制器直接在策略里用执行器名。
      - 两种命名空间都接受，以便兼容旧策略。
    """
    robot = str(robot).strip()
    if not robot:
        raise ValueError(f"robot missing in {policy_yaml}")

    # 从 noetix_robots 包中获取 robot 资源目录
    robots_share = Path(get_package_share_directory("noetix_robots"))
    robot_yaml = robots_share / robot / "config" / "robot.yaml"
    if not robot_yaml.is_file():
        raise FileNotFoundError(
            f"robot resource not found for policy '{policy_yaml}': {robot_yaml}")

    root = _load_yaml(robot_yaml)
    entries = root.get("joints", []) or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"joints must be a list in {robot_yaml}")

    # 构建 关节名 -> 执行器名 的映射
    joint_to_actuator: Dict[str, str] = {}
    actuator_names = set()
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        joint_name = str(item.get("name", "")).strip()
        actuator_name = str(item.get("actuator", "")).strip()
        if not joint_name or not actuator_name:
            continue
        if joint_name in joint_to_actuator:
            raise ValueError(f"duplicate robot joint '{joint_name}' in {robot_yaml}")
        joint_to_actuator[joint_name] = actuator_name
        actuator_names.add(actuator_name)

    # 优先使用 URDF 关节名，找不到再退回执行器名
    actuators: List[str] = []
    missing: List[str] = []
    for name in joints:
        if name in joint_to_actuator:
            actuators.append(joint_to_actuator[name])
        elif name in actuator_names:
            actuators.append(name)
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"policy joints are missing from {robot_yaml}: {missing}")
    if any(not name for name in actuators):
        raise ValueError(f"empty actuator mapping in {robot_yaml}")
    if len(set(actuators)) != len(actuators):
        raise ValueError(
            f"policy joint mapping is not one-to-one in {robot_yaml}: {actuators}")
    return actuators


def _resolve_model(model_cfg: Mapping[str, Any], policy_yaml: Path) -> Path:
    """解析 ONNX 模型路径:支持绝对路径、package:// 相对路径、相对 YAML 路径。"""
    file_name = str(model_cfg.get("file", "")).strip()
    if not file_name:
        raise ValueError(f"model.file missing in {policy_yaml}")

    candidate = Path(file_name).expanduser()
    if candidate.is_absolute():
        return candidate

    # 如果指定了 package，则从 ROS 2 包共享目录查找
    package = str(model_cfg.get("package", "")).strip()
    if package:
        return Path(get_package_share_directory(package)) / candidate

    # 否则相对于 policy YAML 文件位置
    return policy_yaml.parent / candidate


def _vector(value: Any, joints: Sequence[str], field: str) -> np.ndarray:
    """把一个 YAML 字段解析成长度为 len(joints) 的 float32 向量。

    支持三种写法：
      1. mapping：{joint_name: value, ...}，缺一不可，且不能有多余键。
      2. sequence：[v1, v2, ...]，长度必须等于 joints 长度。
      3. 标量：所有关节都用同一个值。
    """
    n = len(joints)
    if isinstance(value, Mapping):
        unknown = set(value.keys()) - set(joints)
        if unknown:
            raise ValueError(f"{field}: unknown joints: {sorted(unknown)}")
        missing = [name for name in joints if name not in value]
        if missing:
            raise ValueError(f"{field}: missing joints: {missing}")
        return np.asarray([float(value[name]) for name in joints], dtype=np.float32)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != n:
            raise ValueError(f"{field}: expected {n} values, got {len(value)}")
        return np.asarray(value, dtype=np.float32)

    return np.full(n, float(value), dtype=np.float32)


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """合并一层嵌套 mapping，override 覆盖 base，同时保留 base 的其他键。"""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(dict(value))
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _inference_spec(runtime_cfg: Mapping[str, Any]) -> InferenceSpec:
    """从 runtime 配置中解析推理相关参数并做一致性校验。"""
    cfg = runtime_cfg.get("inference", {}) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("inference must be a mapping")

    device = str(cfg.get("device", "gpu")).strip().lower()
    provider = str(cfg.get("provider", "auto")).strip().lower()
    if device not in ("gpu", "auto", "cpu"):
        raise ValueError("inference.device must be one of: gpu, auto, cpu")
    if provider not in ("auto", "tensorrt", "cuda", "cpu"):
        raise ValueError("inference.provider must be one of: auto, tensorrt, cuda, cpu")

    # device 和 provider 互相约束：任一是 cpu 时另一个也必须是 cpu
    if device == "cpu":
        provider = "cpu"
    if provider == "cpu":
        device = "cpu"

    cuda_cfg = dict(cfg.get("cuda", {}) or {})
    trt_cfg = dict(cfg.get("tensorrt", {}) or {})
    return InferenceSpec(
        device=device,
        provider=provider,
        device_id=int(cfg.get("device_id", 0)),
        allow_cpu_fallback=bool(cfg.get("allow_cpu_fallback", False)),
        io_binding=bool(cfg.get("io_binding", True)),
        warmup_runs=max(0, int(cfg.get("warmup_runs", 10))),
        cuda=cuda_cfg,
        tensorrt=trt_cfg,
    )


# =============================================================================
# 策略加载主入口
# =============================================================================

def load_policy(policy_yaml: Path, runtime_cfg: Mapping[str, Any]) -> tuple[PolicySpec, RuntimeSpec]:
    """加载 policy YAML 与 manager 传入的 runtime 覆盖配置，返回 (PolicySpec, RuntimeSpec)。

    合并规则：
      - policy YAML 里的 runtime 字段是该策略的默认运行时配置。
      - manager 传入的 runtime_cfg 可以覆盖这些默认值（比如按钮、命令映射、推理选项）。
    """
    root = _load_yaml(policy_yaml)
    if not bool(root.get("enabled", True)):
        raise ValueError(f"policy disabled: {policy_yaml}")

    # policy 里的 runtime 作为默认值，manager 传入的 runtime_cfg 覆盖它
    policy_runtime_cfg = root.get("runtime", {}) or {}
    if not isinstance(policy_runtime_cfg, Mapping):
        raise ValueError(f"runtime must be a mapping: {policy_yaml}")
    effective_runtime_cfg = _merge_mapping(policy_runtime_cfg, runtime_cfg)

    # ---- 基本信息 ----
    name = str(root.get("name", policy_yaml.stem))
    robot = str(root.get("robot", "")).strip()

    # ---- 关节信息 ----
    joints_cfg = root.get("joints", {}) or {}
    joints = [str(x) for x in joints_cfg.get("names", [])]
    if not joints:
        raise ValueError(f"joints.names is empty: {policy_yaml}")

    # 模型关节名，默认与 joints 相同
    model_joints = [str(x) for x in joints_cfg.get("model_names", joints)]
    if not model_joints:
        raise ValueError(f"joints.model_names is empty: {policy_yaml}")
    if len(set(model_joints)) != len(model_joints):
        raise ValueError("joints.model_names contains duplicates")

    # control_model_names 与 joints.names 一一对应，指向 model_joints 中的名字。
    # 用于把训练时有额外头/手等 DOF 的策略，部署到只暴露运动关节的机器人上。
    control_model_names_cfg = joints_cfg.get("control_model_names")
    if control_model_names_cfg is None:
        if len(model_joints) != len(joints):
            raise ValueError(
                "joints.control_model_names is required when model_names and "
                "joints.names have different lengths")
        control_model_names = list(model_joints)
    else:
        control_model_names = [str(x) for x in control_model_names_cfg]

    if len(control_model_names) != len(joints):
        raise ValueError(
            f"joints.control_model_names length={len(control_model_names)} "
            f"!= joints.names length={len(joints)}")
    if len(set(control_model_names)) != len(control_model_names):
        raise ValueError("joints.control_model_names contains duplicates")

    missing_control_model = [name for name in control_model_names if name not in model_joints]
    if missing_control_model:
        raise ValueError(
            f"joints.control_model_names missing from model_names: {missing_control_model}")
    control_model_indices = [model_joints.index(name) for name in control_model_names]

    # 虚拟关节模式：
    #   fixed_default      —— 虚拟关节固定在默认位置，动作输出置零（默认）
    #   commanded_position —— 虚拟关节跟随上一帧命令位置
    virtual_joint_mode = str(joints_cfg.get("virtual_joint_mode", "fixed_default")).strip().lower()
    if virtual_joint_mode not in ("fixed_default", "commanded_position"):
        raise ValueError(
            "joints.virtual_joint_mode must be 'fixed_default' or 'commanded_position'")
    actuators = _resolve_policy_actuators(robot, joints, policy_yaml)

    # ---- 模型信息 ----
    model_cfg = root.get("model", {})
    model_path = _resolve_model(model_cfg, policy_yaml)

    # ---- 输入观测项 ----
    inputs = root.get("inputs", [])
    if len(inputs) != 1:
        raise ValueError(f"exactly one model input is supported: {policy_yaml}")
    inp = inputs[0]
    terms = list(inp.get("terms", []))
    term_dim = sum(int(term.get("dim", 0)) for term in terms)
    frame_size = int(inp.get("frame_size", term_dim))
    if term_dim != frame_size:
        raise ValueError(f"observation term dims={term_dim}, frame_size={frame_size}")

    # 历史堆叠：model_input_size = frame_size * stack_size
    stack_size = int(inp.get("history", {}).get("stack_size", 1))
    model_input_size = int(inp.get("model_input_size", frame_size * stack_size))
    if frame_size * stack_size != model_input_size:
        raise ValueError(
            f"frame_size({frame_size}) * stack_size({stack_size}) != "
            f"model_input_size({model_input_size})")

    # ---- 动作信息 ----
    action = root.get("action", {})
    action_dim = int(action.get("dimension", len(model_joints)))
    if action_dim != len(model_joints):
        raise ValueError(
            f"action.dimension={action_dim}, model_joints={len(model_joints)}")
    if str(action.get("type", "")) != "joint_position_delta":
        raise ValueError("only action.type=joint_position_delta is supported")

    # ---- 安全检查 ----
    safety = root.get("safety", {})
    spec = PolicySpec(
        name=name,
        robot=robot,
        model_path=model_path,
        model_input=str(inp.get("name", "obs")),
        model_output=str(action.get("output_name", "actions")),
        frame_size=frame_size,
        stack_size=stack_size,
        model_input_size=model_input_size,
        observation_clip=float(inp.get("clip", 18.0)),
        action_dimension=action_dim,
        action_clip=float(action.get("clip", 18.0)),
        joints=joints,
        model_joints=model_joints,
        control_model_names=control_model_names,
        control_model_indices=control_model_indices,
        actuators=actuators,
        virtual_joint_mode=virtual_joint_mode,
        terms=terms,
        fall_protection_enabled=bool(safety.get("fall_protection_enabled", False)),
        projected_gravity_z_max=float(safety.get("projected_gravity_z_max", -0.3)),
    )

    # ---- 命令映射 ----
    cmd = effective_runtime_cfg.get("command", {})

    # ---- 入场姿态 ----
    entry_pose_cfg = effective_runtime_cfg.get("entry_pose", {}) or {}
    if not isinstance(entry_pose_cfg, Mapping):
        raise ValueError("runtime.entry_pose must be a mapping")
    entry_pose = None
    if bool(entry_pose_cfg.get("enabled", False)):
        duration_s = float(entry_pose_cfg.get("duration_s", 2.0))
        if duration_s <= 0.0:
            raise ValueError("runtime.entry_pose.duration_s must be > 0")
        entry_pose = EntryPoseSpec(
            enabled=True,
            duration_s=duration_s,
            position=_vector(entry_pose_cfg.get("position", 0.0), joints, "entry_pose.position"),
            kp=_vector(entry_pose_cfg.get("kp", 10.0), joints, "entry_pose.kp"),
            kd=_vector(entry_pose_cfg.get("kd", 0.5), joints, "entry_pose.kd"),
        )

    # ---- RuntimeSpec ----
    runtime = RuntimeSpec(
        button=int(effective_runtime_cfg["button"]),
        # 注意：以下向量都是按 model_joints 顺序
        action_scale=_vector(effective_runtime_cfg.get("action_scale", 0.25), model_joints, "action_scale"),
        default_position=_vector(effective_runtime_cfg.get("default_position", 0.0), model_joints, "default_position"),
        kp=_vector(effective_runtime_cfg.get("kp", 30.0), model_joints, "kp"),
        kd=_vector(effective_runtime_cfg.get("kd", 2.0), model_joints, "kd"),
        max_position_delta=(
            None
            if effective_runtime_cfg.get("max_position_delta", None) is None
            else _vector(effective_runtime_cfg.get("max_position_delta"), model_joints, "max_position_delta")
        ),
        history_init=str(effective_runtime_cfg.get("history_init", "repeat")),
        command_axes={
            "x": int(cmd.get("axes", {}).get("x", 1)),
            "y": int(cmd.get("axes", {}).get("y", 0)),
            "yaw": int(cmd.get("axes", {}).get("yaw", 3)),
        },
        command_signs={
            "x": float(cmd.get("signs", {}).get("x", 1.0)),
            "y": float(cmd.get("signs", {}).get("y", 1.0)),
            "yaw": float(cmd.get("signs", {}).get("yaw", 1.0)),
        },
        command_limits={
            "x": float(cmd.get("limits", {}).get("x", 0.5)),
            "y": float(cmd.get("limits", {}).get("y", 0.3)),
            "yaw": float(cmd.get("limits", {}).get("yaw", 0.5)),
        },
        inference=_inference_spec(effective_runtime_cfg),
        entry_pose=entry_pose,
    )
    if runtime.history_init not in ("repeat", "zero"):
        raise ValueError("history_init must be 'repeat' or 'zero'")
    return spec, runtime


def rotate_world_gravity_to_body(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """把世界坐标系的重力单位向量 [0, 0, -1] 旋转到机体坐标系。

    输入四元数顺序是 [w, x, y, z]，表示从 body 到 world 的旋转。
    输出是 body 坐标系中重力方向的表达，用于观测里的 projected_gravity。
    """
    if len(quaternion_wxyz) != 4:
        raise ValueError("quaternion must be [w, x, y, z]")
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm < 1.0e-9:
        raise ValueError("invalid zero quaternion")
    inv = 1.0 / norm
    w *= inv
    x *= inv
    y *= inv
    z *= inv

    # R^T * [0, 0, -1]，展开后得到下面的公式
    return np.asarray([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


# =============================================================================
# 策略运行时
# =============================================================================

class PolicyRuntime:
    """策略运行时：负责 ONNX 会话管理、观测构造、推理、动作到目标的转换。

    设计要点：
      - GPU 优先(TensorRT > CUDA > CPU),支持 io_binding。
      - 所有中间缓冲区预分配，避免在 50Hz/100Hz 控制循环里分配内存。
      - 观测历史用环形更新方式(shift + 尾插）。
    """

    def __init__(self, spec: PolicySpec, runtime: RuntimeSpec):
        self.spec = spec
        self.runtime = runtime
        self._session = None            # onnxruntime.InferenceSession
        self._ort = None                # onnxruntime 模块
        self._io_binding = None         # io_binding 对象
        self._active_providers: List[str] = []  # 实际生效的 provider

        # 预分配缓冲区（所有在控制循环里复用的数组都提前分配好）
        self._history = np.zeros((spec.stack_size, spec.frame_size), dtype=np.float32)
        self._history_valid = False
        self._frame = np.empty(spec.frame_size, dtype=np.float32)
        self._q = np.empty(spec.action_dimension, dtype=np.float32)          # 模型空间关节位置
        self._dq = np.empty(spec.action_dimension, dtype=np.float32)         # 模型空间关节速度
        self._q_rel = np.empty(spec.action_dimension, dtype=np.float32)      # q - default_position
        self._gravity = np.empty(3, dtype=np.float32)
        self._model_input = np.empty((1, spec.model_input_size), dtype=np.float32)
        self._model_output = np.empty((1, spec.action_dimension), dtype=np.float32)
        self._action = np.empty(spec.action_dimension, dtype=np.float32)     # 裁剪后的动作
        self._delta = np.empty(spec.action_dimension, dtype=np.float32)
        self._position_target = np.empty(spec.action_dimension, dtype=np.float32)
        self.last_action = np.zeros(spec.action_dimension, dtype=np.float32) # 上一帧动作

        # 物理/虚拟关节掩码
        self._control_model_indices = np.asarray(spec.control_model_indices, dtype=np.int64)
        self._physical_model_mask = np.zeros(spec.action_dimension, dtype=bool)
        self._physical_model_mask[self._control_model_indices] = True
        self._virtual_model_mask = ~self._physical_model_mask

        self._joint_indices: List[int] | None = None  # LowState.motor_state 中的下标缓存
        self._control_names: List[str] | None = None  # 当前 LowState 暴露的名字
        self._control_namespace = "unresolved"        # urdf_joint / actuator / unresolved

        # 预编译每个观测项的 indices 选择，避免在 50Hz 循环里做转换
        self._term_indices: List[tuple[int, ...] | None] = []
        for term in spec.terms:
            indices_cfg = term.get("indices")
            if indices_cfg is None:
                self._term_indices.append(None)
                continue
            indices = tuple(int(x) for x in indices_cfg)
            dim = int(term.get("dim", 0))
            if len(indices) != dim:
                raise ValueError(
                    f"term {term.get('type', '')}: indices count={len(indices)}, expected dim={dim}")
            if any(index < 0 for index in indices):
                raise ValueError(
                    f"term {term.get('type', '')}: indices must be non-negative")
            self._term_indices.append(indices)

        # 推理统计
        self._infer_count = 0
        self._infer_total_ns = 0
        self._infer_max_ns = 0
        self._infer_last_ns = 0

    # ---------- Provider 配置 ----------

    @staticmethod
    def _bool_provider_value(value: Any) -> str:
        """ONNX Runtime provider 的布尔选项需要字符串 "0"/"1"。"""
        return "1" if bool(value) else "0"

    def _cuda_provider_options(self) -> Dict[str, str]:
        """构造 CUDAExecutionProvider 的选项字典。"""
        cfg = self.runtime.inference.cuda
        options: Dict[str, str] = {"device_id": str(self.runtime.inference.device_id)}
        if "arena_extend_strategy" in cfg:
            options["arena_extend_strategy"] = str(cfg["arena_extend_strategy"])
        if "gpu_mem_limit_mb" in cfg and int(cfg["gpu_mem_limit_mb"]) > 0:
            options["gpu_mem_limit"] = str(int(cfg["gpu_mem_limit_mb"]) * 1024 * 1024)
        if "cudnn_conv_algo_search" in cfg:
            options["cudnn_conv_algo_search"] = str(cfg["cudnn_conv_algo_search"])
        if "do_copy_in_default_stream" in cfg:
            options["do_copy_in_default_stream"] = self._bool_provider_value(
                cfg["do_copy_in_default_stream"])
        if "use_tf32" in cfg:
            options["use_tf32"] = self._bool_provider_value(cfg["use_tf32"])
        return options

    def _tensorrt_provider_options(self) -> Dict[str, str]:
        """构造 TensorrtExecutionProvider 的选项字典，包含 engine 缓存目录。"""
        cfg = self.runtime.inference.tensorrt
        options: Dict[str, str] = {"device_id": str(self.runtime.inference.device_id)}
        options["trt_fp16_enable"] = self._bool_provider_value(cfg.get("fp16", False))
        options["trt_engine_cache_enable"] = self._bool_provider_value(
            cfg.get("engine_cache_enable", True))
        options["trt_timing_cache_enable"] = self._bool_provider_value(
            cfg.get("timing_cache_enable", True))
        options["trt_context_memory_sharing_enable"] = self._bool_provider_value(
            cfg.get("context_memory_sharing_enable", True))

        cache_path = str(cfg.get("engine_cache_path", "~/.cache/noetix_policy/tensorrt")).strip()
        if cache_path and bool(cfg.get("engine_cache_enable", True)):
            expanded = Path(os.path.expandvars(cache_path)).expanduser()
            expanded.mkdir(parents=True, exist_ok=True)
            options["trt_engine_cache_path"] = str(expanded)
        if "max_workspace_size" in cfg:
            options["trt_max_workspace_size"] = str(int(cfg["max_workspace_size"]))
        if "min_subgraph_size" in cfg:
            options["trt_min_subgraph_size"] = str(int(cfg["min_subgraph_size"]))
        return options

    def _select_providers(self, available: Sequence[str]):
        """根据配置和可用 provider,决定使用哪些执行提供者及其优先级。

        优先级：
          - provider == cpu           -> 只用 CPU
          - provider == tensorrt      -> 只用 TensorRT
          - provider == cuda          -> 只用 CUDA
          - provider == auto          -> TensorRT > CUDA > CPU（视配置）
        """
        cfg = self.runtime.inference
        available_set = set(available)

        # 强制 CPU
        if cfg.device == "cpu" or cfg.provider == "cpu":
            if "CPUExecutionProvider" not in available_set:
                raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
            return ["CPUExecutionProvider"]

        requested = []
        trt_enabled = bool(cfg.tensorrt.get("enabled", True))
        if cfg.provider == "tensorrt" and not trt_enabled:
            raise RuntimeError("TensorRT provider requested but inference.tensorrt.enabled=false")

        # TensorRT 优先
        if cfg.provider in ("auto", "tensorrt") and trt_enabled:
            if "TensorrtExecutionProvider" in available_set:
                requested.append(
                    ("TensorrtExecutionProvider", self._tensorrt_provider_options()))
            elif cfg.provider == "tensorrt":
                raise RuntimeError(
                    "TensorrtExecutionProvider requested but unavailable; "
                    f"available={sorted(available_set)}")

        # TensorRT 之后接 CUDA，让不支持 TensorRT 的子图继续留在 GPU 上
        if cfg.provider in ("auto", "tensorrt", "cuda"):
            if "CUDAExecutionProvider" in available_set:
                requested.append(("CUDAExecutionProvider", self._cuda_provider_options()))
            elif cfg.provider == "cuda":
                raise RuntimeError(
                    "CUDAExecutionProvider requested but unavailable; "
                    f"available={sorted(available_set)}")

        has_gpu = any(
            (item[0] if isinstance(item, tuple) else item)
            in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
            for item in requested)

        # 如果没有 GPU provider 可用
        if not has_gpu:
            if cfg.device == "gpu" and not cfg.allow_cpu_fallback:
                raise RuntimeError(
                    "GPU inference requested but no CUDA/TensorRT execution provider is available. "
                    f"available={sorted(available_set)}")
            if "CPUExecutionProvider" in available_set:
                return ["CPUExecutionProvider"]
            raise RuntimeError(f"No usable ONNX Runtime provider: {sorted(available_set)}")

        # 允许 CPU 回退时，追加 CPU 作为最后兜底
        if cfg.allow_cpu_fallback and "CPUExecutionProvider" in available_set:
            requested.append("CPUExecutionProvider")
        return requested

    # ---------- 模型元数据校验 ----------

    @staticmethod
    def _metadata_vector(metadata: Mapping[str, str], key: str) -> np.ndarray | None:
        """从 ONNX 元数据里解析逗号分隔的浮点向量。"""
        raw = str(metadata.get(key, "")).strip()
        if not raw:
            return None
        return np.asarray([float(x.strip()) for x in raw.split(",")], dtype=np.float32)

    def _validate_model_metadata(self) -> None:
        """校验 ONNX 里的元数据是否与 policy 配置一致。"""
        if self._session is None:
            return
        try:
            metadata = dict(self._session.get_modelmeta().custom_metadata_map or {})
        except Exception:
            return

        # 关节名必须完全一致
        raw_joint_names = str(metadata.get("joint_names", "")).strip()
        if raw_joint_names:
            model_joints = [x.strip() for x in raw_joint_names.split(",") if x.strip()]
            if model_joints != self.spec.model_joints:
                raise ValueError(
                    "ONNX joint_names metadata does not match policy joints.model_names; "
                    f"model={model_joints}, config={self.spec.model_joints}")

        # 以下元数据如果存在，必须和配置一致
        checks = (
            ("joint_stiffness", self.runtime.kp, "runtime.kp"),
            ("joint_damping", self.runtime.kd, "runtime.kd"),
            ("default_joint_pos", self.runtime.default_position, "runtime.default_position"),
            ("action_scale", self.runtime.action_scale, "runtime.action_scale"),
        )
        for metadata_key, configured, config_name in checks:
            expected = self._metadata_vector(metadata, metadata_key)
            if expected is None:
                continue
            if expected.size != configured.size or not np.allclose(
                expected, configured, rtol=1.0e-5, atol=1.0e-4):
                raise ValueError(
                    f"ONNX {metadata_key} metadata does not match {config_name}; "
                    f"model={expected.tolist()}, config={configured.tolist()}")

    def _validate_model_shapes(self) -> None:
        """校验 ONNX 输入/输出最后一维是否与配置一致。"""
        assert self._session is not None
        input_meta = next(x for x in self._session.get_inputs() if x.name == self.spec.model_input)
        output_meta = next(x for x in self._session.get_outputs() if x.name == self.spec.model_output)
        if input_meta.shape:
            last = input_meta.shape[-1]
            if isinstance(last, int) and last > 0 and last != self.spec.model_input_size:
                raise ValueError(
                    f"model input size {last} != configured {self.spec.model_input_size}")
        if output_meta.shape:
            last = output_meta.shape[-1]
            if isinstance(last, int) and last > 0 and last != self.spec.action_dimension:
                raise ValueError(
                    f"model output size {last} != configured {self.spec.action_dimension}")

    # ---------- 模型加载 ----------

    def load_model(self) -> None:
        """创建 ONNX Runtime 会话、绑定 io_binding、做元数据/shape 校验和预热。"""
        if self._session is not None:
            return
        if not self.spec.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.spec.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            package_hint = (
                "a CUDA/JetPack-compatible ONNX Runtime GPU build"
                if self.runtime.inference.device != "cpu"
                else "ONNX Runtime"
            )
            raise RuntimeError(f"Python package for {package_hint} is required") from exc

        self._ort = ort
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # 控制循环本身线程很少，减少 ONNX Runtime 的 CPU 线程占用
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        # 强制 GPU-only 时禁止 CPU 回退
        if (
            self.runtime.inference.device != "cpu"
            and not self.runtime.inference.allow_cpu_fallback
        ):
            try:
                session_options.add_session_config_entry(
                    "session.disable_cpu_ep_fallback", "1")
            except AttributeError as exc:
                raise RuntimeError(
                    "This ONNX Runtime build is too old to enforce GPU-only execution. "
                    "Use ONNX Runtime 1.16 or newer, or explicitly enable CPU fallback.") from exc

        available = ort.get_available_providers()
        providers = self._select_providers(available)
        try:
            self._session = ort.InferenceSession(
                str(self.spec.model_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception:
            # auto 模式下，TensorRT provider 存在但初始化失败时（版本不匹配等），
            # 退回只用 CUDA 再试一次。
            provider_names = [p[0] if isinstance(p, tuple) else p for p in providers]
            can_retry_cuda = (
                self.runtime.inference.provider == "auto"
                and "TensorrtExecutionProvider" in provider_names
                and "CUDAExecutionProvider" in set(available)
            )
            if not can_retry_cuda:
                raise
            cuda_only = [("CUDAExecutionProvider", self._cuda_provider_options())]
            if (
                self.runtime.inference.allow_cpu_fallback
                and "CPUExecutionProvider" in set(available)
            ):
                cuda_only.append("CPUExecutionProvider")
            self._session = ort.InferenceSession(
                str(self.spec.model_path),
                sess_options=session_options,
                providers=cuda_only,
            )

        self._active_providers = list(self._session.get_providers())

        # 检查输入/输出名字是否存在
        input_names = {x.name for x in self._session.get_inputs()}
        output_names = {x.name for x in self._session.get_outputs()}
        if self.spec.model_input not in input_names:
            raise ValueError(
                f"model input '{self.spec.model_input}' not found; available={sorted(input_names)}")
        if self.spec.model_output not in output_names:
            raise ValueError(
                f"model output '{self.spec.model_output}' not found; available={sorted(output_names)}")

        self._validate_model_shapes()
        self._validate_model_metadata()

        # 如果明确要求 GPU，但实际会话没跑在 GPU 上，就直接报错
        if self.runtime.inference.device == "gpu" and not self.using_gpu:
            raise RuntimeError(
                "GPU inference was requested but the created ONNX Runtime session is not using "
                f"CUDA/TensorRT; active={self._active_providers}")

        # 可选：启用 io_binding，减少 CPU<->GPU 的拷贝
        if self.runtime.inference.io_binding:
            self._io_binding = self._session.io_binding()
            self._io_binding.bind_cpu_input(self.spec.model_input, self._model_input)
            self._io_binding.bind_output(
                name=self.spec.model_output,
                device_type="cpu",
                device_id=0,
                element_type=np.float32,
                shape=self._model_output.shape,
                buffer_ptr=self._model_output.ctypes.data,
            )

        # 预热：触发 TensorRT engine 构建 / CUDA kernel 加载
        self._model_input.fill(0.0)
        for _ in range(self.runtime.inference.warmup_runs):
            self._run_session()
        self._model_output.fill(0.0)
        self.reset_inference_stats()

    @property
    def using_gpu(self) -> bool:
        """当前会话是否在使用 GPU provider。"""
        return any(
            p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
            for p in self._active_providers)

    @property
    def provider_summary(self) -> str:
        """返回 provider 链的简短描述，比如 TensorRT>CUDA。"""
        if not self._active_providers:
            return "unloaded"
        return ">".join(self._active_providers)

    def inference_stats(self) -> Dict[str, float | int | str | bool]:
        """返回推理统计信息（用于日志或 ROS 话题）。"""
        count = self._infer_count
        avg_ms = (self._infer_total_ns / count / 1.0e6) if count else 0.0
        return {
            "provider": self.provider_summary,
            "gpu": self.using_gpu,
            "count": count,
            "last_ms": self._infer_last_ns / 1.0e6,
            "avg_ms": avg_ms,
            "max_ms": self._infer_max_ns / 1.0e6,
        }

    def reset_inference_stats(self) -> None:
        """清空推理统计。"""
        self._infer_count = 0
        self._infer_total_ns = 0
        self._infer_max_ns = 0
        self._infer_last_ns = 0

    def _run_session(self) -> None:
        """执行一次 ONNX 推理，结果写入 self._model_output。"""
        assert self._session is not None
        if self._io_binding is not None:
            self._session.run_with_iobinding(self._io_binding)
            # io_binding 输出可能是 GPU 侧的，需要同步到 CPU buffer
            self._io_binding.synchronize_outputs()
        else:
            result = self._session.run(
                [self.spec.model_output], {self.spec.model_input: self._model_input})[0]
            np.copyto(self._model_output, np.asarray(result, dtype=np.float32).reshape(1, -1))

    # ---------- 状态管理 ----------

    def reset(self) -> None:
        """重置内部状态：清空历史、上一帧动作、缓存的索引与统计。"""
        self._history.fill(0.0)
        self._history_valid = False
        self.last_action.fill(0.0)
        self._joint_indices = None
        self._control_names = None
        self._control_namespace = "unresolved"
        self.reset_inference_stats()

    @property
    def control_namespace(self) -> str:
        """当前解析出来的控制命名空间：urdf_joint / actuator / unresolved。"""
        return self._control_namespace

    def resolve_control_names(self, low_state: Any) -> List[str]:
        """判断当前 LowState.motor_state 用的是 URDF 关节名还是执行器名。

        优先 URDF 关节名；如果 URDF 关节名缺失但执行器名齐全，则用执行器名。
        两者都不匹配则报错。
        """
        observed = [str(m.name) for m in low_state.motor_state]
        observed_set = {name for name in observed if name}

        # 如果之前的决定仍然有效，直接复用
        if self._control_names is not None:
            if all(name in observed_set for name in self._control_names):
                return self._control_names
            self._control_names = None
            self._joint_indices = None
            self._control_namespace = "unresolved"

        joint_names = list(self.spec.joints)
        actuator_names = list(self.spec.actuators)
        missing_joints = [name for name in joint_names if name not in observed_set]
        missing_actuators = [name for name in actuator_names if name not in observed_set]

        if not missing_joints:
            self._control_names = joint_names
            self._control_namespace = "urdf_joint"
            return self._control_names
        if not missing_actuators:
            self._control_names = actuator_names
            self._control_namespace = "actuator"
            return self._control_names

        raise ValueError(
            "LowState names match neither supported namespace; "
            f"robot={self.spec.robot}; missing_urdf_joints={missing_joints}; "
            f"missing_actuators={missing_actuators}; observed={observed}")

    def _resolve_joint_indices(self, low_state: Any) -> List[int]:
        """返回 LowState.motor_state 中对应物理控制关节的下标列表，带缓存。"""
        control_names = self.resolve_control_names(low_state)
        # 如果之前的缓存仍然有效，直接复用
        if self._joint_indices is not None:
            if len(self._joint_indices) == len(control_names):
                valid = True
                for expected, index in zip(control_names, self._joint_indices):
                    if (index >= len(low_state.motor_state) or
                            low_state.motor_state[index].name != expected):
                        valid = False
                        break
                if valid:
                    return self._joint_indices

        state_map = {m.name: i for i, m in enumerate(low_state.motor_state) if m.name}
        self._joint_indices = [state_map[name] for name in control_names]
        return self._joint_indices

    # ---------- 观测构造 ----------

    def projected_gravity(self, low_state: Any) -> np.ndarray:
        """从 IMU 四元数计算 body 坐标系下的投影重力。"""
        gravity = rotate_world_gravity_to_body(low_state.imu.quaternion)
        np.copyto(self._gravity, gravity)
        return self._gravity

    def build_frame(
        self,
        low_state: Any,
        command: np.ndarray,
        projected_gravity: np.ndarray | None = None,
    ) -> np.ndarray:
        """构造单帧观测向量（未堆叠历史）。

        步骤：
          1. 用默认姿态初始化模型空间的关节位置和速度（含虚拟关节）。
          2. 用 LowState 里的物理关节值覆盖到模型空间对应下标。
          3. 计算投影重力。
          4. 按 terms 定义逐项填入观测帧。
          5. 对整帧做 clip。
        """
        indices = self._resolve_joint_indices(low_state)

        # 先用默认姿态填满，虚拟关节默认保持默认姿态、速度为零
        np.copyto(self._q, self.runtime.default_position)
        self._dq.fill(0.0)
        if self.spec.virtual_joint_mode == "commanded_position":
            # 虚拟关节跟随上一帧命令位置
            self._q[self._virtual_model_mask] = (
                self.runtime.default_position[self._virtual_model_mask]
                + self.runtime.action_scale[self._virtual_model_mask]
                * self.last_action[self._virtual_model_mask]
            )

        # 用物理状态覆盖模型空间中对应下标
        for control_index, state_index in enumerate(indices):
            model_index = int(self._control_model_indices[control_index])
            state = low_state.motor_state[state_index]
            self._q[model_index] = float(state.q)
            self._dq[model_index] = float(state.dq)

        # 计算投影重力（允许外部传入以免重复计算）
        gravity = projected_gravity
        if gravity is None:
            gravity = self.projected_gravity(low_state)
        else:
            np.copyto(self._gravity, np.asarray(gravity, dtype=np.float32).reshape(3))
            gravity = self._gravity

        # q_rel = q - default_position
        np.subtract(self._q, self.runtime.default_position, out=self._q_rel)

        # 按 terms 定义逐项填观测
        cursor = 0
        for term, selected_indices in zip(self.spec.terms, self._term_indices):
            term_type = str(term.get("type", ""))
            dim = int(term.get("dim", 0))
            end = cursor + dim
            target = self._frame[cursor:end]
            if term_type == "command":
                value = command
            elif term_type == "base_angular_velocity":
                value = low_state.imu.gyroscope
            elif term_type == "projected_gravity":
                value = gravity
            elif term_type == "joint_position_relative":
                value = self._q_rel
            elif term_type == "joint_velocity":
                value = self._dq
            elif term_type == "last_action":
                value = self.last_action
            else:
                raise ValueError(f"unsupported observation term: {term_type}")

            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if selected_indices is not None:
                # 只取指定的下标
                if any(index >= arr.size for index in selected_indices):
                    raise ValueError(
                        f"term {term_type}: indices={list(selected_indices)} "
                        f"out of range for size={arr.size}")
                for target_index, source_index in enumerate(selected_indices):
                    target[target_index] = arr[source_index]
            else:
                if arr.size != dim:
                    raise ValueError(f"term {term_type}: expected dim={dim}, got {arr.size}")
                np.copyto(target, arr)
            cursor = end

        # 观测整体裁剪
        np.clip(
            self._frame,
            -self.spec.observation_clip,
            self.spec.observation_clip,
            out=self._frame,
        )
        return self._frame

    def stacked_observation(self, frame: np.ndarray) -> np.ndarray:
        """把单帧观测压入历史队列，返回展平后的 (stack_size * frame_size) 观测。"""
        frame_arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame_arr.size != self.spec.frame_size:
            raise ValueError(
                f"frame size {frame_arr.size} != configured {self.spec.frame_size}")

        if not self._history_valid:
            # 首次：根据 history_init 选择 repeat 或 zero
            self._history.fill(0.0)
            if self.runtime.history_init == "repeat":
                self._history[:] = frame_arr
            else:
                self._history[-1] = frame_arr
            self._history_valid = True
        else:
            # 正常更新：整体左移一格，最后一行填新帧
            self._history[:-1] = self._history[1:]
            self._history[-1] = frame_arr

        return self._history.reshape(-1)

    # ---------- 推理与动作 ----------

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """执行一次推理：写入观测，运行会话，裁剪动作并更新 last_action。"""
        self.load_model()
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size != self.spec.model_input_size:
            raise ValueError(
                f"observation size {obs.size} != configured {self.spec.model_input_size}")
        np.copyto(self._model_input[0], obs)

        start_ns = time.perf_counter_ns()
        self._run_session()
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._infer_count += 1
        self._infer_total_ns += elapsed_ns
        self._infer_last_ns = elapsed_ns
        self._infer_max_ns = max(self._infer_max_ns, elapsed_ns)

        result = self._model_output.reshape(-1)
        if result.size != self.spec.action_dimension:
            raise ValueError(
                f"model action size {result.size} != {self.spec.action_dimension}")
        # 动作裁剪
        np.clip(result, -self.spec.action_clip, self.spec.action_clip, out=self._action)
        np.copyto(self.last_action, self._action)
        if self.spec.virtual_joint_mode == "fixed_default":
            # 虚拟关节在 LowCmd 里不存在，所以实际生效的动作在这些槽位上是 0
            self.last_action[self._virtual_model_mask] = 0.0
        return self._action

    def position_target(self, action: np.ndarray) -> np.ndarray:
        """动作 -> 目标关节位置：target = default_position + action_scale * action。"""
        # 保留策略契约：模型动作先 clip，再乘 action_scale。
        # max_position_delta 默认关闭，因为额外的部署侧裁剪会改变学到的策略。
        np.multiply(self.runtime.action_scale, action, out=self._delta)
        if self.runtime.max_position_delta is not None:
            np.clip(
                self._delta,
                -self.runtime.max_position_delta,
                self.runtime.max_position_delta,
                out=self._delta,
            )
        np.add(self.runtime.default_position, self._delta, out=self._position_target)
        return self._position_target

    # ---------- 模型空间 -> 物理空间 ----------

    def control_values(self, model_values: np.ndarray) -> np.ndarray:
        """从模型空间向量中选出物理 LowCmd 需要的那部分。"""
        values = np.asarray(model_values, dtype=np.float32).reshape(-1)
        if values.size != self.spec.action_dimension:
            raise ValueError(
                f"model-space vector size {values.size} != {self.spec.action_dimension}")
        return values[self._control_model_indices]

    def control_position_target(self, model_target: np.ndarray) -> np.ndarray:
        """模型空间目标位置 -> 物理 LowCmd 目标位置。"""
        return self.control_values(model_target)

    def control_kp(self) -> np.ndarray:
        """模型空间 kp -> 物理 LowCmd kp。"""
        return self.control_values(self.runtime.kp)

    def control_kd(self) -> np.ndarray:
        """模型空间 kd -> 物理 LowCmd kd。"""
        return self.control_values(self.runtime.kd)