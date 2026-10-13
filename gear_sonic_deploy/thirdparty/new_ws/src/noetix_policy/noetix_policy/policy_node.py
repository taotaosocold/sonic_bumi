from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from noetix_msgs.msg import LowCmd, LowState, MotorCmd, RobotStatus
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import String
import yaml

from .runtime import PolicyRuntime
from .tracking_runtime import load_policy_runtime


class PolicyNode(Node):
    """
    策略管理节点。

    职责：
    1. 从 YAML 配置文件中读取策略列表、话题名、频率、手柄按键映射等；
    2. 订阅 LowState(机器人状态）和 Joy(手柄），按需激活/关闭某个策略；
    3. 以固定频率运行激活中的策略，输出 LowCmd 位置目标指令；
    4. 本节点不切换底层模式；只有 DISABLED 会阻止策略输出。LIN/回零是普通上层工具，
       策略激活后可以立即接管 LowCmd。
    """

    def __init__(self) -> None:
        # 初始化 ROS 节点，节点名为 noetix_policy_node
        super().__init__("noetix_policy_node")

        # ROS 参数可覆盖部署相关的 GPU 选择，不需要为多 GPU 机器复制 YAML。
        self.declare_parameter("manager_config", "")
        self.declare_parameter("inference_provider", "")
        self.declare_parameter("gpu_device_id", -1)
        self.declare_parameter("policy_hz_override", -1.0)
        self.declare_parameter("trajectory_use_ros_time", False)
        # Select one robot profile so policies for different Bumi variants can
        # reuse the same gamepad buttons without colliding.
        self.declare_parameter("robot_name", "bumi4340")

        # 解析配置文件路径，展开 ~ 之类的用户目录
        config_file = Path(self.get_parameter("manager_config").value).expanduser()
        if not config_file.is_file():
            raise RuntimeError(f"manager_config not found: {config_file}")

        # 读取 YAML 配置，取其中的 policy_manager 段落
        with config_file.open("r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
        cfg = root.get("policy_manager", {})
        self.robot_name = str(self.get_parameter("robot_name").value).strip()
        self.trajectory_use_ros_time = bool(
            self.get_parameter("trajectory_use_ros_time").value)
        if not self.robot_name:
            raise ValueError("robot_name must not be empty")

        # ---------------- 从配置中读取话题名、频率、超时、手柄按键等参数 ----------------
        # 机器人低层状态话题（订阅）
        self.low_state_topic = str(cfg.get("low_state_topic", "/noetix/low_state"))
        # 机器人低层命令话题（发布）
        self.low_cmd_topic = str(cfg.get("low_cmd_topic", "/noetix/low_cmd"))
        # 手柄话题（订阅）
        self.joy_topic = str(cfg.get("joy_topic", "/joy"))
        # 策略状态广播话题（发布，用于外部监控当前激活的策略）
        self.policy_state_topic = str(cfg.get("policy_state_topic", "/noetix/policy_state"))
        # 策略运行频率（Hz）
        self.policy_hz = float(cfg.get("policy_hz", 100.0))
        policy_hz_override = float(self.get_parameter("policy_hz_override").value)
        if policy_hz_override > 0.0:
            self.policy_hz = policy_hz_override
        # 用于紧急停止的手柄按钮索引
        self.stop_button = int(cfg.get("stop_button", 1))
        # 推理后端全局配置。单个策略可以在 policies[].inference 中覆盖。
        self.inference_cfg = dict(cfg.get("inference", {}) or {})
        provider_override = str(self.get_parameter("inference_provider").value).strip().lower()
        gpu_id_override = int(self.get_parameter("gpu_device_id").value)
        if provider_override:
            if provider_override not in ("auto", "tensorrt", "cuda", "cpu"):
                raise ValueError(
                    "inference_provider must be one of: auto, tensorrt, cuda, cpu")
            self.inference_cfg["provider"] = provider_override
            self.inference_cfg["device"] = "cpu" if provider_override == "cpu" else "gpu"
        if gpu_id_override >= 0:
            self.inference_cfg["device_id"] = gpu_id_override
        # 启动时预加载模型。GPU/TensorRT 初始化和 warmup 不进入控制中的第一次按键回调。
        self.preload_models = bool(cfg.get("preload_models", True))
        # 周期性输出推理耗时，0 表示关闭。
        self.inference_stats_period_s = float(cfg.get("inference_stats_period_s", 5.0))
        self._last_inference_stats_log = time.monotonic()
        # Strategy signal diagnostics. This is intentionally lightweight and
        # throttled to keep policy diagnostics readable during bring-up.
        self.policy_debug_period_s = float(cfg.get("policy_debug_period_s", 1.0))
        self._last_policy_debug_log = time.monotonic()
        self._last_q_target: Optional[np.ndarray] = None
        self._logged_control_namespace: Dict[str, str] = {}

        if self.policy_hz <= 0.0:
            raise ValueError("policy_hz must be > 0")

        # ---------------- 加载所有策略 ----------------
        self.policies: Dict[str, PolicyRuntime] = {}      # 策略名 -> 运行时对象
        self.policy_buttons: Dict[int, str] = {}          # 手柄按钮索引 -> 策略名
        entries = cfg.get("policies", [])
        if not entries:
            raise ValueError("policy_manager.policies is empty")

        for entry in entries:
            # One manager can contain policies for several robot variants.
            # Load only the selected robot so button assignments stay simple.
            entry_robot = str((entry or {}).get("robot", "")).strip()
            if entry_robot and entry_robot != self.robot_name:
                continue

            # 全局 inference 配置为默认值，策略项中的 inference 只覆盖需要变化的字段。
            entry_cfg = dict(entry)
            entry_inference = dict(self.inference_cfg)
            per_policy_inference = dict(entry_cfg.get("inference", {}) or {})
            # cuda/tensorrt 子配置也做一层合并，避免覆盖全局未修改的选项。
            for section in ("cuda", "tensorrt"):
                merged = dict(entry_inference.get(section, {}) or {})
                merged.update(dict(per_policy_inference.get(section, {}) or {}))
                if merged:
                    per_policy_inference[section] = merged
            entry_inference.update(per_policy_inference)
            entry_cfg["inference"] = entry_inference

            # 策略文件路径支持相对路径（相对于配置文件目录）
            policy_file = Path(str(entry_cfg["file"]))
            if not policy_file.is_absolute():
                policy_file = config_file.parent / policy_file
            # 加载策略描述与运行时。普通 locomotion 和 Mimic/Tracking 策略
            # 共用同一个 manager，但由工厂创建对应 Runtime。
            policy_runtime = load_policy_runtime(policy_file.resolve(), entry_cfg)
            spec = policy_runtime.spec
            runtime_cfg = policy_runtime.runtime
            # 校验：策略名不能重复
            if spec.name in self.policies:
                raise ValueError(f"duplicate policy name: {spec.name}")
            # 校验：手柄按钮不能与 stop 键冲突，也不能与其他策略冲突
            if runtime_cfg.button == self.stop_button or runtime_cfg.button in self.policy_buttons:
                raise ValueError(f"duplicate/reserved gamepad button: {runtime_cfg.button}")
            if spec.robot != self.robot_name:
                raise ValueError(
                    f"policy '{spec.name}' robot={spec.robot} does not match selected robot_name={self.robot_name}")
            self.policies[spec.name] = policy_runtime
            self.policy_buttons[runtime_cfg.button] = spec.name

        if not self.policies:
            raise ValueError(f"no policies configured for robot_name={self.robot_name}")

        # GPU/TensorRT session 创建、TensorRT engine 构建和 warmup 都在节点启动阶段完成。
        # 这样第一次激活策略时不会因为 CUDA context / engine 初始化产生长抖动。
        if self.preload_models:
            for name, runtime in self.policies.items():
                try:
                    runtime.load_model()
                except Exception as exc:
                    raise RuntimeError(f"failed to preload policy '{name}': {exc}") from exc

        # ---------------- 运行状态变量 ----------------
        self.latest_state: Optional[LowState] = None      # 最新一次 LowState
        self.latest_joy: Optional[Joy] = None             # 最新一次 Joy
        self.last_buttons = []                            # 上一次手柄按钮状态，用于检测上升沿
        self.active_name: Optional[str] = None            # 当前激活的策略名
        self.command = np.zeros(3, dtype=np.float32)      # 当前三轴速度指令（x, y, yaw）
        self.sequence = 0                                 # 发布 LowCmd 的递增序列号
        # Some policies require a nominal pose before inference. These fields
        # implement the configured runtime.entry_pose interpolation.
        self._entry_pose_active = False
        self._entry_pose_start_time = 0.0
        self._entry_pose_start_q: Optional[np.ndarray] = None

        # ---------------- 订阅 & 发布 ----------------
        # LowCmd 发布器（队列 20）
        self.cmd_pub = self.create_publisher(LowCmd, self.low_cmd_topic, 20)
        # 策略状态广播（字符串）
        self.state_pub = self.create_publisher(String, self.policy_state_topic, 10)
        # 订阅 LowState（使用传感器 QoS）
        self.state_sub = self.create_subscription(LowState, self.low_state_topic, self._on_low_state, qos_profile_sensor_data)
        # 订阅 Joy（使用传感器 QoS）
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self._on_joy, qos_profile_sensor_data)
        # 控制请求只用于让上层策略在 disable / LIN 工具开始时退出。
        # Bridge 本身只有 ENABLE/DISABLE，不参与策略仲裁。
        self.control_request_sub = self.create_subscription(
            String, "/noetix/control_request", self._on_control_request, 20)
        # 定时器：按 policy_hz 频率运行 _step
        self.timer = self.create_timer(1.0 / self.policy_hz, self._step)

        # ---------------- 启动日志 ----------------
        self.get_logger().info(
            f"Policy manager ready: robot={self.robot_name}, hz={self.policy_hz:.1f}, "
            f"state={self.low_state_topic}, "
            f"cmd={self.low_cmd_topic}, joy={self.joy_topic}")
        self.get_logger().info(
            "Policy output is blocked only by DISABLED; LIN/zero-return is an ordinary upper-layer LowCmd publisher")
        self.get_logger().info(f"stop button={self.stop_button}")
        for button, name in sorted(self.policy_buttons.items()):
            runtime = self.policies[name]
            command_names = list(getattr(runtime.spec, "joints", []) or [])
            robot = getattr(runtime.spec, "robot", "")
            robot_text = f", robot={robot}" if robot else ""
            if getattr(runtime, "is_mimic_tracking", False):
                tracking = runtime.tracking
                self.get_logger().info(
                    f"policy button={button} -> {name}{robot_text}, type=mimic_tracking, "
                    f"model={runtime.spec.model_path}, provider={runtime.provider_summary}, "
                    f"gpu={runtime.using_gpu}, io_binding={runtime.runtime.inference.io_binding}, "
                    f"policy_joints={len(command_names)}, "
                    f"inputs={tracking.command_input}[{tracking.command_size}]+"
                    f"{tracking.policy_input}[{tracking.policy_size}], "
                    f"reference={tracking.reference_path}")
            else:
                self.get_logger().info(
                    f"policy button={button} -> {name}{robot_text}, type=locomotion, "
                    f"model={runtime.spec.model_path}, provider={runtime.provider_summary}, "
                    f"gpu={runtime.using_gpu}, io_binding={runtime.runtime.inference.io_binding}, "
                    f"policy_joints={len(command_names)}, "
                    f"joy_axes(x/y/yaw)={runtime.runtime.command_axes['x']}/"
                    f"{runtime.runtime.command_axes['y']}/{runtime.runtime.command_axes['yaw']}, "
                    f"command_signs={runtime.runtime.command_signs}, "
                    f"command_limits={runtime.runtime.command_limits}")

    # ------------------------------------------------------------------
    # 手柄工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _pressed(msg: Joy, index: int) -> bool:
        """判断手柄的某个按钮当前是否被按下。越界返回 False。"""
        return 0 <= index < len(msg.buttons) and msg.buttons[index] != 0

    def _rising(self, msg: Joy, index: int) -> bool:
        """检测按钮的上升沿（从上一次未按 -> 本次按下）。"""
        if not self._pressed(msg, index):
            return False
        # 上一次按钮数组中该索引不存在或为 0，说明是从未按到按下的跳变
        return index >= len(self.last_buttons) or self.last_buttons[index] == 0

    # ------------------------------------------------------------------
    # 回调：LowState
    # ------------------------------------------------------------------
    def _on_low_state(self, msg: LowState) -> None:
        """收到新 LowState 时更新缓存；如果当前有策略在跑，且机器人已经不满足条件，则自动停策略。"""
        self.latest_state = msg
        # Only DISABLED blocks upper-layer control.  LIN is not a mode and the
        # transient status.enabled bit is not used for policy arbitration.
        if self.active_name is not None and msg.status.mode == RobotStatus.MODE_DISABLED:
            self._deactivate("robot entered DISABLED", send_zero=False)

    # ------------------------------------------------------------------
    # 回调：Joy
    # ------------------------------------------------------------------
    def _on_joy(self, msg: Joy) -> None:
        """处理手柄事件：紧急停止、按键激活策略。"""
        self.latest_joy = msg

        # 1) 紧急停止按键优先处理
        if self._rising(msg, self.stop_button):
            self._deactivate("gamepad stop", send_zero=True)
            self.last_buttons = list(msg.buttons)
            return

        # 2) 遍历所有策略按键，取第一个上升沿的按键进行激活
        for button, name in self.policy_buttons.items():
            if not self._rising(msg, button):
                continue
            # The only mode that blocks a strategy is DISABLED.  Zero-return
            # is an upper-layer tool, so a policy may take over immediately.
            if not self._robot_accepts_policy():
                self.get_logger().warning(
                    f"Refuse policy '{name}': no LowState yet or robot is DISABLED.")
                break
            self._activate(name)
            break

        # 记录本次按钮状态，用于下一次上升沿检测
        self.last_buttons = list(msg.buttons)

    def _on_control_request(self, msg: String) -> None:
        request = str(msg.data).strip().lower()
        if self.active_name is None:
            return
        if request in ("disable", "stop", "estop", "safe_stop"):
            self._deactivate(f"control request: {request}", send_zero=False)
        elif request == "lin":
            # LIN is an ordinary upper-layer tool. Stop the currently running
            # policy so the zero-return publisher has a single writer. A new
            # strategy may be activated immediately afterwards, even while the
            # zero-return trajectory is still in progress.
            self._deactivate("zero-return tool requested", send_zero=False)

    # ------------------------------------------------------------------
    # 内部状态检查
    # ------------------------------------------------------------------
    def _robot_accepts_policy(self) -> bool:
        """Return True once LowState exists and the public mode is not DISABLED.

        Policy arbitration deliberately ignores LIN state, command freshness,
        and the transient status.enabled flag.  The bridge owns only the
        ENABLE/DISABLE gate; every other upper-layer command may take over.
        """
        return (
            self.latest_state is not None
            and self.latest_state.status.mode != RobotStatus.MODE_DISABLED
        )

    # ------------------------------------------------------------------
    # 策略激活 / 关闭
    # ------------------------------------------------------------------
    def _trajectory_time(self) -> float:
        """Time base used by deployment pose interpolation.

        Real hardware keeps the historical monotonic-clock behaviour. Gazebo
        can opt into ROS time so entry-pose timing follows /clock and pauses
        with the simulator.
        """
        if self.trajectory_use_ros_time:
            return self.get_clock().now().nanoseconds * 1.0e-9
        return time.monotonic()

    def _activate(self, name: str) -> None:
        """激活指定策略：加载模型、重置运行时状态、记录激活名、广播状态。"""
        runtime = self.policies[name]
        try:
            runtime.load_model()
        except Exception as exc:
            self.get_logger().error(f"Cannot activate policy '{name}': {exc}")
            return
        runtime.reset()
        if getattr(runtime, "is_mimic_tracking", False):
            try:
                runtime.prepare_activation()
            except Exception as exc:
                runtime.reset()
                self.get_logger().error(
                    f"Cannot activate policy '{name}': {exc}")
                return
        self.active_name = name
        now = time.monotonic()
        trajectory_now = self._trajectory_time()
        self._last_inference_stats_log = now
        self._last_policy_debug_log = now
        self._last_q_target = None
        self._entry_pose_active = False
        self._entry_pose_start_q = None

        entry_pose = runtime.runtime.entry_pose
        if entry_pose is not None and entry_pose.enabled:
            if self.latest_state is None:
                self.get_logger().error(
                    f"Cannot activate policy '{name}': no LowState for entry pose")
                self.active_name = None
                return
            state_by_name = {m.name: m for m in self.latest_state.motor_state if m.name}
            command_names = runtime.resolve_control_names(self.latest_state)
            missing = [joint for joint in command_names if joint not in state_by_name]
            if missing:
                self.get_logger().error(
                    f"Cannot activate policy '{name}': entry-pose joints missing: {missing}")
                self.active_name = None
                return
            self._entry_pose_start_q = np.asarray(
                [float(state_by_name[joint].q) for joint in command_names], dtype=np.float32)
            self._entry_pose_start_time = trajectory_now
            self._entry_pose_active = True
            self._publish_policy_state(
                f"active={name} phase=entry_pose duration={entry_pose.duration_s:.3f}s")
            self.get_logger().info(
                f"Policy -> {name}: entry pose for {entry_pose.duration_s:.3f}s, "
                f"then inference; provider={runtime.provider_summary}; gpu={runtime.using_gpu}")
            return

        self._publish_policy_state(
            f"active={name} provider={runtime.provider_summary} gpu={runtime.using_gpu}")
        self.get_logger().info(
            f"Policy -> {name}; provider={runtime.provider_summary}; gpu={runtime.using_gpu}")

    def _deactivate(self, reason: str, send_zero: bool) -> None:
        """关闭当前策略：
        - send_zero=True 且机器人仍可用时，先发一条全零命令（安全）再退出策略；
        - 重置策略运行时状态；
        - 广播状态变化。
        """
        if self.active_name is None:
            return
        old = self.active_name
        # 发送零命令：把 q 设为当前关节位置，kp/kd 为 0，相当于“卸力”
        if send_zero and self._robot_accepts_policy():
            self._publish_zero_command()
        self.policies[old].reset()
        self.active_name = None
        self._entry_pose_active = False
        self._entry_pose_start_q = None
        self._publish_policy_state(f"active=none reason={reason}")
        self.get_logger().info(f"Policy stopped: {old}; reason={reason}")

    def _publish_policy_state(self, text: str) -> None:
        """通过 policy_state_topic 发布字符串状态。"""
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

    # ------------------------------------------------------------------
    # 手柄轴/指令处理
    # ------------------------------------------------------------------
    def _axis(self, index: int) -> float:
        """读取手柄指定轴的数值；越界或未收到 Joy 时返回 0.0。"""
        if self.latest_joy is None or index < 0 or index >= len(self.latest_joy.axes):
            return 0.0
        return float(self.latest_joy.axes[index])

    def _command_for(self, runtime: PolicyRuntime) -> np.ndarray:
        """Build the policy command in the same physical units used at training.

        The policy contract's deadzone is defined on the final velocity command,
        not on the normalized joystick axis.  Therefore the joystick is first
        mapped to [vx, vy, yaw_rate], then the policy deadzone and exclusivity
        rules are applied.  This also keeps command semantics independent from
        the particular gamepad hardware.
        """
        term_cfg: Dict[str, Any] = {}
        for term in runtime.spec.terms:
            if str(term.get("type", "")) == "command":
                term_cfg = term
                break

        raw = np.asarray([
            self._axis(runtime.runtime.command_axes["x"])
            * runtime.runtime.command_signs["x"],
            self._axis(runtime.runtime.command_axes["y"])
            * runtime.runtime.command_signs["y"],
            self._axis(runtime.runtime.command_axes["yaw"])
            * runtime.runtime.command_signs["yaw"],
        ], dtype=np.float32)

        c = np.asarray([
            raw[0] * runtime.runtime.command_limits["x"],
            raw[1] * runtime.runtime.command_limits["y"],
            raw[2] * runtime.runtime.command_limits["yaw"],
        ], dtype=np.float32)

        deadzone = float(term_cfg.get("deadzone", 0.0))
        if deadzone > 0.0:
            for i in range(3):
                if abs(float(c[i])) < deadzone:
                    c[i] = 0.0

        min_x = term_cfg.get("min_x")
        if min_x is not None:
            c[0] = max(float(min_x), float(c[0]))

        if bool(term_cfg.get("lateral_exclusive", False)):
            # Lateral locomotion is a separate command regime for this policy.
            # If lateral motion is requested, do not simultaneously request
            # forward/yaw motion.  When forward and yaw are both commanded,
            # suppress lateral motion.
            if abs(float(c[1])) > deadzone:
                c[0] = 0.0
                c[2] = 0.0
            elif abs(float(c[0])) > deadzone and abs(float(c[2])) > deadzone:
                c[1] = 0.0

        return c

    # ------------------------------------------------------------------
    # 主控制循环
    # ------------------------------------------------------------------
    def _step(self) -> None:
        """定时器回调：按频率运行激活的策略。"""
        # 未激活任何策略则直接返回
        if self.active_name is None:
            return
        # Only DISABLED blocks strategy output. There is intentionally no
        # freshness/LIN/owner gate in the policy layer.
        if not self._robot_accepts_policy():
            self._deactivate("robot is DISABLED", send_zero=False)
            return

        assert self.latest_state is not None
        runtime = self.policies[self.active_name]

        # IMU 无效时出于安全停止策略，并发送零命令。即使当前还在
        # legacy entry pose，也不在无有效姿态数据时继续进入行走策略。
        if not self.latest_state.imu.valid:
            self._deactivate("IMU invalid", send_zero=True)
            return

        try:
            # Policies with runtime.entry_pose interpolate from the measured
            # current pose before history reset and ONNX inference.
            if self._entry_pose_active:
                entry_pose = runtime.runtime.entry_pose
                if entry_pose is None or self._entry_pose_start_q is None:
                    raise RuntimeError("entry-pose state is incomplete")
                elapsed = max(0.0, self._trajectory_time() - self._entry_pose_start_time)
                alpha = min(1.0, elapsed / entry_pose.duration_s)
                entry_target = (
                    runtime.entry_pose_target
                    if getattr(runtime, "is_mimic_tracking", False)
                    else entry_pose.position
                )
                q_target = (
                    self._entry_pose_start_q * (1.0 - alpha)
                    + entry_target * alpha
                ).astype(np.float32, copy=False)
                self._publish_position_command(
                    runtime, q_target, entry_pose.kp, entry_pose.kd)
                if alpha >= 1.0:
                    self._entry_pose_active = False
                    self._entry_pose_start_q = None
                    runtime.reset()
                    self.command.fill(0.0)
                    self._last_q_target = None
                    self._last_policy_debug_log = time.monotonic()
                    self._publish_policy_state(
                        f"active={runtime.spec.name} phase=inference "
                        f"provider={runtime.provider_summary}")
                    self.get_logger().info(
                        f"Policy {runtime.spec.name}: entry pose complete; "
                        "history reset, inference starts on next tick")
                return

            # 1) 计算重力投影向量，用于跌倒保护判断
            gravity = runtime.projected_gravity(self.latest_state)
            if (
                runtime.spec.fall_protection_enabled
                and float(gravity[2]) > runtime.spec.projected_gravity_z_max
            ):
                self._deactivate(
                    f"fall protection: projected_gravity_z={gravity[2]:.3f}", send_zero=True)
                return

            # 2) 构造策略输入并推理。
            #    - locomotion: joy command + 单输入 obs
            #    - mimic_tracking: reference command[240] + policy[378]
            if getattr(runtime, "is_mimic_tracking", False):
                self.command.fill(0.0)
                tracking_command, policy_observation = runtime.build_tracking_inputs(
                    self.latest_state, projected_gravity=gravity)
                action = runtime.infer_tracking(tracking_command, policy_observation)
            else:
                self.command = self._command_for(runtime)
                frame = runtime.build_frame(
                    self.latest_state, self.command, projected_gravity=gravity)
                observation = runtime.stacked_observation(frame)
                action = runtime.infer(observation)

            resolved_ns = runtime.control_namespace
            if self._logged_control_namespace.get(runtime.spec.name) != resolved_ns:
                names = runtime.resolve_control_names(self.latest_state)
                self.get_logger().info(
                    f"policy={runtime.spec.name} LowState/LowCmd namespace={resolved_ns} "
                    f"names={names}")
                self._logged_control_namespace[runtime.spec.name] = resolved_ns
            if self.inference_stats_period_s > 0.0:
                now = time.monotonic()
                if now - self._last_inference_stats_log >= self.inference_stats_period_s:
                    stats = runtime.inference_stats()
                    self.get_logger().info(
                        f"policy={runtime.spec.name} provider={stats['provider']} "
                        f"infer_ms(last/avg/max)={stats['last_ms']:.3f}/"
                        f"{stats['avg_ms']:.3f}/{stats['max_ms']:.3f} "
                        f"count={stats['count']}")
                    self._last_inference_stats_log = now
            # 6) 动作 -> 关节位置目标
            q_target = runtime.position_target(action)

            # Runtime diagnostics: prove each stage changes when the joystick
            # moves. This is much more useful than inspecting kp/kd alone.
            now = time.monotonic()
            if (self.policy_debug_period_s > 0.0 and
                    now - self._last_policy_debug_log >= self.policy_debug_period_s):
                joy_axes = list(self.latest_joy.axes) if self.latest_joy is not None else []
                q_change = 0.0
                if self._last_q_target is not None:
                    q_change = float(np.max(np.abs(q_target - self._last_q_target)))
                if getattr(runtime, "is_mimic_tracking", False):
                    self.get_logger().info(
                        f"policy_signal type=mimic_tracking "
                        f"ref={runtime.reference_status} "
                        f"gyro={np.asarray(self.latest_state.imu.gyroscope, dtype=np.float32).round(3).tolist()} "
                        f"gravity={gravity.round(3).tolist()} "
                        f"q[min,max]=[{float(np.min(runtime._q)):.4f},"
                        f"{float(np.max(runtime._q)):.4f}] "
                        f"dq[min,max]=[{float(np.min(runtime._dq)):.4f},"
                        f"{float(np.max(runtime._dq)):.4f}] "
                        f"action[min,max,norm]=[{float(np.min(action)):.4f},"
                        f"{float(np.max(action)):.4f},{float(np.linalg.norm(action)):.4f}] "
                        f"q_target[min,max]=[{float(np.min(q_target)):.4f},"
                        f"{float(np.max(q_target)):.4f}] q_change={q_change:.6f}")
                else:
                    self.get_logger().info(
                        f"policy_signal joy={np.asarray(joy_axes, dtype=np.float32).round(3).tolist()} "
                        f"cmd={self.command.round(3).tolist()} "
                        f"gyro={np.asarray(self.latest_state.imu.gyroscope, dtype=np.float32).round(3).tolist()} "
                        f"gravity={gravity.round(3).tolist()} "
                        f"q_rel[min,max]=[{float(np.min(runtime._q_rel)):.4f},"
                        f"{float(np.max(runtime._q_rel)):.4f}] "
                        f"dq[min,max]=[{float(np.min(runtime._dq)):.4f},"
                        f"{float(np.max(runtime._dq)):.4f}] "
                        f"action[min,max,norm]=[{float(np.min(action)):.4f},"
                        f"{float(np.max(action)):.4f},{float(np.linalg.norm(action)):.4f}] "
                        f"q_target[min,max]=[{float(np.min(q_target)):.4f},"
                        f"{float(np.max(q_target)):.4f}] q_change={q_change:.6f}")
                self._last_policy_debug_log = now

            if self._last_q_target is None:
                self._last_q_target = np.array(q_target, dtype=np.float32, copy=True)
            else:
                np.copyto(self._last_q_target, q_target)

            # 7) 发布位置指令
            self._publish_policy_command(runtime, q_target)
        except Exception as exc:
            # 任何异常都停策略并发零命令，确保安全
            self.get_logger().error(f"Policy '{self.active_name}' failed: {exc}")
            self._deactivate("runtime error", send_zero=True)

    # ------------------------------------------------------------------
    # LowCmd 构造与发布
    # ------------------------------------------------------------------
    def _base_low_cmd(self) -> LowCmd:
        """构造基础 LowCmd：
        - 递增 sequence；
        - 用 LowState 中的关节名字建立 MotorCmd 列表；
        - 默认 q 为当前位置，dq/tau/kp/kd 全为 0（相当于卸力）。
        """
        msg = LowCmd()
        self.sequence += 1
        msg.sequence = self.sequence
        if self.latest_state is not None:
            for state in self.latest_state.motor_state:
                cmd = MotorCmd()
                cmd.name = state.name
                cmd.q = float(state.q)
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = 0.0
                cmd.kd = 0.0
                msg.motor_cmd.append(cmd)
        return msg

    def _publish_zero_command(self) -> None:
        """发布全零命令（卸力），用于安全停止。"""
        self.cmd_pub.publish(self._base_low_cmd())

    def _publish_position_command(
        self,
        runtime: PolicyRuntime,
        q_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        """Publish one complete policy-order position command."""
        assert self.latest_state is not None
        msg = self._base_low_cmd()
        by_name = {cmd.name: cmd for cmd in msg.motor_cmd}

        command_names = runtime.resolve_control_names(self.latest_state)
        n = len(command_names)
        if len(q_target) != n or len(kp) != n or len(kd) != n:
            raise ValueError(
                f"position command size mismatch: names={n}, q={len(q_target)}, "
                f"kp={len(kp)}, kd={len(kd)}")

        for i, command_name in enumerate(command_names):
            if command_name not in by_name:
                raise ValueError(f"LowState joint disappeared: {command_name}")
            cmd = by_name[command_name]
            cmd.q = float(q_target[i])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = float(kp[i])
            cmd.kd = float(kd[i])

        self.cmd_pub.publish(msg)

    def _publish_policy_command(self, runtime: PolicyRuntime, q_target: np.ndarray) -> None:
        """Publish the learned policy target in the physical control-joint space."""
        self._publish_position_command(
            runtime,
            runtime.control_position_target(q_target),
            runtime.control_kp(),
            runtime.control_kd(),
        )


def main(args=None) -> None:
    """程序入口：初始化 rclpy、创建节点、自旋, 最后关闭。"""
    rclpy.init(args=args)
    try:
        node = PolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C 正常退出
        pass
    except Exception as exc:
        print(f"noetix_policy_node fatal: {exc}")
        raise
    finally:
        # 只在仍处于 ok 状态时关闭，避免重复关闭报错
        if rclpy.ok():
            rclpy.shutdown()