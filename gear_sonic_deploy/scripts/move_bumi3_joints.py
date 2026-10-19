#!/usr/bin/env python3
"""通过 ROS 2 依次小幅活动 BUMI3 关节，人工核对策略顺序与真实关节的对应关系。

默认只打印检查顺序；--run 才允许发布动作，每个关节开始前均等待 Enter。
机器人必须被可靠支撑，本工具不提供站立平衡控制。--start-bridge 可单独启动
已编译的硬件桥；所有配置临时复制，不改原配置，也不加载策略或运动序列。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import termios
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.bumi3_deploy_common import joint_mappings, load_deploy_config, repo_path
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import Bumi3Contract


def load_mapping(args):
    """复用训练契约和部署映射，补充 SDK 标定及地址；关节检查顺序固定为策略顺序。"""
    config = load_deploy_config(args.config)
    if args.robot_root:
        config['hardware']['robot_root'] = args.robot_root
    contract = Bumi3Contract.from_yaml(repo_path(args.obs_config or config['contract']))
    mappings = joint_mappings(config, contract)
    hardware_path = repo_path(config['hardware']['robot_root']) / 'config/hardware.yaml'
    hardware = yaml.safe_load(hardware_path.read_text(encoding='utf-8'))
    calibrations = {item['name']: item for item in hardware['actuator_list']}
    expected = {item['actuator'] for item in mappings}
    if len(calibrations) != 21 or len(hardware['actuator_list']) != 21 or set(calibrations) != expected:
        raise ValueError('SDK 标定与 21 个逻辑执行器不一致')
    topology, addresses = {}, set()
    for canu in hardware['canu_network']:
        if not canu.get('enable', True):
            continue
        for channel in (1, 2, 3):
            for motor in canu.get(f'channel_{channel}', []):
                address = (canu['ecat_id'], channel, motor['can_id'])
                if motor['name'] in topology or address in addresses:
                    raise ValueError(f'执行器或总线地址重复: {motor["name"]}, {address}')
                topology[motor['name']] = f'{canu["name"]}/EC{address[0]}/CH{channel}/ID{address[2]}'
                addresses.add(address)
    if set(topology) != expected:
        raise ValueError('CANU 拓扑与逻辑执行器集合不一致')
    print('索引 模型关节                 ROS执行器          ROS方向/偏移  SDK方向/偏移  总线')
    for i, item in enumerate(mappings):
        calibration = calibrations[item['actuator']]
        if calibration['direction'] not in (-1, 1) or not math.isfinite(float(calibration['bias'])):
            raise ValueError(f'SDK 标定无效: {item["actuator"]}')
        item['bus'] = topology[item['actuator']]
        print(f'{i:02} {item["name"]:24} {item["actuator"]:18} '
              f'{item["direction"]:+}/{item["offset"]:+.3f} '
              f'{calibration["direction"]:+}/{calibration["bias"]:+.3f} {item["bus"]}')
    print('SDK 标定由桥执行，不在此重复应用；并联踝按逻辑 pitch/roll 检查，由 SDK 完成转换。')
    return config, contract, mappings


class HardwareBridge:
    """启动一份独立硬件桥；标准输入关闭、日志用管道转发，避免 sudo 争用键盘。"""

    def __init__(self, args, config, contract, mappings):
        self.args, self.config = args, config
        self.contract, self.mappings = contract, mappings
        self.directory = None
        self.process = None
        self.reader = None

    def start(self):
        """复制本机资源与网络配置，为桥传入训练力矩上限和正确的浮点参数类型。"""
        install = os.environ.get('BUMI_ROS_INSTALL')
        if not install:
            raise RuntimeError('请先 source gear_sonic_deploy/scripts/setup_env_bumi3.sh')
        executable = Path(install) / 'lib/noetix_bridge/noetix_bridge_node'
        if not executable.is_file():
            raise RuntimeError('硬件桥未编译，请先执行 bash gear_sonic_deploy/scripts/setup_env_bumi3.sh real')
        if self.args.sudo:
            subprocess.run(['sudo', '-v'], check=True)
        network = yaml.safe_load(repo_path(self.config['hardware']['network']).read_text(encoding='utf-8'))
        if self.args.interface:
            network['ethercat']['ifname'] = self.args.interface
        if not network['ethercat'].get('ifname'):
            raise ValueError('启动桥需要 --interface 或网络配置中的 ethercat.ifname')
        cycle = float(network['ethercat'].get('cycle_time_ns', 1000000))
        frequency = float(self.config['hardware']['io_frequency'])
        if not math.isclose(cycle * frequency, 1e9, rel_tol=1e-5):
            raise ValueError('EtherCAT cycle_time_ns 与 hardware.io_frequency 不匹配')
        self.directory = Path(tempfile.mkdtemp(prefix='bumi3-joint-check-'))
        root = self.directory
        robot_root = root / 'robots/bumi3'
        shutil.copytree(repo_path(self.config['hardware']['robot_root']) / 'config', robot_root / 'config')
        network_path = root / 'robots/common/bumi/network.yaml'
        network_path.parent.mkdir(parents=True)
        network_path.write_text(yaml.safe_dump(network, allow_unicode=True), encoding='utf-8')
        robot = yaml.safe_load((robot_root / 'config/robot.yaml').read_text(encoding='utf-8'))
        efforts = dict(zip(self.contract.mujoco_joint_names, self.contract.effort_mujoco))
        params = {'/**': {'ros__parameters': {
            'robot_name': 'bumi3', 'robot_root': str(robot_root),
            'control_frequency_hz': frequency,
            'command_timeout': float(self.config['ros2']['command_timeout']),
            'effort_limits': [float(efforts[name]) for name in robot['robot']['joint_order']],
        }}}
        params_path = root / 'bridge.yaml'
        params_path.write_text(yaml.safe_dump(params, allow_unicode=True), encoding='utf-8')
        ros = self.config['ros2']
        command = [str(executable), '--ros-args', '--params-file', str(params_path)]
        for original, setting in [('/noetix/low_state', 'state_topic'),
                                  ('/noetix/low_cmd', 'command_topic'),
                                  ('/noetix/control_request', 'request_topic')]:
            command += ['-r', f'{original}:={ros[setting]}']
        if self.args.sudo:
            # 只传递运行库与 ROS 通信相关变量，保留当前终端已配置的依赖查找路径。
            keys = ['ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'RMW_IMPLEMENTATION',
                    'FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE',
                    'FASTDDS_BUILTIN_TRANSPORTS', 'ROS_DISCOVERY_SERVER', 'ROS_SUPER_CLIENT',
                    'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH']
            command = ['sudo', '-n', 'env'] + [f'{key}={os.environ[key]}' for key in keys if key in os.environ] + command
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)

        def relay():
            """只转发本工具启动的桥日志，不读取操作者键盘。"""
            for line in self.process.stdout:
                print(line, end='', flush=True)

        self.reader = threading.Thread(target=relay, daemon=True)
        self.reader.start()

    def check(self):
        """检测桥提前退出，避免策略式超时掩盖缺库或 EtherCAT 初始化错误。"""
        if self.process and self.process.poll() is not None:
            raise RuntimeError(f'硬件桥已退出，返回码 {self.process.returncode}，请查看前面的桥日志')

    def close(self):
        """只停止本工具创建的桥；未退出时保留临时配置并报告，不强杀硬件进程。"""
        if self.process and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                print(f'BRIDGE_STOP_UNCONFIRMED: 桥进程未确认退出，PID={self.process.pid}；请用原厂急停确认。', flush=True)
                if self.directory:
                    # 仍运行的桥可能使用临时资源，交还给操作者处理，避免自动删除配置。
                    print(f'桥的临时配置保留在 {self.directory}', flush=True)
                return
        if self.reader:
            self.reader.join(timeout=1)
        if self.directory:
            shutil.rmtree(self.directory)


class JointCheck:
    """独占命令话题，持续保活并检查实测反馈；交互等待期间仍保持当前目标。"""

    def __init__(self, args, config, contract, mappings):
        import mujoco
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from noetix_msgs.msg import LowCmd, LowState, MotorCmd
        from std_msgs.msg import String

        self.args, self.config, self.contract, self.mappings = args, config, contract, mappings
        self.rclpy, self.LowCmd, self.MotorCmd, self.String = rclpy, LowCmd, MotorCmd, String
        self.node = rclpy.create_node(f'bumi3_joint_sequence_{os.getpid()}')
        self.cmd_pub = self.node.create_publisher(LowCmd, config['ros2']['command_topic'], 1)
        self.req_pub = self.node.create_publisher(String, config['ros2']['request_topic'], 10)
        self.subscription = self.node.create_subscription(LowState, config['ros2']['state_topic'], self.receive, qos_profile_sensor_data)
        self.message, self.received, self.sequence = None, 0.0, None
        self.target, self.origin, self.enabled_by_us = None, None, False
        self.active_joint = None
        self.sent, self.last_graph, self.counter = 0.0, 0.0, 0
        self.trial_origin, self.peak, self.peak_signed = None, np.zeros(21), np.zeros(21)
        self.bridge = HardwareBridge(args, config, contract, mappings)
        self.directions = np.array([item['direction'] for item in mappings])
        self.offsets = np.array([item['offset'] for item in mappings])
        self.kp = contract.stiffness_mujoco[contract.mujoco_to_policy] * args.gain_scale
        self.kd = contract.damping_mujoco[contract.mujoco_to_policy] * args.gain_scale
        model = mujoco.MjModel.from_xml_path(str(contract.model_path))
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, item['name']) for item in mappings]
        self.lower, self.upper = model.jnt_range[joints, 0].copy(), model.jnt_range[joints, 1].copy()

    def receive(self, message):
        """只将变化的反馈序号计作新状态，禁止用重复缓存维持控制。"""
        if message.sequence != self.sequence:
            self.message, self.sequence, self.received = message, message.sequence, time.monotonic()

    def graph(self, require_bridge=True):
        """拒绝仿真、多份环境或其他命令／控制请求发布者，避免多源争用机器人。"""
        ros = self.config['ros2']
        publishers = self.node.get_publishers_info_by_topic(ros['state_topic'])
        if len(publishers) > 1 or any(p.node_name != 'noetix_bridge_node' for p in publishers):
            raise RuntimeError('状态来源不是唯一真机桥: ' + str([(p.node_namespace, p.node_name) for p in publishers]))
        if require_bridge and len(publishers) != 1:
            raise RuntimeError('未发现唯一真机状态发布者')
        for key in ('command_topic', 'request_topic'):
            others = [p.node_name for p in self.node.get_publishers_info_by_topic(ros[key])
                      if p.node_name != self.node.get_name() or p.node_namespace != self.node.get_namespace()]
            if others:
                raise RuntimeError(f'{ros[key]} 有其他发布者 {others}；请退出策略／姿态／手柄控制程序')

    def state(self):
        """按真实部署的名称和方向规则得到策略顺序角度，并检查连接与电机反馈。"""
        message = self.message
        if message is None or time.monotonic() - self.received > self.config['ros2']['state_timeout']:
            raise RuntimeError('真机反馈缺失、过期或序号停止更新')
        names = [motor.name for motor in message.motor_state]
        expected = {item['actuator'] for item in self.mappings}
        if len(names) != 21 or len(set(names)) != 21 or set(names) != expected:
            raise RuntimeError(f'执行器名称不匹配: missing={expected-set(names)}, extra={set(names)-expected}')
        if not message.status.connected or message.status.fault or not message.imu.valid:
            raise RuntimeError('连接、IMU 或机器人故障状态无效')
        motors = {motor.name: motor for motor in message.motor_state}
        if any(motor.error or not all(math.isfinite(v) for v in (motor.q, motor.dq, motor.tau_est)) for motor in motors.values()):
            raise RuntimeError('电机反馈含错误或非有限值')
        q = np.array([motors[item['actuator']].q for item in self.mappings]) * self.directions + self.offsets
        dq = np.array([motors[item['actuator']].dq for item in self.mappings]) * self.directions
        if np.any(q < self.lower - 0.01) or np.any(q > self.upper + 0.01):
            raise RuntimeError('实测角度超出模型限位，请先检查零位与方向，未继续发送动作')
        if self.enabled_by_us:
            if not message.status.enabled:
                raise RuntimeError('环境端关闭了驱动门控')
            if np.max(np.abs(dq)) > 1.0:
                raise RuntimeError('实测关节速度超过检查上限 1 rad/s')
            # 仅当前检查关节按动作幅度增加偏差容限，保证无响应时能完成测量并报告；
            # 其余保持姿态的关节仍使用 8° 上限，避免放宽整机的异常位移检查。
            limits = np.full(len(q), math.radians(8))
            if self.active_joint is not None:
                limits[self.active_joint] = math.radians(max(8, self.args.angle_deg + 2))
            errors = np.abs(q - self.target)
            if np.any(errors > limits):
                index = int(np.argmax(errors - limits))
                raise RuntimeError(f'{self.mappings[index]["name"]} 目标与实测偏差 '
                                   f'{math.degrees(errors[index]):.1f}° 超过 '
                                   f'{math.degrees(limits[index]):.1f}°，检查中止')
        return q

    def publish(self):
        """每次发送完整 21 关节命令；其他关节保持初始角度，不使用默认站姿或零角。"""
        message = self.LowCmd()
        message.stamp = self.node.get_clock().now().to_msg()
        self.counter += 1
        message.sequence = self.counter
        for i, item in enumerate(self.mappings):
            motor = self.MotorCmd()
            motor.name = item['actuator']
            motor.q = float((self.target[i] - item['offset']) * item['direction'])
            motor.kp, motor.kd = float(self.kp[i]), float(self.kd[i])
            message.motor_cmd.append(motor)
        self.cmd_pub.publish(message)

    def pump(self):
        """非阻塞处理 ROS、检查反馈并持续下发目标，交互等待不能阻塞看门狗保活。"""
        if not self.rclpy.ok():
            raise KeyboardInterrupt
        self.rclpy.spin_once(self.node, timeout_sec=0.005)
        self.bridge.check()
        now = time.monotonic()
        if now - self.last_graph >= 0.5:
            self.graph()
            self.last_graph = now
        q = self.state()
        if self.trial_origin is not None:
            delta = np.rad2deg(q - self.trial_origin)
            changed = np.abs(delta) > self.peak
            self.peak[changed], self.peak_signed[changed] = np.abs(delta[changed]), delta[changed]
        if self.target is not None and now - self.sent >= self.contract.control_dt:
            self.publish()
            self.sent = now
        return q

    def prompt(self, text, armed=False):
        """等待确认时继续运行反馈检查与保持命令，避免 input() 阻塞导致超时。"""
        # 丢弃上一动作期间积压的回车，每个新关节都需要看到提示后再次明确确认。
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        print(text + ' [Enter 继续 / q 然后 Enter 退出；Ctrl+C 立即中止]', flush=True)
        while True:
            if armed:
                self.pump()
            else:
                self.rclpy.spin_once(self.node, timeout_sec=0.02)
                self.bridge.check()
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if ready:
                line = sys.stdin.readline()
                if not line or line.strip().lower() == 'q':
                    raise KeyboardInterrupt
                if not line.strip():
                    return
                print('请输入 Enter 或 q。', flush=True)

    def move(self, target, seconds):
        """采用端点速度为零的余弦轨迹；时间结束时强制发送准确终点，再保持一帧。"""
        initial, started = self.target.copy(), time.monotonic()
        while True:
            ratio = min((time.monotonic() - started) / seconds, 1.0)
            self.target = initial + (target - initial) * (0.5 - 0.5 * math.cos(math.pi * ratio))
            self.pump()
            if ratio >= 1:
                break
        self.target = target.copy()
        self.hold(self.contract.control_dt * 2)

    def hold(self, seconds):
        """定时保持当前目标，期间仍可通过 Ctrl+C 中止。"""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.pump()

    def run(self, indices):
        """先建立唯一控制来源，再使能并逐项检查；没有反馈响应时不自动增加增益。"""
        discovery_end = time.monotonic() + 1.5
        while time.monotonic() < discovery_end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        self.graph(require_bridge=False)
        if self.args.start_bridge:
            if self.node.get_publishers_info_by_topic(self.config['ros2']['state_topic']):
                raise RuntimeError('已有真机桥；请去掉 --start-bridge，避免重复启动 EtherCAT 主站')
            self.bridge.start()
        deadline = time.monotonic() + 10
        while self.message is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            self.bridge.check()
        self.graph()
        self.state()
        self.prompt('请确认机器人已可靠支撑、急停可用；本工具只检查关节，不维持站立平衡。确认后请求使能。')
        self.graph()
        self.origin = self.state().copy()
        self.target = self.origin.copy()
        # SDK 的使能查询可能误报；这里只将 enabled 用作门控，运动效果必须看实测变化。
        self.enabled_by_us = True
        self.req_pub.publish(self.String(data='enable'))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)
            self.bridge.check()
            self.state_before_enable()
            self.publish()
            if self.message.status.enabled:
                break
        else:
            raise RuntimeError('未收到 ROS 驱动门控开启反馈')
        self.hold(0.5)
        print('门控已开启；这不是实际电机使能成功的证明，将通过小幅动作观察反馈。', flush=True)
        for index in indices:
            item = self.mappings[index]
            target = self.origin.copy()
            amplitude = math.radians(self.args.angle_deg)
            if target[index] + amplitude > self.upper[index]:
                amplitude = -amplitude
            if not self.lower[index] <= target[index] + amplitude <= self.upper[index]:
                raise RuntimeError(f'{item["name"]} 附近没有足够的检查范围')
            target[index] += amplitude
            self.prompt(f'\n[{index:02d}] {item["name"]} → {item["actuator"]} / {item["bus"]}；'
                        f'模型方向 {math.degrees(amplitude):+.1f}°，随后回原位。', armed=True)
            trial_start = self.state().copy()
            self.active_joint = index
            self.trial_origin = trial_start
            self.peak[:], self.peak_signed[:] = 0, 0
            self.move(target, self.args.ramp_seconds)
            self.hold(0.5)
            at_target = self.state().copy()
            self.move(self.origin, self.args.ramp_seconds)
            self.hold(0.5)
            self.trial_origin = None
            self.active_joint = None
            print(f'RESULT joint={item["name"]} requested={math.degrees(amplitude):+.2f}° '
                  f'measured_at_target={math.degrees(at_target[index]-trial_start[index]):+.2f}° '
                  f'peak={self.peak[index]:.2f}°', flush=True)
            for i in np.argsort(-self.peak)[:3]:
                print(f'  {self.mappings[i]["name"]}: peak={self.peak[i]:.2f}° signed={self.peak_signed[i]:+.2f}°', flush=True)
            measured = at_target[index] - trial_start[index]
            if abs(measured) < abs(amplitude) * 0.2:
                raise RuntimeError('NO_RESPONSE: 指定关节反馈没有足够响应；需检查实际使能、增益和电机供电，未继续下一关节')
            if measured * amplitude <= 0:
                raise RuntimeError('OPPOSITE_FEEDBACK: 指定关节反馈与目标符号相反，未继续下一关节')
            others = self.peak.copy()
            others[index] = 0
            if np.max(others) > max(1.0, self.peak[index] * 1.5):
                raise RuntimeError('OTHER_JOINT_MOVED: 其他关节变化更大，请核对映射或机械耦合，未继续下一关节')
            print('请确认真实活动的关节及方向与名称一致；反馈响应不等于实体映射已被自动验证。', flush=True)
        self.prompt('所选关节已检查完毕，当前保持初始姿态；按 Enter 结束并请求去使能。', armed=True)

    def state_before_enable(self):
        """等待使能时仍校验反馈，允许门控尚未开启，保留清理所需的已请求标记。"""
        self.enabled_by_us = False
        try:
            return self.state()
        finally:
            self.enabled_by_us = True

    def close(self):
        """请求关闭门控，不发送可能覆盖去使能的后续 MIT 命令，不宣称物理卸力成功。"""
        try:
            if self.enabled_by_us and self.rclpy.ok():
                self.req_pub.publish(self.String(data='disable'))
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline and self.rclpy.ok():
                    self.rclpy.spin_once(self.node, timeout_sec=0.02)
                print('已发送 disable 请求；SDK 真实去使能反馈存在已知问题，请用原厂急停确认电机状态。', flush=True)
        finally:
            self.bridge.close()
            self.node.destroy_node()


def main():
    """默认仅展示计划；明确指定 --run 并在终端确认后才开始真实关节检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='gear_sonic/utils/mujoco_sim/wbc_configs/bumi3_21dof_sonic_model12.yaml')
    parser.add_argument('--obs-config')
    parser.add_argument('--robot-root')
    parser.add_argument('--run', action='store_true', help='允许真实 ROS 动作，仍逐关节等待 Enter')
    parser.add_argument('--start-bridge', action='store_true', help='单独启动已编译的硬件桥，无需运行 deploy.sh')
    parser.add_argument('--interface', help='启动桥时使用的 EtherCAT 网卡')
    parser.add_argument('--sudo', action='store_true', help='启动硬件桥时使用 sudo，仅桥提权')
    parser.add_argument('--joint', help='仅检查这个模型关节；默认按策略顺序检查全部 21 个')
    parser.add_argument('--angle-deg', type=float, default=10.0, help='检查幅度，默认 10°，最大 10°')
    parser.add_argument('--ramp-seconds', type=float, default=1.5, help='单程平滑过渡时间，至少 1 秒')
    parser.add_argument('--gain-scale', type=float, default=0.25, help='训练 KP/KD 的比例，范围 (0,1]')
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in (args.angle_deg, args.ramp_seconds, args.gain_scale))
            or not 0 < args.angle_deg <= 10 or args.ramp_seconds < 1 or not 0 < args.gain_scale <= 1):
        parser.error('angle-deg 需在 (0,10]，ramp-seconds 至少 1，gain-scale 需在 (0,1]，且均为有限数')
    if (args.interface or args.sudo) and not args.start_bridge:
        parser.error('--interface 和 --sudo 只用于 --start-bridge')
    config, contract, mappings = load_mapping(args)
    if args.joint and args.joint not in contract.policy_joint_names:
        parser.error('未知模型关节，可用名称见上方映射表')
    indices = [contract.policy_joint_names.index(args.joint)] if args.joint else list(range(21))
    print('\n检查顺序:', ' → '.join(f'{i}:{mappings[i]["name"]}' for i in indices), flush=True)
    if not args.run:
        print('当前只展示计划。添加 --run 后才会请求使能与发送真实动作。')
        return
    if not sys.stdin.isatty():
        parser.error('--run 需要前台交互终端，逐关节按 Enter 确认')
    os.environ['ROS_DOMAIN_ID'] = str(config['ros2']['domain_id'])
    os.environ.setdefault('ROS_LOCALHOST_ONLY', '0')
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    profile = str(repo_path('gear_sonic_deploy/bumi3/ros2_udp.xml'))
    os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', profile)
    os.environ.setdefault('FASTDDS_DEFAULT_PROFILES_FILE', os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'])
    import rclpy
    # Ctrl+C 由 Python 抛出异常，以便 ROS 上下文仍可用于发送退出请求。
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def stop_on_signal(signum, frame):
        """终止信号也进入退出清理流程，使 ROS 去使能请求有机会发送。"""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    checker = None
    try:
        checker = JointCheck(args, config, contract, mappings)
        checker.run(indices)
    except KeyboardInterrupt:
        print('\n检查已由操作者中止。', flush=True)
    finally:
        if checker:
            checker.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'BUMI3_JOINT_CHECK_ERROR: {error}', file=sys.stderr)
        sys.exit(1)
