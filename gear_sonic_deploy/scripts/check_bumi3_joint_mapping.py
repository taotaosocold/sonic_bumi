#!/usr/bin/env python3
"""只读检查 BUMI3 关节映射，将真实反馈按部署规则重排并可同步显示到 MuJoCo。

--list 只核对本地配置；默认订阅反馈，--joint 指定人工检查的实体关节。
本脚本没有控制发布器，不发送使能、去使能或运动指令，不运行策略或物理步进。
人工活动关节须先通过机器人原有流程确认可安全手动活动，不能仅凭 enabled 标志判断。
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.bumi3_deploy_common import (
    joint_mappings, load_deploy_config, repo_path,
)
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import Bumi3Contract


def unique(items, key, label):
    """按名称建立索引，重复项直接报错，避免后出现的配置静默覆盖前项。"""
    result = {}
    for item in items:
        name = item[key]
        if name in result:
            raise ValueError(f'{label} 重复: {name}')
        result[name] = item
    return result


def load_mapping(args):
    """复用部署配置与契约，核对每个逻辑执行器的标定和总线地址。"""
    config = load_deploy_config(args.config)
    if args.robot_root:
        config['hardware']['robot_root'] = args.robot_root
    contract = Bumi3Contract.from_yaml(repo_path(args.obs_config or config['contract']))
    mappings = joint_mappings(config, contract)
    resource = repo_path(config['hardware']['robot_root']) / 'config'
    robot_path, hardware_path = resource / 'robot.yaml', resource / 'hardware.yaml'
    robot = yaml.safe_load(robot_path.read_text(encoding='utf-8'))
    hardware = yaml.safe_load(hardware_path.read_text(encoding='utf-8'))
    if robot['robot']['name'] != 'bumi3' or robot['robot']['dof'] != 21:
        raise ValueError('需要 BUMI3 的 21 自由度机器人配置')
    if tuple(robot['robot']['joint_order']) != contract.mujoco_joint_names:
        raise ValueError('robot.yaml joint_order 与契约 MuJoCo 顺序不一致')
    calibrations = unique(hardware['actuator_list'], 'name', 'SDK 标定')
    expected = {item['actuator'] for item in mappings}
    if set(calibrations) != expected:
        raise ValueError('SDK actuator_list 与部署执行器集合不一致')
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
        raise ValueError('CANU 拓扑与部署执行器集合不一致')
    ankle = hardware['parallel_ankle']
    ankle_names = {ankle[side][axis] for side in ('left', 'right') for axis in ('pitch', 'roll')}
    for item in mappings:
        cal = calibrations[item['actuator']]
        if cal['direction'] not in (-1, 1) or not math.isfinite(float(cal['bias'])):
            raise ValueError(f'SDK 标定无效: {item["actuator"]}')
        item.update(sdk_direction=cal['direction'], sdk_bias=float(cal['bias']),
                    bus=topology[item['actuator']], ankle=item['actuator'] in ankle_names)
    for path in (contract.config_path, robot_path, hardware_path):
        print(f'CONFIG {path}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
    print('\nP=策略索引，M=MuJoCo索引，Dmap=ROS→模型方向，Dsdk=电机→SDK方向；偏移单位 rad。')
    print(' P  M  模型关节                 ROS逻辑执行器       Dmap offset   Dsdk bias    总线地址')
    for i, item in enumerate(mappings):
        m = contract.mujoco_joint_names.index(item['name'])
        print(f'{i:2} {m:2}  {item["name"]:24} {item["actuator"]:18} '
              f'{item["direction"]:+4} {item["offset"]:+7.3f} '
              f'{item["sdk_direction"]:+5} {item["sdk_bias"]:+7.3f} {item["bus"]}'
              f'{" [并联踝逻辑关节]" if item["ankle"] else ""}')
    print('\n反馈使用 q_model = Dmap * q_ros + offset，SDK 标定不重复应用。')
    print('非踝 SDK 标定为 q_sdk = Dsdk * q_motor + bias；并联踝还经过 FK/Jacobian 转换。')
    print('配置一致只证明软件连接关系自洽，实体接线、方向和零位须结合真实姿态确认。')
    return config, contract, mappings


class PoseViewer:
    """只做实测关节姿态的正运动学显示，基座保持直立，不模拟动力学或施加控制。"""

    def __init__(self, contract, selected):
        import mujoco
        import mujoco.viewer

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(contract.model_path))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = contract.initial_root_position
        self.data.qpos[3:7] = [1, 0, 0, 0]
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in contract.policy_joint_names]
        self.addresses = [self.model.jnt_qposadr[j] for j in joints]
        if selected is not None:
            body = self.model.jnt_bodyid[joints[selected]]
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == body and self.model.geom_group[g] == 1:
                    self.model.geom_rgba[g] = [1, 0.5, 0.1, 1]
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def update(self, q):
        """直接按名称对应的地址更新实测角度，不裁剪限位，以便暴露零位和方向问题。"""
        if not self.viewer.is_running():
            return False
        with self.viewer.lock():
            self.data.qpos[self.addresses] = q
            self.mj.mj_forward(self.model, self.data)
        self.viewer.sync()
        return True

    def close(self):
        """关闭本脚本创建的显示窗口，不改变机器人运行状态。"""
        self.viewer.close()


def monitor(args, config, contract, mappings):
    """按执行器名称读取反馈与现有命令，报告实际变化；不把命令回显当成运动成功。"""
    os.environ['ROS_DOMAIN_ID'] = str(config['ros2']['domain_id'])
    profile = repo_path('gear_sonic_deploy/bumi3/ros2_udp.xml')
    os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    os.environ.setdefault('FASTDDS_DEFAULT_PROFILES_FILE', os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'])
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import qos_profile_sensor_data
        from noetix_msgs.msg import LowCmd, LowState
    except ImportError as error:
        raise RuntimeError('请先 source .venv_sim/bin/activate 和 source gear_sonic_deploy/scripts/setup_bumi3_ros2.sh') from error

    selected = contract.policy_joint_names.index(args.joint) if args.joint else None
    expected = {item['actuator'] for item in mappings}
    directions = np.array([item['direction'] for item in mappings])
    offsets = np.array([item['offset'] for item in mappings])
    sample = dict(state=None, received=0.0, sequence=None, command=None, command_time=0.0)

    def receive_state(message):
        """按消息序号更新鲜度；重复的缓存状态不能冒充持续的真实反馈。"""
        if message.sequence != sample['sequence']:
            sample.update(state=message, received=time.monotonic(), sequence=message.sequence)

    def receive_command(message):
        """记录其他控制程序发出的目标用于对照，本脚本始终只订阅。"""
        sample.update(command=message, command_time=time.monotonic())

    rclpy.init()
    node = rclpy.create_node('bumi3_joint_mapping_check')
    viewer, baseline, peaks = None, None, np.zeros(21)
    peak_signed = np.zeros(21)
    begin, first, last_print, last_graph = time.monotonic(), None, 0.0, 0.0
    try:
        subscriptions = [
            node.create_subscription(LowState, config['ros2']['state_topic'], receive_state, qos_profile_sensor_data),
            node.create_subscription(LowCmd, config['ros2']['command_topic'], receive_command, qos_profile_sensor_data),
        ]
        print(f'\n只读订阅 domain={os.environ["ROS_DOMAIN_ID"]} state={config["ros2"]["state_topic"]}')
        if selected is not None:
            print(f'人工检查目标: {args.joint}。先确认机器人可安全手动活动，再单独活动该实体关节。')
        print('可选 --viewer 对照真机与模型姿态；Ctrl+C 只退出检查，不停止机器人。', flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now - last_graph >= 1.0:
                publishers = node.get_publishers_info_by_topic(config['ros2']['state_topic'])
                if len(publishers) > 1:
                    raise RuntimeError('LowState 存在多个发布者: ' + str([(p.node_namespace, p.node_name) for p in publishers]))
                if any(p.node_name == 'bumi3_mujoco' for p in publishers):
                    raise RuntimeError('当前来源是 MuJoCo，无法据此检查实体映射；请退出仿真并保留真机桥')
                if publishers and first is None:
                    print('状态发布者:', [(p.node_namespace, p.node_name) for p in publishers], flush=True)
                last_graph = now
            message = sample['state']
            if message is None:
                if now - begin > args.timeout:
                    raise RuntimeError('未收到状态。请保持真机桥运行，核对 domain/topic；检查脚本不会启动 SDK')
                continue
            if now - sample['received'] > config['ros2']['state_timeout']:
                raise RuntimeError('反馈过期或序号不再更新，停止使用缓存姿态')
            names = [motor.name for motor in message.motor_state]
            if len(names) != 21 or len(set(names)) != 21 or set(names) != expected:
                raise RuntimeError(f'反馈名称不匹配: missing={expected-set(names)}, extra={set(names)-expected}, count={len(names)}')
            motors = {motor.name: motor for motor in message.motor_state}
            q_ros = np.array([motors[item['actuator']].q for item in mappings])
            q = directions * q_ros + offsets
            if not np.isfinite(q).all() or not message.status.connected or message.status.fault:
                raise RuntimeError('反馈非有限、连接失效或存在机器人故障，不能判定映射')
            errors = {name: motor.error for name, motor in motors.items() if motor.error}
            if errors:
                raise RuntimeError(f'关节反馈存在错误: {errors}')
            if baseline is None:
                baseline, first = q.copy(), now
                print('首帧关节角度（deg）:', dict(zip(contract.policy_joint_names, np.round(np.rad2deg(q), 2))), flush=True)
                if args.viewer:
                    viewer = PoseViewer(contract, selected)
            delta = np.rad2deg(q - baseline)
            changed = np.abs(delta) > peaks
            peaks[changed], peak_signed[changed] = np.abs(delta[changed]), delta[changed]
            if viewer and not viewer.update(q):
                break
            if now - last_print >= 1.0:
                ranked = list(np.argsort(-peaks)[:3])
                indices = ([selected] if selected is not None else []) + [int(i) for i in ranked if i != selected]
                command = sample['command'] if now - sample['command_time'] <= config['ros2']['state_timeout'] else None
                targets = {motor.name: motor for motor in command.motor_cmd} if command else {}
                print(f'\n反馈 seq={message.sequence} enabled={message.status.enabled} imu={message.imu.valid}；此使能标志不证明物理驱动已使能')
                for i in indices:
                    item = mappings[i]
                    target = targets.get(item['actuator'])
                    comparison = ''
                    if target:
                        target_q = math.degrees(target.q * item['direction'] + item['offset'])
                        comparison = f' target={target_q:+.2f} error={target_q-math.degrees(q[i]):+.2f} kp={target.kp:.2f} kd={target.kd:.2f}'
                    print(f'{item["name"]:24} q={math.degrees(q[i]):+8.2f} delta={delta[i]:+7.2f} peak={peaks[i]:6.2f}{comparison}', flush=True)
                last_print = now
            if args.seconds and now - first >= args.seconds:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if viewer:
            viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if baseline is not None:
            print('\n变化排名（相对首帧的最大绝对变化，deg）：')
            for i in np.argsort(-peaks)[:5]:
                print(f'{contract.policy_joint_names[i]}: peak={peaks[i]:.2f}, signed={peak_signed[i]:+.2f}')
            if max(peaks) < args.threshold_deg:
                print('NO_MOTION: 未检测到足够变化，不能据此判定映射或使能成功。')
            elif selected is not None:
                dominant = int(np.argmax(peaks))
                print(f'指定实体关节={args.joint}，最大反馈变化={contract.policy_joint_names[dominant]}。')
                print('仅在实际单独活动指定关节时才有辨识意义；并联踝与身体运动可能产生耦合。')
            print('本次为反馈检查；正负方向和绝对零位需对照实体姿态，命令到电机的执行仍需独立确认。')


def main():
    """解析少量检查参数，静态清单模式不连接 ROS，也不创建机器人控制进程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='gear_sonic_deploy/bumi3/deploy.yaml', help='仓库相对部署配置')
    parser.add_argument('--obs-config', help='与实际策略相同的观测配置')
    parser.add_argument('--robot-root', help='桥实际使用的机器人资源目录，覆盖部署配置')
    parser.add_argument('--list', action='store_true', help='只打印并核对本地映射配置')
    parser.add_argument('--joint', help='人工活动并观察的模型关节名称，如 l_arm_pitch_joint')
    parser.add_argument('--viewer', action='store_true', help='把实测关节角度同步到 MuJoCo 静态模型')
    parser.add_argument('--seconds', type=float, default=20.0, help='首帧后观察秒数，0 表示持续观察')
    parser.add_argument('--timeout', type=float, default=10.0, help='等待首条反馈的秒数')
    parser.add_argument('--threshold-deg', type=float, default=1.0, help='报告无明显变化的角度阈值')
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in (args.seconds, args.timeout, args.threshold_deg))
            or args.seconds < 0 or args.timeout <= 0 or args.threshold_deg <= 0):
        parser.error('seconds 必须非负，timeout 和 threshold-deg 必须为正且有限')
    config, contract, mappings = load_mapping(args)
    if args.joint and args.joint not in contract.policy_joint_names:
        parser.error('未知关节，可选: ' + ', '.join(contract.policy_joint_names))
    if not args.list:
        monitor(args, config, contract, mappings)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'BUMI3_MAPPING_ERROR: {error}', file=sys.stderr)
        sys.exit(1)
