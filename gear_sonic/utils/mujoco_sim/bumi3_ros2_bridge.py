"""MuJoCo environment using the same Noetix ROS 2 API as the hardware bridge."""
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
import json
import os
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from noetix_msgs.msg import LowCmd, LowState, MotorState, RobotStatus
from std_msgs.msg import Float64MultiArray, String
import tyro

from gear_sonic.utils.mujoco_sim.bumi3_deploy_common import (
    DEFAULT_DEPLOY_CONFIG, contract_id, joint_mappings, load_deploy_config, repo_path,
)
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract, Bumi3SonicSim2Sim, ZeroPolicy, make_static_reference_motion,
)


@dataclass
class Args:
    config: Path = DEFAULT_DEPLOY_CONFIG
    obs_config: Path | None = None
    headless: bool = False
    show_reference: bool = True


class Bumi3Ros2Bridge(Node):
    def __init__(self, config, contract, runner):
        super().__init__('bumi3_mujoco')
        self.settings = config['ros2']
        self.contract = contract
        self.runner = runner
        self.mapping = joint_mappings(config, contract)
        self.keys = SimpleQueue()
        self.command = None
        self.received = 0.0
        self.enabled = False
        self.ghost = None
        self.reset_sequence = 0
        self.reset_state_sequence = 0
        self.key_sequence = 0
        self.key = 0
        self.contract_id = contract_id(contract, config)
        self.sequence = 0
        self.state_publisher = self.create_publisher(LowState, self.settings['state_topic'], 10)
        self.status_publisher = self.create_publisher(String, self.settings['sim_status_topic'], 10)
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(LowCmd, self.settings['command_topic'], self.receive, reliable)
        self.create_subscription(String, self.settings['request_topic'], self.request, 20)
        self.create_subscription(Float64MultiArray, self.settings['sim_reset_topic'], self.reset, reliable)
        self.create_subscription(Float64MultiArray, self.settings['sim_reference_topic'], self.reference, reliable)
        self.imu_to_base = np.asarray(config['hardware']['imu_to_base_wxyz'], dtype=float)
        if self.imu_to_base.shape != (4,) or not np.isfinite(self.imu_to_base).all() or not np.isclose(np.linalg.norm(self.imu_to_base), 1):
            raise ValueError('hardware.imu_to_base_wxyz must be a unit quaternion')

    def receive(self, message):
        if not self.enabled:
            return
        by_name = {item.name: item for item in message.motor_cmd}
        if len(message.motor_cmd) != 21 or len(by_name) != 21 or set(by_name) != {item['actuator'] for item in self.mapping}:
            self.get_logger().error('LowCmd must contain every configured logical actuator exactly once')
            self.enabled = False
            return
        values = np.array([[by_name[item['actuator']].q, by_name[item['actuator']].dq,
                            by_name[item['actuator']].tau, by_name[item['actuator']].kp,
                            by_name[item['actuator']].kd] for item in self.mapping], dtype=float)
        if not np.isfinite(values).all() or (values[:, 3:] < 0).any():
            self.enabled = False
            self.get_logger().error('Invalid LowCmd; disabled')
            return
        self.command = values
        self.received = time.monotonic()

    def request(self, message):
        request = message.data.lower()
        if request == 'enable' and not self.enabled:
            self.command = None
            self.received = time.monotonic()
            self.enabled = True
        elif request in ('disable', 'stop', 'estop', 'safe_stop'):
            self.enabled = False
            self.command = None
            self.runner.data.ctrl[:] = 0

    def reset(self, message):
        # Simulator-only: [contract CRC32, request sequence, qpos(28), qvel(27)].
        values = np.asarray(message.data, dtype=float)
        if values.shape != (57,) or not np.isfinite(values).all() or values[0] != self.contract_id or values[1] != int(values[1]) or not 0 < values[1] < 2**53:
            self.get_logger().error('Invalid simulation reset/config; ignored')
            return
        if int(values[1]) == self.reset_sequence:
            return
        qpos, qvel = values[2:30], values[30:]
        if not np.isclose(np.linalg.norm(qpos[3:7]), 1, atol=1e-5):
            self.get_logger().error('Invalid reset quaternion; ignored')
            return
        self.enabled = False
        self.command = None
        mujoco.mj_resetData(self.runner.model, self.runner.data)
        self.runner.data.qpos[:] = qpos
        self.runner.data.qvel[:] = qvel
        mujoco.mj_forward(self.runner.model, self.runner.data)
        self.reset_sequence = int(values[1])
        self.reset_state_sequence = self.sequence + 1
        self.ghost = None

    def reference(self, message):
        values = np.asarray(message.data, dtype=float)
        if values.size == 0:
            self.ghost = None
        elif values.shape == (28,) and np.isfinite(values).all() and np.isclose(np.linalg.norm(values[3:7]), 1, atol=1e-5):
            self.ghost = values

    def step(self):
        if self.enabled and time.monotonic() - self.received > self.settings['command_timeout']:
            self.enabled = False
            self.command = None
            self.get_logger().error('LowCmd timeout; simulation disabled')
        if not self.enabled or self.command is None:
            self.runner.data.ctrl[:] = 0
            return
        q, dq = self.runner._joint_state_policy()
        directions = np.array([item['direction'] for item in self.mapping])
        offsets = np.array([item['offset'] for item in self.mapping])
        cmd = self.command
        target = cmd[:, 0] * directions + offsets
        target_velocity = cmd[:, 1] * directions
        torque = cmd[:, 3] * (target - q) + cmd[:, 4] * (target_velocity - dq) + cmd[:, 2] * directions
        effort = self.contract.effort_mujoco[self.contract.mujoco_to_policy]
        torque = np.clip(torque, -effort, effort)
        self.runner.data.ctrl[self.runner.actuator_ids] = torque[self.contract.policy_to_mujoco]
        mujoco.mj_step(self.runner.model, self.runner.data)
        mujoco.mj_forward(self.runner.model, self.runner.data)
        if not np.isfinite(self.runner.data.qpos).all() or not np.isfinite(self.runner.data.qvel).all():
            raise FloatingPointError('MuJoCo produced nonfinite state')

    def publish(self):
        q, dq = self.runner._joint_state_policy()
        message = LowState()
        message.stamp = self.get_clock().now().to_msg()
        self.sequence += 1
        message.sequence = self.sequence
        for index, item in enumerate(self.mapping):
            motor = MotorState()
            motor.name = item['actuator']
            motor.q = float((q[index] - item['offset']) * item['direction'])
            motor.dq = float(dq[index] * item['direction'])
            motor.tau_est = float(self.runner.data.actuator_force[self.runner.actuator_ids[self.contract.mujoco_to_policy[index]]] * item['direction'])
            message.motor_state.append(motor)
        # Publish sensor coordinates just like the SDK; the policy rotates to base_link.
        sensor_quat = np.empty(4)
        mujoco.mju_mulQuat(sensor_quat, self.runner.data.xquat[self.runner.anchor_body_id], self.imu_to_base)
        inverse = self.imu_to_base.copy()
        inverse[1:] *= -1
        gyro = np.empty(3)
        mujoco.mju_rotVecQuat(gyro, self.runner._base_angular_velocity_local(), inverse)
        message.imu.valid = True
        message.imu.quaternion = sensor_quat.astype(float).tolist()
        message.imu.gyroscope = gyro.astype(float).tolist()
        message.status.connected = True
        message.status.enabled = self.enabled
        message.status.mode = RobotStatus.MODE_ENABLED if self.enabled else RobotStatus.MODE_DISABLED
        self.state_publisher.publish(message)
        try:
            self.key = self.keys.get_nowait()
            self.key_sequence += 1
        except Empty:
            pass
        status = String()
        status.data = json.dumps(dict(contract_id=self.contract_id, reset_sequence=self.reset_sequence,
                                      state_sequence=self.reset_state_sequence,
                                      key_sequence=self.key_sequence, key=self.key))
        self.status_publisher.publish(status)


def main(args: Args):
    config = load_deploy_config(args.config)
    os.environ['ROS_DOMAIN_ID'] = str(config['ros2']['domain_id'])
    rclpy.init(args=[])
    contract = Bumi3Contract.from_yaml(repo_path(args.obs_config or config['contract']))
    runner = Bumi3SonicSim2Sim(contract, make_static_reference_motion(contract), ZeroPolicy(contract), start_paused=True)
    bridge = Bumi3Ros2Bridge(config, contract, runner)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    viewer = None
    try:
        if not args.headless:
            from mujoco import viewer as mujoco_viewer
            viewer = mujoco_viewer.launch_passive(runner.model, runner.data, key_callback=bridge.keys.put)
        print(f"BUMI3_SIM_READY ROS2 domain={config['ros2']['domain_id']} physics_hz={1/contract.sim_dt:g}", flush=True)
        last_view = 0.0
        while rclpy.ok() and (viewer is None or viewer.is_running()):
            tick = time.monotonic()
            for _ in range(8):
                executor.spin_once(timeout_sec=0)
            bridge.step()
            bridge.publish()
            if viewer is not None and tick - last_view >= 1/60:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if args.show_reference and bridge.ghost is not None:
                        runner.reference_visual_data.qpos[:] = bridge.ghost
                        mujoco.mj_forward(runner.reference_visual_model, runner.reference_visual_data)
                        markers = runner._resolved_visual_marker_specs(runner.reference_visual_data,
                            runner.reference_visual_scene, runner.reference_visual_option,
                            runner.reference_visual_perturb, runner.reference_visual_camera,
                            alpha=0.32, tint=np.array([1., 0.15, 0.15], dtype=np.float32))
                        runner._append_visual_markers_to_scene(viewer.user_scn, markers)
                viewer.sync()
                last_view = tick
            time.sleep(max(0, contract.sim_dt - (time.monotonic() - tick)))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        runner.data.ctrl[:] = 0
        executor.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main(tyro.cli(Args))
