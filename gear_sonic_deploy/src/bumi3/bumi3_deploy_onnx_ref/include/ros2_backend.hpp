#pragma once

#include "robot_parameters.hpp"
#include "math_utils.hpp"
#include "policy_parameters.hpp"
#include <rclcpp/rclcpp.hpp>
#include <noetix_msgs/msg/low_cmd.hpp>
#include <noetix_msgs/msg/low_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace bumi3 {
// 仿真和实机共用 Noetix ROS2 消息；通信后端不依赖 ONNX 或观测模块。
class RosBackend {
 public:
  explicit RosBackend(const Json& config)
      : config_(config), settings_(config.at("ros2")), parameters_(PolicyParameters::Parse(config.at("contract"))),
        real_(config.at("mode") == "real") {
    sequence_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Time::now().time_since_epoch()).count()) + uint64_t(getpid());
    const auto domain = std::to_string(settings_.at("domain_id").get<int>());
    setenv("ROS_DOMAIN_ID", domain.c_str(), 1);
    rclcpp::init(0, nullptr, rclcpp::InitOptions{}, rclcpp::SignalHandlerOptions::None);
    node_ = std::make_shared<rclcpp::Node>("bumi3_policy");
    command_pub_ = node_->create_publisher<noetix_msgs::msg::LowCmd>(settings_.at("command_topic"), rclcpp::QoS(1).reliable());
    request_pub_ = node_->create_publisher<std_msgs::msg::String>(settings_.at("request_topic"), rclcpp::QoS(20).reliable());
    state_sub_ = node_->create_subscription<noetix_msgs::msg::LowState>(
        settings_.at("state_topic"), rclcpp::QoS(1).best_effort(),
        [this](noetix_msgs::msg::LowState::ConstSharedPtr message) { LowStateHandler(*message); });
    const auto rotation = config_.at("imu_to_base_wxyz").get<std::vector<double>>();
    if (rotation.size() != 4) throw std::runtime_error("imu_to_base_wxyz must have four values");
    imu_to_base_ = QuaternionFromWxyz(rotation.data());
    if (!real_) {
      reset_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(settings_.at("sim_reset_topic"), rclcpp::QoS(1).reliable());
      reference_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(settings_.at("sim_reference_topic"), rclcpp::QoS(1).reliable());
      sim_sub_ = node_->create_subscription<std_msgs::msg::String>(
          settings_.at("sim_status_topic"), rclcpp::QoS(1).best_effort(),
          [this](std_msgs::msg::String::ConstSharedPtr message) { ReceiveSimulationStatus(message->data); });
    }
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    spin_ = std::thread([this] {
      try { executor_->spin(); }
      catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        fault_ = error.what();
        valid_ = false;
      }
    });
  }

  ~RosBackend() {
    try { Disable(false); }
    catch (const std::exception& error) { std::cerr << "BUMI3_STOP_ERROR: " << error.what() << '\n'; }
    executor_->cancel();
    if (spin_.joinable()) spin_.join();
    executor_->remove_node(node_);
    rclcpp::shutdown();
  }

  RobotState GetState() {
    std::lock_guard<std::mutex> lock(mutex_);
    const double timeout = settings_.at("state_timeout");
    if (!valid_ || std::chrono::duration<double>(Time::now() - received_).count() > timeout)
      throw std::runtime_error("ROS2 LowState unavailable/stale: " + fault_);
    if (!real_ && (!sim_matches_ || std::chrono::duration<double>(Time::now() - sim_received_).count() > timeout))
      throw std::runtime_error("MuJoCo config/status mismatch; use matching --config");
    return state_;
  }

  RobotState WaitForState() {
    const auto deadline = Time::now() + std::chrono::seconds(5);
    std::string last_error;
    while (Time::now() < deadline && !stopping) {
      try { return GetState(); }
      catch (const std::exception& error) { last_error = error.what(); }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // 保留实际校验失败原因与接收数量，区分 DDS 未送达和已收到但电机、IMU 状态不合格。
    std::lock_guard<std::mutex> lock(mutex_);
    throw std::runtime_error("No valid Noetix ROS2 state; topic=" + settings_.at("state_topic").get<std::string>() +
                             " messages=" + std::to_string(state_messages_) + "; " + last_error);
  }

  std::vector<int> ReadKeys() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto keys = std::move(keys_);
    keys_.clear();
    return keys;
  }

  void Enable() {
    const auto measured = GetState();
    const auto hold = parameters_.CreatePositionCommand(measured.q.data());
    const auto deadline = Time::now() + std::chrono::seconds(2);
    while (Time::now() < deadline && !stopping) {
      SendRequest("enable");
      WriteTarget(hold);
      try { if (GetState().enabled) return; } catch (const std::exception&) {}
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("ROS2 enable not acknowledged");
  }

  void Disable(bool require_ack = true) {
    noetix_msgs::msg::LowCmd zero;
    zero.stamp = node_->now();
    zero.sequence = ++sequence_;
    for (const auto& mapping : config_.at("joint_mappings")) {
      noetix_msgs::msg::MotorCmd motor;
      motor.name = mapping.at("actuator");
      zero.motor_cmd.push_back(motor);
    }
    const auto requested = Time::now();
    const auto deadline = requested + std::chrono::milliseconds(300);
    do {
      command_pub_->publish(zero);
      SendRequest("disable");
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      std::lock_guard<std::mutex> lock(mutex_);
      if (status_received_ >= requested && !state_.enabled) return;
    } while (Time::now() < deadline);
    std::cerr << "BUMI3_DISABLE_UNCONFIRMED: no ROS2 disabled acknowledgment; environment watchdog remains active\n";
    if (require_ack) throw std::runtime_error("ROS2 disable not acknowledged");
  }

  void Reset(const std::vector<double>& qpos, const std::vector<double>& qvel) {
    if (real_) throw std::runtime_error("Hardware cannot teleport");
    if (qpos.size() != 28 || qvel.size() != 27) throw std::runtime_error("Reset dimensions must be 28/27");
    std_msgs::msg::Float64MultiArray message;
    const auto sequence = ++sequence_;
    if (sequence >= (uint64_t(1) << 53)) throw std::runtime_error("Reset sequence exceeds float64 integer range");
    message.data = {config_.at("contract_id").get<double>(), double(sequence)};
    message.data.insert(message.data.end(), qpos.begin(), qpos.end());
    message.data.insert(message.data.end(), qvel.begin(), qvel.end());
    const auto deadline = Time::now() + std::chrono::seconds(2);
    while (Time::now() < deadline && !stopping) {
      reset_pub_->publish(message);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        // 必须等 reset 对应的新反馈，防止用重置前的机器人状态初始化观测历史。
        if (ack_ == sequence && valid_ && state_sequence_ >= ack_state_sequence_) return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("ROS2 simulation reset not acknowledged");
  }

  void WriteTarget(const MotorCommand& command) {
    if (real_) GetState();
    noetix_msgs::msg::LowCmd message;
    message.stamp = node_->now();
    message.sequence = ++sequence_;
    for (int i = 0; i < BUMI3_NUM_MOTOR; ++i) {
      const auto& mapping = config_.at("joint_mappings")[i];
      double q = command.q_target[i];
      if (!std::isfinite(q) || !std::isfinite(command.dq_target[i]) || !std::isfinite(command.tau_ff[i]) ||
          !std::isfinite(command.kp[i]) || !std::isfinite(command.kd[i]) || command.kp[i] < 0 || command.kd[i] < 0)
        throw std::runtime_error("Invalid motor command");
      if (real_) q = std::clamp(q, config_.at("joint_min")[i].get<double>(), config_.at("joint_max")[i].get<double>());
      noetix_msgs::msg::MotorCmd motor;
      motor.name = mapping.at("actuator");
      motor.q = float((q - mapping.at("offset").get<double>()) * mapping.at("direction").get<double>());
      // 与 G1 的 MotorCommand 字段对应；方向只作用于运动量和力矩，增益不改变符号。
      motor.dq = command.dq_target[i] * mapping.at("direction").get<double>();
      motor.tau = command.tau_ff[i] * mapping.at("direction").get<double>();
      motor.kp = command.kp[i];
      motor.kd = command.kd[i];
      message.motor_cmd.push_back(motor);
    }
    command_pub_->publish(message);
  }

  // 只发送环境端的参考影子；动作缩放已由 CreatePolicyCommand 完成，后端不再参与策略处理。
  void WriteReference(const std::vector<double>& reference) {
    if (!reference_pub_) return;
    if (!reference.empty() && reference.size() != 28) throw std::runtime_error("Reference qpos requires 28 values");
    std_msgs::msg::Float64MultiArray message;
    message.data = reference;
    reference_pub_->publish(message);
  }

 private:
  void SendRequest(const std::string& value) {
    std_msgs::msg::String message;
    message.data = value;
    request_pub_->publish(message);
  }

  void ReceiveSimulationStatus(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
      const auto status = Json::parse(message);
      sim_matches_ = status.at("contract_id") == config_.at("contract_id");
      sim_received_ = Time::now();
      ack_ = status.at("reset_sequence").get<uint64_t>();
      ack_state_sequence_ = status.at("state_sequence").get<uint64_t>();
      const auto sequence = status.at("key_sequence").get<uint64_t>();
      if (seen_key_ && sequence != key_sequence_) {
        const int key = status.at("key");
        if (key) keys_.push_back(key);
      }
      key_sequence_ = sequence;
      seen_key_ = true;
    } catch (const std::exception& error) {
      sim_matches_ = false;
      fault_ = error.what();
    }
  }

  void LowStateHandler(const noetix_msgs::msg::LowState& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++state_messages_;
    state_.enabled = message.status.enabled;
    status_received_ = Time::now();
    try {
      if (message.motor_state.size() != 21)
        throw std::runtime_error("Expected 21 motor states, received " + std::to_string(message.motor_state.size()));
      if (!message.status.connected) throw std::runtime_error("Robot status.connected=false");
      if (message.status.fault)
        throw std::runtime_error("Robot status.fault=" + std::to_string(message.status.fault));
      if (!message.imu.valid) throw std::runtime_error("IMU status.valid=false");
      std::unordered_map<std::string, const noetix_msgs::msg::MotorState*> motors;
      for (const auto& motor : message.motor_state)
        if (!motors.emplace(motor.name, &motor).second) throw std::runtime_error("Duplicate actuator feedback: " + motor.name);
      RobotState state;
      for (int i = 0; i < 21; ++i) {
        const auto& mapping = config_.at("joint_mappings")[i];
        const auto name = mapping.at("actuator").get<std::string>();
        const auto found = motors.find(name);
        if (found == motors.end()) throw std::runtime_error("Missing actuator feedback: " + name);
        const auto& motor = *found->second;
        // 反馈异常保留执行器名称与错误码，便于区分设备故障和策略观测问题。
        if (motor.error) throw std::runtime_error("Motor fault " + name + ": code=" + std::to_string(motor.error));
        if (!std::isfinite(motor.q) || !std::isfinite(motor.dq)) throw std::runtime_error("Nonfinite motor feedback: " + name);
        state.q[i] = mapping.at("direction").get<double>() * motor.q + mapping.at("offset").get<double>();
        state.dq[i] = mapping.at("direction").get<double>() * motor.dq;
      }
      std::array<double, 4> rotation{};
      for (int i = 0; i < 4; ++i) rotation[i] = message.imu.quaternion[i];
      state.root = QuaternionFromWxyz(rotation.data()) * imu_to_base_.conjugate();
      Eigen::Vector3d angular;
      for (int i = 0; i < 3; ++i) angular[i] = message.imu.gyroscope[i];
      if (!angular.allFinite()) throw std::runtime_error("Nonfinite gyro");
      angular = imu_to_base_ * angular;
      for (int i = 0; i < 3; ++i) state.gyro[i] = angular[i];
      state.enabled = message.status.enabled;
      state_ = state;
      fault_.clear();
      if (!valid_ || state_sequence_ != message.sequence) {
        received_ = Time::now();
        valid_ = true;
        state_sequence_ = message.sequence;
      }
    } catch (const std::exception& error) {
      fault_ = error.what();
      valid_ = false;
    }
  }

  Json config_, settings_;
  PolicyParameters parameters_;
  bool real_;
  Quat imu_to_base_ = Quat::Identity();
  std::mutex mutex_;
  RobotState state_;
  bool valid_ = false, sim_matches_ = false, seen_key_ = false;
  Time::time_point received_{}, sim_received_{}, status_received_{};
  std::string fault_ = "No LowState message received";
  uint64_t state_messages_ = 0;
  uint64_t sequence_ = 0, state_sequence_ = 0, ack_ = 0, ack_state_sequence_ = 0, key_sequence_ = 0;
  std::vector<int> keys_;
  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_;
  rclcpp::Publisher<noetix_msgs::msg::LowCmd>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr request_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr reset_pub_, reference_pub_;
  rclcpp::Subscription<noetix_msgs::msg::LowState>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sim_sub_;
};
}  // BUMI3 ROS2 通信后端的命名空间。
