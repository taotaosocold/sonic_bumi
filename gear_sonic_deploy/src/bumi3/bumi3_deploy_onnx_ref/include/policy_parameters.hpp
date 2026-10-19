#pragma once

#include "robot_parameters.hpp"
#include <string>

namespace bumi3 {
// 与 G1 的 policy_parameters.hpp 对应，集中说明策略维度与动作参数；
// 数值从 BUMI3 训练配置解析后的契约读取，避免用 G1 常量覆盖本机器人训练参数。
inline constexpr int HISTORY_LENGTH = 10;
inline constexpr int TOKEN_DIM = 64;
inline constexpr int ROBOT_REFERENCE_DIM = 480;
inline constexpr int SMPL_REFERENCE_DIM = 780;
inline constexpr int PROPRIOCEPTION_DIM = 690;

struct PolicyParameters {
  std::vector<std::string> policy_joint_names;
  std::array<double, BUMI3_NUM_MOTOR> default_angles{}, action_scale{}, kps{}, kds{}, effort_limits{};
  double control_dt = 0.02, action_clip = 20.0;
  int history_length = HISTORY_LENGTH, future_frames = 10, future_stride = 5;

  // JSON 仅作为启动配置边界；主循环和硬件写入使用下面这些类型明确的字段。
  static PolicyParameters Parse(const Json& contract) {
    PolicyParameters result;
    result.policy_joint_names = contract.at("policy_joint_names").get<std::vector<std::string>>();
    result.default_angles = contract.at("default_policy").get<decltype(result.default_angles)>();
    result.action_scale = contract.at("action_scale_policy").get<decltype(result.action_scale)>();
    result.kps = contract.at("stiffness_policy").get<decltype(result.kps)>();
    result.kds = contract.at("damping_policy").get<decltype(result.kds)>();
    result.effort_limits = contract.at("effort_policy").get<decltype(result.effort_limits)>();
    result.control_dt = contract.at("control_dt").get<double>();
    result.action_clip = contract.at("action_clip").get<double>();
    result.history_length = contract.at("history_length").get<int>();
    result.future_frames = contract.at("future_frames").get<int>();
    result.future_stride = contract.at("future_stride").get<int>();
    if (result.policy_joint_names.size() != BUMI3_NUM_MOTOR ||
        std::abs(result.control_dt - 0.02) > 1e-8 || result.history_length != HISTORY_LENGTH ||
        result.future_frames != 10 || result.future_stride != 5)
      throw std::runtime_error("Native BUMI3 requires 21 joints, 50 Hz and the trained ten-frame layout");
    return result;
  }

  // positions 指向已按策略顺序排列的 21 个角度；速度和前馈保持原实现的零值。
  MotorCommand CreatePositionCommand(const double* positions) const {
    MotorCommand command;
    std::copy_n(positions, BUMI3_NUM_MOTOR, command.q_target.begin());
    command.kp = kps;
    command.kd = kds;
    return command;
  }
};
}  // BUMI3 策略参数的命名空间。
