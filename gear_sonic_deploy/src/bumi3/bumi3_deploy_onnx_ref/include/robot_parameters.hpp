#pragma once

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace bumi3 {
// 21 个受控关节不包含浮动基座；所有策略命令按训练顺序保存，SDK 顺序由名称映射处理。
inline constexpr int BUMI3_NUM_MOTOR = 21;
using Json = nlohmann::json;
using Time = std::chrono::steady_clock;
using Quat = Eigen::Quaterniond;

// 信号处理器只写信号安全的标志；禁用电机和释放通信资源由正常控制流程完成。
inline volatile std::sig_atomic_t stopping = 0;
inline void signalHandler(int) { stopping = 1; }

// 反馈已转为策略关节顺序和基座坐标系； SDK 的标定、并联踝转换留在环境端。
struct RobotState {
  std::array<double, BUMI3_NUM_MOTOR> q{}, dq{};
  std::array<double, 3> gyro{};
  Quat root = Quat::Identity();
  bool enabled = false;
};

// 沿用 G1 的命令字段名称；每次下发完整目标、增益和前馈，参考影子仅供 MuJoCo 显示。
struct MotorCommand {
  std::array<double, BUMI3_NUM_MOTOR> q_target{}, dq_target{}, kp{}, kd{}, tau_ff{};
  std::vector<double> reference_qpos;
};

// 操作状态与 G1 分工相同：输入阶段设置标志，由控制状态机决定何时接管与播放。
struct OperatorState {
  bool stop = false, start = false, play = false;
};

// BUMI3 的朝向基准同时保存机器人和参考首帧，保持既有水平朝向对齐方式。
struct HeadingState {
  Quat init_base_quat = Quat::Identity();
  Quat init_ref_data_root_rot = Quat::Identity();
};

// 完整 SMPL 参考由十帧局部人体关节和根方向组成；离线和在线输入共用观测布局。
struct PoseWindow {
  std::array<double, 720> joints{};
  std::array<Quat, 10> roots;
  int64_t last = -1;
  PoseWindow() { roots.fill(Quat::Identity()); }
};
}  // BUMI3 状态与姿态工具的命名空间。
