#pragma once

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace bumi3 {
using Json = nlohmann::json;
using Time = std::chrono::steady_clock;
using Quat = Eigen::Quaterniond;

// 信号处理器只写信号安全的标志；禁用电机和释放通信资源由正常控制流程完成。
inline volatile std::sig_atomic_t stopping = 0;
inline void signalHandler(int) { stopping = 1; }

inline Quat QuaternionFromWxyz(const double* values) {
  Quat result(values[0], values[1], values[2], values[3]);
  if (!result.coeffs().allFinite() || result.norm() < 1e-8)
    throw std::runtime_error("Invalid quaternion");
  return result.normalized();
}

// 参考动作只对齐水平朝向，不能将机器人的实际俯仰和横滚一并消去。
inline Quat HeadingQuaternion(const Quat& rotation) {
  const double yaw = std::atan2(2 * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
                                1 - 2 * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));
  return Quat(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
}

// 反馈已转为策略关节顺序和基座坐标系； SDK 的标定、并联踝转换留在环境端。
struct State {
  std::array<double, 21> q{}, dq{};
  std::array<double, 3> gyro{};
  Quat root = Quat::Identity();
  bool enabled = false;
};

// 完整 SMPL 参考由十帧局部人体关节和根方向组成；离线和在线输入共用观测布局。
struct PoseWindow {
  std::array<double, 720> joints{};
  std::array<Quat, 10> roots;
  int64_t last = -1;
  PoseWindow() { roots.fill(Quat::Identity()); }
};
}  // BUMI3 状态与姿态工具的命名空间。
