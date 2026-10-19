#pragma once

#include "robot_parameters.hpp"

namespace bumi3 {
// 四元数归一化与水平朝向提取供观测、参考读取和 ROS 状态转换共用。
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

}  // BUMI3 姿态数学工具的命名空间。
