#pragma once

#include "policy_parameters.hpp"
#include <deque>

namespace bumi3 {
// 对应 G1 的 StateLogger 历史职责：每个控制周期保存同一份反馈及上一步动作。
// 此处只提供模型所需的内存历史，不包含磁盘录制、手柄或其他尚未接入的功能。
class StateLogger {
 public:
  struct Entry {
    Quat base_quat = Quat::Identity();
    std::array<double, 3> base_ang_vel{}, gravity_dir{};
    std::array<double, BUMI3_NUM_MOTOR> body_q{}, body_dq{}, last_action{};
  };

  explicit StateLogger(const PolicyParameters& parameters)
      : defaults_(parameters.default_angles), control_dt_(parameters.control_dt) {}

  void ResetHistory(const RobotState& state) {
    const std::vector<double> zero_action(BUMI3_NUM_MOTOR, 0);
    entries_.assign(HISTORY_LENGTH, MakeEntry(state, zero_action));
  }

  bool LogFullState(const RobotState& state, const std::vector<double>& last_action) {
    if (last_action.size() != BUMI3_NUM_MOTOR) return false;
    entries_.push_back(MakeEntry(state, last_action));
    if (entries_.size() > HISTORY_LENGTH) entries_.pop_front();
    return true;
  }

  // 历史按固定 50 Hz 控制采样；采样步距只选择已有帧，不根据抖动的墙钟时间重新插值。
  // 与 G1 一样可指定从旧到新，启动时缺少的历史由 ResetHistory 的首帧重复值补齐。
  std::vector<Entry> GetLatest(std::size_t count, double sample_dt, bool newest_first = true) const {
    if (entries_.empty() || !count || !std::isfinite(sample_dt) || sample_dt <= 0) return {};
    const int stride = int(std::round(sample_dt / control_dt_));
    if (stride < 1 || std::abs(sample_dt - stride * control_dt_) > 1e-8) return {};
    std::vector<Entry> history;
    history.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const int index = std::max(0, int(entries_.size()) - 1 - int(i) * stride);
      history.push_back(entries_[index]);
    }
    if (!newest_first) std::reverse(history.begin(), history.end());
    return history;
  }

 private:
  Entry MakeEntry(const RobotState& state, const std::vector<double>& last_action) const {
    Entry entry;
    entry.base_quat = state.root;
    entry.base_ang_vel = state.gyro;
    const Eigen::Vector3d gravity = state.root.conjugate() * Eigen::Vector3d(0, 0, -1);
    for (int i = 0; i < 3; ++i) entry.gravity_dir[i] = gravity[i];
    for (int i = 0; i < BUMI3_NUM_MOTOR; ++i) {
      entry.body_q[i] = state.q[i] - defaults_[i];
      entry.body_dq[i] = state.dq[i];
      entry.last_action[i] = last_action[i];
    }
    return entry;
  }

  std::array<double, BUMI3_NUM_MOTOR> defaults_;
  double control_dt_;
  std::deque<Entry> entries_;
};
}  // BUMI3 策略反馈历史的命名空间。
