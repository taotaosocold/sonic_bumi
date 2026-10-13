#include "noetix_bridge/io_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace noetix_bridge {

namespace {
std::uint64_t steadyNowNs()
{
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace

IoLoop::IoLoop(std::shared_ptr<noetix::RobotInterface> robot,
               std::vector<std::string> actuator_names,
               double command_timeout,
               std::vector<double> effort_limits)
: robot_(std::move(robot)),
  actuator_names_(std::move(actuator_names)),
  dof_(actuator_names_.size()),
  command_timeout_(command_timeout),
  effort_limits_(std::move(effort_limits))
{
  if (!std::isfinite(command_timeout_) || command_timeout_ <= 0)
    throw std::invalid_argument("command_timeout must be finite and positive");
  if (!effort_limits_.empty() && effort_limits_.size() != dof_)
    throw std::invalid_argument("effort_limits must match actuator order");
  for (double limit : effort_limits_)
    if (!std::isfinite(limit) || limit <= 0)
      throw std::invalid_argument("effort_limits must be finite and positive");
  actuator_index_.reserve(dof_);
  for (std::size_t i = 0; i < dof_; ++i) {
    actuator_index_.emplace(actuator_names_[i], i);
  }

  state_.resize(dof_);
  for (std::size_t i = 0; i < dof_; ++i) {
    state_.motor_state[i].name = actuator_names_[i];
  }

  zero_command_.resize(dof_);
  command_snapshot_.resize(dof_);
  limited_command_.resize(dof_);
  initializeCommand(zero_command_);
  initializeCommand(command_snapshot_);
  initializeCommand(limited_command_);
  zeroCommand(zero_command_);
  zeroCommand(command_snapshot_);

  state_buffer_.initialize(state_);
  command_buffer_.initialize(zero_command_);
}

IoLoop::~IoLoop()
{
  stop();
}

void IoLoop::initializeCommand(noetix::LowCmd& cmd) const
{
  if (cmd.motor_cmd.size() != dof_) {
    cmd.resize(dof_);
  }
  for (std::size_t i = 0; i < dof_; ++i) {
    cmd.motor_cmd[i].name = actuator_names_[i];
  }
}

void IoLoop::zeroCommand(noetix::LowCmd& cmd) const noexcept
{
  for (auto& c : cmd.motor_cmd) {
    c.q = 0.0F;
    c.dq = 0.0F;
    c.tau = 0.0F;
    c.kp = 0.0F;
    c.kd = 0.0F;
  }
}

bool IoLoop::finiteCommand(const noetix::LowCmd& cmd) const noexcept
{
  for (const auto& c : cmd.motor_cmd) {
    if (!std::isfinite(c.q) || !std::isfinite(c.dq) ||
        !std::isfinite(c.tau) || !std::isfinite(c.kp) ||
        !std::isfinite(c.kd) || c.kp < 0 || c.kd < 0) {
      return false;
    }
  }
  return true;
}

bool IoLoop::start(double frequency,
                   const noetix::PeriodicThreadOptions& options)
{
  if (running_.exchange(true, std::memory_order_acq_rel)) {
    return true;
  }
  if (frequency <= 0.0) {
    running_.store(false, std::memory_order_release);
    return false;
  }

  const auto period_ns = static_cast<std::uint64_t>(1.0e9 / frequency);
  if (!io_thread_.start(
          std::chrono::nanoseconds(period_ns),
          [this]() { cycle(); },
          options)) {
    running_.store(false, std::memory_order_release);
    return false;
  }
  return true;
}

void IoLoop::stop()
{
  if (!running_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  io_thread_.stop();
  disableDrive();
}

bool IoLoop::setCommand(const noetix::LowCmd& command)
{
  if (!drive_enabled_.load(std::memory_order_acquire)) {
    return false;
  }
  if (command.motor_cmd.size() != dof_) {
    return false;
  }

  noetix::LowCmd ordered;
  ordered.resize(dof_);
  ordered.sequence = command.sequence;
  ordered.timestamp_ns = command.timestamp_ns;
  initializeCommand(ordered);

  std::vector<std::uint8_t> seen(dof_, 0U);
  for (const auto& src : command.motor_cmd) {
    const auto it = actuator_index_.find(src.name);
    if (src.name.empty() || it == actuator_index_.end()) {
      return false;
    }
    const std::size_t index = it->second;
    if (seen[index] != 0U) {
      return false;
    }
    seen[index] = 1U;

    auto& dst = ordered.motor_cmd[index];
    dst.q = src.q;
    dst.dq = src.dq;
    dst.tau = src.tau;
    dst.kp = src.kp;
    dst.kd = src.kd;
  }

  if (!finiteCommand(ordered)) {
    return false;
  }

  if (!command_buffer_.write(ordered)) {
    return false;
  }
  last_command_ns_.store(steadyNowNs(), std::memory_order_release);
  command_valid_.store(true, std::memory_order_release);
  return true;
}

void IoLoop::clearCommand() noexcept
{
  command_valid_.store(false, std::memory_order_release);
}

bool IoLoop::hasCommand() const noexcept
{
  return command_valid_.load(std::memory_order_acquire);
}

bool IoLoop::enableDrive()
{
  clearCommand();
  std::lock_guard<std::mutex> lock(robot_mutex_);
  if (!robot_) {
    return false;
  }
  if (!state_.status.connected || state_.status.fault || !state_.imu.valid ||
      state_.sequence == 0 || steadyNowNs() - state_.timestamp_ns > 100000000U)
    return false;
  double quat_norm = 0;
  for (float value : state_.imu.quaternion) {
    if (!std::isfinite(value)) return false;
    quat_norm += value * value;
  }
  if (quat_norm < 1e-8) return false;
  for (float value : state_.imu.gyroscope) if (!std::isfinite(value)) return false;
  for (const auto& motor : state_.motor_state)
    if (!std::isfinite(motor.q) || !std::isfinite(motor.dq) || motor.error) return false;
  const bool ok = robot_->enableActuators();
  const bool enabled = ok && robot_->actuatorsEnabled();
  last_command_ns_.store(steadyNowNs(), std::memory_order_release);
  drive_enabled_.store(enabled, std::memory_order_release);
  return enabled;
}

bool IoLoop::disableDrive()
{
  clearCommand();
  drive_enabled_.store(false, std::memory_order_release);

  std::lock_guard<std::mutex> lock(robot_mutex_);
  if (!robot_) {
    return false;
  }

  // One explicit zero packet before the vendor disable request.  This is not
  // a safety controller; it simply guarantees that DISABLE has no stale MIT
  // gains left in the hardware command buffer.
  disableLocked();
  return !robot_->actuatorsEnabled();
}

void IoLoop::disableLocked()
{
  clearCommand();
  drive_enabled_.store(false, std::memory_order_release);
  robot_->write(zero_command_);
  const bool ok = robot_->disableActuators();
  if (!ok || robot_->actuatorsEnabled())
    std::cerr << "NOETIX_DISABLE_FAILED: hardware did not confirm disable" << std::endl;
}

bool IoLoop::driveEnabled() const noexcept
{
  return drive_enabled_.load(std::memory_order_acquire);
}

bool IoLoop::getLatestState(noetix::LowState& state) const
{
  return state_buffer_.read(state);
}

noetix::PeriodicThreadStats IoLoop::stats() const noexcept
{
  return io_thread_.stats();
}

std::uint64_t IoLoop::readErrors() const noexcept
{
  return read_errors_.load(std::memory_order_acquire);
}

std::uint64_t IoLoop::writeErrors() const noexcept
{
  return write_errors_.load(std::memory_order_acquire);
}

void IoLoop::cycle()
{
  std::lock_guard<std::mutex> lock(robot_mutex_);
  if (!robot_) {
    return;
  }

  bool healthy = robot_->read(state_);
  if (healthy) {
    state_.timestamp_ns = steadyNowNs();
    state_.sequence = ++state_sequence_;
    state_buffer_.write(state_);
  } else {
    read_errors_.fetch_add(1U, std::memory_order_relaxed);
  }
  healthy = healthy && state_.status.connected && state_.status.fault == 0 && state_.imu.valid;
  double quat_norm = 0;
  for (float value : state_.imu.quaternion) { healthy = healthy && std::isfinite(value); quat_norm += value * value; }
  for (float value : state_.imu.gyroscope) healthy = healthy && std::isfinite(value);
  healthy = healthy && quat_norm > 1e-8;
  for (const auto& motor : state_.motor_state)
    healthy = healthy && std::isfinite(motor.q) && std::isfinite(motor.dq) && motor.error == 0;
  if (drive_enabled_.load(std::memory_order_acquire)) {
    const double age = double(steadyNowNs() - last_command_ns_.load(std::memory_order_acquire)) / 1e9;
    if (!healthy || age > command_timeout_) {
      std::cerr << "NOETIX_DRIVE_STOP: " << (!healthy ? "invalid feedback/IMU" : "command timeout") << std::endl;
      disableLocked();
    }
  }

  const noetix::LowCmd* output = &zero_command_;
  if (drive_enabled_.load(std::memory_order_acquire) &&
      command_valid_.load(std::memory_order_acquire)) {
    if (command_buffer_.read(command_snapshot_)) {
      output = &command_snapshot_;
    } else {
      // Keep the previous snapshot.  Do not synthesize a different command.
      output = &command_snapshot_;
    }
  }

  if (output != &zero_command_ && !effort_limits_.empty()) {
    limited_command_.sequence = output->sequence;
    for (std::size_t i = 0; i < dof_; ++i) {
      const auto& cmd = output->motor_cmd[i];
      auto& dst = limited_command_.motor_cmd[i];
      dst.q = dst.dq = dst.kp = dst.kd = 0;
      dst.tau = static_cast<float>(std::clamp(
          double(cmd.tau) + cmd.kp * (cmd.q - state_.motor_state[i].q)
              + cmd.kd * (cmd.dq - state_.motor_state[i].dq),
          -effort_limits_[i], effort_limits_[i]));
    }
    output = &limited_command_;
  }

  if (!robot_->write(*output)) {
    write_errors_.fetch_add(1U, std::memory_order_relaxed);
    if (drive_enabled_.load(std::memory_order_acquire)) disableLocked();
  }
}

}  // namespace noetix_bridge
