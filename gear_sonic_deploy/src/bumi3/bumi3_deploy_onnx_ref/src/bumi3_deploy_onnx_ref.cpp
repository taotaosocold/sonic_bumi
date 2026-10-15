#include "deploy_config.hpp"
#include "input_interface.hpp"
#include "motion_data_reader.hpp"
#include "policy_engine.hpp"
#include "ros2_backend.hpp"
#include "state_logger.hpp"
#include "zmq_input.hpp"
#include <functional>
#include <optional>

using namespace bumi3;

// 保留 G1 的部署组织方式：状态机、参考观测采集、历史观测采集、策略命令与命令下发。
// BUMI3 的联合 ONNX 在模型内部编码和解码，外部仍必须按训练契约拼接参考与历史。
class Bumi3Deploy {
 public:
  enum class ProgramState { INIT, WAIT_FOR_CONTROL, CONTROL, STOPPED };

  explicit Bumi3Deploy(Json config) : config_(std::move(config)), contract_(config_.at("contract")) {}
  ~Bumi3Deploy() { Stop(false); }

  void Initialize() {
    real_ = config_.at("mode") == "real";
    has_offline_ = !config_.at("motions").empty();
    has_live_ = config_.at("input_type") == "zmq";
    offline_smpl_ = config_.at("encoder") == "smpl";
    live_ = !has_offline_;
    current_encoder_mode_ = live_ || offline_smpl_ ? 1 : 0;
    control_dt_ = contract_.at("control_dt");
    if (std::abs(control_dt_ - 0.02) > 1e-8 || contract_.at("policy_joint_names").size() != 21 ||
        contract_.at("history_length") != 10 || contract_.at("future_frames") != 10 || contract_.at("future_stride") != 5)
      throw std::runtime_error("Native BUMI3 requires 21 joints, 50 Hz and the trained ten-frame layout");

    // 与 G1 一样先预加载动作，再初始化模型与观测函数；控制循环和切换动作不读取 CSV。
    if (has_offline_) {
      if (!motion_reader_.ReadFromCSV(config_.at("motions"), offline_smpl_,
                                      config_.at("reference_contract_id"), contract_.at("policy_joint_names")))
        throw std::runtime_error("No reference motions loaded");
      current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
    }
    const Json inference = config_.value("inference", Json{{"provider", config_.at("provider")}});
    if (has_offline_)
      offline_policy_ = std::make_unique<PolicyEngine>(config_.at("policy"), offline_smpl_ ? 1470 : 1170, inference);
    if (has_live_) live_policy_ = std::make_unique<PolicyEngine>(config_.at("smpl_policy"), 1470, inference);
    state_logger_ = std::make_unique<StateLogger>(contract_);
    last_action.assign(21, 0);
    obs_config_ = config_.value("observations", DefaultObservations());
    encoder_config_ = config_.value("encoder_config", DefaultEncoderConfig());
    InitializeObservationFunctions();
    if (config_.at("validate_only") == true) return;

    backend_ = std::make_unique<RosBackend>(config_);
    used_low_state_data_ = backend_->WaitForState();
    // 初始化只确认反馈有效，不把电机去使能作为启动前置条件。
    // 真机仍等待操作员按 ]，再由 InitControl 请求使能并平滑进入参考起始姿态。
    input_interface_ = std::make_unique<KeyboardInput>();
    if (has_live_) zmq_input_ = std::make_unique<ZmqInput>(config_.at("teleop"));
    operator_state.play = config_.at("autoplay");
    reset_with_reference_velocity_ = operator_state.play;
    if (live_) {
      std::cout << "Waiting for ten-frame SMPL stream\n";
      frozen_pose_ = zmq_input_->WaitForPose(config_.at("teleop").at("startup_timeout"));
      live_pose_snapshot_ = frozen_pose_;
    }
    program_state_ = ProgramState::INIT;
    std::cout << "BUMI3_CONTROL_READY native_cpp: ] prepare-pose (real) / T start-policy/play-pause / P next / R restart / ENTER offline-live / O stop\n";
  }

  void Run() {
    Initialize();
    if (config_.at("validate_only") == true) {
      std::cout << "BUMI3_NATIVE_CONFIG_OK: policy/observations validated; no device connected\n";
      return;
    }
    const auto started = Time::now();
    while (!stopping && !operator_state.stop && program_state_ != ProgramState::STOPPED) {
      const auto tick = Time::now();
      if (!config_.at("duration").is_null() &&
          std::chrono::duration<double>(tick - started).count() >= config_.at("duration").get<double>()) break;
      Input();
      if (stopping || operator_state.stop) break;
      Control();
      LowCommandWriter();
      std::this_thread::sleep_until(tick + std::chrono::duration<double>(control_dt_));
    }
    Stop();
  }

  void Stop(bool require_ack = true) {
    if (!backend_ || program_state_ == ProgramState::STOPPED) return;
    program_state_ = ProgramState::STOPPED;
    operator_state.stop = true;
    enabled_ = false;
    if (require_ack) backend_->Disable();
    else {
      try { backend_->Disable(false); }
      catch (const std::exception& error) { std::cerr << "BUMI3_STOP_ERROR: " << error.what() << '\n'; }
    }
  }

 private:
  struct OperatorState {
    bool start = false, play = false, stop = false;
  } operator_state;

  using ObservationFunction = std::function<bool(std::vector<double>&, std::size_t)>;
  struct ObservationRegistry {
    std::string name;
    std::size_t dimension;
    ObservationFunction function;
  };
  struct ActiveObservation {
    std::string name;
    ObservationFunction function;
    std::size_t offset, dimension;
  };
  struct MotorCommand {
    // 目标关节仍按策略顺序保存，执行器名称、方向、限位和 PD 增益由 ROS2 后端处理。
    std::vector<double> q_target;
    std::vector<double> reference_qpos;
  };

  // 兼容先前的临时配置；新启动流程从 observation_config.yaml 读取同名注册项。
  static Json DefaultObservations() {
    Json result = Json::array();
    for (const auto* name : {"his_base_angular_velocity_10frame_step1", "his_body_joint_positions_10frame_step1",
                            "his_body_joint_velocities_10frame_step1", "his_last_actions_10frame_step1",
                            "his_gravity_dir_10frame_step1"})
      result.push_back(Json{{"name", name}, {"enabled", true}});
    return result;
  }

  static Json DefaultEncoderConfig() {
    const Json robot = {"motion_joint_positions_10frame_step5", "motion_joint_velocities_10frame_step5",
                        "motion_anchor_orientation_10frame_step5"};
    const Json smpl = {"smpl_joints_10frame_step1", "smpl_anchor_orientation_10frame_step1"};
    Json entries = Json::array();
    for (const auto& name : robot) entries.push_back(Json{{"name", name}, {"enabled", true}});
    for (const auto& name : smpl) entries.push_back(Json{{"name", name}, {"enabled", true}});
    return Json{{"dimension", 64}, {"encoder_observations", entries},
                {"encoder_modes", Json::array({Json{{"name", "robot"}, {"mode_id", 0}, {"required_observations", robot}},
                                                Json{{"name", "smpl"}, {"mode_id", 1}, {"required_observations", smpl}}})}};
  }

  // 参考观测与 G1 一样按帧、步距和输出偏移采集；所有采集函数使用本周期同一份参考快照。
  int GetFutureFrame(int frame_index, int step_size) const {
    if (live_) return operator_state.play ? frame_index * step_size : 0;
    const int frame = current_frame_ + (operator_state.play ? frame_index * step_size : 0);
    return config_.at("loop_motion") == true ? frame % current_motion_->timesteps
                                             : std::min(frame, current_motion_->timesteps - 1);
  }

  bool CheckObservationWindow(std::size_t size, std::size_t offset, int frames, int stride, int dimension) {
    if (frames <= 0 || frames > 10 || stride <= 0 || stride > 5 || offset > size ||
        std::size_t(frames * dimension) > size - offset) return false;
    const int required = (frames - 1) * stride + 1;
    saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, required);
    if (live_) return live_pose_snapshot_.last >= 0 && required <= 10;
    return current_motion_ && current_motion_->timesteps > 0;
  }

  bool GatherMotionJointPositionsMultiFrame(std::vector<double>& target_buffer, std::size_t offset,
                                            int num_frames = 10, int step_size = 5) {
    if (live_ || !CheckObservationWindow(target_buffer.size(), offset, num_frames, step_size, 21)) return false;
    for (int i = 0; i < num_frames; ++i) {
      const auto* positions = current_motion_->JointPositions(GetFutureFrame(i, step_size));
      std::copy_n(positions, 21, target_buffer.begin() + offset + i * 21);
    }
    return true;
  }

  bool GatherMotionJointVelocitiesMultiFrame(std::vector<double>& target_buffer, std::size_t offset,
                                             int num_frames = 10, int step_size = 5) {
    if (live_ || !CheckObservationWindow(target_buffer.size(), offset, num_frames, step_size, 21)) return false;
    for (int i = 0; i < num_frames; ++i) {
      const auto destination = target_buffer.begin() + offset + i * 21;
      if (operator_state.play) std::copy_n(current_motion_->JointVelocities(GetFutureFrame(i, step_size)), 21, destination);
      else std::fill_n(destination, 21, 0.0);
    }
    return true;
  }

  Quat GetReferenceQuaternion(int frame) const {
    if (live_) return live_pose_snapshot_.roots.at(frame);
    return QuaternionFromWxyz((current_encoder_mode_ == 1 ? current_motion_->SmplRootQuaternions(frame)
                                                         : current_motion_->BodyQuaternions(frame))[0].data());
  }

  // 使用 G1 的朝向重置和相对朝向计算分工；只有初始化或显式切换参考时更新基准。
  // 实时 SMPL 每周期变化的根朝向必须保留，不能每帧重新对齐而消除操作者的转身目标。
  void UpdateHeadingState() {
    if (!reinitialize_heading_) return;
    init_base_quat_ = used_low_state_data_.root;
    init_ref_data_root_rot_ = GetReferenceQuaternion(live_ ? 0 : current_frame_);
    reinitialize_heading_ = false;
  }

  Quat ComputeApplyDeltaHeading() const {
    return HeadingQuaternion(init_base_quat_) * HeadingQuaternion(init_ref_data_root_rot_).conjugate();
  }

  // 函数名沿用 G1 的 MutiFrame 拼写，含义仍为多帧采集；不引入未训练的 heading-only 模式。
  bool GatherMotionAnchorOrientationMutiFrame(std::vector<double>& target_buffer, std::size_t offset,
                                              int num_frames = 10, int step_size = 5) {
    if (!CheckObservationWindow(target_buffer.size(), offset, num_frames, step_size, 6)) return false;
    const auto alignment = ComputeApplyDeltaHeading();
    for (int i = 0; i < num_frames; ++i) {
      const auto relative = used_low_state_data_.root.conjugate() * alignment * GetReferenceQuaternion(GetFutureFrame(i, step_size));
      const auto matrix = relative.normalized().toRotationMatrix();
      // 训练采用矩阵前两列的行优先展平，顺序为 r00,r01,r10,r11,r20,r21。
      for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 2; ++column)
          target_buffer[offset + i * 6 + row * 2 + column] = matrix(row, column);
    }
    return true;
  }

  bool GatherMotionSmplJointsMultiFrame(std::vector<double>& target_buffer, std::size_t offset,
                                        int num_frames = 10, int step_size = 1) {
    if (current_encoder_mode_ != 1 || !CheckObservationWindow(target_buffer.size(), offset, num_frames, step_size, 72)) return false;
    for (int i = 0; i < num_frames; ++i) {
      const int frame = GetFutureFrame(i, step_size);
      if (live_) std::copy_n(live_pose_snapshot_.joints.begin() + frame * 72, 72, target_buffer.begin() + offset + i * 72);
      else {
        const auto* joints = current_motion_->SmplJoints(frame);
        for (int joint = 0; joint < 24; ++joint)
          for (int axis = 0; axis < 3; ++axis)
            target_buffer[offset + i * 72 + joint * 3 + axis] = joints[joint][axis];
      }
    }
    return true;
  }

  template <std::size_t Dimension>
  bool GatherHistory(std::vector<double>& target_buffer, std::size_t offset, int num_frames, int step_size,
                     bool newest_first, std::array<double, Dimension> StateLogger::Entry::* field) {
    if (num_frames < 1 || num_frames > 10 || step_size < 1 || step_size > 10 || offset > target_buffer.size() ||
        std::size_t(num_frames) * Dimension > target_buffer.size() - offset) return false;
    const auto history = state_logger_->GetLatest(num_frames, control_dt_ * step_size, newest_first);
    if (history.size() != std::size_t(num_frames)) return false;
    for (int i = 0; i < num_frames; ++i) {
      const auto& value = history[i].*field;
      std::copy(value.begin(), value.end(), target_buffer.begin() + offset + i * Dimension);
    }
    return true;
  }

  bool GatherHisBaseAngularVelocity(std::vector<double>& buffer, std::size_t offset, int num_frames = 10,
                                    int step_size = 1, bool newest_first = false) {
    return GatherHistory(buffer, offset, num_frames, step_size, newest_first, &StateLogger::Entry::base_ang_vel);
  }
  bool GatherHisBodyJointPositions(std::vector<double>& buffer, std::size_t offset, int num_frames = 10,
                                   int step_size = 1, bool newest_first = false) {
    return GatherHistory(buffer, offset, num_frames, step_size, newest_first, &StateLogger::Entry::body_q);
  }
  bool GatherHisBodyJointVelocities(std::vector<double>& buffer, std::size_t offset, int num_frames = 10,
                                    int step_size = 1, bool newest_first = false) {
    return GatherHistory(buffer, offset, num_frames, step_size, newest_first, &StateLogger::Entry::body_dq);
  }
  bool GatherHisLastActions(std::vector<double>& buffer, std::size_t offset, int num_frames = 10,
                            int step_size = 1, bool newest_first = false) {
    return GatherHistory(buffer, offset, num_frames, step_size, newest_first, &StateLogger::Entry::last_action);
  }
  bool GatherHisGravityDir(std::vector<double>& buffer, std::size_t offset, int num_frames = 10,
                           int step_size = 1, bool newest_first = false) {
    return GatherHistory(buffer, offset, num_frames, step_size, newest_first, &StateLogger::Entry::gravity_dir);
  }

  std::vector<ObservationRegistry> GetObservationRegistry() {
    return {{"motion_joint_positions_10frame_step5", 210, [this](auto& b, auto o) { return GatherMotionJointPositionsMultiFrame(b, o, 10, 5); }},
            {"motion_joint_velocities_10frame_step5", 210, [this](auto& b, auto o) { return GatherMotionJointVelocitiesMultiFrame(b, o, 10, 5); }},
            {"motion_anchor_orientation_10frame_step5", 60, [this](auto& b, auto o) { return GatherMotionAnchorOrientationMutiFrame(b, o, 10, 5); }},
            {"smpl_joints_10frame_step1", 720, [this](auto& b, auto o) { return GatherMotionSmplJointsMultiFrame(b, o, 10, 1); }},
            {"smpl_anchor_orientation_10frame_step1", 60, [this](auto& b, auto o) { return GatherMotionAnchorOrientationMutiFrame(b, o, 10, 1); }},
            {"his_base_angular_velocity_10frame_step1", 30, [this](auto& b, auto o) { return GatherHisBaseAngularVelocity(b, o); }},
            {"his_body_joint_positions_10frame_step1", 210, [this](auto& b, auto o) { return GatherHisBodyJointPositions(b, o); }},
            {"his_body_joint_velocities_10frame_step1", 210, [this](auto& b, auto o) { return GatherHisBodyJointVelocities(b, o); }},
            {"his_last_actions_10frame_step1", 210, [this](auto& b, auto o) { return GatherHisLastActions(b, o); }},
            {"his_gravity_dir_10frame_step1", 30, [this](auto& b, auto o) { return GatherHisGravityDir(b, o); }}};
  }

  void InitializeObservationFunctions() {
    const auto registry = GetObservationRegistry();
    active_obs_functions_.clear();
    active_encoder_obs_functions_.clear();
    const auto append = [&](const std::string& name, std::size_t offset, auto& active) {
      const auto found = std::find_if(registry.begin(), registry.end(), [&](const auto& entry) { return entry.name == name; });
      if (found == registry.end()) throw std::runtime_error("Unknown BUMI3 observation function: " + name);
      active.push_back(ActiveObservation{name, found->function, offset, found->dimension});
      return found->dimension;
    };
    const auto defaults = DefaultEncoderConfig();
    const auto& expected_mode = defaults.at("encoder_modes")[current_encoder_mode_];
    std::optional<Json> mode_config;
    for (const auto& mode : encoder_config_.at("encoder_modes")) {
      if (mode.at("name") != expected_mode.at("name")) continue;
      if (mode_config) throw std::runtime_error("Duplicate BUMI3 encoder mode");
      mode_config = mode;
    }
    if (!mode_config || mode_config->at("mode_id") != current_encoder_mode_ ||
        mode_config->at("required_observations") != expected_mode.at("required_observations") ||
        encoder_config_.at("dimension") != 64) throw std::runtime_error("Encoder layout differs from trained BUMI3 model");
    std::size_t encoder_offset = 0;
    for (const auto& name_json : mode_config->at("required_observations")) {
      const auto name = name_json.get<std::string>();
      int matches = 0;
      for (const auto& entry : encoder_config_.at("encoder_observations"))
        if (entry.at("name") == name && entry.value("enabled", true)) ++matches;
      if (matches != 1) throw std::runtime_error("Required encoder observation disabled or duplicated: " + name);
      encoder_offset += append(name, encoder_offset, active_encoder_obs_functions_);
    }
    // 联合模型布局为参考在前、历史在后，不能照搬 G1 独立 decoder 的 token+历史输入。
    std::size_t policy_offset = encoder_offset;
    Json active_names = Json::array(), expected_names = Json::array();
    for (const auto& entry : DefaultObservations()) expected_names.push_back(entry.at("name"));
    for (const auto& entry : obs_config_) {
      if (!entry.value("enabled", true)) continue;
      const auto name = entry.at("name").get<std::string>();
      active_names.push_back(name);
      policy_offset += append(name, policy_offset, active_obs_functions_);
    }
    if (active_names != expected_names || encoder_offset != std::size_t(current_encoder_mode_ == 1 ? 780 : 480) ||
        policy_offset - encoder_offset != 690) throw std::runtime_error("Observation layout differs from trained BUMI3 model");
    encoder_obs_buffer_.assign(encoder_offset, 0);
    obs_buffer_.assign(policy_offset, 0);
    const auto& policy = live_ ? live_policy_ : offline_policy_;
    if (!policy || policy->GetInputBuffer().size() != policy_offset)
      throw std::runtime_error("Observation buffer differs from selected policy input");
    std::cout << "BUMI3_OBSERVATIONS encoder=" << expected_mode.at("name").get<std::string>()
              << " reference=" << encoder_offset << " history=690 total=" << policy_offset << '\n';
  }

  bool GatherEncoderObservations() {
    std::fill(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), 0);
    saved_frame_for_observation_window_ = 0;
    for (const auto& observation : active_encoder_obs_functions_) {
      if (!observation.function(encoder_obs_buffer_, observation.offset)) {
        std::cerr << "Failed to gather encoder observation: " << observation.name << '\n';
        return false;
      }
    }
    return true;
  }

  bool GatherObservations() {
    std::fill(obs_buffer_.begin(), obs_buffer_.end(), 0);
    if (!GatherEncoderObservations()) return false;
    std::copy(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), obs_buffer_.begin());
    for (const auto& observation : active_obs_functions_) {
      if (!observation.function(obs_buffer_, observation.offset)) {
        std::cerr << "Failed to gather observation: " << observation.name << '\n';
        return false;
      }
    }
    return true;
  }

  bool CheckSafety() {
    used_low_state_data_ = backend_->GetState();
    if (enabled_ && !used_low_state_data_.enabled) throw std::runtime_error("Environment disabled by watchdog or feedback fault");
    return true;
  }

  bool GatherRobotStateToLogger() {
    return state_logger_->LogFullState(used_low_state_data_, last_action);
  }

  bool GatherInputInterfaceData() {
    if (live_) live_pose_snapshot_ = operator_state.play ? zmq_input_->GetLatestPose() : frozen_pose_;
    return !live_ || live_pose_snapshot_.last >= 0;
  }

  void ResetObservationHistory() {
    used_low_state_data_ = backend_->GetState();
    last_action.assign(21, 0);
    state_logger_->ResetHistory(used_low_state_data_);
    reinitialize_heading_ = true;
  }

  // 仿真初始化属于环境职责，独立于朝向对齐和观测采集；真机不允许执行位置瞬移。
  void ResetSimulation() {
    if (real_) throw std::runtime_error("Hardware cannot teleport");
    const auto positions = live_ ? config_.at("static_qpos").get<std::vector<double>>()
                                : std::vector<double>(current_motion_->MujocoPositions(current_frame_),
                                                      current_motion_->MujocoPositions(current_frame_) + 28);
    std::vector<double> velocities(27, 0);
    if (reset_with_reference_velocity_ && !live_)
      velocities.assign(current_motion_->MujocoVelocities(current_frame_), current_motion_->MujocoVelocities(current_frame_) + 27);
    backend_->Reset(positions, velocities);
    backend_->Enable();
    enabled_ = true;
    // qpos 的驱动关节顺序不是策略顺序，初始化目标只能从策略参考或默认姿态读取。
    MotorCommand command;
    command.q_target = live_ ? contract_.at("default_policy").get<std::vector<double>>()
                            : std::vector<double>(current_motion_->JointPositions(current_frame_), current_motion_->JointPositions(current_frame_) + 21);
    motor_command_buffer_ = std::move(command);
  }

  bool InitControl() {
    if (!real_) {
      ResetSimulation();
      ResetObservationHistory();
      UpdateHeadingState();
      operator_state.start = true;
      program_state_ = ProgramState::WAIT_FOR_CONTROL;
      return true;
    }
    if (!operator_state.start) {
      ResetObservationHistory();
      program_state_ = ProgramState::WAIT_FOR_CONTROL;
      return true;
    }
    if (!entry_started_) {
      backend_->Enable();
      enabled_ = true;
      used_low_state_data_ = backend_->GetState();
      entry_start_ = used_low_state_data_.q;
      entry_step_ = 0;
      entry_started_ = true;
    }
    const int steps = std::max(1, int(config_.at("entry_duration").get<double>() / control_dt_));
    if (entry_step_ < steps) {
      MotorCommand command{std::vector<double>(21), {}};
      ++entry_step_;
      for (int i = 0; i < 21; ++i) {
        const double end = live_ ? contract_.at("default_policy")[i].get<double>() : current_motion_->JointPositions(current_frame_)[i];
        command.q_target[i] = entry_start_[i] + (end - entry_start_[i]) * entry_step_ / steps;
      }
      motor_command_buffer_ = std::move(command);
      return true;
    }
    // 姿态过渡完成只进入 PD 保持，不沿用按 ] 设置的启动标志自动运行策略。
    // 保留最后一帧目标，由 LowCommandWriter 持续下发；只有之后按 T 才允许策略接管。
    operator_state.start = false;
    operator_state.play = false;
    program_state_ = ProgramState::WAIT_FOR_CONTROL;
    entry_started_ = false;
    std::cout << "BUMI3_REAL_HOLDING: preparation pose held; policy inactive; press T to start policy and motion" << std::endl;
    return true;
  }

  std::vector<int> ReadKeys() {
    auto keys = input_interface_->ReadKeys();
    const auto gui = backend_->ReadKeys();
    keys.insert(keys.end(), gui.begin(), gui.end());
    for (auto& key : keys) {
      if (key == 257) key = '\n';
      if (key >= 'a' && key <= 'z') key -= 'a' - 'A';
    }
    return keys;
  }

  void Input() {
    for (const auto key : ReadKeys()) {
      if (key == 'O') { operator_state.stop = true; return; }
      // 与 G1 的 INIT 状态对应，进入姿态期间仅处理停止，不允许切换正在过渡的参考。
      if (real_ && entry_started_) continue;
      if (key == ']' && real_ && !enabled_) {
        std::cout << "BUMI3_ENABLE_KEY_RECEIVED: requesting enable and entry pose" << std::endl;
        operator_state.start = true;
        operator_state.play = false;
        program_state_ = ProgramState::INIT;
      } else if (key == 'T') {
        if (real_ && !enabled_) { std::cout << "Press ] before playing\n"; continue; }
        if (live_) {
          frozen_pose_ = zmq_input_->GetLatestPose();
          live_pose_snapshot_ = frozen_pose_;
          if (!operator_state.play) {
            // 恢复实时播放只清空历史动作，不重新对齐朝向，避免把当前转身目标消掉。
            used_low_state_data_ = backend_->GetState();
            last_action.assign(21, 0);
            state_logger_->ResetHistory(used_low_state_data_);
          }
        } else if (!operator_state.play && current_frame_ == current_motion_->timesteps - 1) {
          if (real_) { std::cout << "Press R then ] to restart\n"; continue; }
          current_frame_ = 0;
          reset_with_reference_velocity_ = false;
          program_state_ = ProgramState::INIT;
        }
        operator_state.play = !operator_state.play;
        // 真机保持准备姿态时，T 才设置独立的策略启动请求；播放状态本身不触发自动接管。
        // 进入 CONTROL 后，T 继续仅切换参考播放，保留原有的暂停与恢复行为。
        if (real_ && program_state_ == ProgramState::WAIT_FOR_CONTROL)
          operator_state.start = operator_state.play;
      } else if ((key == 'P' || key == 'R') && !live_) {
        backend_->Disable();
        enabled_ = false;
        motor_command_buffer_.reset();
        operator_state.play = false;
        operator_state.start = false;
        current_frame_ = 0;
        if (key == 'P') {
          motion_reader_.current_motion_index_ = (motion_reader_.current_motion_index_ + 1) % motion_reader_.motions.size();
          current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
        }
        reset_with_reference_velocity_ = false;
        program_state_ = ProgramState::INIT;
        std::cout << "BUMI3_MOTION=" << current_motion_->name << '\n';
      } else if ((key == '\n' || key == '\r') && has_live_ && has_offline_) {
        backend_->Disable();
        enabled_ = false;
        motor_command_buffer_.reset();
        operator_state.play = false;
        operator_state.start = false;
        live_ = !live_;
        if (live_) {
          frozen_pose_ = zmq_input_->WaitForPose(config_.at("teleop").at("startup_timeout"));
          live_pose_snapshot_ = frozen_pose_;
        }
        current_encoder_mode_ = live_ || offline_smpl_ ? 1 : 0;
        InitializeObservationFunctions();
        reset_with_reference_velocity_ = false;
        program_state_ = ProgramState::INIT;
        std::cout << "BUMI3_SOURCE=" << (live_ ? "live_smpl" : "offline") << '\n';
      }
    }
  }

  bool CreatePolicyCommand() {
    auto& policy_engine = live_ ? live_policy_ : offline_policy_;
    auto& obs_buffer_float = policy_engine->GetInputBuffer();
    std::transform(obs_buffer_.begin(), obs_buffer_.end(), obs_buffer_float.begin(), [](double value) { return float(value); });
    if (!policy_engine->Infer(contract_.at("action_clip"))) return false;
    last_action = policy_engine->GetActionBuffer();
    MotorCommand command{std::vector<double>(21), {}};
    for (int i = 0; i < 21; ++i)
      command.q_target[i] = contract_.at("default_policy")[i].get<double>() + contract_.at("action_scale_policy")[i].get<double>() * last_action[i];
    if (!real_ && !live_ && current_motion_->has_robot_reference)
      command.reference_qpos.assign(current_motion_->MujocoPositions(current_frame_), current_motion_->MujocoPositions(current_frame_) + 28);
    motor_command_buffer_ = std::move(command);
    return true;
  }

  void LowCommandWriter() {
    if (!enabled_ || !motor_command_buffer_ || operator_state.stop) return;
    backend_->WriteTarget(motor_command_buffer_->q_target);
    // 参考影子保存于命令快照，不能在游标推进后再读取下一帧来显示上一条命令。
    backend_->WriteReference(motor_command_buffer_->reference_qpos);
  }

  bool CurrentFrameAdvancement() {
    if (!operator_state.play || live_) return true;
    if (config_.at("loop_motion") == true) current_frame_ = (current_frame_ + 1) % current_motion_->timesteps;
    else {
      current_frame_ = std::min(current_frame_ + 1, current_motion_->timesteps - 1);
      if (current_frame_ == current_motion_->timesteps - 1) operator_state.play = false;
    }
    return true;
  }

  // 与 G1 的 CONTROL 执行顺序对应：反馈历史 → 输入快照 → 朝向 → 观测 → 策略命令 → 游标。
  // 命令写入后由 Run 中的 LowCommandWriter 下发；SDK 环境端维持自身的高频 I/O。
  void Control() {
    if (operator_state.stop) return;
    if (!CheckSafety()) throw std::runtime_error("BUMI3 safety check failed");
    switch (program_state_) {
      case ProgramState::INIT:
        if (!InitControl()) throw std::runtime_error("BUMI3 InitControl failed");
        break;
      case ProgramState::WAIT_FOR_CONTROL:
        // 没有新的启动请求就保留最后的 PD 目标，不采集策略观测、不推理，也不推进参考帧。
        // 真机在 T 触发接管时重新初始化历史与朝向，避免使用漫长保持阶段之前的状态。
        if (operator_state.start) {
          if (real_) {
            if (!enabled_) throw std::runtime_error("Prepare pose with ] before starting policy");
            ResetObservationHistory();
            UpdateHeadingState();
            std::cout << "BUMI3_POLICY_STARTED: policy control and reference playback active" << std::endl;
          }
          operator_state.start = false;
          program_state_ = ProgramState::CONTROL;
        }
        break;
      case ProgramState::CONTROL:
        if (!GatherRobotStateToLogger()) throw std::runtime_error("Failed to gather robot state history");
        if (!GatherInputInterfaceData()) throw std::runtime_error("Failed to gather input interface data");
        UpdateHeadingState();
        if (!GatherObservations()) throw std::runtime_error("Failed to gather observations");
        if (!CreatePolicyCommand()) throw std::runtime_error("Failed to create policy command");
        if (!CurrentFrameAdvancement()) throw std::runtime_error("Failed to advance motion frame");
        break;
      case ProgramState::STOPPED:
        break;
    }
  }

  Json config_, contract_, obs_config_, encoder_config_;
  ProgramState program_state_ = ProgramState::INIT;
  bool real_ = false, has_offline_ = false, has_live_ = false, offline_smpl_ = false;
  bool live_ = false, enabled_ = false, entry_started_ = false, reset_with_reference_velocity_ = false;
  bool reinitialize_heading_ = true;
  double control_dt_ = 0.02;
  int current_frame_ = 0, current_encoder_mode_ = 0, saved_frame_for_observation_window_ = 0, entry_step_ = 0;
  Quat init_base_quat_ = Quat::Identity(), init_ref_data_root_rot_ = Quat::Identity();
  State used_low_state_data_;
  std::array<double, 21> entry_start_{};
  std::vector<double> last_action, obs_buffer_, encoder_obs_buffer_;
  std::vector<ActiveObservation> active_obs_functions_, active_encoder_obs_functions_;
  std::optional<MotorCommand> motor_command_buffer_;
  MotionDataReader motion_reader_;
  std::shared_ptr<const MotionSequence> current_motion_;
  std::unique_ptr<PolicyEngine> offline_policy_, live_policy_;
  std::unique_ptr<StateLogger> state_logger_;
  std::unique_ptr<RosBackend> backend_;
  std::unique_ptr<InputInterface> input_interface_;
  std::unique_ptr<ZmqInput> zmq_input_;
  PoseWindow frozen_pose_, live_pose_snapshot_;
};

int main(int argc, char** argv) {
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  try {
    const auto options = DeployOptions::Parse(argc, argv);
    if (options.help) { DeployOptions::PrintHelp(); return 0; }
    if (options.capabilities) {
      std::cout << Json{{"policy", "native_cpp"}, {"sim_transport", "ros2"}, {"real_transport", "ros2"},
                        {"inference_backends", {"cpu", "cuda", "tensorrt"}},
                        {"available_providers", Ort::GetAvailableProviders()}}.dump() << '\n';
      return 0;
    }
    DeploymentConfigParser parser(options);
    auto config = parser.Load();
    if (options.validate_only) config["validate_only"] = true;
    if (!options.output_dir.empty()) { parser.PrepareLaunch(config); return 0; }
    Bumi3Deploy deploy(std::move(config));
    deploy.Run();
  } catch (const std::exception& error) {
    if (!stopping) { std::cerr << "BUMI3_DEPLOY_ERROR: " << error.what() << '\n'; return 1; }
  }
  return 0;
}
