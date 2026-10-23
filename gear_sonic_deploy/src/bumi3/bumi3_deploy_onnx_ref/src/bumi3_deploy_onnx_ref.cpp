#include "deploy_config.hpp"
#include "input_interface/keyboard_handler.hpp"
#include "motion_data_reader.hpp"
#include "observation_config.hpp"
#include "policy_parameters.hpp"
#include "math_utils.hpp"
#include "control_policy.hpp"
#include "ros2_backend.hpp"
#include "state_logger.hpp"
#include "utils.hpp"
#include "input_interface/zmq_endpoint_interface.hpp"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <optional>

using namespace bumi3;

// 保留 G1 的部署组织方式：状态机、参考观测采集、历史观测采集、策略命令与命令下发。
// BUMI3 的联合 ONNX 在模型内部编码和解码，外部仍必须按训练契约拼接参考与历史。
class Bumi3Deploy {
 public:
  enum class ProgramState { INIT, WAIT_FOR_CONTROL, CONTROL, STOPPED };

  explicit Bumi3Deploy(Json config)
      : config_(std::move(config)), policy_parameters_(PolicyParameters::Parse(config_.at("contract"))) {}
  ~Bumi3Deploy() { Stop(false); }

  void Initialize() {
    real_ = config_.at("mode") == "real";
    has_offline_ = !config_.at("motions").empty();
    has_live_ = config_.at("input_type") == "zmq";
    offline_smpl_ = config_.at("encoder") == "smpl";
    const auto smpl_input = config_.value("smpl_input", std::string("full"));
    if (smpl_input != "full" && smpl_input != "five_point") throw std::runtime_error("Unknown SMPL input layout");
    five_point_ = smpl_input == "five_point";
    smpl_reference_dim_ = five_point_ ? SMPL_FIVE_POINT_REFERENCE_DIM : SMPL_REFERENCE_DIM;
    live_ = !has_offline_;
    current_encoder_mode_ = live_ || offline_smpl_ ? 1 : 0;
    control_dt_ = policy_parameters_.control_dt;
    duration_ = config_.at("entry_duration").get<double>();

    // 与 G1 一样先预加载动作，再初始化模型与观测函数；控制循环和切换动作不读取 CSV。
    if (has_offline_) {
      if (!motion_reader_.ReadFromCSV(config_.at("motions"), offline_smpl_,
                                      config_.at("reference_contract_id"), policy_parameters_.policy_joint_names, five_point_))
        throw std::runtime_error("No reference motions loaded");
      current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
    }
    const Json inference = config_.value("inference", Json{{"provider", config_.at("provider")}});
    if (has_offline_) {
      offline_policy_ = std::make_unique<PolicyEngine>();
      if (!offline_policy_->Initialize(config_.at("policy"),
          (offline_smpl_ ? smpl_reference_dim_ : ROBOT_REFERENCE_DIM) + PROPRIOCEPTION_DIM, inference))
        throw std::runtime_error("Failed to initialize offline BUMI3 policy");
    }
    if (has_live_) {
      live_policy_ = std::make_unique<PolicyEngine>();
      if (!live_policy_->Initialize(config_.at("smpl_policy"), smpl_reference_dim_ + PROPRIOCEPTION_DIM, inference))
        throw std::runtime_error("Failed to initialize live BUMI3 policy");
    }
    state_logger_ = std::make_unique<StateLogger>(policy_parameters_);
    last_action.assign(BUMI3_NUM_MOTOR, 0);
    const auto observation_config = ObservationConfigParser::ParseFullConfig(config_);
    obs_config_ = observation_config.observations;
    encoder_config_ = observation_config.encoder;
    InitializeObservationFunctions();
    if (config_.at("validate_only") == true) return;

    backend_ = std::make_unique<RosBackend>(config_);
    used_low_state_data_ = backend_->WaitForState();
    // 与 G1 一样由 INIT 自动进入默认站姿；键盘仅请求策略接管与参考播放。
    // Noetix 的使能确认仍由后端负责，默认站姿使用本机器人的训练参数。
    input_interface_ = std::make_unique<SimpleKeyboard>();
    if (has_live_) zmq_interface_ = std::make_unique<ZMQEndpointInterface>(config_.at("teleop"), five_point_);
    operator_state.play = false;
    if (live_) {
      std::cout << "Waiting for ten-frame SMPL stream\n";
      frozen_pose_ = zmq_interface_->WaitForPose(config_.at("teleop").at("startup_timeout"));
      live_pose_snapshot_ = frozen_pose_;
    }
    program_state_ = ProgramState::INIT;
    std::cout << "BUMI3_STARTUP: enabling drives and preparing default standing pose; support real robot before launch\n"
              << "BUMI3_CONTROL_READY native_cpp: ] start-policy / T play / R restart-paused / N next / P previous / I align-heading / ENTER offline-live / O stop\n";
  }

  void Run() {
    Initialize();
    if (config_.at("validate_only") == true) {
      std::cout << "BUMI3_NATIVE_CONFIG_OK: policy/observations validated; no device connected\n";
      return;
    }
    try {
      // 输入、控制和命令发布分别调度，频率与 G1 对应；ROS2 回调继续由后端接收线程执行。
      input_thread_ = CreateRecurrentThread("Input", input_dt_, &Bumi3Deploy::Input);
      command_writer_thread_ = CreateRecurrentThread("command_writer", publish_dt_, &Bumi3Deploy::LowCommandWriter);
      control_thread_ = CreateRecurrentThread("control", control_dt_, &Bumi3Deploy::Control);
      std::cout << "BUMI3_THREADS_READY input_hz=100 control_hz=50 command_hz=500\n";
      const auto started = Time::now();
      while (!StopRequested()) {
        if (!config_.at("duration").is_null() &&
            std::chrono::duration<double>(Time::now() - started).count() >= config_.at("duration").get<double>()) break;
        // 信号处理器只设置标志，因此主线程定期检查信号；普通停止通过条件变量立即唤醒。
        std::unique_lock<std::mutex> lock(wakeup_mutex_);
        wakeup_.wait_for(lock, std::chrono::milliseconds(10), [this] { return StopRequested(); });
      }
    } catch (...) {
      // 即使只成功创建了部分线程，也先关闭下发入口并回收这些线程，再向调用者报告错误。
      Stop(false);
      throw;
    }
    std::exception_ptr failure;
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      failure = thread_error_;
    }
    Stop(!failure);
    if (failure) std::rethrow_exception(failure);
  }

  void Stop(bool require_ack = true) {
    if (stopped_) return;
    stopped_ = true;
    RequestStop();
    std::exception_ptr failure;
    if (backend_) {
      // 先禁止新的使能、重置与目标发布，再请求去使能；不等待可能尚在返回的 GPU 推理。
      backend_->RequestStop();
      ClearMotorCommand();
      try { backend_->Disable(require_ack); }
      catch (...) { failure = std::current_exception(); }
    }
    if (input_thread_.joinable()) input_thread_.join();
    if (control_thread_.joinable()) control_thread_.join();
    if (command_writer_thread_.joinable()) command_writer_thread_.join();
    // 状态机只由控制线程修改；回收线程后主线程才能写入终止状态或释放输入接口。
    program_state_ = ProgramState::STOPPED;
    operator_state.stop = true;
    enabled_ = false;
    input_interface_.reset();
    if (failure && require_ack) std::rethrow_exception(failure);
    if (failure) {
      try { std::rethrow_exception(failure); }
      catch (const std::exception& error) { std::cerr << "BUMI3_STOP_ERROR: " << error.what() << '\n'; }
    }
  }

 private:
  OperatorState operator_state;

  bool StopRequested() const { return stopping || stop_requested_.load(); }

  void RequestStop() {
    stop_requested_.store(true);
    wakeup_.notify_all();
  }

  std::thread CreateRecurrentThread(const char* name, double dt, void (Bumi3Deploy::*callback)()) {
    return std::thread([this, name, dt, callback] {
      try {
        const auto period = std::chrono::duration_cast<Time::duration>(std::chrono::duration<double>(dt));
        auto next = Time::now();
        while (!StopRequested()) {
          (this->*callback)();
          next += period;
          // 沿固定周期网格跳过已经错过的时刻，不补发旧命令；超时后只等待到下一个周期，
          // 避免在回调耗时之外再完整等待一次周期，进一步降低实际控制频率。
          const auto now = Time::now();
          if (next <= now) next += period * ((now - next) / period + 1);
          std::unique_lock<std::mutex> lock(wakeup_mutex_);
          wakeup_.wait_until(lock, next, [this] { return StopRequested(); });
        }
      } catch (...) {
        // 工作线程不得让异常逃逸触发 terminate；保留首个错误，由主线程停止硬件并报告。
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!StopRequested()) {
          if (!thread_error_) thread_error_ = std::current_exception();
          std::cerr << "BUMI3_THREAD_ERROR thread=" << name << '\n';
          RequestStop();
        }
      }
    });
  }

  void SetMotorCommand(MotorCommand command) {
    std::lock_guard<std::mutex> lock(command_publish_mutex_);
    if (!StopRequested()) motor_command_buffer_.SetData(std::move(command));
  }

  void ClearMotorCommand() {
    // 清空操作与实际发布串行，返回后不会再发出切换动作或重置之前取得的命令快照。
    std::lock_guard<std::mutex> lock(command_publish_mutex_);
    motor_command_buffer_.Clear();
  }

  void RefreshHoldingCommand() {
    std::lock_guard<std::mutex> lock(command_publish_mutex_);
    const auto snapshot = motor_command_buffer_.GetDataWithTime();
    if (!StopRequested() && snapshot.data) motor_command_buffer_.SetData(*snapshot.data);
  }

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
    heading_state_.init_base_quat = used_low_state_data_.root;
    heading_state_.init_ref_data_root_rot = GetReferenceQuaternion(live_ ? 0 : current_frame_);
    reinitialize_heading_ = false;
  }

  Quat ComputeApplyDeltaHeading() const {
    return HeadingQuaternion(heading_state_.init_base_quat) * HeadingQuaternion(heading_state_.init_ref_data_root_rot).conjugate();
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

  // CSV 和实时消息都已按训练布局保存局部五点姿态；这里只采集十帧，不重复变换坐标。
  // 根朝向仍由 GatherMotionAnchorOrientationMutiFrame 对齐后独立拼接在五点特征之后。
  bool GatherMotionSmplFivePointPoseMultiFrame(std::vector<double>& target_buffer, std::size_t offset,
                                               int num_frames = 10, int step_size = 1) {
    if (!five_point_ || current_encoder_mode_ != 1 ||
        !CheckObservationWindow(target_buffer.size(), offset, num_frames, step_size, 45)) return false;
    for (int i = 0; i < num_frames; ++i) {
      const int frame = GetFutureFrame(i, step_size);
      const double* pose = live_ ? live_pose_snapshot_.five_point_pose.data() + frame * 45
                                : current_motion_->SmplFivePointPose(frame);
      std::copy_n(pose, 45, target_buffer.begin() + offset + i * 45);
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
            {"smpl_five_point_pose_10frame_step1", 450, [this](auto& b, auto o) { return GatherMotionSmplFivePointPoseMultiFrame(b, o, 10, 1); }},
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
    const auto defaults = ObservationConfigParser::GetDefaultEncoderConfig(five_point_);
    const auto& expected_mode = defaults.encoder_modes.at(current_encoder_mode_);
    std::optional<EncoderModeConfig> mode_config;
    for (const auto& mode : encoder_config_.encoder_modes) {
      if (mode.name != expected_mode.name) continue;
      if (mode_config) throw std::runtime_error("Duplicate BUMI3 encoder mode");
      mode_config = mode;
    }
    if (!mode_config || mode_config->mode_id != current_encoder_mode_ ||
        mode_config->required_observations != expected_mode.required_observations ||
        encoder_config_.dimension != TOKEN_DIM) throw std::runtime_error("Encoder layout differs from trained BUMI3 model");
    std::size_t encoder_offset = 0;
    for (const auto& name : mode_config->required_observations) {
      int matches = 0;
      for (const auto& entry : encoder_config_.encoder_observations)
        if (entry.name == name && entry.enabled) ++matches;
      if (matches != 1) throw std::runtime_error("Required encoder observation disabled or duplicated: " + name);
      encoder_offset += append(name, encoder_offset, active_encoder_obs_functions_);
    }
    // 联合模型布局为参考在前、历史在后，不能照搬 G1 独立 decoder 的 token+历史输入。
    std::size_t policy_offset = encoder_offset;
    std::vector<std::string> active_names, expected_names;
    for (const auto& entry : ObservationConfigParser::GetDefaultConfig()) expected_names.push_back(entry.name);
    for (const auto& entry : obs_config_) {
      if (!entry.enabled) continue;
      const auto& name = entry.name;
      active_names.push_back(name);
      policy_offset += append(name, policy_offset, active_obs_functions_);
    }
    if (active_names != expected_names || encoder_offset != std::size_t(current_encoder_mode_ == 1 ? smpl_reference_dim_ : ROBOT_REFERENCE_DIM) ||
        policy_offset - encoder_offset != PROPRIOCEPTION_DIM) throw std::runtime_error("Observation layout differs from trained BUMI3 model");
    encoder_obs_buffer_.assign(encoder_offset, 0);
    obs_buffer_.assign(policy_offset, 0);
    const auto& policy = live_ ? live_policy_ : offline_policy_;
    if (!policy || policy->GetInputBuffer().size() != policy_offset)
      throw std::runtime_error("Observation buffer differs from selected policy input");
    std::cout << "BUMI3_OBSERVATIONS encoder=" << expected_mode.name
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
    if (live_) live_pose_snapshot_ = operator_state.play ? zmq_interface_->GetLatestPose() : frozen_pose_;
    return !live_ || live_pose_snapshot_.last >= 0;
  }

  void ResetObservationHistory() {
    used_low_state_data_ = backend_->GetState();
    last_action.assign(BUMI3_NUM_MOTOR, 0);
    state_logger_->ResetHistory(used_low_state_data_);
    reinitialize_heading_ = true;
  }

  // INIT 与 G1 使用相同职责：从实测姿态平滑进入默认站姿，随后只做 PD 保持。
  // 仿真与真机走同一条路径，不读取动作首帧初始化机器人，也不自动启动策略。
  bool InitControl() {
    if (!enabled_) {
      backend_->Enable();
      enabled_ = true;
      used_low_state_data_ = backend_->GetState();
      time_ = 0.0;
    }
    MotorCommand command = policy_parameters_.CreatePositionCommand(policy_parameters_.default_angles.data());
    time_ += control_dt_;
    if (time_ < duration_) {
      const double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
      for (int i = 0; i < BUMI3_NUM_MOTOR; ++i)
        command.q_target[i] = used_low_state_data_.q[i] * (1.0 - ratio) + policy_parameters_.default_angles[i] * ratio;
    } else {
      operator_state.start = false;
      operator_state.play = false;
      program_state_ = ProgramState::WAIT_FOR_CONTROL;
      std::cout << "BUMI3_INIT_DONE: default standing pose held; press ] to start policy with reference paused" << std::endl;
    }
    SetMotorCommand(std::move(command));
    return true;
  }

  // 输入接口与 G1 一样使用 update/handle_input，将按键翻译为操作请求。
  // 请求经队列交给控制线程应用，避免输入线程与推理线程同时修改参考及朝向。
  void Input() {
    input_interface_->update(backend_->ReadKeys());
    const auto input = input_interface_->handle_input();
    if (input.stop_control) {
      // 停止请求直接唤醒主线程，不等待推理结束，也不排在普通按键队列后面。
      RequestStop();
      return;
    }
    if (!(input.start_control || input.play_motion || input.motion_prev || input.motion_next ||
          input.motion_restart || input.reinitialize || input.switch_source)) return;
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (pending_input_commands_.size() >= 128) throw std::runtime_error("BUMI3 input queue overflow");
    pending_input_commands_.push_back(input);
  }

  void ApplyInputCommands() {
    std::vector<InputCommand> inputs;
    {
      std::lock_guard<std::mutex> lock(input_mutex_);
      inputs.swap(pending_input_commands_);
    }
    for (const auto& input : inputs) {
      if (StopRequested()) return;
      // 与 G1 一样，切换动作只改变内存中的参考和播放游标；机器人本体不瞬移、
      // 不重新去使能，也不再次进入准备站姿，上一周期动作继续作为历史反馈。
      if (!live_ && (input.motion_prev || input.motion_next)) {
        const auto count = motion_reader_.motions.size();
        const auto index = motion_reader_.current_motion_index_;
        motion_reader_.current_motion_index_ = input.motion_next ? (index + 1) % count : (index + count - 1) % count;
        current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
        operator_state.play = false;
        current_frame_ = 0;
        reinitialize_heading_ = true;
        std::cout << "BUMI3_MOTION=" << current_motion_->name << " frame=0 paused\n";
      }
      if (input.switch_source && has_live_ && has_offline_) {
        if (!live_) {
          // 来源切换必须立即获得有效窗口，不能在控制线程等待数分钟并中断命令续期。
          frozen_pose_ = zmq_interface_->GetLatestPose();
          live_pose_snapshot_ = frozen_pose_;
        }
        live_ = !live_;
        operator_state.play = false;
        current_frame_ = 0;
        reinitialize_heading_ = true;
        current_encoder_mode_ = live_ || offline_smpl_ ? 1 : 0;
        InitializeObservationFunctions();
        std::cout << "BUMI3_SOURCE=" << (live_ ? "live_smpl" : "offline") << '\n';
      }
      if (input.motion_restart) {
        operator_state.play = false;
        current_frame_ = 0;
        if (live_) {
          frozen_pose_ = zmq_interface_->GetLatestPose();
          live_pose_snapshot_ = frozen_pose_;
        }
        reinitialize_heading_ = true;
        std::cout << "BUMI3_REFERENCE_PAUSED: frame=0; policy state unchanged\n";
      }
      if (input.reinitialize) reinitialize_heading_ = true;
      if (input.start_control) {
        if (program_state_ == ProgramState::WAIT_FOR_CONTROL) operator_state.start = true;
        else if (program_state_ == ProgramState::INIT) std::cout << "Wait for BUMI3_INIT_DONE before pressing ]\n";
      }
      if (input.play_motion && !operator_state.play) {
        if (program_state_ != ProgramState::CONTROL) {
          std::cout << "Press ] to start policy before pressing T to play\n";
        } else if (!live_ && current_motion_->timesteps > 1 && current_frame_ == current_motion_->timesteps - 1) {
          std::cout << "Motion finished; press R to restart paused, then T to play\n";
        } else {
          // T 只启动或继续参考播放，不清空历史、不重新对齐朝向、不切换策略状态。
          if (live_) live_pose_snapshot_ = zmq_interface_->GetLatestPose();
          operator_state.play = true;
          std::cout << "BUMI3_PLAYING: source=" << (live_ ? "live_smpl" : "offline") << " frame=" << current_frame_ << '\n';
        }
      }
    }
  }

  bool CreatePolicyCommand() {
    auto& policy_engine = live_ ? live_policy_ : offline_policy_;
    auto& obs_buffer_float = policy_engine->GetInputBuffer();
    std::transform(obs_buffer_.begin(), obs_buffer_.end(), obs_buffer_float.begin(), [](double value) { return float(value); });
    if (!policy_engine->Infer(policy_parameters_.action_clip)) return false;
    last_action = policy_engine->GetActionBuffer();
    MotorCommand command = policy_parameters_.CreatePositionCommand(policy_parameters_.default_angles.data());
    for (int i = 0; i < BUMI3_NUM_MOTOR; ++i)
      command.q_target[i] = policy_parameters_.default_angles[i] + policy_parameters_.action_scale[i] * last_action[i];
    if (!real_ && !live_ && current_motion_->has_robot_reference)
      command.reference_qpos.assign(current_motion_->MujocoPositions(current_frame_), current_motion_->MujocoPositions(current_frame_) + 28);
    SetMotorCommand(std::move(command));
    return true;
  }

  void LowCommandWriter() {
    std::lock_guard<std::mutex> lock(command_publish_mutex_);
    if (StopRequested()) return;
    const auto snapshot = motor_command_buffer_.GetDataWithTime();
    if (!snapshot.data) return;
    // 时间戳来自控制线程，500 Hz 重发不续期；推理停滞不能靠重复发送掩盖环境端超时。
    if (snapshot.GetAgeMs() > config_.at("command_timeout").get<double>() * 1000.)
      throw std::runtime_error("BUMI3 control command stale; stopping publication");
    backend_->WriteTarget(*snapshot.data);
    // 参考影子保存于命令快照，不能在游标推进后再读取下一帧来显示上一条命令。
    backend_->WriteReference(snapshot.data->reference_qpos);
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
  // 控制线程生成不可变命令快照，由独立发送线程重发；SDK 环境端维持自身的高频 I/O。
  void Control() {
    if (StopRequested()) return;
    ApplyInputCommands();
    if (StopRequested()) return;
    if (!CheckSafety()) throw std::runtime_error("BUMI3 safety check failed");
    switch (program_state_) {
      case ProgramState::INIT:
        if (!InitControl()) throw std::runtime_error("BUMI3 InitControl failed");
        break;
      case ProgramState::WAIT_FOR_CONTROL:
        // 与 G1 相同，等待阶段只保持默认站姿；] 单独请求接管，T 单独请求播放。
        // 两种环境在接管时用同一份实测状态填充历史，不依赖仿真首帧瞬移或提前推理。
        if (operator_state.start) {
          if (!enabled_) throw std::runtime_error("Drives are not enabled after preparation");
          if (live_) {
            frozen_pose_ = zmq_interface_->GetLatestPose();
            live_pose_snapshot_ = frozen_pose_;
          }
          ResetObservationHistory();
          UpdateHeadingState();
          operator_state.start = false;
          // 显式指定仿真自动播放时，仍必须先按 ] 接管；真机始终等待单独的 T 请求。
          operator_state.play = !real_ && config_.at("autoplay").get<bool>();
          program_state_ = ProgramState::CONTROL;
          last_control_time_.reset();
          logging_counter_ = 0;
          max_control_interval_ms_ = 0.0;
          std::cout << "BUMI3_POLICY_STARTED: reference=" << (operator_state.play ? "playing" : "paused")
                    << "; release simulation band / lower real support, then press T to play" << std::endl;
        }
        // 等待可无限持续；控制线程确认反馈正常后续期，发送线程单独重发不掩盖控制停滞。
        if (enabled_) RefreshHoldingCommand();
        break;
      case ProgramState::CONTROL: {
        const auto obs_start_time = Time::now();
        const double interval_ms = last_control_time_
            ? std::chrono::duration<double, std::milli>(obs_start_time - *last_control_time_).count() : 0.0;
        last_control_time_ = obs_start_time;
        max_control_interval_ms_ = std::max(max_control_interval_ms_, interval_ms);
        if (!GatherRobotStateToLogger()) throw std::runtime_error("Failed to gather robot state history");
        if (!GatherInputInterfaceData()) throw std::runtime_error("Failed to gather input interface data");
        UpdateHeadingState();
        if (!GatherObservations()) throw std::runtime_error("Failed to gather observations");
        const auto obs_end_time = Time::now();
        if (!CreatePolicyCommand()) throw std::runtime_error("Failed to create policy command");
        const auto motor_command_end_time = Time::now();
        if (!CurrentFrameAdvancement()) throw std::runtime_error("Failed to advance motion frame");
        // 沿用 G1 的周期耗时输出，区分观测采集、推理与实际周期，便于定位动态部署延迟。
        if (++logging_counter_ % 50 == 0) {
          std::cout << "BUMI3_CONTROL_TIMING obs_us="
                    << std::chrono::duration_cast<std::chrono::microseconds>(obs_end_time - obs_start_time).count()
                    << " policy_us="
                    << std::chrono::duration_cast<std::chrono::microseconds>(motor_command_end_time - obs_end_time).count()
                    << " interval_ms=" << interval_ms << " max_interval_ms=" << max_control_interval_ms_ << '\n';
          max_control_interval_ms_ = 0.0;
        }
        break;
      }
      case ProgramState::STOPPED:
        break;
    }
  }

  Json config_;
  PolicyParameters policy_parameters_;
  std::vector<ObservationConfig> obs_config_;
  EncoderConfig encoder_config_;
  ProgramState program_state_ = ProgramState::INIT;
  bool real_ = false, has_offline_ = false, has_live_ = false, offline_smpl_ = false;
  bool five_point_ = false;
  int smpl_reference_dim_ = SMPL_REFERENCE_DIM;
  bool live_ = false, enabled_ = false;
  bool reinitialize_heading_ = true;
  double control_dt_ = 0.02, time_ = 0.0, duration_ = 3.0;
  const double input_dt_ = 0.01, publish_dt_ = 0.002;
  int current_frame_ = 0, current_encoder_mode_ = 0, saved_frame_for_observation_window_ = 0;
  std::optional<Time::time_point> last_control_time_;
  uint64_t logging_counter_ = 0;
  double max_control_interval_ms_ = 0.0;
  HeadingState heading_state_;
  RobotState used_low_state_data_;
  std::vector<double> last_action, obs_buffer_, encoder_obs_buffer_;
  std::vector<ActiveObservation> active_obs_functions_, active_encoder_obs_functions_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  // 动作与观测只由控制线程拥有；跨线程仅共享输入请求、命令快照、停止标志和异常。
  std::thread input_thread_, command_writer_thread_, control_thread_;
  std::atomic<bool> stop_requested_{false};
  bool stopped_ = false;
  std::mutex input_mutex_, command_publish_mutex_, error_mutex_, wakeup_mutex_;
  std::condition_variable wakeup_;
  std::vector<InputCommand> pending_input_commands_;
  std::exception_ptr thread_error_;
  MotionDataReader motion_reader_;
  std::shared_ptr<const MotionSequence> current_motion_;
  std::unique_ptr<PolicyEngine> offline_policy_, live_policy_;
  std::unique_ptr<StateLogger> state_logger_;
  std::unique_ptr<RosBackend> backend_;
  std::unique_ptr<InputInterface> input_interface_;
  std::unique_ptr<ZMQEndpointInterface> zmq_interface_;
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
