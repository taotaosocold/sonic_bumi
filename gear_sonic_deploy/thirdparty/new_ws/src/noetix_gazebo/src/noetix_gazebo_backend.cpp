#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

#include <noetix_msgs/msg/low_cmd.hpp>
#include <noetix_msgs/msg/low_state.hpp>
#include <noetix_msgs/msg/robot_status.hpp>

namespace noetix_gazebo {
namespace {

bool finite(double value)
{
  return std::isfinite(value);
}

builtin_interfaces::msg::Time toRosTime(double seconds)
{
  if (!std::isfinite(seconds) || seconds < 0.0) {
    seconds = 0.0;
  }

  const double sec_floor = std::floor(seconds);
  auto sec = static_cast<std::int32_t>(sec_floor);
  auto nsec = static_cast<std::uint32_t>(
      std::llround((seconds - sec_floor) * 1.0e9));

  if (nsec >= 1000000000u) {
    ++sec;
    nsec -= 1000000000u;
  }

  builtin_interfaces::msg::Time out;
  out.sec = sec;
  out.nanosec = nsec;
  return out;
}

std::string lower(std::string value)
{
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

}  // namespace

class NoetixGazeboBackend final : public gazebo::ModelPlugin {
 public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = std::move(model);
    world_ = model_->GetWorld();
    ros_node_ = gazebo_ros::Node::Get(sdf);

    if (!model_ || !world_ || !ros_node_) {
      gzerr << "[NoetixGazeboBackend] invalid Gazebo/ROS handles\n";
      return;
    }

    base_link_name_ = readString(sdf, "base_link", "base_link");
    imu_link_name_ = readString(sdf, "imu_link", "base_imu");
    state_publish_hz_ = readDouble(sdf, "state_publish_hz", 200.0);
    joint_viscous_damping_ = readDouble(sdf, "joint_viscous_damping", 0.0);
    joint_coulomb_friction_ = readDouble(sdf, "joint_coulomb_friction", 0.0);
    hold_active_ = readBool(sdf, "hold_on_start", true);

    if (state_publish_hz_ <= 0.0 || !finite(state_publish_hz_)) {
      throw std::runtime_error("state_publish_hz must be finite and > 0");
    }
    if (joint_viscous_damping_ < 0.0 || !finite(joint_viscous_damping_)) {
      throw std::runtime_error("joint_viscous_damping must be finite and >= 0");
    }
    if (joint_coulomb_friction_ < 0.0 || !finite(joint_coulomb_friction_)) {
      throw std::runtime_error("joint_coulomb_friction must be finite and >= 0");
    }

    base_link_ = model_->GetLink(base_link_name_);
    imu_link_ = model_->GetLink(imu_link_name_);
    if (!base_link_) {
      throw std::runtime_error("base link not found: " + base_link_name_);
    }
    if (!imu_link_) {
      throw std::runtime_error("IMU link not found: " + imu_link_name_);
    }

    loadJoints(sdf);
    if (joints_.empty()) {
      throw std::runtime_error("no controlled joints configured");
    }

    hold_pose_ = model_->WorldPose();
    initial_hold_pose_ = hold_pose_;
    last_imu_world_linear_vel_ = imu_link_->WorldLinearVel();
    last_imu_update_s_ = world_->SimTime().Double();

    // Apply the configured initial pose exactly once before the physics loop.
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      joints_[i].joint->SetPosition(0, joints_[i].initial_position, true);
    }

    low_cmd_sub_ = ros_node_->create_subscription<noetix_msgs::msg::LowCmd>(
        "/noetix/low_cmd", rclcpp::QoS(20).reliable(),
        [this](noetix_msgs::msg::LowCmd::SharedPtr msg) {
          onLowCmd(*msg);
        });

    request_sub_ = ros_node_->create_subscription<std_msgs::msg::String>(
        "/noetix/control_request", rclcpp::QoS(20).reliable(),
        [this](std_msgs::msg::String::SharedPtr msg) {
          onControlRequest(msg->data);
        });

    hoist_sub_ = ros_node_->create_subscription<std_msgs::msg::String>(
        "/noetix/sim/hoist_request",
        rclcpp::QoS(1).reliable().transient_local(),
        [this](std_msgs::msg::String::SharedPtr msg) {
          onHoistRequest(msg->data);
        });

    low_state_pub_ = ros_node_->create_publisher<noetix_msgs::msg::LowState>(
        "/noetix/low_state", rclcpp::QoS(10).reliable());
    joint_state_pub_ = ros_node_->create_publisher<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::QoS(10).reliable());
    imu_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Imu>(
        "/noetix/imu", rclcpp::QoS(10).reliable());
    control_state_pub_ = ros_node_->create_publisher<std_msgs::msg::String>(
        "/noetix/control_state", rclcpp::QoS(10).reliable());
    hoist_state_pub_ = ros_node_->create_publisher<std_msgs::msg::String>(
        "/noetix/sim/hoist_state",
        rclcpp::QoS(1).reliable().transient_local());

    last_state_publish_s_ = -std::numeric_limits<double>::infinity();
    last_stats_publish_s_ = -std::numeric_limits<double>::infinity();

    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
        std::bind(&NoetixGazeboBackend::onUpdate, this));

    publishHoistState();

    RCLCPP_INFO(
        ros_node_->get_logger(),
        "Noetix Gazebo Classic backend online model=%s joints=%zu physics=Gazebo "
        "state=%.1fHz drive=DISABLED hoist=%s",
        model_->GetName().c_str(), joints_.size(), state_publish_hz_,
        hold_active_ ? "HOLD" : "RELEASED");
  }

 private:
  struct JointRuntime {
    std::string name;
    gazebo::physics::JointPtr joint;
    double effort_limit{0.0};
    double initial_position{0.0};
    double last_tau{0.0};
  };

  struct CommandRuntime {
    double q{0.0};
    double dq{0.0};
    double tau{0.0};
    double kp{0.0};
    double kd{0.0};
  };

  static std::string readString(
      const sdf::ElementPtr& sdf,
      const std::string& key,
      const std::string& fallback)
  {
    return sdf->HasElement(key) ? sdf->Get<std::string>(key) : fallback;
  }

  static double readDouble(
      const sdf::ElementPtr& sdf,
      const std::string& key,
      double fallback)
  {
    return sdf->HasElement(key) ? sdf->Get<double>(key) : fallback;
  }

  static bool readBool(
      const sdf::ElementPtr& sdf,
      const std::string& key,
      bool fallback)
  {
    return sdf->HasElement(key) ? sdf->Get<bool>(key) : fallback;
  }

  void loadJoints(const sdf::ElementPtr& sdf)
  {
    if (!sdf->HasElement("controlled_joint")) {
      return;
    }

    auto element = sdf->GetElement("controlled_joint");
    std::unordered_set<std::string> unique;

    while (element) {
      if (!element->HasElement("name")) {
        throw std::runtime_error("controlled_joint missing name");
      }

      JointRuntime runtime;
      runtime.name = element->Get<std::string>("name");
      if (!unique.emplace(runtime.name).second) {
        throw std::runtime_error("duplicate controlled_joint: " + runtime.name);
      }

      runtime.joint = model_->GetJoint(runtime.name);
      if (!runtime.joint) {
        throw std::runtime_error("Gazebo joint not found: " + runtime.name);
      }

      runtime.effort_limit = element->HasElement("effort_limit")
          ? element->Get<double>("effort_limit")
          : 0.0;
      runtime.initial_position = element->HasElement("initial_position")
          ? element->Get<double>("initial_position")
          : 0.0;

      if (!finite(runtime.effort_limit) || runtime.effort_limit <= 0.0) {
        throw std::runtime_error(
            "invalid effort limit for joint: " + runtime.name);
      }
      if (!finite(runtime.initial_position)) {
        throw std::runtime_error(
            "invalid initial position for joint: " + runtime.name);
      }

      joint_index_.emplace(runtime.name, joints_.size());
      joints_.push_back(std::move(runtime));
      element = element->GetNextElement("controlled_joint");
    }

    command_.resize(joints_.size());
  }

  bool validateAndOrderCommand(
      const noetix_msgs::msg::LowCmd& msg,
      std::vector<CommandRuntime>& ordered) const
  {
    if (msg.motor_cmd.size() != joints_.size()) {
      return false;
    }

    ordered.assign(joints_.size(), CommandRuntime{});
    std::vector<bool> seen(joints_.size(), false);

    for (const auto& src : msg.motor_cmd) {
      const auto it = joint_index_.find(src.name);
      if (it == joint_index_.end()) {
        return false;
      }
      const std::size_t index = it->second;
      if (seen[index]) {
        return false;
      }
      seen[index] = true;

      if (!finite(src.q) || !finite(src.dq) || !finite(src.tau) ||
          !finite(src.kp) || !finite(src.kd) ||
          src.kp < 0.0 || src.kd < 0.0) {
        return false;
      }

      ordered[index] = CommandRuntime{
          src.q, src.dq, src.tau, src.kp, src.kd};
    }

    return std::all_of(seen.begin(), seen.end(), [](bool value) {
      return value;
    });
  }

  void onLowCmd(const noetix_msgs::msg::LowCmd& msg)
  {
    ++low_cmd_received_;

    if (!drive_enabled_) {
      ++low_cmd_rejected_;
      return;
    }

    std::vector<CommandRuntime> ordered;
    if (!validateAndOrderCommand(msg, ordered)) {
      ++low_cmd_rejected_;
      RCLCPP_WARN_THROTTLE(
          ros_node_->get_logger(), *ros_node_->get_clock(), 1000,
          "LowCmd ignored: command must contain exactly %zu unique configured "
          "joints and only finite values",
          joints_.size());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      command_ = std::move(ordered);
      have_command_ = true;
    }

    ++low_cmd_accepted_;
  }

  void onControlRequest(std::string request)
  {
    request = lower(std::move(request));

    if (request == "enable") {
      if (!drive_enabled_) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        have_command_ = false;
        drive_enabled_ = true;
        RCLCPP_WARN(ros_node_->get_logger(), "SIM Mode -> ENABLED");
      }
      return;
    }

    if (request == "disable" || request == "stop" ||
        request == "estop" || request == "safe_stop") {
      {
        std::lock_guard<std::mutex> lock(command_mutex_);
        have_command_ = false;
        drive_enabled_ = false;
      }
      RCLCPP_WARN(ros_node_->get_logger(), "SIM Mode -> DISABLED");
      return;
    }

    // LIE/STAND remain ordinary upper-layer LowCmd generators, exactly like
    // the real bridge.  The existing pose-mode node consumes these requests.
    if (request == "lie" || request == "stand") {
      return;
    }
  }

  void onHoistRequest(std::string request)
  {
    request = lower(std::move(request));
    if (request != "release" && request != "hold" && request != "reset") {
      RCLCPP_WARN(
          ros_node_->get_logger(),
          "Unknown hoist request '%s' (valid: hold, release, reset)",
          request.c_str());
      return;
    }

    // Gazebo physics objects are only mutated from onUpdate(). ROS callbacks
    // merely enqueue the request, avoiding cross-thread physics access.
    std::lock_guard<std::mutex> lock(hoist_request_mutex_);
    pending_hoist_request_ = std::move(request);
  }

  void processHoistRequest()
  {
    std::string request;
    {
      std::lock_guard<std::mutex> lock(hoist_request_mutex_);
      request.swap(pending_hoist_request_);
    }

    if (request.empty()) {
      return;
    }

    if (request == "release") {
      if (hold_active_) {
        base_link_->SetLinearVel(ignition::math::Vector3d::Zero);
        base_link_->SetAngularVel(ignition::math::Vector3d::Zero);
        hold_active_ = false;
        publishHoistState();
        RCLCPP_WARN(
            ros_node_->get_logger(),
            "Virtual hoist RELEASED at model_z=%.4f; gravity/contact now fully dynamic",
            model_->WorldPose().Pos().Z());
      }
      return;
    }

    if (request == "hold") {
      hold_pose_ = model_->WorldPose();
      hold_active_ = true;
      publishHoistState();
      RCLCPP_WARN(
          ros_node_->get_logger(),
          "Virtual hoist HOLD at model_z=%.4f",
          hold_pose_.Pos().Z());
      return;
    }

    if (request == "reset") {
      hold_pose_ = initial_hold_pose_;
      model_->SetWorldPose(hold_pose_, true, true);
      base_link_->SetLinearVel(ignition::math::Vector3d::Zero);
      base_link_->SetAngularVel(ignition::math::Vector3d::Zero);
      hold_active_ = true;
      publishHoistState();
      RCLCPP_WARN(
          ros_node_->get_logger(),
          "Virtual hoist RESET to initial pose z=%.4f",
          hold_pose_.Pos().Z());
    }
  }

  void zeroJointEfforts()
  {
    for (auto& joint : joints_) {
      joint.last_tau = 0.0;
      joint.joint->SetForce(0, 0.0);
    }
  }

  void applyHoistConstraint()
  {
    if (!hold_active_) {
      return;
    }

    // Only the floating base is constrained.  The 21 controlled joints remain
    // normal Gazebo dynamic joints and therefore STAND uses the same LowCmd PD
    // path as the real robot.  Once released, no pose override remains.
    model_->SetWorldPose(hold_pose_, true, true);
    base_link_->SetLinearVel(ignition::math::Vector3d::Zero);
    base_link_->SetAngularVel(ignition::math::Vector3d::Zero);
  }

  void applyJointControl()
  {
    std::vector<CommandRuntime> command;
    bool have_command = false;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (drive_enabled_ && have_command_) {
        command = command_;
        have_command = true;
      }
    }

    for (std::size_t i = 0; i < joints_.size(); ++i) {
      auto& runtime = joints_[i];
      const double q = runtime.joint->Position(0);
      const double dq = runtime.joint->GetVelocity(0);

      double tau = 0.0;
      if (have_command) {
        const auto& c = command[i];
        tau = c.kp * (c.q - q) + c.kd * (c.dq - dq) + c.tau;

        if (joint_viscous_damping_ > 0.0) {
          tau -= joint_viscous_damping_ * dq;
        }
        if (joint_coulomb_friction_ > 0.0 && std::abs(dq) > 1.0e-5) {
          tau -= std::copysign(joint_coulomb_friction_, dq);
        }

        tau = std::clamp(tau, -runtime.effort_limit, runtime.effort_limit);
      }

      runtime.last_tau = tau;
      runtime.joint->SetForce(0, tau);
    }
  }

  void publishHoistState()
  {
    if (!hoist_state_pub_) {
      return;
    }
    std_msgs::msg::String msg;
    msg.data = hold_active_ ? "hold" : "released";
    hoist_state_pub_->publish(msg);
  }

  void updateImu(double sim_s)
  {
    const auto pose = imu_link_->WorldPose();
    imu_orientation_ = pose.Rot();

    const auto angular_world = imu_link_->WorldAngularVel();
    imu_angular_body_ =
        imu_orientation_.RotateVectorReverse(angular_world);

    const auto linear_world = imu_link_->WorldLinearVel();
    const double dt = sim_s - last_imu_update_s_;

    ignition::math::Vector3d acceleration_world =
        ignition::math::Vector3d::Zero;
    if (dt > 1.0e-9 && dt < 0.1) {
      acceleration_world =
          (linear_world - last_imu_world_linear_vel_) / dt;
    }

    // Match Gazebo Classic IMU semantics: specific force is kinematic
    // acceleration minus gravity, rotated into the IMU/body frame.
    imu_specific_body_ = imu_orientation_.RotateVectorReverse(
        acceleration_world - world_->Gravity());

    last_imu_world_linear_vel_ = linear_world;
    last_imu_update_s_ = sim_s;
  }

  void publishState(double sim_s)
  {
    const auto stamp = toRosTime(sim_s);

    noetix_msgs::msg::LowState low_state;
    low_state.stamp = stamp;
    low_state.sequence = ++state_sequence_;
    low_state.motor_state.resize(joints_.size());

    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp = stamp;
    joint_state.name.resize(joints_.size());
    joint_state.position.resize(joints_.size());
    joint_state.velocity.resize(joints_.size());
    joint_state.effort.resize(joints_.size());

    for (std::size_t i = 0; i < joints_.size(); ++i) {
      const auto& runtime = joints_[i];
      const double q = runtime.joint->Position(0);
      const double dq = runtime.joint->GetVelocity(0);

      auto& motor = low_state.motor_state[i];
      motor.name = runtime.name;
      motor.q = static_cast<float>(q);
      motor.dq = static_cast<float>(dq);
      // Real LowState.tau_est is a motor torque estimate.  The closest Gazebo
      // equivalent is the actuator effort actually requested after saturation.
      motor.tau_est = static_cast<float>(runtime.last_tau);
      motor.temperature = 0.0f;
      motor.error = 0;

      joint_state.name[i] = runtime.name;
      joint_state.position[i] = q;
      joint_state.velocity[i] = dq;
      joint_state.effort[i] = runtime.last_tau;
    }

    low_state.imu.valid = true;
    low_state.imu.quaternion[0] = static_cast<float>(imu_orientation_.W());
    low_state.imu.quaternion[1] = static_cast<float>(imu_orientation_.X());
    low_state.imu.quaternion[2] = static_cast<float>(imu_orientation_.Y());
    low_state.imu.quaternion[3] = static_cast<float>(imu_orientation_.Z());
    low_state.imu.gyroscope[0] = static_cast<float>(imu_angular_body_.X());
    low_state.imu.gyroscope[1] = static_cast<float>(imu_angular_body_.Y());
    low_state.imu.gyroscope[2] = static_cast<float>(imu_angular_body_.Z());
    low_state.imu.accelerometer[0] = static_cast<float>(imu_specific_body_.X());
    low_state.imu.accelerometer[1] = static_cast<float>(imu_specific_body_.Y());
    low_state.imu.accelerometer[2] = static_cast<float>(imu_specific_body_.Z());

    low_state.status.mode = drive_enabled_
        ? noetix_msgs::msg::RobotStatus::MODE_ENABLED
        : noetix_msgs::msg::RobotStatus::MODE_DISABLED;
    low_state.status.connected = true;
    low_state.status.enabled = drive_enabled_;
    low_state.status.fault = 0;

    low_state_pub_->publish(low_state);
    joint_state_pub_->publish(joint_state);

    sensor_msgs::msg::Imu imu;
    imu.header.stamp = stamp;
    imu.header.frame_id = imu_link_name_;
    imu.orientation.w = imu_orientation_.W();
    imu.orientation.x = imu_orientation_.X();
    imu.orientation.y = imu_orientation_.Y();
    imu.orientation.z = imu_orientation_.Z();
    imu.angular_velocity.x = imu_angular_body_.X();
    imu.angular_velocity.y = imu_angular_body_.Y();
    imu.angular_velocity.z = imu_angular_body_.Z();
    imu.linear_acceleration.x = imu_specific_body_.X();
    imu.linear_acceleration.y = imu_specific_body_.Y();
    imu.linear_acceleration.z = imu_specific_body_.Z();
    imu_pub_->publish(imu);
  }

  void publishStats(double sim_s)
  {
    std_msgs::msg::String msg;
    msg.data =
        std::string("mode=") + (drive_enabled_ ? "ENABLED" : "DISABLED") +
        " command=" + (have_command_ ? "present" : "none") +
        " sim_time=" + std::to_string(sim_s) +
        " hoist=" + (hold_active_ ? "HOLD" : "RELEASED") +
        " cmd_rx=" + std::to_string(low_cmd_received_) +
        " cmd_ok=" + std::to_string(low_cmd_accepted_) +
        " cmd_reject=" + std::to_string(low_cmd_rejected_);
    control_state_pub_->publish(msg);
    RCLCPP_INFO(ros_node_->get_logger(), "%s", msg.data.c_str());
  }

  void onUpdate()
  {
    const double sim_s = world_->SimTime().Double();

    processHoistRequest();
    applyHoistConstraint();
    applyJointControl();
    updateImu(sim_s);

    const double state_period = 1.0 / state_publish_hz_;
    if (sim_s - last_state_publish_s_ >= state_period) {
      last_state_publish_s_ = sim_s;
      publishState(sim_s);
    }

    if (sim_s - last_stats_publish_s_ >= 2.0) {
      last_stats_publish_s_ = sim_s;
      publishStats(sim_s);
    }
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::WorldPtr world_;
  gazebo::physics::LinkPtr base_link_;
  gazebo::physics::LinkPtr imu_link_;
  gazebo_ros::Node::SharedPtr ros_node_;
  gazebo::event::ConnectionPtr update_connection_;

  std::string base_link_name_{"base_link"};
  std::string imu_link_name_{"base_imu"};
  double state_publish_hz_{200.0};
  double joint_viscous_damping_{0.0};
  double joint_coulomb_friction_{0.0};

  std::vector<JointRuntime> joints_;
  std::unordered_map<std::string, std::size_t> joint_index_;

  std::mutex command_mutex_;
  std::vector<CommandRuntime> command_;
  std::atomic_bool have_command_{false};
  std::atomic_bool drive_enabled_{false};

  std::atomic_bool hold_active_{true};
  ignition::math::Pose3d hold_pose_;
  ignition::math::Pose3d initial_hold_pose_;

  ignition::math::Quaterniond imu_orientation_{1.0, 0.0, 0.0, 0.0};
  ignition::math::Vector3d imu_angular_body_{0.0, 0.0, 0.0};
  ignition::math::Vector3d imu_specific_body_{0.0, 0.0, 9.81};
  ignition::math::Vector3d last_imu_world_linear_vel_{0.0, 0.0, 0.0};
  double last_imu_update_s_{0.0};

  std::uint64_t state_sequence_{0};
  std::atomic<std::uint64_t> low_cmd_received_{0};
  std::atomic<std::uint64_t> low_cmd_accepted_{0};
  std::atomic<std::uint64_t> low_cmd_rejected_{0};
  double last_state_publish_s_{0.0};
  double last_stats_publish_s_{0.0};

  std::mutex hoist_request_mutex_;
  std::string pending_hoist_request_;

  rclcpp::Subscription<noetix_msgs::msg::LowCmd>::SharedPtr low_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr request_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr hoist_sub_;
  rclcpp::Publisher<noetix_msgs::msg::LowState>::SharedPtr low_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr control_state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr hoist_state_pub_;
};

GZ_REGISTER_MODEL_PLUGIN(NoetixGazeboBackend)

}  // namespace noetix_gazebo
