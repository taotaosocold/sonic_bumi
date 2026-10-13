#ifndef NOETIX_BRIDGE_ROBOT_RESOURCE_HPP_
#define NOETIX_BRIDGE_ROBOT_RESOURCE_HPP_

#include <cstddef>
#include <string>
#include <vector>

namespace noetix_bridge {

struct JointResource {
  std::string joint_name;
  std::string actuator_name;
  int direction{1};
  double offset{0.0};
};

/**
 * One fully resolved command entry for a configured robot pose mode.
 *
 * pos/vel/ff are expressed in logical joint coordinates in robot.yaml.
 * The pose-mode node converts them to actuator coordinates before publishing
 * LowCmd. kp/kd are scalar actuator gains and therefore are not sign-flipped.
 */
struct JointModeResource {
  std::string joint_name;
  std::string actuator_name;
  double pos{0.0};
  double vel{0.0};
  double kp{0.0};
  double kd{0.0};
  double ff{0.0};
};

/**
 * Robot-local LIE/STAND profile parsed from config/robot.yaml.
 *
 * joints is always ordered exactly like robot.joint_order.  A configured
 * profile is required to contain one command for every configured joint.
 */
struct JointControlProfile {
  bool configured{false};

  double kp{0.0};
  double kd{0.0};
  double kd_ankle_roll{0.0};
  double vel{0.0};
  double ff{0.0};
  double duration_s{0.0};
  double publish_hz{0.0};

  std::vector<JointModeResource> joints;
};

struct RobotResourceInfo {
  std::string robot_name;
  std::string robot_root;
  std::string robot_yaml;
  std::string hardware_yaml;
  std::string joy_yaml;
  std::vector<std::string> joint_names;
  std::vector<std::string> actuator_names;
  std::vector<JointResource> joints;
  std::size_t dof{0};
  std::string base_frame{"base_link"};
  std::string imu_frame{"base_imu"};

  // Optional robot-local pose modes.  configured=false means the mode was not
  // provided by that robot's robot.yaml.
  JointControlProfile lie_control;
  JointControlProfile stand_control;
};

RobotResourceInfo loadRobotResource(const std::string& robot_name,
                                   const std::string& robot_root = "");

}  // namespace noetix_bridge
#endif
