#include "noetix_bridge/robot_resource.hpp"

#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

namespace noetix_bridge {
namespace {

std::string frame(const YAML::Node& robot,
                  const char* key,
                  const char* fallback)
{
  return (robot["frames"] && robot["frames"][key])
      ? robot["frames"][key].as<std::string>()
      : fallback;
}

bool isAnkleRollJoint(const std::string& name)
{
  return name == "l_ankle_roll_joint" ||
         name == "r_ankle_roll_joint";
}

double finiteScalar(const YAML::Node& node,
                    const char* key,
                    double fallback,
                    const std::string& label)
{
  if (!node || !node[key]) {
    return fallback;
  }

  const double value = node[key].as<double>();
  if (!std::isfinite(value)) {
    throw std::runtime_error(label + "." + key + " must be finite");
  }
  return value;
}

double requiredFiniteScalar(const YAML::Node& node,
                            const char* key,
                            const std::string& label)
{
  if (!node || !node[key]) {
    throw std::runtime_error(label + "." + key + " is required");
  }
  return finiteScalar(node, key, 0.0, label);
}

void validateNonNegative(double value, const std::string& label)
{
  if (!std::isfinite(value) || value < 0.0) {
    throw std::runtime_error(label + " must be finite and >= 0");
  }
}

void validatePositive(double value, const std::string& label)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(label + " must be finite and > 0");
  }
}

JointControlProfile loadPoseProfile(
    const YAML::Node& root_yaml,
    const char* control_key,
    const char* joint_list_key,
    const std::vector<JointResource>& joints)
{
  JointControlProfile profile;
  const YAML::Node control = root_yaml[control_key];
  const YAML::Node top_level_joint_list = root_yaml[joint_list_key];

  YAML::Node joint_list = top_level_joint_list;
  if (!joint_list && control) {
    if (control["joint_commands"]) {
      joint_list = control["joint_commands"];
    } else if (control["joints"]) {
      joint_list = control["joints"];
    }
  }

  if (!control && !joint_list) {
    return profile;
  }
  if (control && !control.IsMap()) {
    throw std::runtime_error(std::string(control_key) + " must be a map");
  }
  if (!joint_list || !joint_list.IsSequence()) {
    throw std::runtime_error(
        std::string(control_key) + " requires a complete " + joint_list_key +
        " sequence");
  }

  // These are the previous RL LIE/STAND command defaults.  A robot may
  // override any of them in its own robot.yaml.
  profile.configured = true;
  profile.kp = finiteScalar(control, "kp", 10.0, control_key);
  profile.kd = finiteScalar(control, "kd", 0.5, control_key);
  profile.kd_ankle_roll = finiteScalar(
      control, "kd_ankle_roll", 0.2, control_key);
  profile.vel = finiteScalar(control, "vel", 0.0, control_key);
  profile.ff = finiteScalar(control, "ff", 0.0, control_key);
  profile.duration_s = finiteScalar(control, "duration_s", 2.0, control_key);
  profile.publish_hz = finiteScalar(control, "publish_hz", 500.0, control_key);

  validateNonNegative(profile.kp, std::string(control_key) + ".kp");
  validateNonNegative(profile.kd, std::string(control_key) + ".kd");
  validateNonNegative(
      profile.kd_ankle_roll,
      std::string(control_key) + ".kd_ankle_roll");
  validatePositive(
      profile.duration_s,
      std::string(control_key) + ".duration_s");
  validatePositive(
      profile.publish_hz,
      std::string(control_key) + ".publish_hz");

  std::unordered_map<std::string, std::size_t> joint_index;
  joint_index.reserve(joints.size());
  profile.joints.resize(joints.size());

  for (std::size_t i = 0; i < joints.size(); ++i) {
    const auto& mapping = joints[i];
    joint_index.emplace(mapping.joint_name, i);

    auto& command = profile.joints[i];
    command.joint_name = mapping.joint_name;
    command.actuator_name = mapping.actuator_name;
    command.vel = profile.vel;
    command.kp = profile.kp;
    command.kd = isAnkleRollJoint(mapping.joint_name)
        ? profile.kd_ankle_roll
        : profile.kd;
    command.ff = profile.ff;
  }

  std::unordered_set<std::string> seen;
  seen.reserve(joints.size());

  for (const auto& node : joint_list) {
    if (!node["name"]) {
      throw std::runtime_error(
          std::string(joint_list_key) + " entry requires name");
    }

    const std::string name = node["name"].as<std::string>();
    const auto index_it = joint_index.find(name);
    if (index_it == joint_index.end()) {
      throw std::runtime_error(
          std::string(joint_list_key) + " contains unknown joint: " + name);
    }
    if (!seen.emplace(name).second) {
      throw std::runtime_error(
          std::string(joint_list_key) + " contains duplicate joint: " + name);
    }

    const std::string label = std::string(joint_list_key) + "." + name;
    auto& command = profile.joints[index_it->second];
    command.pos = requiredFiniteScalar(node, "pos", label);
    command.vel = finiteScalar(node, "vel", command.vel, label);
    command.kp = finiteScalar(node, "kp", command.kp, label);
    command.kd = finiteScalar(node, "kd", command.kd, label);
    command.ff = finiteScalar(node, "ff", command.ff, label);

    validateNonNegative(command.kp, label + ".kp");
    validateNonNegative(command.kd, label + ".kd");
  }

  if (seen.size() != joints.size()) {
    std::string missing;
    for (const auto& joint : joints) {
      if (seen.find(joint.joint_name) == seen.end()) {
        if (!missing.empty()) {
          missing += ", ";
        }
        missing += joint.joint_name;
      }
    }
    throw std::runtime_error(
        std::string(joint_list_key) +
        " must contain every robot joint; missing: " + missing);
  }

  return profile;
}

}  // namespace

RobotResourceInfo loadRobotResource(const std::string& robot_name,
                                   const std::string& robot_root)
{
  if (robot_name.empty()) {
    throw std::invalid_argument("robot_name is empty");
  }

  RobotResourceInfo info;
  info.robot_name = robot_name;

  const std::filesystem::path root = robot_root.empty()
      ? std::filesystem::path(ament_index_cpp::get_package_share_directory("noetix_robots")) / robot_name
      : std::filesystem::path(robot_root);
  const std::filesystem::path config = root / "config";

  info.robot_root = root.string();
  info.robot_yaml = (config / "robot.yaml").string();
  info.hardware_yaml = (config / "hardware.yaml").string();
  info.joy_yaml = (config / "joy.yaml").string();

  for (const auto* p : {&info.robot_yaml, &info.hardware_yaml}) {
    if (!std::filesystem::is_regular_file(*p)) {
      throw std::runtime_error("Required config not found: " + *p);
    }
  }

  const YAML::Node root_yaml = YAML::LoadFile(info.robot_yaml);
  const YAML::Node robot = root_yaml["robot"];
  const YAML::Node joints = root_yaml["joints"];

  if (!robot || !robot["joint_order"] || !joints || !joints.IsSequence()) {
    throw std::runtime_error(
        "robot.yaml requires robot.joint_order and joints[]");
  }
  if (robot["name"] && robot["name"].as<std::string>() != robot_name) {
    throw std::runtime_error("robot.yaml name mismatch");
  }

  std::set<std::string> order_set;
  for (const auto& n : robot["joint_order"]) {
    const std::string name = n.as<std::string>();
    if (!order_set.insert(name).second) {
      throw std::runtime_error("duplicate joint_order: " + name);
    }
    info.joint_names.push_back(name);
  }

  info.dof = info.joint_names.size();
  if (info.dof == 0) {
    throw std::runtime_error("joint_order is empty");
  }
  if (robot["dof"] && robot["dof"].as<std::size_t>() != info.dof) {
    throw std::runtime_error("robot.dof mismatch");
  }

  std::unordered_map<std::string, JointResource> by_joint;
  std::set<std::string> actuator_unique;

  for (const auto& n : joints) {
    if (!n["name"] || !n["actuator"] || !n["direction"] || !n["offset"]) {
      throw std::runtime_error(
          "every joints[] entry requires name/actuator/direction/offset");
    }

    JointResource j;
    j.joint_name = n["name"].as<std::string>();
    j.actuator_name = n["actuator"].as<std::string>();
    j.direction = n["direction"].as<int>();
    j.offset = n["offset"].as<double>();

    if (j.direction != 1 && j.direction != -1) {
      throw std::runtime_error(
          "direction must be +/-1: " + j.joint_name);
    }
    if (!std::isfinite(j.offset)) {
      throw std::runtime_error("invalid offset: " + j.joint_name);
    }
    if (!by_joint.emplace(j.joint_name, j).second) {
      throw std::runtime_error(
          "duplicate joint mapping: " + j.joint_name);
    }
    if (!actuator_unique.insert(j.actuator_name).second) {
      throw std::runtime_error(
          "duplicate actuator mapping: " + j.actuator_name);
    }
  }

  for (const auto& joint_name : info.joint_names) {
    const auto it = by_joint.find(joint_name);
    if (it == by_joint.end()) {
      throw std::runtime_error("missing mapping for: " + joint_name);
    }
    info.joints.push_back(it->second);
    info.actuator_names.push_back(it->second.actuator_name);
  }

  if (info.joints.size() != by_joint.size()) {
    throw std::runtime_error(
        "joints[] contains unknown joint_order entries");
  }

  info.base_frame = frame(robot, "base", "base_link");
  info.imu_frame = frame(robot, "imu", "base_imu");

  info.lie_control = loadPoseProfile(
      root_yaml,
      "lie_control",
      "lie_joint_angles",
      info.joints);

  info.stand_control = loadPoseProfile(
      root_yaml,
      "stand_control",
      "stand_joint_angles",
      info.joints);

  return info;
}

}  // namespace noetix_bridge
