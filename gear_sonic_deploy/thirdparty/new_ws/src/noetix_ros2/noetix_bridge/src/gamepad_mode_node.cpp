#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>

#include "noetix_bridge/robot_resource.hpp"

using namespace std::chrono_literals;

namespace noetix_bridge {
namespace {

struct Binding {
  std::string name;
  std::string request;
  int button{-1};
  std::vector<int> require_buttons;
};

std::string requestOf(const YAML::Node& node)
{
  if (!node["action"]) {
    throw std::runtime_error("joy binding missing action");
  }

  const std::string action = node["action"].as<std::string>();

  if (action == "enable" ||
      action == "lie" ||
      action == "stand" ||
      action == "disable" ||
      action == "stop") {
    return action == "stop" ? "disable" : action;
  }

  if (action == "mode" && node["mode"]) {
    const std::string mode = node["mode"].as<std::string>();
    if (mode == "enable" ||
        mode == "lie" ||
        mode == "stand" ||
        mode == "disable" ||
        mode == "stop") {
      return mode == "stop" ? "disable" : mode;
    }
  }

  throw std::runtime_error(
      "unsupported joy action/mode: " + action +
      " (valid: enable, lie, stand, disable)");
}


}  // namespace


class GamepadModeNode final :
    public rclcpp::Node {
 public:
  GamepadModeNode()
  : rclcpp::Node(
        "noetix_gamepad_mode_node")
  {
    declare_parameter<std::string>(
        "robot_name", "n3");

    declare_parameter<std::string>(
        "joy_topic", "");

    const auto resource =
        loadRobotResource(
            get_parameter(
                "robot_name").as_string());

    const auto root =
        YAML::LoadFile(
            resource.joy_yaml);

    const auto joy =
        root["joy"];

    if (!joy) {
      throw std::runtime_error(
          "joy root missing: " +
          resource.joy_yaml);
    }

    if (joy["enabled"] &&
        !joy["enabled"].as<bool>()) {
      throw std::runtime_error(
          "joy.enabled=false in " +
          resource.joy_yaml);
    }

    joy_topic_ =
        get_parameter(
            "joy_topic").as_string();

    if (joy_topic_.empty()) {
      joy_topic_ =
          joy["topic"]
              ? joy["topic"].as<std::string>()
              : "/joy";
    }

    const auto bindings =
        joy["bindings"];

    if (!bindings ||
        !bindings.IsSequence()) {
      throw std::runtime_error(
          "joy.bindings must be a sequence");
    }

    std::set<
        std::pair<
            int,
            std::vector<int>>>
        unique;

    for (const auto& node : bindings) {
      Binding binding;

      binding.name =
          node["name"]
              ? node["name"].as<std::string>()
              : "unnamed";

      if (!node["button"]) {
        throw std::runtime_error(
            "joy binding missing button: " +
            binding.name);
      }

      binding.button =
          node["button"].as<int>();

      binding.request =
          requestOf(node);

      if (node["require_buttons"]) {
        for (const auto& item :
             node["require_buttons"]) {
          binding.require_buttons.push_back(
              item.as<int>());
        }
      }

      auto chord =
          binding.require_buttons;

      std::sort(
          chord.begin(),
          chord.end());

      if (!unique.emplace(
              binding.button,
              chord).second) {
        throw std::runtime_error(
            "duplicate joy button/chord");
      }

      bindings_.push_back(
          std::move(binding));
    }

    request_pub_ =
        create_publisher<
            std_msgs::msg::String>(
            "/noetix/control_request",
            20);

    joy_sub_ =
        create_subscription<
            sensor_msgs::msg::Joy>(
            joy_topic_,
            rclcpp::SensorDataQoS(),
            [this](
                sensor_msgs::msg::Joy::SharedPtr msg) {
              onJoy(*msg);
            });

    watchdog_ =
        create_wall_timer(
            3s,
            [this]() {
              if (!received_joy_) {
                RCLCPP_WARN(
                    get_logger(),
                    "No Joy messages received on %s. "
                    "Check joy_node, device_id and permissions.",
                    joy_topic_.c_str());
              }
            });

    RCLCPP_INFO(
        get_logger(),
        "Gamepad mode adapter active: topic=%s bindings=%zu",
        joy_topic_.c_str(),
        bindings_.size());

    for (const auto& b : bindings_) {
      RCLCPP_INFO(
          get_logger(),
          "  button=%d -> %s",
          b.button,
          b.request.c_str());
    }
  }

 private:
  static bool pressed(
      const sensor_msgs::msg::Joy& msg,
      int index)
  {
    return
        index >= 0 &&
        static_cast<std::size_t>(index) <
            msg.buttons.size() &&
        msg.buttons[
            static_cast<std::size_t>(index)] != 0;
  }

  bool rising(
      const sensor_msgs::msg::Joy& msg,
      int index) const
  {
    if (!pressed(msg, index)) {
      return false;
    }

    const auto i =
        static_cast<std::size_t>(index);

    return
        i >= last_buttons_.size() ||
        last_buttons_[i] == 0;
  }

  void onJoy(
      const sensor_msgs::msg::Joy& msg)
  {
    if (!received_joy_) {
      received_joy_ = true;

      RCLCPP_INFO(
          get_logger(),
          "First Joy frame: axes=%zu buttons=%zu",
          msg.axes.size(),
          msg.buttons.size());
    }

    for (const auto& binding :
         bindings_) {
      bool modifiers_ok = true;

      for (const int index :
           binding.require_buttons) {
        modifiers_ok =
            modifiers_ok &&
            pressed(msg, index);
      }

      if (modifiers_ok &&
          rising(
              msg,
              binding.button)) {
        std_msgs::msg::String request;
        request.data =
            binding.request;

        request_pub_->publish(
            request);

        RCLCPP_WARN(
            get_logger(),
            "Gamepad button %d -> mode request '%s'",
            binding.button,
            binding.request.c_str());

        break;
      }
    }

    last_buttons_ =
        msg.buttons;
  }

  std::string joy_topic_;

  bool received_joy_{false};

  std::vector<Binding>
      bindings_;

  std::vector<int32_t>
      last_buttons_;

  rclcpp::Publisher<
      std_msgs::msg::String>::SharedPtr
      request_pub_;

  rclcpp::Subscription<
      sensor_msgs::msg::Joy>::SharedPtr
      joy_sub_;

  rclcpp::TimerBase::SharedPtr
      watchdog_;
};

}  // namespace noetix_bridge


int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(
        std::make_shared<
            noetix_bridge::GamepadModeNode>());
  }
  catch (const std::exception& e) {
    std::cerr
        << "noetix_gamepad_mode_node fatal: "
        << e.what()
        << std::endl;

    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
