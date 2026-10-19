#pragma once

#include "policy_parameters.hpp"
#include <string>

namespace bumi3 {
// 类型名称沿用 G1；配置只描述输入项与顺序，具体采集函数仍由部署主类注册。
struct ObservationConfig {
  std::string name;
  bool enabled = true;
};

struct EncoderModeConfig {
  std::string name;
  int mode_id = -1;
  std::vector<std::string> required_observations;
};

struct EncoderConfig {
  int dimension = TOKEN_DIM;
  bool use_fp16 = false;
  std::vector<ObservationConfig> encoder_observations;
  std::vector<EncoderModeConfig> encoder_modes;
};

struct FullObservationConfig {
  std::vector<ObservationConfig> observations;
  EncoderConfig encoder;
};

class ObservationConfigParser {
 public:
  static std::vector<ObservationConfig> GetDefaultConfig() {
    return {{"his_base_angular_velocity_10frame_step1"}, {"his_body_joint_positions_10frame_step1"},
            {"his_body_joint_velocities_10frame_step1"}, {"his_last_actions_10frame_step1"},
            {"his_gravity_dir_10frame_step1"}};
  }

  static EncoderConfig GetDefaultEncoderConfig() {
    EncoderConfig result;
    const std::vector<std::string> robot{"motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5", "motion_anchor_orientation_10frame_step5"};
    const std::vector<std::string> smpl{"smpl_joints_10frame_step1", "smpl_anchor_orientation_10frame_step1"};
    for (const auto& name : robot) result.encoder_observations.push_back({name, true});
    for (const auto& name : smpl) result.encoder_observations.push_back({name, true});
    result.encoder_modes = {{"robot", 0, robot}, {"smpl", 1, smpl}};
    return result;
  }

  // 输入是 YAML 已解析并随启动配置传递的观测段，避免运行循环再读文件或访问 JSON 键。
  // 缺省项沿用原 BUMI3 默认值，已提供但错误的结构仍报错，不能静默改变训练输入。
  static FullObservationConfig ParseFullConfig(const Json& runtime) {
    FullObservationConfig result{GetDefaultConfig(), GetDefaultEncoderConfig()};
    if (runtime.contains("observations")) result.observations = ParseObservations(runtime.at("observations"));
    if (runtime.contains("encoder_config")) {
      const auto& encoder = runtime.at("encoder_config");
      result.encoder.dimension = encoder.at("dimension").get<int>();
      result.encoder.use_fp16 = encoder.value("use_fp16", false);
      result.encoder.encoder_observations = ParseObservations(encoder.at("encoder_observations"));
      result.encoder.encoder_modes.clear();
      for (const auto& mode : encoder.at("encoder_modes"))
        result.encoder.encoder_modes.push_back({mode.at("name").get<std::string>(), mode.at("mode_id").get<int>(),
            mode.at("required_observations").get<std::vector<std::string>>()});
    }
    return result;
  }

 private:
  static std::vector<ObservationConfig> ParseObservations(const Json& entries) {
    if (!entries.is_array()) throw std::runtime_error("Observation configuration must be an array");
    std::vector<ObservationConfig> result;
    for (const auto& entry : entries)
      result.push_back({entry.at("name").get<std::string>(), entry.value("enabled", true)});
    return result;
  }
};
}  // BUMI3 观测与编码器配置的命名空间。
