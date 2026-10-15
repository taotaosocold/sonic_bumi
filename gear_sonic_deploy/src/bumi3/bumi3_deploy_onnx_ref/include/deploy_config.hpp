#pragma once

#include "robot_state.hpp"
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <zlib.h>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace bumi3 {
namespace fs = std::filesystem;

// 启动参数与 G1 一样由原生控制程序解析；相对资产路径从仓库根目录解析，允许迁移机器。
struct DeployOptions {
  fs::path repository;
  std::string mode = "sim", encoder = "robot", input_type = "keyboard", provider;
  std::string config = "gear_sonic_deploy/bumi3/deploy.yaml";
  std::string obs_config, policy, smpl_policy, dataset, motion, motion_name, interface;
  std::string zmq_url, zmq_host, zmq_topic, output_dir, runtime_config;
  std::string engine_cache_dir;
  int zmq_port = 5556;
  int gpu_device = -1, trt_workspace_mb = 0;
  double duration = 0;
  bool autoplay = false, loop_motion = false, external_bridge = false, validate_only = false;
  bool help = false, capabilities = false;
  bool policy_fp16 = false, policy_fp32 = false, disable_cuda_graph = false;

  static DeployOptions Parse(int argc, char** argv) {
    DeployOptions options;
    // build/bumi3/bin 下的可执行文件向上查找配置锚点，不嵌入开发机器的编译路径。
    for (auto path = fs::canonical("/proc/self/exe").parent_path(); !path.empty(); path = path.parent_path()) {
      if (fs::is_regular_file(path / "gear_sonic_deploy/bumi3/deploy.yaml")) {
        options.repository = path;
        break;
      }
      if (path == path.root_path()) break;
    }
    if (options.repository.empty()) throw std::runtime_error("Cannot locate sonic_bumi repository from executable");
    std::map<std::string, std::string*> strings{
        {"--config", &options.config}, {"--obs-config", &options.obs_config},
        {"--policy", &options.policy}, {"--smpl-policy", &options.smpl_policy},
        {"--dataset", &options.dataset}, {"--motion", &options.motion},
        {"--motion-name", &options.motion_name}, {"--encoder", &options.encoder},
        {"--input-type", &options.input_type}, {"--provider", &options.provider},
        {"--interface", &options.interface}, {"--zmq-url", &options.zmq_url},
        {"--zmq-host", &options.zmq_host}, {"--zmq-topic", &options.zmq_topic},
        {"--engine-cache-dir", &options.engine_cache_dir},
        {"--prepare-launch", &options.output_dir}, {"--runtime-config", &options.runtime_config}};
    bool mode_seen = false;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i], value;
      const auto equals = key.find('=');
      const bool inline_value = equals != std::string::npos;
      if (inline_value) { value = key.substr(equals + 1); key.resize(equals); }
      const auto argument = [&]() {
        if (inline_value) return value;
        if (++i == argc) throw std::runtime_error("Missing value for " + key);
        return std::string(argv[i]);
      };
      if (strings.count(key)) *strings.at(key) = argument();
      else if (key == "--duration") {
        const auto text = argument(); std::size_t end = 0;
        options.duration = std::stod(text, &end);
        if (end != text.size() || !std::isfinite(options.duration) || options.duration <= 0)
          throw std::runtime_error("--duration must be finite and positive");
      } else if (key == "--zmq-port") {
        const auto text = argument(); std::size_t end = 0;
        options.zmq_port = std::stoi(text, &end);
        if (end != text.size() || options.zmq_port < 1 || options.zmq_port > 65535)
          throw std::runtime_error("--zmq-port must be in [1, 65535]");
      } else if (key == "--gpu-device" || key == "--trt-workspace-mb") {
        const auto text = argument();
        std::size_t end = 0;
        const int number = std::stoi(text, &end);
        if (end != text.size() || number < (key == "--gpu-device" ? 0 : 1) ||
            (key == "--trt-workspace-mb" && number > 65536))
          throw std::runtime_error("Invalid value for " + key);
        if (key == "--gpu-device") options.gpu_device = number;
        else options.trt_workspace_mb = number;
      } else if (inline_value) throw std::runtime_error("Unexpected value for " + key);
      else if (key == "--autoplay") options.autoplay = true;
      else if (key == "--loop-motion") options.loop_motion = true;
      else if (key == "--external-bridge") options.external_bridge = true;
      else if (key == "--validate-only") options.validate_only = true;
      else if (key == "--policy-fp16") options.policy_fp16 = true;
      else if (key == "--policy-fp32") options.policy_fp32 = true;
      else if (key == "--disable-cuda-graph") options.disable_cuda_graph = true;
      else if (key == "--help" || key == "-h") options.help = true;
      else if (key == "--capabilities") options.capabilities = true;
      else if ((key == "sim" || key == "real") && !mode_seen) { options.mode = key; mode_seen = true; }
      else throw std::runtime_error("Unknown argument: " + key);
    }
    return options;
  }

  fs::path Resolve(const std::string& path) const {
    const fs::path value(path);
    return fs::absolute(value.is_absolute() ? value : repository / value).lexically_normal();
  }

  static void PrintHelp() {
    std::cout << "BUMI3 native deployment: deploy.sh --robot-type bumi3 [sim|real] [options]\n"
                 "  --config YAML           deployment settings (repository relative)\n"
                 "  --obs-config YAML       observation/control contract\n"
                 "  --encoder robot|smpl    offline reference encoder\n"
                 "  --policy ONNX           offline policy (or live policy in live-only mode)\n"
                 "  --smpl-policy ONNX      live SMPL policy\n"
                 "  --dataset JSON          converted CSV manifest\n"
                 "  --motion PATH           converted motion directory or metadata.json\n"
                 "  --motion-name NAME      starting motion in manifest\n"
                 "  --input-type keyboard|zmq\n"
                 "  --zmq-url URL | --zmq-host HOST --zmq-port PORT\n"
                 "  --zmq-topic TOPIC       packed SMPL topic\n"
                 "  --provider tensorrt|cuda|cpu  inference backend (default: tensorrt; overrides YAML)\n"
                 "  --gpu-device INDEX      visible GPU index, default 0\n"
                 "  --policy-fp16           enable TensorRT FP16 (G1-style option)\n"
                 "  --policy-fp32           override YAML and use TensorRT FP32\n"
                 "  --engine-cache-dir DIR  TensorRT cache (repository relative)\n"
                 "  --trt-workspace-mb MB   TensorRT builder workspace, default 1024\n"
                 "  --disable-cuda-graph    disable TensorRT CUDA graph capture\n"
                 "  --interface NAME        local hardware EtherCAT interface\n"
                 "  --external-bridge       connect to an existing ROS2 hardware bridge\n"
                 "  --autoplay              start simulation reference playback\n"
                 "  --loop-motion           repeat offline reference\n"
                 "  --duration SECONDS      stop after the given time\n"
                 "  --validate-only         validate models/config without connecting a device\n";
  }
};

class DeploymentConfigParser {
 public:
  explicit DeploymentConfigParser(const DeployOptions& options) : options_(options) {}

  Json Load() {
    if (!options_.runtime_config.empty()) {
      if (!options_.output_dir.empty()) throw std::runtime_error("--runtime-config cannot be combined with --prepare-launch");
      return ReadJson(options_.Resolve(options_.runtime_config));
    }
    ValidateOptions();
    deploy_ = YAML::LoadFile(options_.Resolve(options_.config).string());
    if (deploy_["version"].as<int>() != 1) throw std::runtime_error("Deployment version must be 1");
    const auto inference = ReadInference();
    const auto obs_path = options_.Resolve(options_.obs_config.empty() ? deploy_["contract"].as<std::string>() : options_.obs_config);
    obs_ = YAML::LoadFile(obs_path.string());
    const Json fields = ReadContract();
    Json transport_fields = fields;
    transport_fields["joint_mappings"] = ReadJointMappings(fields.at("policy_joint_names"));
    transport_fields["imu_to_base_wxyz"] = ToJson(deploy_["hardware"]["imu_to_base_wxyz"]);
    const auto reference_id = ContractId(fields);
    const auto ros = ToJson(deploy_["ros2"]);
    const auto hardware = ToJson(deploy_["hardware"]);
    ValidateTransport(ros, hardware);
    const bool offline = options_.input_type == "keyboard" || !options_.motion.empty() || !options_.dataset.empty();
    Json runtime{{"version", 1}, {"mode", options_.mode}, {"encoder", options_.encoder},
                 {"input_type", options_.input_type}, {"provider", inference.at("provider")},
                 {"inference", inference},
                 {"contract", fields}, {"reference_contract_id", reference_id},
                 {"contract_id", ContractId(transport_fields)}, {"ros2", ros},
                 {"joint_mappings", transport_fields.at("joint_mappings")},
                 {"imu_to_base_wxyz", transport_fields.at("imu_to_base_wxyz")},
                 {"teleop", ToJson(deploy_["teleop"])}, {"autoplay", options_.autoplay},
                 {"loop_motion", options_.loop_motion}, {"validate_only", options_.validate_only},
                 {"duration", options_.duration > 0 ? Json(options_.duration) : Json(nullptr)},
                 {"start_bridge", options_.mode == "real" && !options_.external_bridge},
                 {"command_timeout", ros.at("command_timeout")},
                 {"io_frequency", hardware.at("io_frequency")}, {"entry_duration", hardware.at("entry_duration")},
                 {"motions", Json::array()}, {"policy", nullptr}, {"smpl_policy", nullptr}};
    if (offline) {
      runtime["motions"] = ReadPlaylist(reference_id, fields.at("policy_joint_names"));
      runtime["policy"] = ModelPath(options_.policy, options_.encoder);
    }
    if (options_.input_type == "zmq")
      runtime["smpl_policy"] = ModelPath(!options_.smpl_policy.empty() ? options_.smpl_policy : (!offline ? options_.policy : ""), "smpl");
    if (!options_.zmq_url.empty()) runtime["teleop"]["endpoint"] = options_.zmq_url;
    else if (!options_.zmq_host.empty()) runtime["teleop"]["endpoint"] = "tcp://" + options_.zmq_host + ":" + std::to_string(options_.zmq_port);
    if (!options_.zmq_topic.empty()) runtime["teleop"]["topic"] = options_.zmq_topic;
    ReadRobotModel(obs_path.parent_path() / obs_["model_path"].as<std::string>(), runtime);
    // 与 G1 一样将部署 YAML 的注册项交给主类绑定采集函数；这些字段不改变训练契约 CRC。
    if (obs_["observations"]) runtime["observations"] = ToJson(obs_["observations"]);
    if (obs_["encoder"]) runtime["encoder_config"] = ToJson(obs_["encoder"]);
    return runtime;
  }

  void PrepareLaunch(Json& runtime) const {
    const auto directory = fs::absolute(options_.output_dir);
    fs::create_directories(directory);
    if (runtime.at("start_bridge") == true && runtime.at("validate_only") == false) {
      const auto hardware = deploy_["hardware"];
      const auto root = directory / "robots/bumi3";
      fs::create_directories(root);
      fs::copy(options_.Resolve(hardware["robot_root"].as<std::string>()) / "config", root / "config", fs::copy_options::recursive);
      auto network = YAML::LoadFile(options_.Resolve(hardware["network"].as<std::string>()).string());
      if (!options_.interface.empty()) network["ethercat"]["ifname"] = options_.interface;
      if (!network["ethercat"]["ifname"] || network["ethercat"]["ifname"].as<std::string>().empty())
        throw std::runtime_error("Real deployment requires --interface or configured ethercat.ifname");
      const double cycle = network["ethercat"]["cycle_time_ns"] ? network["ethercat"]["cycle_time_ns"].as<double>() : 1000000.;
      if (std::abs(cycle * runtime.at("io_frequency").get<double>() - 1e9) > 1e4)
        throw std::runtime_error("hardware.io_frequency must match EtherCAT cycle_time_ns");
      const auto network_path = directory / "robots/common/bumi/network.yaml";
      fs::create_directories(network_path.parent_path());
      WriteYaml(network_path, network);
      const auto robot = YAML::LoadFile((root / "config/robot.yaml").string());
      YAML::Node params, node;
      node["robot_name"] = "bumi3"; node["robot_root"] = root.string();
      node["control_frequency_hz"] = RosDouble(runtime.at("io_frequency").get<double>());
      node["command_timeout"] = RosDouble(runtime.at("command_timeout").get<double>());
      std::map<std::string, double> efforts;
      const auto& names = runtime.at("contract").at("policy_joint_names");
      for (std::size_t i = 0; i < names.size(); ++i) efforts[names[i].get<std::string>()] = runtime.at("contract").at("effort_policy")[i];
      for (const auto& name : robot["robot"]["joint_order"])
        node["effort_limits"].push_back(RosDouble(efforts.at(name.as<std::string>())));
      params["/**"]["ros__parameters"] = node;
      const auto params_path = directory / "bridge_params.yaml";
      WriteYaml(params_path, params);
      runtime["robot_root"] = root.string(); runtime["bridge_params"] = params_path.string();
    }
    std::ofstream output(directory / "runtime.json");
    output << runtime.dump(2); output.close();
    if (!output) throw std::runtime_error("Cannot write launch configuration");
    // Shell 只接收启动 ROS2 桥所需的固定字段，策略推理不依赖 Python 启动预处理。
    std::cout << runtime.at("ros2").at("domain_id") << '\n'
              << (runtime.at("start_bridge") == true && runtime.at("validate_only") == false ? 1 : 0) << '\n';
    for (const auto* key : {"command_topic", "state_topic", "request_topic"})
      std::cout << runtime.at("ros2").at(key).get<std::string>() << '\n';
  }

 private:
  // yaml-cpp 默认会把整数值的 double 写成 1000，ROS 2 随即将其识别为整数参数。
  // 使用 JSON 的浮点文本构造普通 YAML 标量，保留 1000.0 等小数或指数形式，避免改变参数类型。
  // 标量不加引号，也不依赖 ROS 2 参数解析器忽略的 YAML 类型标签；数组元素同样适用。
  static YAML::Node RosDouble(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("ROS2 floating-point parameter must be finite");
    return YAML::Node(Json(value).dump());
  }

  static Json ReadJson(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open " + path.string());
    return Json::parse(stream);
  }

  static void WriteYaml(const fs::path& path, const YAML::Node& node) {
    std::ofstream stream(path); stream << node; stream.close();
    if (!stream) throw std::runtime_error("Cannot write " + path.string());
  }

  static Json ToJson(const YAML::Node& node) {
    if (!node || node.IsNull()) return nullptr;
    if (node.IsMap()) {
      Json result = Json::object();
      for (const auto& entry : node) result[entry.first.as<std::string>()] = ToJson(entry.second);
      return result;
    }
    if (node.IsSequence()) {
      Json result = Json::array();
      for (const auto& item : node) result.push_back(ToJson(item));
      return result;
    }
    const auto scalar = node.Scalar();
    if (node.Tag() == "!" || node.Tag() == "tag:yaml.org,2002:str") return scalar;
    if (scalar == "true" || scalar == "True") return true;
    if (scalar == "false" || scalar == "False") return false;
    try { return Json::parse(scalar); } catch (const Json::exception&) { return scalar; }
  }

  // 与离线转换器和仿真端的 Python JSON 序列化保持同一 CRC32：键排序、空格和浮点格式均参与签名。
  static std::string PythonJson(const Json& value) {
    if (value.is_object()) {
      std::string text = "{";
      for (auto it = value.begin(); it != value.end(); ++it) {
        if (it != value.begin()) text += ", ";
        text += Json(it.key()).dump(-1, ' ', true) + ": " + PythonJson(it.value());
      }
      return text + "}";
    }
    if (value.is_array()) {
      std::string text = "[";
      for (std::size_t i = 0; i < value.size(); ++i) { if (i) text += ", "; text += PythonJson(value[i]); }
      return text + "]";
    }
    if (!value.is_number_float()) return value.dump(-1, ' ', true);
    const double number = value.get<double>();
    if (!std::isfinite(number)) throw std::runtime_error("Nonfinite contract value");
    char buffer[128];
    const auto encoded = std::to_chars(buffer, buffer + sizeof(buffer), number, std::chars_format::general);
    if (encoded.ec != std::errc{}) throw std::runtime_error("Cannot serialize contract float");
    std::string text(buffer, encoded.ptr);
    const auto exponent_at = text.find('e');
    if (exponent_at != std::string::npos) {
      const int exponent = std::stoi(text.substr(exponent_at + 1));
      // Python 对 [1e-4, 1e16) 范围使用普通小数，其他值的指数至少保留两位。
      if (exponent >= -4 && exponent < 16) {
        const bool negative = text[0] == '-';
        std::string digits = text.substr(negative ? 1 : 0, exponent_at - (negative ? 1 : 0));
        const auto dot = digits.find('.');
        const int before = dot == std::string::npos ? int(digits.size()) : int(dot);
        if (dot != std::string::npos) digits.erase(dot, 1);
        const int position = before + exponent;
        if (position <= 0) text = "0." + std::string(-position, '0') + digits;
        else if (position >= int(digits.size())) text = digits + std::string(position - digits.size(), '0') + ".0";
        else { digits.insert(position, "."); text = digits; }
        if (negative) text.insert(0, "-");
      } else {
        std::ostringstream exp;
        exp << 'e' << (exponent < 0 ? '-' : '+') << std::setfill('0') << std::setw(2) << std::abs(exponent);
        text = text.substr(0, exponent_at) + exp.str();
      }
    } else if (text.find('.') == std::string::npos) text += ".0";
    return text;
  }

  static uint32_t ContractId(const Json& fields) {
    const auto bytes = PythonJson(fields);
    return static_cast<uint32_t>(crc32(0, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size()));
  }

  void ValidateOptions() const {
    if ((options_.encoder != "robot" && options_.encoder != "smpl") ||
        (options_.input_type != "keyboard" && options_.input_type != "zmq") ||
        (!options_.provider.empty() && options_.provider != "cpu" && options_.provider != "cuda" &&
         options_.provider != "tensorrt")) throw std::runtime_error("Unsupported encoder, input type or provider");
    if (options_.policy_fp16 && options_.policy_fp32) throw std::runtime_error("Choose --policy-fp16 or --policy-fp32");
    if (!options_.motion.empty() && (!options_.dataset.empty() || !options_.motion_name.empty()))
      throw std::runtime_error("--motion cannot be combined with --dataset or --motion-name");
    if (options_.mode == "real" && options_.autoplay) throw std::runtime_error("Real deployment requires ] to enable");
    if (options_.external_bridge && !options_.interface.empty()) throw std::runtime_error("External bridge cannot configure EtherCAT interface");
    if (options_.mode == "sim" && (options_.external_bridge || !options_.interface.empty())) throw std::runtime_error("Hardware bridge options are real-only");
  }

  static void Positive(double value, const std::string& name) {
    if (!std::isfinite(value) || value <= 0) throw std::runtime_error(name + " must be finite and positive");
  }

  // 与 G1 一样默认选择 TensorRT、GPU 0 和 FP32；配置文件和显式命令行参数可覆盖。
  // 当前会话没有固定设备输入输出绑定，默认关闭 CUDA Graph，避免图回放跳过临时输出分配。
  // 缺少 inference 配置时仍使用 GPU 默认值，防止旧配置静默切回 CPU；训练控制契约不参与更改。
  Json ReadInference() const {
    Json settings{{"provider", "tensorrt"}, {"device_id", 0}, {"fp16", false}, {"cuda_graph", false},
                  {"engine_cache_dir", ".local/bumi3/tensorrt"}, {"workspace_size_mb", 1024}};
    if (deploy_["inference"]) {
      const auto overrides = ToJson(deploy_["inference"]);
      if (!overrides.is_object()) throw std::runtime_error("inference must be a YAML mapping");
      for (const auto& item : overrides.items()) {
        if (!settings.contains(item.key())) throw std::runtime_error("Unknown inference option: " + item.key());
        settings[item.key()] = item.value();
      }
    }
    if (!options_.provider.empty()) settings["provider"] = options_.provider;
    if (options_.gpu_device >= 0) settings["device_id"] = options_.gpu_device;
    if (options_.trt_workspace_mb > 0) settings["workspace_size_mb"] = options_.trt_workspace_mb;
    if (options_.policy_fp16) settings["fp16"] = true;
    if (options_.policy_fp32) settings["fp16"] = false;
    if (options_.disable_cuda_graph) settings["cuda_graph"] = false;
    if (!options_.engine_cache_dir.empty()) settings["engine_cache_dir"] = options_.engine_cache_dir;
    const auto provider = settings.at("provider").get<std::string>();
    if (provider != "cpu" && provider != "cuda" && provider != "tensorrt")
      throw std::runtime_error("inference.provider must be cpu, cuda or tensorrt");
    if (!settings.at("device_id").is_number_integer() || settings.at("device_id").get<int>() < 0)
      throw std::runtime_error("inference.device_id must be a nonnegative integer");
    if (!settings.at("workspace_size_mb").is_number_integer() ||
        settings.at("workspace_size_mb").get<int>() < 1 || settings.at("workspace_size_mb").get<int>() > 65536)
      throw std::runtime_error("inference.workspace_size_mb must be an integer in [1,65536]");
    if (!settings.at("fp16").is_boolean() || !settings.at("cuda_graph").is_boolean())
      throw std::runtime_error("inference.fp16 and inference.cuda_graph must be boolean");
    if (settings.at("fp16") == true && provider != "tensorrt")
      throw std::runtime_error("Policy FP16 requires --provider tensorrt");
    const auto cache = settings.at("engine_cache_dir").get<std::string>();
    if (cache.empty()) throw std::runtime_error("inference.engine_cache_dir cannot be empty");
    settings["engine_cache_dir"] = options_.Resolve(cache).string();
    return settings;
  }

  void ValidateTransport(const Json& ros, const Json& hardware) const {
    const auto& domain = ros.at("domain_id");
    if (!domain.is_number_integer() || domain.get<int>() < 0 || domain.get<int>() > 232)
      throw std::runtime_error("ROS domain must be an integer in [0,232]");
    for (const auto* key : {"state_topic", "command_topic", "request_topic", "sim_reset_topic", "sim_status_topic", "sim_reference_topic"}) {
      const auto topic = ros.at(key).get<std::string>();
      if (topic.empty() || topic.front() != '/' || topic.find_first_of("\r\n") != std::string::npos) throw std::runtime_error("Invalid ROS topic");
    }
    for (const auto* key : {"state_timeout", "command_timeout"}) Positive(ros.at(key), key);
    for (const auto* key : {"entry_duration", "io_frequency"}) Positive(hardware.at(key), key);
    const auto teleop = ToJson(deploy_["teleop"]);
    for (const auto* key : {"startup_timeout", "stream_timeout"}) Positive(teleop.at(key), key);
    const auto rotation = hardware.at("imu_to_base_wxyz").get<std::vector<double>>();
    if (rotation.size() != 4) throw std::runtime_error("IMU rotation requires four values");
    QuaternionFromWxyz(rotation.data());
    double norm = 0;
    for (double value : rotation) norm += value * value;
    if (std::abs(std::sqrt(norm) - 1) > 1e-5) throw std::runtime_error("IMU rotation must be a unit quaternion");
  }

  Json ReadContract() const {
    const auto names = obs_["policy_joint_names"].as<std::vector<std::string>>();
    const auto mujoco_names = obs_["mujoco_joint_names"].as<std::vector<std::string>>();
    if (obs_["robot_type"].as<std::string>() != "bumi3" || names.size() != 21 || mujoco_names.size() != 21)
      throw std::runtime_error("Expected BUMI3 21-joint observation contract");
    const std::set<std::string> unique(names.begin(), names.end()), mj_unique(mujoco_names.begin(), mujoco_names.end());
    if (unique.size() != 21 || unique != mj_unique || obs_["default_joint_positions"].size() != 21)
      throw std::runtime_error("Joint names/defaults must cover 21 unique joints");
    // 与原 Python 契约一样检查既有训练顺序；只校验映射，不从另一机器人复制参数。
    const std::array<int, 21> expected{9,15,0,10,16,1,5,11,17,2,6,12,18,3,7,13,19,4,8,14,20};
    for (int i = 0; i < 21; ++i)
      if (std::distance(mujoco_names.begin(), std::find(mujoco_names.begin(), mujoco_names.end(), names[i])) != expected[i])
        throw std::runtime_error("BUMI3 training joint order mismatch");
    std::map<std::string, YAML::Node> parameters;
    for (const auto& entry : obs_["actuator_groups"]) {
      const auto group = entry.second;
      for (const auto& joint : group["joint_names"])
        if (!parameters.emplace(joint.as<std::string>(), group).second) throw std::runtime_error("Duplicate actuator group joint");
    }
    if (parameters.size() != 21) throw std::runtime_error("Actuator groups must cover 21 joints");
    const auto sim = obs_["simulation"], sonic = obs_["sonic"];
    Json fields{{"policy_joint_names", names}, {"sim_dt", sim["sim_dt"].as<double>()},
                {"decimation", sim["decimation"].as<int>()},
                {"control_dt", sim["sim_dt"].as<double>() * sim["decimation"].as<int>()},
                {"action_clip", sonic["action_clip"].as<double>()},
                {"history_length", sonic["history_length"].as<int>()},
                {"future_frames", sonic["num_future_frames"].as<int>()},
                {"future_stride", sonic["future_frame_stride"].as<int>()}};
    for (const auto& name : names) {
      const auto group = parameters.at(name);
      const double kp = group["stiffness"].as<double>(), effort = group["effort_limit"].as<double>();
      Positive(kp, "stiffness"); Positive(effort, "effort_limit"); Positive(group["velocity_limit"].as<double>(), "velocity_limit");
      const double kd = group["damping"].as<double>(), armature = group["armature"].as<double>();
      const double value = obs_["default_joint_positions"][name].as<double>();
      if (!std::isfinite(value) || !std::isfinite(kd) || kd < 0 || !std::isfinite(armature) || armature < 0)
        throw std::runtime_error("Invalid default posture or actuator parameter");
      fields["default_policy"].push_back(value); fields["stiffness_policy"].push_back(kp);
      fields["damping_policy"].push_back(kd); fields["effort_policy"].push_back(effort);
      fields["action_scale_policy"].push_back(.25 * effort / kp);
    }
    for (const auto& name : mujoco_names) fields["armature"].push_back(parameters.at(name)["armature"].as<double>());
    if (std::abs(fields.at("sim_dt").get<double>() - .005) > 1e-10 || fields.at("decimation") != 4 ||
        sim["target_fps"].as<double>() != 50 || fields.at("history_length") != 10 ||
        fields.at("future_frames") != 10 || fields.at("future_stride") != 5 ||
        sonic["action_dim"].as<int>() != 21 || sonic["token_dim"].as<int>() != 64 ||
        sonic["actor_proprioception_dim"].as<int>() != 690 || sonic["robot_tokenizer_dim"].as<int>() != 480 ||
        sonic["combined_policy_input_dim"].as<int>() != 1170) throw std::runtime_error("Unsupported BUMI3 training dimensions or timing");
    Positive(fields.at("action_clip"), "action_clip");
    return fields;
  }

  Json ReadJointMappings(const Json& names) const {
    const auto root = options_.Resolve(deploy_["hardware"]["robot_root"].as<std::string>());
    const auto robot = YAML::LoadFile((root / "config/robot.yaml").string());
    std::map<std::string, Json> mappings;
    for (const auto& item : robot["joints"]) {
      const Json mapping = ToJson(item);
      if (!mappings.emplace(mapping.at("name").get<std::string>(), mapping).second) throw std::runtime_error("Duplicate SDK joint");
    }
    if (mappings.size() != 21) throw std::runtime_error("SDK mapping requires 21 joints");
    Json result = Json::array(); std::set<std::string> actuators;
    for (const auto& name : names) {
      const auto& mapping = mappings.at(name.get<std::string>());
      const double direction = mapping.at("direction"), offset = mapping.at("offset");
      // 驱动器名称在 JSON 中是字符串，显式提取后插入集合，避免隐式转换导致 insert 重载歧义。
      if ((direction != 1 && direction != -1) || !std::isfinite(offset) ||
          !actuators.insert(mapping.at("actuator").get<std::string>()).second)
        throw std::runtime_error("Invalid SDK actuator mapping");
      result.push_back(mapping);
    }
    return result;
  }

  std::string ModelPath(const std::string& override, const std::string& encoder) const {
    const auto path = options_.Resolve(override.empty() ? deploy_["policies"][encoder].as<std::string>() : override);
    if (!fs::is_regular_file(path)) throw std::runtime_error("Missing ONNX model: " + path.string());
    return path.string();
  }

  Json ReadPlaylist(uint32_t reference_id, const Json& names) const {
    Json entries = Json::array(); std::set<std::string> seen;
    const auto append = [&](fs::path path, const std::string& name) {
      if (fs::is_directory(path)) path /= "metadata.json";
      const auto data = ReadJson(path);
      if (data.at("version") != 1 || data.at("format") != "bumi3_csv_v1" || data.at("encoder") != options_.encoder ||
          data.at("reference_contract_id") != reference_id || data.at("policy_joint_names") != names ||
          (!name.empty() && data.at("name") != name)) throw std::runtime_error("Converted motion contract mismatch: " + path.string());
      const std::string motion_name = data.at("name");
      if (!seen.insert(motion_name).second) throw std::runtime_error("Duplicate motion name: " + motion_name);
      entries.push_back(Json{{"name", motion_name}, {"reference", fs::absolute(path).string()}});
    };
    if (!options_.motion.empty()) append(options_.Resolve(options_.motion), "");
    else {
      const auto path = options_.Resolve(options_.dataset.empty() ? deploy_["dataset"].as<std::string>() : options_.dataset);
      const auto manifest = ReadJson(path);
      if (manifest.at("version") != 1 || manifest.at("format") != "bumi3_csv_v1" || manifest.at("robot_type") != "bumi3")
        throw std::runtime_error("--dataset requires a converted BUMI3 CSV manifest");
      for (const auto& entry : manifest.at("motions")) {
        const fs::path reference(entry.at(options_.encoder).get<std::string>());
        append(reference.is_absolute() ? reference : path.parent_path() / reference, entry.at("name"));
      }
      if (!options_.motion_name.empty()) {
        auto found = std::find_if(entries.begin(), entries.end(), [&](const Json& motion) { return motion.at("name") == options_.motion_name; });
        if (found == entries.end()) throw std::runtime_error("Motion not found: " + options_.motion_name);
        std::rotate(entries.begin(), found, entries.end());
      }
    }
    if (entries.empty()) throw std::runtime_error("Converted motion manifest is empty");
    return entries;
  }

  void ReadRobotModel(const fs::path& path, Json& runtime) const {
    char error[1024]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)), &mj_deleteModel);
    if (!model) throw std::runtime_error(std::string("Cannot load BUMI3 MJCF: ") + error);
    if (model->nq != 28 || model->nv != 27) throw std::runtime_error("BUMI3 MJCF requires nq=28, nv=27");
    std::vector<double> qpos(28, 0), lower, upper;
    const auto position = obs_["simulation"]["initial_root_position"].as<std::vector<double>>();
    const auto rotation = obs_["simulation"]["initial_root_quaternion_wxyz"].as<std::vector<double>>();
    if (position.size() != 3 || rotation.size() != 4) throw std::runtime_error("Invalid initial root pose");
    for (double value : position) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite initial position");
    const auto root = QuaternionFromWxyz(rotation.data());
    std::copy(position.begin(), position.end(), qpos.begin());
    qpos[3] = root.w(); qpos[4] = root.x(); qpos[5] = root.y(); qpos[6] = root.z();
    const auto& names = runtime.at("contract").at("policy_joint_names");
    for (int i = 0; i < 21; ++i) {
      const int joint = mj_name2id(model.get(), mjOBJ_JOINT, names[i].get<std::string>().c_str());
      if (joint < 0 || !model->jnt_limited[joint] || model->jnt_type[joint] != mjJNT_HINGE)
        throw std::runtime_error("Missing, unlimited or non-hinge BUMI3 joint");
      const double min = model->jnt_range[2 * joint], max = model->jnt_range[2 * joint + 1];
      if (!std::isfinite(min) || !std::isfinite(max) || min >= max) throw std::runtime_error("Invalid joint limits");
      lower.push_back(min); upper.push_back(max);
      qpos[model->jnt_qposadr[joint]] = runtime.at("contract").at("default_policy")[i];
    }
    runtime["static_qpos"] = qpos; runtime["joint_min"] = lower; runtime["joint_max"] = upper;
  }

  const DeployOptions& options_;
  YAML::Node deploy_, obs_;
};
}  // BUMI3 原生启动配置的命名空间。
