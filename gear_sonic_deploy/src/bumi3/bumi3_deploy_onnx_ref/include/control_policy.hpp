#pragma once

#include "robot_parameters.hpp"
#include <onnxruntime_cxx_api.h>
#include <zlib.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace bumi3 {
// 联合模型已包含编码器和动态解码器；此处仅管理会话、张量契约和动作限幅。
class PolicyEngine {
 public:
  PolicyEngine() : env_(ORT_LOGGING_LEVEL_WARNING, "bumi3") {}

  // 沿用 G1 的显式 Initialize 生命周期，在机器人连接前加载和预热模型；
  // 联合 ONNX 的参考维度由所选 Robot／SMPL 模式决定，不拆分或重新导出模型。
  bool Initialize(const std::string& path, int input_dimension, const Json& settings) {
    input_dimension_ = input_dimension;
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(2);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    const auto provider = settings.value("provider", std::string("tensorrt"));
    ConfigureProviders(options, settings, path);
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options);
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
      throw std::runtime_error("Joint ONNX requires one input/output");
    auto input_info = session_->GetInputTypeInfo(0);
    auto output_info = session_->GetOutputTypeInfo(0);
    const auto input_tensor = input_info.GetTensorTypeAndShapeInfo();
    const auto output_tensor = output_info.GetTensorTypeAndShapeInfo();
    const auto input_shape = input_tensor.GetShape(), output_shape = output_tensor.GetShape();
    if (input_shape.size() != 2 || output_shape.size() != 2 ||
        input_shape[1] != input_dimension || output_shape[1] != BUMI3_NUM_MOTOR ||
        (input_shape[0] != 1 && input_shape[0] != -1) || (output_shape[0] != 1 && output_shape[0] != -1) ||
        input_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("BUMI3 ONNX input/output contract mismatch");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    policy_input_buffer_.assign(input_dimension_, 0);
    action_buffer_.assign(BUMI3_NUM_MOTOR, 0);
    if (provider != "cpu") {
      // 首次 GPU 推理可能构建 TensorRT 引擎和分配设备内存；在连接机器人前预热，默认关闭图捕获。
      // 此处的零输入只初始化会话，不进入观测历史，也不会发送任何机器人命令。
      Infer(20.0);
      Infer(20.0);
    }
    std::cout << "BUMI3_POLICY_BACKEND=" << provider
              << " device=" << settings.value("device_id", 0)
              << " precision=" << (provider == "tensorrt" && settings.value("fp16", false) ? "fp16" : "fp32")
              << " model=" << path << '\n';
    return true;
  }

  // 与 G1 的 PolicyEngine 一样持有输入和动作缓冲区；GPU 预热与控制循环复用同一份输入存储。
  std::vector<float>& GetInputBuffer() { return policy_input_buffer_; }
  const std::vector<double>& GetActionBuffer() const { return action_buffer_; }

  bool Infer(double clip = 20.0) {
    auto& observation = policy_input_buffer_;
    if (observation.size() != std::size_t(input_dimension_)) throw std::runtime_error("Observation dimension mismatch");
    for (float value : observation)
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite observation");
    const std::array<int64_t, 2> shape{1, input_dimension_};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, observation.data(), observation.size(), shape.data(), 2);
    const char* input = input_name_.c_str();
    const char* output = output_name_.c_str();
    auto results = session_->Run(Ort::RunOptions{nullptr}, &input, &tensor, 1, &output, 1);
    if (results[0].GetTensorTypeAndShapeInfo().GetElementCount() != BUMI3_NUM_MOTOR) throw std::runtime_error("Invalid ONNX output");
    const auto* values = results[0].GetTensorData<float>();
    for (int i = 0; i < BUMI3_NUM_MOTOR; ++i) {
      if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite policy action");
      action_buffer_[i] = std::clamp(double(values[i]), -clip, clip);
    }
    return true;
  }

 private:
  static void RequireProvider(const std::string& name) {
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), name) == providers.end()) {
      throw std::runtime_error(name + " is unavailable in the linked ONNX Runtime C++ SDK. "
                               "Install a GPU SDK and set ONNX_INSTALL_PATH before rebuilding deploy.sh.");
    }
  }

  // 模型内容和推理参数参与缓存目录命名，避免更新同名 ONNX 或切换精度后误用旧引擎。
  // GPU 架构与 TensorRT 引擎兼容性由执行后端检查；更换环境后仍应使用本机生成的缓存。
  static std::string ModelCacheKey(const std::string& path, const Json& settings, int dimension) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open ONNX for engine cache: " + path);
    uLong checksum = crc32(0, nullptr, 0);
    std::array<char, 65536> bytes;
    while (stream.read(bytes.data(), bytes.size()) || stream.gcount())
      checksum = crc32(checksum, reinterpret_cast<const Bytef*>(bytes.data()), stream.gcount());
    if (!stream.eof()) throw std::runtime_error("Cannot read ONNX for engine cache: " + path);
    std::ostringstream key;
    key << std::hex << checksum << "_ort" << Ort::GetVersionString()
        << "_in" << std::dec << dimension << "_gpu" << settings.value("device_id", 0)
        << (settings.value("fp16", false) ? "_fp16" : "_fp32")
        << "_ws" << settings.value("workspace_size_mb", 1024)
        << (settings.value("cuda_graph", false) ? "_graph" : "_nograph");
    return key.str();
  }

  void ConfigureProviders(Ort::SessionOptions& options, const Json& settings, const std::string& path) {
    const auto provider = settings.value("provider", std::string("tensorrt"));
    const int device = settings.value("device_id", 0);
    const int workspace = settings.value("workspace_size_mb", 1024);
    if (device < 0 || workspace < 1 || workspace > 65536)
      throw std::runtime_error("Invalid GPU device or TensorRT workspace size");
    if (settings.value("fp16", false) && provider != "tensorrt")
      throw std::runtime_error("Policy FP16 requires --provider tensorrt");
    if (provider == "cpu") return;
    if (provider != "cuda" && provider != "tensorrt")
      throw std::runtime_error("Unsupported ONNX Runtime provider: " + provider);
    RequireProvider("CUDAExecutionProvider");
    if (provider == "tensorrt") {
      RequireProvider("TensorrtExecutionProvider");
      const auto root = settings.value("engine_cache_dir", std::string());
      if (root.empty()) throw std::runtime_error("TensorRT requires inference.engine_cache_dir");
      const auto cache = std::filesystem::path(root) / ModelCacheKey(path, settings, input_dimension_);
      std::filesystem::create_directories(cache);
      Ort::TensorRTProviderOptions tensor_rt;
      tensor_rt.Update({{"device_id", std::to_string(device)},
                        {"trt_max_workspace_size", std::to_string(uint64_t(workspace) * 1024 * 1024)},
                        {"trt_fp16_enable", settings.value("fp16", false) ? "1" : "0"},
                        {"trt_engine_cache_enable", "1"}, {"trt_engine_cache_path", cache.string()},
                        {"trt_timing_cache_enable", "1"}, {"trt_timing_cache_path", cache.string()},
                        {"trt_force_timing_cache", "0"},
                        {"trt_cuda_graph_enable", settings.value("cuda_graph", false) ? "1" : "0"}});
      options.AppendExecutionProvider_TensorRT_V2(*static_cast<OrtTensorRTProviderOptionsV2*>(tensor_rt));
      std::cout << "BUMI3_TENSORRT_CACHE=" << cache.string() << '\n';
    }
    // TensorRT 优先，其不支持的算子交给同一设备的 CUDA 后端；CPU 模式不会注册 GPU 后端。
    // 使用 V2 配置 API 即可在 CPU SDK 上编译，无需此机器安装 CUDA/TensorRT 开发头文件。
    Ort::CUDAProviderOptions cuda;
    cuda.Update({{"device_id", std::to_string(device)}, {"do_copy_in_default_stream", "1"}});
    options.AppendExecutionProvider_CUDA_V2(*static_cast<OrtCUDAProviderOptionsV2*>(cuda));
  }

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  int input_dimension_ = 0;
  std::string input_name_, output_name_;
  std::vector<float> policy_input_buffer_;
  std::vector<double> action_buffer_;
};
}  // BUMI3 原生策略推理的命名空间。
