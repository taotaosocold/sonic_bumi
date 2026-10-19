#pragma once

#include "../robot_parameters.hpp"
#include "../math_utils.hpp"
#include <zmq.hpp>
#include <cstring>
#include <string>
#include <set>

namespace bumi3 {
// 与 G1 共用 PICO 的打包消息协议；BUMI3 只使用已训练的完整 SMPL 输入。
class ZMQEndpointInterface {
 public:
  explicit ZMQEndpointInterface(const Json& config)
      : socket_(context_, zmq::socket_type::sub), topic_(config.at("topic")), timeout_(config.at("stream_timeout")) {
    socket_.set(zmq::sockopt::linger, 0);
    socket_.set(zmq::sockopt::conflate, 1);
    socket_.set(zmq::sockopt::subscribe, topic_);
    socket_.connect(config.at("endpoint").get<std::string>());
  }

  PoseWindow WaitForPose(double timeout) {
    valid_ = false;
    window_.last = -1;
    const auto deadline = Time::now() + std::chrono::duration<double>(timeout);
    while (Time::now() < deadline && !stopping) {
      Poll(20);
      if (valid_) return window_;
    }
    throw std::runtime_error("No complete SMPL window; start PICO with --smpl-only --num_frames_to_send 10");
  }

  PoseWindow GetLatestPose() {
    Poll(0);
    if (!valid_ || std::chrono::duration<double>(Time::now() - received_).count() > timeout_)
      throw std::runtime_error("SMPL stream stale; stopping control");
    return window_;
  }

 private:
  template <typename T>
  static double ReadNumber(const char* bytes, bool swap) {
    std::array<char, sizeof(T)> storage;
    std::memcpy(storage.data(), bytes, sizeof(T));
    if (swap) std::reverse(storage.begin(), storage.end());
    T value;
    std::memcpy(&value, storage.data(), sizeof(T));
    return double(value);
  }

  void Poll(int timeout) {
    zmq::pollitem_t item{socket_.handle(), 0, ZMQ_POLLIN, 0};
    zmq::poll(&item, 1, std::chrono::milliseconds(timeout));
    if (!(item.revents & ZMQ_POLLIN)) return;
    zmq::message_t packet;
    if (!socket_.recv(packet)) return;
    const auto* bytes = static_cast<const char*>(packet.data());
    const std::size_t size = packet.size(), header_start = topic_.size(), payload = header_start + 1280;
    if (size < payload || std::memcmp(bytes, topic_.data(), topic_.size()) != 0)
      throw std::runtime_error("Invalid pose packet");
    const std::string header_text(bytes + header_start, strnlen(bytes + header_start, 1280));
    const auto header = Json::parse(header_text);
    const auto endian = header.value("endian", std::string("le"));
    if (endian != "le" && endian != "be") throw std::runtime_error("Invalid pose endianness");
    const uint16_t marker = 1;
    const bool host_little = *reinterpret_cast<const unsigned char*>(&marker) == 1;
    const bool swap = host_little != (endian == "le");
    std::size_t offset = payload;
    Json shapes;
    std::set<std::string> names;
    std::vector<double> joints, roots, indices;
    for (const auto& field : header.at("fields")) {
      const auto dtype = field.at("dtype").get<std::string>();
      const auto name = field.at("name").get<std::string>();
      if (!names.insert(name).second) throw std::runtime_error("Duplicate pose field: " + name);
      std::size_t count = 1;
      for (int dimension : field.at("shape")) {
        if (dimension < 0 || dimension > 100000) throw std::runtime_error("Invalid pose shape");
        count *= dimension;
        if (count > 1000000) throw std::runtime_error("Pose field too large");
      }
      const int width = (dtype == "f64" || dtype == "i64") ? 8 :
                        (dtype == "f32" || dtype == "i32") ? 4 : dtype == "bool" ? 1 : 0;
      if (!width || offset + count * width > size) throw std::runtime_error("Invalid pose payload");
      auto* output = name == "smpl_joints" ? &joints : name == "body_quat_w" ? &roots : name == "frame_index" ? &indices : nullptr;
      if (output) {
        shapes[name] = field.at("shape");
        output->resize(count);
        for (std::size_t i = 0; i < count; ++i) {
          const auto* source = bytes + offset + i * width;
          double value;
          if (dtype == "f32") value = ReadNumber<float>(source, swap);
          else if (dtype == "f64") value = ReadNumber<double>(source, swap);
          else if (dtype == "i64") value = ReadNumber<int64_t>(source, swap);
          else if (dtype == "i32") value = ReadNumber<int32_t>(source, swap);
          else throw std::runtime_error("Invalid pose numeric dtype");
          if (!std::isfinite(value)) throw std::runtime_error("Nonfinite SMPL reference");
          (*output)[i] = value;
        }
      }
      offset += count * width;
    }
    const auto frames = indices.size();
    if (offset != size || frames < 10 || shapes["smpl_joints"] != Json::array({frames, 24, 3}) ||
        shapes["body_quat_w"] != Json::array({frames, 4}) || shapes["frame_index"] != Json::array({frames}))
      throw std::runtime_error("Pose requires ten joints/orientation/index samples");
    // 帧序号必须是可精确表示的非负整数；最后十帧必须连续，不能把缺帧窗口直接送给策略。
    for (std::size_t i = frames - 10; i < frames; ++i)
      if (indices[i] < 0 || indices[i] >= 9007199254740992. || indices[i] != std::floor(indices[i]))
        throw std::runtime_error("Invalid SMPL frame index");
    for (std::size_t i = frames - 9; i < frames; ++i)
      if (indices[i] - indices[i - 1] != 1) throw std::runtime_error("SMPL window must have consecutive 50 Hz frames");
    // 重复帧不能刷新接收时间，否则停止发送的上游会被误判为仍然在线。
    if (indices.back() <= window_.last) return;
    PoseWindow next;
    next.last = int64_t(indices.back());
    std::copy(joints.begin() + (frames - 10) * 72, joints.end(), next.joints.begin());
    for (int i = 0; i < 10; ++i) next.roots[i] = QuaternionFromWxyz(roots.data() + (frames - 10 + i) * 4);
    window_ = next;
    valid_ = true;
    received_ = Time::now();
  }

  zmq::context_t context_{1};
  zmq::socket_t socket_;
  std::string topic_;
  double timeout_;
  PoseWindow window_;
  bool valid_ = false;
  Time::time_point received_{};
};
}  // BUMI3 SMPL 流输入的命名空间。
