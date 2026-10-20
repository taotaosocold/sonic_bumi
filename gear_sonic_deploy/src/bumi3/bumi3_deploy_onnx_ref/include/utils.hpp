#pragma once

#include "robot_parameters.hpp"
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace bumi3 {
// 沿用 G1 的带时间戳快照接口；读者持有不可变对象，后续写入不会改变已取出的命令。
template <typename T>
struct TimestampedData {
  std::shared_ptr<const T> data;
  Time::time_point timestamp{};

  double GetAgeMs() const {
    return data ? std::chrono::duration<double, std::milli>(Time::now() - timestamp).count() : -1.0;
  }
};

// 控制线程更新目标，发送线程只读取最近快照；时间戳只由生产者更新，重发不能续期旧目标。
template <typename T>
class DataBuffer {
 public:
  void SetData(T value) {
    auto data = std::make_shared<const T>(std::move(value));
    std::unique_lock<std::shared_mutex> lock(mutex_);
    snapshot_ = {std::move(data), Time::now()};
  }

  TimestampedData<T> GetDataWithTime() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return snapshot_;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    snapshot_ = {};
  }

 private:
  mutable std::shared_mutex mutex_;
  TimestampedData<T> snapshot_;
};
}  // BUMI3 部署缓冲工具的命名空间。
