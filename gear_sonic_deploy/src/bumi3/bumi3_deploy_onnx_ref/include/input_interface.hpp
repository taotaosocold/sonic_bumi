#pragma once

#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace bumi3 {
// 输入接口只负责取得操作键，不负责切换策略或修改控制状态；GUI 键由部署主类合并。
class InputInterface {
 public:
  virtual ~InputInterface() = default;
  virtual std::vector<int> ReadKeys() = 0;
};

class KeyboardInput final : public InputInterface {
 public:
  KeyboardInput() {
    if (isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &saved_) == 0) {
      auto settings = saved_;
      settings.c_lflag &= ~(ICANON | ECHO);
      active_ = tcsetattr(STDIN_FILENO, TCSANOW, &settings) == 0;
    }
  }
  ~KeyboardInput() override {
    if (active_) tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
  }
  KeyboardInput(const KeyboardInput&) = delete;
  KeyboardInput& operator=(const KeyboardInput&) = delete;

  std::vector<int> ReadKeys() override {
    std::vector<int> keys;
    if (!active_) return keys;
    while (true) {
      fd_set descriptors;
      FD_ZERO(&descriptors);
      FD_SET(STDIN_FILENO, &descriptors);
      timeval timeout{};
      if (select(STDIN_FILENO + 1, &descriptors, nullptr, nullptr, &timeout) <= 0) break;
      unsigned char key;
      if (::read(STDIN_FILENO, &key, 1) != 1) break;
      keys.push_back(key);
    }
    return keys;
  }

 private:
  termios saved_{};
  bool active_ = false;
};
}  // BUMI3 操作输入的命名空间。
