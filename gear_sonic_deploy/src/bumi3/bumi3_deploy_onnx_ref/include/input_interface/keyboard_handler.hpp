#pragma once

#include "input_interface.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace bumi3 {
class SimpleKeyboard final : public InputInterface {
 public:
  SimpleKeyboard() {
    // 按键负责真机启动与停止，缺少交互终端时必须明确报错，不能静默忽略全部操作。
    if (!isatty(STDIN_FILENO))
      throw std::runtime_error("Keyboard stdin is not a terminal; run deploy.sh in a foreground terminal (SSH: ssh -t)");
    const auto foreground = tcgetpgrp(STDIN_FILENO);
    if (foreground < 0)
      throw std::runtime_error(std::string("Keyboard tcgetpgrp failed: ") + std::strerror(errno));
    if (foreground != getpgrp())
      throw std::runtime_error("Keyboard process is not in the terminal foreground; run deploy.sh without backgrounding");
    if (tcgetattr(STDIN_FILENO, &saved_) != 0)
      throw std::runtime_error(std::string("Keyboard tcgetattr failed: ") + std::strerror(errno));
    PrepareTerminal();
    active_ = true;
    std::cerr << "BUMI3_KEYBOARD_READY: foreground terminal; press keys without ENTER\n";
  }
  ~SimpleKeyboard() override {
    if (active_) tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
  }
  SimpleKeyboard(const SimpleKeyboard&) = delete;
  SimpleKeyboard& operator=(const SimpleKeyboard&) = delete;

  std::vector<int> ReadKeys() override {
    std::vector<int> keys;
    if (!active_) return keys;
    // 后台 sudo 或终端工具可能恢复规范输入；每轮读取前重新确认，保证单键无需回车。
    PrepareTerminal();
    while (true) {
      fd_set descriptors;
      FD_ZERO(&descriptors);
      FD_SET(STDIN_FILENO, &descriptors);
      timeval timeout{};
      const auto ready = select(STDIN_FILENO + 1, &descriptors, nullptr, nullptr, &timeout);
      if (ready < 0 && errno != EINTR)
        throw std::runtime_error(std::string("Keyboard select failed: ") + std::strerror(errno));
      if (ready <= 0) break;
      unsigned char key;
      if (::read(STDIN_FILENO, &key, 1) != 1) break;
      keys.push_back(key);
    }
    return keys;
  }

 private:
  // 保留终端其余设置，只关闭按行输入与回显，并令一次读取立即返回已有字节。
  // 退出时恢复构造时保存的设置；配置失败时抛出实际原因，避免显示就绪却无法操作。
  void PrepareTerminal() {
    termios settings{};
    if (tcgetattr(STDIN_FILENO, &settings) != 0)
      throw std::runtime_error(std::string("Keyboard tcgetattr failed: ") + std::strerror(errno));
    if (!(settings.c_lflag & (ICANON | ECHO)) && settings.c_cc[VMIN] == 0 && settings.c_cc[VTIME] == 0) return;
    settings.c_lflag &= ~(ICANON | ECHO);
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &settings) != 0)
      throw std::runtime_error(std::string("Keyboard tcsetattr failed: ") + std::strerror(errno));
  }

  termios saved_{};
  bool active_ = false;
};
}  // BUMI3 操作输入的命名空间。
