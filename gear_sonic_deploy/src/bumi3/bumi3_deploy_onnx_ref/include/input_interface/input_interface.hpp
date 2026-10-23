#pragma once

#include <vector>

namespace bumi3 {
// 与 G1 键盘输入同名的单次操作请求；控制线程决定何时应用到机器人与参考状态。
struct InputCommand {
  bool start_control = false;
  bool stop_control = false;
  bool motion_prev = false;
  bool motion_next = false;
  bool play_motion = false;
  bool motion_restart = false;
  bool reinitialize = false;
  bool switch_source = false;
};

// update 采集操作，handle_input 取出本次请求；输入线程不直接改变控制状态。
class InputInterface {
 public:
  virtual ~InputInterface() = default;
  virtual void update(const std::vector<int>& external_keys = {}) = 0;
  virtual InputCommand handle_input() = 0;
  virtual std::vector<int> ReadKeys() = 0;

  // 终端与 MuJoCo 窗口共用键位：] 接管、T 播放、R 重置暂停、N/P 下一条/上一条。
  static InputCommand DecodeKeys(const std::vector<int>& keys) {
    InputCommand command;
    for (auto key : keys) {
      if (key >= 'a' && key <= 'z') key -= 'a' - 'A';
      switch (key) {
        case ']': command.start_control = true; break;
        case 'O': command.stop_control = true; break;
        case 'P': command.motion_prev = true; break;
        case 'N': command.motion_next = true; break;
        case 'T': command.play_motion = true; break;
        case 'R': command.motion_restart = true; break;
        case 'I': command.reinitialize = true; break;
        case '\n':
        case '\r':
        case 257: command.switch_source = true; break;
        default: break;
      }
    }
    return command;
  }
};

}  // BUMI3 操作输入接口的命名空间。
