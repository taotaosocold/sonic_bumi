#pragma once

#include <vector>

namespace bumi3 {
// 输入接口只负责取得操作键，不负责切换策略或修改控制状态；GUI 键由部署主类合并。
class InputInterface {
 public:
  virtual ~InputInterface() = default;
  virtual std::vector<int> ReadKeys() = 0;
};

}  // BUMI3 操作输入接口的命名空间。
