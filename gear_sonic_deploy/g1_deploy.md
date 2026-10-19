# G1 部署简明指南

本文整理当前仓库的 G1 环境搭建、离线 sim2sim／sim2real 和 PICO 遥操作流程。除注明 `cd gear_sonic_deploy` 的控制端命令外，其余命令均在 **`sonic_bumi` 仓库根目录**执行。

G1 是默认机器人，不需要添加 `--robot-type`。机器人状态与电机命令通过 **Unitree SDK／DDS** 通信，PICO 参考输入通过 **ZMQ** 传给策略控制端。

## 1. 首次环境配置

### 1.1 GPU 与 C++ 控制端

先安装 NVIDIA 驱动、CUDA Toolkit 和 TensorRT。仓库现有部署说明要求：

| 平台 | TensorRT |
| --- | --- |
| x86_64 部署电脑 | 10.13 |
| G1 板载 Jetson Orin | 10.7，配套 JetPack 6 |

详见 [原始部署环境说明](../docs/source/getting_started/installation_deploy.md)。TensorRT 需要自行安装，`deploy.sh` **不会自动下载 TensorRT**。下方假设已经将匹配版本解压到 `$HOME/TensorRT`，目录内包含 `include/`、`lib/` 等内容。

首次安装控制端依赖：

```bash
cd gear_sonic_deploy
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build
```

依赖安装会调用系统包管理器、下载组件；`just build` 编译 C++ 控制端。G1 的 `setup_env.sh` 还会尝试 Git LFS 等初始化操作。

以后每个新的控制终端设置实际的 TensorRT 根目录即可：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
```

`deploy.sh` 内部会自动 source `scripts/setup_env.sh` 并构建，无需重复手动 source。未设置根目录时，`deploy.sh` 会尝试已有的 `$HOME/TensorRT`；它不会搜索整台电脑。直接运行 C++ 可执行文件而绕过 `deploy.sh` 时，先手动 source 环境。

G1 的 DDS 仿真不依赖 Noetix ROS 消息包；基础 DDS 控制路径无需专门安装 ROS 2，ROS 输入等扩展功能另行配置。

### 1.2 MuJoCo Python 环境

在仓库根目录执行：

```bash
bash install_scripts/install_mujoco_sim.sh
```

该命令创建 **`.venv_sim`** 并安装 MuJoCo、Unitree Python SDK 等依赖。安装脚本会重建已有同名环境，正常使用时不必重复安装。

G1 的 MuJoCo 终端通常只需激活这个环境，不需要 source C++ 控制端的 `setup_env.sh`。BUMI3 目前也按自身安装命令配置同名 `.venv_sim`，额外 source `setup_env_bumi3.sh` 加载 ROS；切换机器人时应确认各自依赖已满足。

### 1.3 模型与参考数据

默认资产位置如下（相对于 `gear_sonic_deploy/`）：

| 内容 | 默认位置 |
| --- | --- |
| 策略编码器 | `policy/release/model_encoder.onnx` |
| 策略解码器 | `policy/release/model_decoder.onnx` |
| 观测配置 | `policy/release/observation_config.yaml` |
| 运动规划模型 | `planner/target_vel/V2/planner_sonic.onnx` |
| 离线参考 CSV | `reference/example/` |
| G1 机器人资源 | `g1/` |

克隆后先确认大文件已获取；使用 Git LFS 的文件需执行 `git lfs pull`。如果缺少官方模型，可在仓库根目录执行：

```bash
source .venv_sim/bin/activate
python -m pip install huggingface_hub
python download_from_hf.py
```

下载脚本准备策略模型、配置和 planner；参考运动目录仍需存在。已有 `reference/example/` CSV 可直接使用。自定义 G1 PKL 可通过 `gear_sonic_deploy/reference/convert_motions.py` 转换，控制端用 `--motion-data` 指定转换后的目录。

## 2. 离线 sim2sim

### 终端一：MuJoCo 环境

在仓库根目录：

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

### 终端二：键盘策略控制端

在仓库根目录：

```bash
cd gear_sonic_deploy
export TensorRT_ROOT="$HOME/TensorRT"
bash deploy.sh --input-type keyboard sim
```

脚本会加载环境、编译并询问是否部署，确认后进入 C++ 控制程序。若只执行 `bash deploy.sh sim`，当前默认输入类型是 `manager`；这里显式选择 `keyboard`，用于离线运动播放。

操作顺序：

1. 等待 `Init Done`：G1 初始化时会用约 3 秒进入配置中的默认站姿，**不是将关节直接设为运动第一帧**。
2. 在控制端终端按英文半角 **`]`** 启动策略。
3. 聚焦 MuJoCo 窗口，按 **`9`** 关闭默认悬挂辅助，让机器人落地；该键为开关，再次按下会恢复辅助。
4. 回到控制端按 **`T`** 播放当前运动。
5. **`N`／`P`** 切换下一条／上一条运动，按 `T` 播放；**`R`** 重置当前参考到首帧。
6. **`O`** 停止控制并退出。

指定自定义模型或动作目录，例如：

```bash
bash deploy.sh --input-type keyboard \
  --checkpoint policy/release/model \
  --obs-config policy/release/observation_config.yaml \
  --motion-data reference/example/ sim
```

`--checkpoint` 填模型公共前缀，脚本自动追加 `_encoder.onnx` 和 `_decoder.onnx`。

## 3. 离线 sim2real

先在仿真中熟悉启动和停止流程。真机运行前关闭 MuJoCo 和其他控制进程，确认机械支撑、供电、网络和实体急停可用；程序初始化阶段就会发送姿态控制命令。

部署电脑与 G1 的 DDS 网络连接正常后，在仓库根目录执行：

```bash
cd gear_sonic_deploy
export TensorRT_ROOT="$HOME/TensorRT"
bash deploy.sh --input-type keyboard real
```

`real` 自动查找 `192.168.123.x` 网络的本机接口。若自动识别不正确，显式传本机连接 G1 的网卡名，例如：

```bash
bash deploy.sh --input-type keyboard enp3s0
```

这里是 G1 的网卡位置参数，不使用 BUMI3 的 `--interface` 参数。程序会通过 Unitree `MotionSwitcherClient` 查询并释放原运动控制服务，然后进行默认站姿初始化。

等待 `Init Done` 后按 **`]`** 启动策略，按 **`T`** 播放运动；`N`／`P` 切换动作，`O` 停止退出。真机不需要运行 `run_sim_loop.py`，也没有 MuJoCo 窗口的 `9` 操作。

## 4. PICO 实时遥操作

### 4.1 首次准备

按 [PICO 硬件与服务配置](../docs/source/getting_started/vr_teleop_setup.md)安装 XRoboToolkit PC 服务、PICO 头显应用和全身追踪设备。PICO 与 PC 服务电脑连通，在头显填写该电脑 IP，完成校准并开启全身数据发送。

在未激活 venv／Conda 的新终端，从仓库根目录安装遥操作环境：

```bash
bash install_scripts/install_pico.sh
```

该脚本创建 `.venv_teleop`，会重建已有同名环境。如果提示从 `.venv_sim/bin/python` 查找 `Python.h` 失败，先退出已激活的虚拟环境／Conda 再安装。

### 4.2 PICO → G1 仿真：三个终端

**终端一：MuJoCo**，在仓库根目录：

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**终端二：策略控制端**，在仓库根目录：

```bash
cd gear_sonic_deploy
export TensorRT_ROOT="$HOME/TensorRT"
bash deploy.sh --input-type zmq_manager sim
```

**终端三：PICO 发送端**，在仓库根目录：

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

这里使用 G1 原有的 `--manager`，无需 BUMI3 的 `--smpl-only --num_frames_to_send 10`。可选添加 `--vis_vr3pt --vis_smpl` 显示可视化。

控制端显示 `Init Done`、发送端已收到全身追踪数据后：

1. 操作者直立，双脚并拢，上臂自然垂下，肘部弯曲约 90°、前臂向前，手掌相对，准备校准。
2. 同时按 PICO 的 **`A+B+X+Y`** 启动控制并校准。
3. 仿真需要时在 MuJoCo 窗口按 **`9`** 解除悬挂辅助。
4. 手臂与机器人当前姿态对齐后，按 **`A+X`** 进入全身 SMPL 跟随。
5. 再按 **`A+X`** 返回 planner 模式；这不代表退出策略。
6. 再按 **`A+B+X+Y`** 停止，或在控制端按 **`O`** 停止退出。

### 4.3 PICO → G1 真机

关闭 MuJoCo 和仿真控制端，保留 PICO 发送端。控制端改为：

```bash
cd gear_sonic_deploy
export TensorRT_ROOT="$HOME/TensorRT"
bash deploy.sh --input-type zmq_manager real
```

按上一节完成校准、启动和模式切换；真机不执行 MuJoCo 的 `9` 操作。发送端与策略分属两台电脑时，控制端追加 `--zmq-host <PICO发送端电脑IP>`；默认连接本机 5556 端口。

## 5. 关键区别与排错

- **环境加载：**G1 MuJoCo 端通常只激活 `.venv_sim`；C++ 控制端由 `deploy.sh` 自动加载 `setup_env.sh`，不传 BUMI3 参数。
- **通信：**PICO 到策略用 ZMQ；策略到 G1 真机／MuJoCo 用 DDS。不要按 BUMI3 的 Noetix ROS 话题流程配置 G1。
- **TensorRT 找不到：**确认已安装，并使 `TensorRT_ROOT` 指向包含 `lib/` 和 `include/` 的真实根目录。若布局不同，需要额外补充实际库路径。
- **模型缺失：**核对上述默认资产位置，补齐 Git LFS 或模型下载；G1 的 encoder、decoder 和 planner 要与配置配套。
- **默认输入模式：**当前脚本实际默认 `manager`；键盘离线用 `--input-type keyboard`，PICO 用 `--input-type zmq_manager`。部分旧帮助文字的默认值与代码不同，以实际配置为准。
- **停止含义：**G1 控制程序退出时发送阻尼命令；不要将停止程序等同于电机完全断力，实体急停仍按机器人规范使用。
