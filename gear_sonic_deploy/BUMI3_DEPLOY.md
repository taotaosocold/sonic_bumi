# BUMI3 部署简明指南

支持离线 sim2sim、离线 sim2real 和 PICO 实时 SMPL 遥操作。以下命令均在 **`sonic_bumi` 仓库根目录**执行；配置和资产路径均相对于仓库根目录。

## 1. 环境搭建

### 基础环境

以下以 **Ubuntu 22.04、x86_64、NVIDIA GPU、ROS 2 Humble** 为例。先按 [ROS 2 官方说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)配置软件源，然后安装：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-ros-base ros-humble-ament-cmake \
  ros-humble-rosidl-default-generators \
  ros-humble-rclcpp ros-humble-rclpy ros-humble-std-msgs \
  build-essential cmake pkg-config curl \
  python3-dev python3-venv python3-pip \
  libeigen3-dev libyaml-cpp-dev nlohmann-json3-dev \
  libzmq3-dev zlib1g-dev libgl1 libegl1 libglfw3

bash install_scripts/install_mujoco_sim.sh --robot-type bumi3
```

该脚本创建 `.venv_sim`、编译 ROS 消息包并安装 ONNX Runtime GPU C++ SDK，默认位置为 `/opt/onnxruntime-gpu/1.23.2`。

G1 和 BUMI3 当前使用同名 `.venv_sim`，按实际机器人选择安装参数，并确认对应依赖完整。G1 安装会重建该目录，切换前应注意已有环境。BUMI3 安装目录可通过 `BUMI_DEPLOY_VENV` 自定义；自定义控制端解释器需另外设置 `BUMI_DEPLOY_PYTHON`。

**NVIDIA 驱动、CUDA 12、cuDNN 9 和兼容的 TensorRT 10 运行库需要另外安装**，脚本不会自动安装。版本应与 ONNX Runtime 匹配，参见 [TensorRT 后端依赖说明](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)。`nvidia-smi` 显示的 CUDA 版本不代表相应运行库已安装。

### 每个新的仿真／控制终端

```bash
source .venv_sim/bin/activate
source gear_sonic_deploy/scripts/setup_env_bumi3.sh
```

仿真和控制端统一读取 `gear_sonic/utils/mujoco_sim/wbc_configs/bumi3_21dof_sonic_model12.yaml`。顶部 `DOMAIN_ID` 默认是 `0`，通过 YAML 引用提供给 `ros2.domain_id`，修改一处即可。两端自动设置通信域，无需手动 export；另开终端用 `ros2 topic` 查看话题时可执行 `export ROS_DOMAIN_ID=0`。本机设置可复制到 `.local/bumi3.yaml`，仿真和控制端均通过 `--config .local/bumi3.yaml` 选择同一配置。

G1 使用原有 `setup_env.sh`；BUMI3 使用 `setup_env_bumi3.sh`。`deploy.sh --robot-type bumi3 ...` 已自动加载 BUMI3 环境，控制端可省略手动 source；独立运行的 MuJoCo 端和关节检查脚本仍需在自己的终端加载。

`setup_env_bumi3.sh` 被 source 时只加载 ROS 环境；直接执行 `bash gear_sonic_deploy/scripts/setup_env_bumi3.sh sim` 或 `real` 时编译相应 ROS 包。安装脚本和 `deploy.sh` 会自动调用编译流程。

**策略控制端还需要 GPU 库路径**。`deploy.sh` 会检查已设置的 TensorRT 根目录，以及 `$HOME/TensorRT/lib`、仓库 `.local/bumi3/tensorrt-sdk` 等约定位置。安装在其他位置时按实际目录设置，例如：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TensorRT_ROOT="$HOME/TensorRT"
# 此目录应包含 libnvinfer.so.10 和 libnvonnxparser.so.10；按实际安装布局选择。
export BUMI_TRT_LIB="$TensorRT_ROOT/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$BUMI_TRT_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

若 TensorRT 位于 `targets/x86_64-linux-gnu/lib`，将 `BUMI_TRT_LIB` 改为该子目录；若 cuDNN 安装在其他目录，也将其库目录加入 `LD_LIBRARY_PATH`。默认使用 **TensorRT GPU 推理**；首次启动构建引擎可能较慢，缓存保存于 `.local/bumi3/tensorrt/`。

## 2. 模型与离线运动数据

| 内容 | 仓库内位置 |
| --- | --- |
| 统一仿真／部署配置、模型路径与 ROS 话题 | `gear_sonic/utils/mujoco_sim/wbc_configs/bumi3_21dof_sonic_model12.yaml` |
| Robot 模型 | `gear_sonic_deploy/policy/bumi3/robot/` |
| SMPL 模型 | `gear_sonic_deploy/policy/bumi3/smpl/` |
| 观测与控制配置 | `gear_sonic_deploy/policy/bumi3/observation_config.yaml` |
| 原始配对 PKL 与数据清单 | `gear_sonic_deploy/reference/bumi3_example/pkl/` |
| 转换后的 CSV 与播放清单 | `gear_sonic_deploy/reference/bumi3_example/converted/` |

克隆后确认 ONNX、PKL 或已转换的 CSV 确实存在；若文件由 Git LFS 管理，执行 `git lfs pull`。未随仓库分发的模型和数据需要另行复制到上述位置，实际 ONNX 文件名需与统一配置中的 `policies` 一致。

已有完整 `converted/` 时无需重新转换；否则准备好默认 `pkl/dataset_smoke_24.json` 引用的 Robot／SMPL PKL 后执行：

```bash
source .venv_sim/bin/activate
python gear_sonic_deploy/reference/bumi3_convert_motion.py
```

默认同时转换 Robot 和 SMPL 数据。自定义原始清单使用 `--dataset <原始清单.json>`；转换会重写输出目录的 `dataset.json`，扩充已有播放清单时应先备份或通过 `--output-dir` 输出到新目录。

## 3. 离线 sim2sim

打开两个终端，分别执行第 1 节的环境加载；**终端二另外加载 GPU 库路径**。

### 终端一：MuJoCo 环境

```bash
python gear_sonic/scripts/run_sim_loop.py --robot-type bumi3
```

看到 `BUMI3_SIM_READY` 后启动控制端。

### 终端二：策略控制端

Robot 模型：

```bash
bash gear_sonic_deploy/deploy.sh --robot-type bumi3 sim
```

SMPL 模型：

```bash
bash gear_sonic_deploy/deploy.sh --robot-type bumi3 sim --encoder smpl
```

两条命令任选一条。指定起始运动时追加 `--motion-name '<播放清单中的运动名称>'`；未指定则从清单第一条开始。自定义转换清单可追加 `--dataset <转换后的dataset.json>`。

仿真自动重置到选中运动第一帧，策略开始运行；按 **`T`** 开始播放参考运动，**无需按 `]`**。

## 4. 离线 sim2real

本节使用 **部署电脑直接连接 EtherCAT 网卡和 USB IMU** 的方式，环境桥与策略在同一台电脑运行，不需要启动 MuJoCo。

首次使用先确认：

- 内置 SDK 位于 `gear_sonic_deploy/thirdparty/new_ws/src/noetix_sdk2/`，具有部署电脑架构对应的厂商库；x86_64 需要 `third-party/vendor/hw_controller/bumi/x86_64/lib/libhw_controller.a`，ARM 库不能用于 x86_64。
- 按实际硬件设置 `gear_sonic_deploy/thirdparty/new_ws/src/noetix_robots/common/bumi/network.yaml`。USB IMU 示例为 `imu.usb.enable: true`、`imu.usb.port: /dev/ttyACM0`；网卡通过启动参数指定。
- 机器人供电、网线、IMU 和实体急停正常；首次动作前完成关节映射检查，并提供机械支撑。一次只运行一个机器人控制程序。

运行前先导入安装的TensorRT的路径和CUDA路径，不同电脑上如果路径不一样去修改一下两个变量的路径即可：
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TensorRT_ROOT="$HOME/sonic/.deps/TensorRT-10.13.0.35"
```
然后执行命令
```bash
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$TensorRT_ROOT/targets/x86_64-linux-gnu/lib:$TensorRT_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:$HOME/app/miniconda3/lib"
```

新终端加载第 1 节的环境和 GPU 库路径，再执行（替换为本机 EtherCAT 网卡）：

```bash
BUMI_BRIDGE_SUDO=1 bash gear_sonic_deploy/deploy.sh \
  --robot-type bumi3 real --interface enp132s0
```

启动脚本会编译并启动 Noetix ROS 2 硬件桥和 C++ 策略端。可追加 `--encoder smpl` 选择离线 SMPL 模型，或追加 `--motion-name '<运动名称>'` 选择动作。

操作顺序：

1. 等待 `BUMI3_CONTROL_READY`，按英文半角 **`]`**。
2. 机器人用约 **3 秒**平滑进入所选运动第一帧的**关节姿态**，随后持续 PD 保持，策略尚未启动。
3. 看到 `BUMI3_REAL_HOLDING` 后，按 **`T`** 启动策略并播放运动。
4. 按 **`O`** 停止退出；退出会请求去使能，但软件日志不能证明电机已实际断力，必要时使用实体急停。

这一步只控制关节，不会将真机的世界位置或朝向重置成参考值。

## 5. PICO 实时 SMPL 遥操作

数据通路：`PICO → Python SMPL 发送端 → ZMQ → C++ SMPL 策略 → ROS 2 → MuJoCo／真机`。

### 首次准备

按 [PICO 环境配置](../docs/source/getting_started/vr_teleop_setup.md)安装 XRoboToolkit PC 服务和头显应用，连接同一局域网，在头显填写 PC 服务电脑的 IP，开启全身追踪并确认正在发送数据。

在没有激活 venv／Conda 的新终端安装遥操作环境（若已激活，先执行相应的 `deactivate`、`conda deactivate`）：

```bash
SKIP_SIM_AND_UNITREE=1 bash install_scripts/install_pico.sh
```

该脚本会重建 `.venv_teleop`；已有环境正常时不必重复安装。如果报错路径指向 `.venv_sim/bin/python` 且提示缺少 `Python.h`，先退出已激活的环境再运行。

### 三个终端启动 PICO 仿真

**终端一：**加载第 1 节的仿真环境，然后启动 MuJoCo：

```bash
source .venv_sim/bin/activate
source gear_sonic_deploy/scripts/setup_env_bumi3.sh
python gear_sonic/scripts/run_sim_loop.py --robot-type bumi3
```

**终端二：**使用独立遥操作环境，启动发送端：

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --smpl-only --num_frames_to_send 10
```

默认 50 Hz、端口 5556；不要添加 G1 的 `--manager` 参数。发送端可加 `--cuda` 加速数据处理，与控制端的 TensorRT 推理设置相互独立。

**终端三：**加载第 1 节的仿真环境和 GPU 库路径，然后启动实时控制端：

```bash
bash gear_sonic_deploy/deploy.sh --robot-type bumi3 sim --input-type zmq
```

该命令自动选择 SMPL 模型，接收本机 `tcp://127.0.0.1:5556` 的实时数据，不需要离线 CSV。等待有效 SMPL 数据并显示 `BUMI3_CONTROL_READY` 后，在控制端按 **`T`** 跟随，按 **`O`** 退出。

### 换成 PICO 真机遥操作

关闭 MuJoCo 和仿真控制端，保留 PICO 发送端。控制终端加载相同环境后改用：

```bash
BUMI_BRIDGE_SUDO=1 bash gear_sonic_deploy/deploy.sh \
  --robot-type bumi3 real --interface enp132s0 --input-type zmq
```

先按 **`]`**，在约 3 秒内进入**训练默认关节姿态并保持**；再按 **`T`** 启动策略跟随 PICO。实时模式没有离线运动第一帧。发送端与策略分属两台电脑时，控制命令追加 `--zmq-host <发送端电脑IP>`。

## 6. 按键与常见问题

| 按键 | 功能 |
| --- | --- |
| `]` | 真机请求使能并平滑进入准备姿态 |
| `T` | 真机首次启动策略；之后开始／暂停参考播放或跟随，暂停参考不会退出策略 |
| `P` | 离线模式切换下一个运动；真机需重新按 `]`、`T` |
| `R` | 离线模式重置当前运动；真机需重新按 `]`、`T` |
| `O` | 停止并退出控制端 |

按键应在控制端终端输入，无需回车。

- **`libnvinfer.so.10` 找不到：**确认 TensorRT 已安装，`TensorRT_ROOT` 指向实际安装根目录，`BUMI_TRT_LIB` 指向实际库目录，并在当前控制终端重新加载 `LD_LIBRARY_PATH`。
- **`GLIBCXX_3.4.30` 找不到：**旧 Conda C++ 库可能抢先加载；在当前终端执行 `export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`，避免全局覆盖其他项目的环境。
- **`libyaml-cpp.so.0.8` 找不到：**已有 SDK 可能链接了 Conda 库；将实际包含该文件的目录追加到 `LD_LIBRARY_PATH` 末尾，保持系统 C++ 库在前；或在干净环境重新编译 SDK。不要将 0.7 库伪装成 0.8。
- **没有 ROS 状态／机器人不动：**先确认环境端仍运行，真机网线、供电和 IMU 正常，两端 domain／话题一致。`POLICY_STARTED` 只说明控制程序进入策略阶段，不能证明实体电机已经动作。

## 7. C++ 代码对应关系

BUMI3 源码位于 `gear_sonic_deploy/src/bumi3/bumi3_deploy_onnx_ref/`，以下文件名与 G1 对齐：

| 文件 | 职责 |
| --- | --- |
| `src/bumi3_deploy_onnx_ref.cpp` | 状态机、观测采集、策略调用和命令下发 |
| `include/control_policy.hpp` | `PolicyEngine` 模型初始化、输入／动作缓冲区及推理 |
| `include/robot_parameters.hpp` | `RobotState`、`MotorCommand`、`OperatorState`、`HeadingState` |
| `include/policy_parameters.hpp` | BUMI3 策略维度、默认角度、动作缩放和 PD 参数 |
| `include/observation_config.hpp` | `ObservationConfig`、`EncoderConfig` 等配置与解析 |
| `include/input_interface/keyboard_handler.hpp` | `SimpleKeyboard` 终端按键输入 |
| `include/input_interface/zmq_endpoint_interface.hpp` | `ZMQEndpointInterface` 实时 SMPL 窗口输入 |
| `include/motion_data_reader.hpp`、`include/state_logger.hpp` | 离线参考读取和反馈历史 |
| `include/math_utils.hpp` | 朝向和四元数数学工具 |

主控制路径保持：反馈历史 → 输入快照 → 朝向 → 观测 → 策略动作 → 参考游标 → 命令下发。BUMI3 的联合 ONNX 输入、21 关节映射及 Noetix ROS 2 后端仍按本机器人实现；`ros2_backend.hpp` 负责具名执行器消息转换，`deploy_config.hpp` 负责启动参数和配置装配。
