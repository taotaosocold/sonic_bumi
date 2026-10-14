#!/bin/bash
set -e

# ============================================================================
# G1 Deploy - Deployment Script
# ============================================================================
# This script handles the complete setup and deployment process for g1_deploy
# Following the steps from the README.md
#
# Usage: ./deploy.sh [sim|real|<interface_name>|<ip_address>]
#   sim   - Use loopback interface for simulation (MuJoCo)
#   real  - Auto-detect robot network interface (192.168.123.x)
#   <interface_name> - Use specific interface (e.g., enP8p1s0, eth0)
#   <ip_address> - Use interface with specific IP
#
# Default: real
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Select the robot while preserving the original G1 entry and its defaults.
ROBOT_TYPE=g1
BUMI_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-type)
            [[ $# -ge 2 ]] || { echo "--robot-type requires g1 or bumi3" >&2; exit 2; }
            ROBOT_TYPE="$2"; shift 2 ;;
        --robot-type=*) ROBOT_TYPE="${1#*=}"; shift ;;
        *) BUMI_ARGS+=("$1"); shift ;;
    esac
done
set -- "${BUMI_ARGS[@]}"
if [[ "$ROBOT_TYPE" == bumi3 ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    PYTHON_BIN="${BUMI_DEPLOY_PYTHON:-$REPO_ROOT/.venv_sim/bin/python}"
    [[ -x "$PYTHON_BIN" ]] || {
        echo "BUMI3 environment missing. Run bash install_scripts/install_mujoco_sim.sh --robot-type bumi3" >&2
        exit 2
    }
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    cd "$REPO_ROOT"
    # 与 G1 一样先构建机器人的 C++ 控制程序，参数和 YAML 均交给原生入口解析。
    bash "$SCRIPT_DIR/scripts/setup_bumi3_ros2.sh" sim
    source "$SCRIPT_DIR/scripts/setup_bumi3_ros2.sh"
    # 与 G1 一样链接环境安装的 C++ SDK；自定义路径可通过安装参数或标准根目录变量指定。
    # BUMI3 与 G1 一样面向 GPU 部署，默认链接带 CUDA/TensorRT 后端的 SDK。
    # 显式指定 ONNX_VARIANT=cpu 时使用独立调试 SDK；不根据当前机器是否有显卡改变默认值。
    BUMI_ORT_VARIANT="${ONNX_VARIANT:-gpu}"
    case "$BUMI_ORT_VARIANT" in
        gpu) ORT_DEFAULT=/opt/onnxruntime-gpu/1.23.2 ;;
        cpu) ORT_DEFAULT=/opt/onnxruntime/1.23.2 ;;
        *) echo "ONNX_VARIANT must be gpu or cpu" >&2; exit 2 ;;
    esac
    ORT_ROOT="${ONNX_INSTALL_PATH:-${onnxruntime_ROOT:-$ORT_DEFAULT}}"
    ORT_ROOT="$(realpath -m "$ORT_ROOT")"
    # 在控制程序的进程环境内加载 GPU 运行库，避免每次新开终端都要手动 export。
    # 支持系统安装、仓库内解压的 deb 包和 TensorRT 安装包；所有仓库路径均根据脚本位置计算。
    if [[ "$BUMI_ORT_VARIANT" == gpu ]]; then
        if [[ -z "${TensorRT_ROOT:-}${TENSORRT_ROOT:-}" && -d "$HOME/TensorRT/lib" ]]; then
            export TensorRT_ROOT="$HOME/TensorRT"
        fi
        BUMI_GPU_ARCH="$(uname -m)"
        [[ "$BUMI_GPU_ARCH" != arm64 ]] || BUMI_GPU_ARCH=aarch64
        # 后加入的目录具有更高优先级：用户指定的 TensorRT 根目录优先于仓库本地依赖。
        # 本地解压目录不写入系统配置，也不修改其他终端或仓库使用的 GPU 库路径。
        for BUMI_GPU_ROOT in "${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}" \
            "$REPO_ROOT/.local/bumi3/tensorrt-sdk" "${TensorRT_ROOT:-${TENSORRT_ROOT:-}}"; do
            [[ -n "$BUMI_GPU_ROOT" ]] || continue
            for BUMI_GPU_LIB_DIR in "$BUMI_GPU_ROOT/lib" "$BUMI_GPU_ROOT/lib64" \
                "$BUMI_GPU_ROOT/targets/$BUMI_GPU_ARCH-linux/lib" \
                "$BUMI_GPU_ROOT/usr/lib/$BUMI_GPU_ARCH-linux-gnu"; do
                if [[ -d "$BUMI_GPU_LIB_DIR" ]]; then
                    export LD_LIBRARY_PATH="$BUMI_GPU_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                fi
            done
        done
    fi
    # 编译与运行必须选中同一个 SDK，避免 G1 环境的旧 CPU 动态库覆盖新 GPU 后端。
    export LD_LIBRARY_PATH="$ORT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    BUILD_DIR="$SCRIPT_DIR/build/bumi3"
    cmake -U 'onnxruntime_*' -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DROBOT_TYPE=bumi3 \
        -DCMAKE_BUILD_TYPE=Release -Donnxruntime_ROOT="$ORT_ROOT" \
        -Dnoetix_msgs_DIR="$BUMI_ROS_INSTALL/share/noetix_msgs/cmake" \
        -DPython3_EXECUTABLE="${BUMI_ROS_PYTHON:-/usr/bin/python3}" -DBUILD_TESTING=OFF
    cmake --build "$BUILD_DIR" --parallel "${BUMI_BUILD_JOBS:-2}"
    DEPLOY_EXECUTABLE="$BUILD_DIR/bin/bumi3_deploy_onnx_ref"
    if [[ " $* " == *" --help "* || " $* " == *" -h "* ]]; then
        "$DEPLOY_EXECUTABLE" --help
        exit 0
    fi
    if [[ " $* " == *" --capabilities "* ]]; then
        "$DEPLOY_EXECUTABLE" --capabilities
        exit 0
    fi
    RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sonic_bumi3.XXXXXX")"
    BRIDGE_PID=""
    cleanup_bumi3() {
        if [[ -n "$BRIDGE_PID" ]]; then
            kill -INT "$BRIDGE_PID" 2>/dev/null || true
            wait "$BRIDGE_PID" 2>/dev/null || true
        fi
        rm -rf -- "$RUNTIME_DIR"
    }
    trap cleanup_bumi3 EXIT
    # C++ 准备临时桥配置并输出固定启动字段；捕获失败状态，避免配置失败后继续启动。
    LAUNCH_INFO="$("$DEPLOY_EXECUTABLE" "$@" --prepare-launch "$RUNTIME_DIR")"
    mapfile -t ROS_RUNTIME <<< "$LAUNCH_INFO"
    export ROS_DOMAIN_ID="${ROS_RUNTIME[0]}"
    if [[ "${ROS_RUNTIME[1]}" == 1 ]]; then
        # 在启动真机环境前，由原生程序加载并检查模型和离线动作。
        "$BUILD_DIR/bin/bumi3_deploy_onnx_ref" --runtime-config "$RUNTIME_DIR/runtime.json" --validate-only
        VENDOR_ARCH="$(uname -m)"
        [[ "$VENDOR_ARCH" != arm64 ]] || VENDOR_ARCH=aarch64
        SDK_SOURCE="${BUMI3_SDK_ROOT:-$SCRIPT_DIR/thirdparty/new_ws/src/noetix_sdk2}"
        if [[ ! -f "$SDK_SOURCE/third-party/vendor/hw_controller/bumi/$VENDOR_ARCH/lib/libhw_controller.a" ]]; then
            echo "Missing $VENDOR_ARCH Bumi vendor archive. Install the matching SDK library, or use --external-bridge for a remote hardware node." >&2
            exit 2
        fi
        bash "$SCRIPT_DIR/scripts/setup_bumi3_ros2.sh" real
        source "$SCRIPT_DIR/scripts/setup_bumi3_ros2.sh"
        BRIDGE_COMMAND=("$BUMI_ROS_INSTALL/lib/noetix_bridge/noetix_bridge_node" --ros-args \
            --params-file "$RUNTIME_DIR/bridge_params.yaml" \
            -r "/noetix/low_cmd:=${ROS_RUNTIME[2]}" \
            -r "/noetix/low_state:=${ROS_RUNTIME[3]}" \
            -r "/noetix/control_request:=${ROS_RUNTIME[4]}")
        if [[ "${BUMI_BRIDGE_SUDO:-0}" == 1 ]]; then
            sudo -v
            BRIDGE_ENV=("ROS_DOMAIN_ID=$ROS_DOMAIN_ID" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
                "AMENT_PREFIX_PATH=${AMENT_PREFIX_PATH:-}" "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}")
            [[ -z "${RMW_IMPLEMENTATION:-}" ]] || BRIDGE_ENV+=("RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION")
            sudo env "${BRIDGE_ENV[@]}" "${BRIDGE_COMMAND[@]}" &
        else
            "${BRIDGE_COMMAND[@]}" &
        fi
        BRIDGE_PID=$!
    fi
    "$BUILD_DIR/bin/bumi3_deploy_onnx_ref" --runtime-config "$RUNTIME_DIR/runtime.json"
    exit $?
elif [[ "$ROBOT_TYPE" != g1 ]]; then
    echo "--robot-type must be g1 or bumi3" >&2
    exit 2
fi
cd "$SCRIPT_DIR"

# ============================================================================
# Interface Resolution Functions
# ============================================================================

# Get all network interfaces and their IPs
# Returns lines of: interface_name:ip_address
get_network_interfaces() {
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        ifconfig | awk '
            /^[a-z]/ { iface=$1; gsub(/:$/, "", iface) }
            /inet / { print iface ":" $2 }
        '
    else
        # Linux
        ip -4 addr show 2>/dev/null | awk '
            /^[0-9]+:/ { gsub(/:$/, "", $2); iface=$2 }
            /inet / { split($2, a, "/"); print iface ":" a[1] }
        ' 2>/dev/null || \
        ifconfig 2>/dev/null | awk '
            /^[a-z]/ { iface=$1; gsub(/:$/, "", iface) }
            /inet / { 
                for (i=1; i<=NF; i++) {
                    if ($i == "inet") { print iface ":" $(i+1); break }
                    if ($i ~ /^addr:/) { split($i, a, ":"); print iface ":" a[2]; break }
                }
            }
        '
    fi
}

# Find interface by IP address
# Returns interface name or empty string
find_interface_by_ip() {
    local target_ip="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$target_ip" ]]; then
            echo "$iface"
            return 0
        fi
    done
}

# Find interface with IP matching a prefix
# Returns interface name or empty string
find_interface_by_ip_prefix() {
    local prefix="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$prefix"* ]]; then
            echo "$iface"
            return 0
        fi
    done
}

# Check if string is an IP address
is_ip_address() {
    local input="$1"
    if [[ "$input" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    fi
    return 1
}

# Check if interface has a specific IP
interface_has_ip() {
    local iface="$1"
    local target_ip="$2"
    get_network_interfaces | while IFS=: read -r name ip; do
        if [[ "$name" == "$iface" ]] && [[ "$ip" == "$target_ip" ]]; then
            echo "yes"
            return 0
        fi
    done
}

# Resolve interface parameter to actual network interface name and environment type
# Arguments: interface - "sim", "real", or direct interface name or IP address
# Outputs: Sets TARGET and ENV_TYPE variables
resolve_interface() {
    local interface="$1"
    local os_type="$(uname)"
    
    # Check if interface is an IP address
    if is_ip_address "$interface"; then
        if [[ "$interface" == "127.0.0.1" ]]; then
            TARGET="$interface"
            ENV_TYPE="sim"
        else
            TARGET="$interface"
            ENV_TYPE="real"
        fi
        return 0
    fi
    
    if [[ "$interface" == "sim" ]]; then
        local lo_interface
        lo_interface=$(find_interface_by_ip "127.0.0.1")
        
        if [[ -n "$lo_interface" ]]; then
            # macOS uses lo0 instead of lo
            if [[ "$os_type" == "Darwin" ]] && [[ "$lo_interface" == "lo" ]]; then
                TARGET="lo0"
            else
                TARGET="$lo_interface"
            fi
        else
            # Fallback
            if [[ "$os_type" == "Darwin" ]]; then
                TARGET="lo0"
            else
                TARGET="lo"
            fi
        fi
        ENV_TYPE="sim"
        return 0
    
    elif [[ "$interface" == "real" ]]; then
        # Try to find interface with 192.168.123.x IP (Unitree robot network)
        local real_interface
        real_interface=$(find_interface_by_ip_prefix "192.168.123.")
        
        if [[ -n "$real_interface" ]]; then
            TARGET="$real_interface"
        else
            # Fallback to common interface names
            # Try to find any non-loopback interface
            local fallback_interface
            fallback_interface=$(get_network_interfaces | grep -v "127.0.0.1" | head -1 | cut -d: -f1)
            
            if [[ -n "$fallback_interface" ]]; then
                TARGET="$fallback_interface"
                echo -e "${YELLOW}⚠️  Could not find 192.168.123.x interface, using: $TARGET${NC}" >&2
            else
                # Ultimate fallback
                TARGET="enP8p1s0"
                echo -e "${YELLOW}⚠️  Could not auto-detect interface, using default: $TARGET${NC}" >&2
            fi
        fi
        ENV_TYPE="real"
        return 0
    
    else
        # Direct interface name - check if it has 127.0.0.1 to determine env_type
        local has_loopback
        has_loopback=$(interface_has_ip "$interface" "127.0.0.1")
        
        if [[ "$has_loopback" == "yes" ]]; then
            TARGET="$interface"
            ENV_TYPE="sim"
            return 0
        fi
        
        # macOS lo interface handling
        if [[ "$os_type" == "Darwin" ]] && [[ "$interface" == "lo" ]]; then
            TARGET="lo0"
            ENV_TYPE="sim"
            return 0
        fi
        
        # Default to real for unknown interfaces
        TARGET="$interface"
        ENV_TYPE="real"
        return 0
    fi
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS] [sim|real|<interface>]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --cp, --checkpoint PATH Set the checkpoint path (default: policy/checkpoints/example/model_step_000000)"
    echo "  --obs-config PATH       Set the observation config file (default: policy/configs/example.yaml)"
    echo "  --planner PATH          Set the planner model path (default: planner/example.onnx)"
    echo "  --motion-data PATH      Set the motion data path (default: reference/example_motion/)"
    echo "  --input-type TYPE       Set the input type (default: zmq_manager)"
    echo "  --output-type TYPE      Set the output type (default: ros2)"
    echo "  --zmq-host HOST         Set the ZMQ host (default: localhost)"
    echo ""
    echo "Interface modes:"
    echo "  sim              Use loopback interface for simulation (MuJoCo)"
    echo "  real             Auto-detect robot network (192.168.123.x)"
    echo "  <interface>      Use specific interface (e.g., enP8p1s0, eth0)"
    echo "  <ip_address>     Use interface by IP address"
    echo ""
    echo "Default: real"
    echo ""
    echo "Examples:"
    echo "  $0 sim           # Run in simulation mode"
    echo "  $0 real          # Auto-detect real robot interface"
    echo "  $0 enP8p1s0      # Use specific interface"
    echo "  $0 192.168.x.x # Use interface with this IP"
    echo "  $0 --cp policy/checkpoints/custom/model_step_123456 real  # Use custom checkpoint"
    echo "  $0 --obs-config policy/configs/custom.yaml sim  # Use custom obs config"
    echo "  $0 --planner planner/custom.onnx --input-type keyboard real  # Use custom planner and input"
    echo "  $0 --motion-data reference/custom_motion/ sim  # Use custom motion data"
}

# Default interface mode
INTERFACE_MODE="real"

# Default configuration values (can be overridden by command line)
CHECKPOINT_DEFAULT="policy/release/model"
OBS_CONFIG_DEFAULT="policy/release/observation_config.yaml"
PLANNER_DEFAULT="planner/target_vel/V2/planner_sonic.onnx"
MOTION_DATA_DEFAULT="reference/example/"
INPUT_TYPE_DEFAULT="manager"
OUTPUT_TYPE_DEFAULT="all"
ZMQ_HOST_DEFAULT="localhost"

# Initialize with defaults (will be set after parsing)
CHECKPOINT="$CHECKPOINT_DEFAULT"
OBS_CONFIG="$OBS_CONFIG_DEFAULT"
PLANNER="$PLANNER_DEFAULT"
MOTION_DATA="$MOTION_DATA_DEFAULT"
INPUT_TYPE="$INPUT_TYPE_DEFAULT"
OUTPUT_TYPE="$OUTPUT_TYPE_DEFAULT"
ZMQ_HOST="$ZMQ_HOST_DEFAULT"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --cp|--checkpoint)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --cp/--checkpoint requires a path argument${NC}" >&2
                exit 1
            fi
            CHECKPOINT="$2"
            shift 2
            ;;
        --obs-config)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --obs-config requires a path argument${NC}" >&2
                exit 1
            fi
            OBS_CONFIG="$2"
            shift 2
            ;;
        --planner)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --planner requires a path argument${NC}" >&2
                exit 1
            fi
            PLANNER="$2"
            shift 2
            ;;
        --motion-data)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --motion-data requires a path argument${NC}" >&2
                exit 1
            fi
            MOTION_DATA="$2"
            shift 2
            ;;
        --input-type)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --input-type requires a type argument${NC}" >&2
                exit 1
            fi
            INPUT_TYPE="$2"
            shift 2
            ;;
        --output-type)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --output-type requires a type argument${NC}" >&2
                exit 1
            fi
            OUTPUT_TYPE="$2"
            shift 2
            ;;
        --zmq-host)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --zmq-host requires a host argument${NC}" >&2
                exit 1
            fi
            ZMQ_HOST="$2"
            shift 2
            ;;
        sim|real)
            INTERFACE_MODE="$1"
            shift
            ;;
        *)
            # Could be interface name or IP
            INTERFACE_MODE="$1"
            shift
            ;;
    esac
done

# ============================================================================
# Display Header
# ============================================================================

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                         G1 DEPLOY LAUNCHER                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# Resolve Interface
# ============================================================================

echo -e "${BLUE}[Interface Resolution]${NC}"
echo "Requested mode: $INTERFACE_MODE"

resolve_interface "$INTERFACE_MODE"

echo -e "Resolved interface: ${GREEN}$TARGET${NC}"
echo -e "Environment type:   ${GREEN}$ENV_TYPE${NC}"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint path (set via command line or default)
# CHECKPOINT and OBS_CONFIG are already set from argument parsing above

# Decoder and Encoder ONNX models
CHECKPOINT_DECODER="${CHECKPOINT}_decoder.onnx"
CHECKPOINT_ENCODER="${CHECKPOINT}_encoder.onnx"

# Motion data path (set via command line or default)
# MOTION_DATA is already set from argument parsing above

# Observation config (set via command line or default)
# OBS_CONFIG is already set from argument parsing above

# Planner model (set via command line or default)
# PLANNER is already set from argument parsing above

# Input type (set via command line or default)
# INPUT_TYPE is already set from argument parsing above

# Output type (set via command line or default)
# OUTPUT_TYPE is already set from argument parsing above

# ZMQ host (set via command line or default)
# ZMQ_HOST is already set from argument parsing above

# Additional flags for simulation mode
EXTRA_ARGS=""
if [[ "$ENV_TYPE" == "sim" ]]; then
    EXTRA_ARGS="--disable-crc-check"
    echo -e "${YELLOW}📋 Simulation mode: CRC check will be disabled${NC}"
    echo ""
fi

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

echo -e "${BLUE}[Step 1/4]${NC} Checking prerequisites..."

# Check for TensorRT
if [ -z "$TensorRT_ROOT" ]; then
    echo -e "${YELLOW}⚠️  TensorRT_ROOT is not set.${NC}"
    echo "   Please ensure TensorRT is installed and add to your ~/.bashrc:"
    echo "   export TensorRT_ROOT=\$HOME/TensorRT"
    echo ""
    echo "   Get TensorRT from: https://developer.nvidia.com/tensorrt/download/10x"
    
    # Check if it exists in common locations
    if [ -d "$HOME/TensorRT" ]; then
        echo -e "${GREEN}   Found TensorRT at ~/TensorRT - setting temporarily${NC}"
        export TensorRT_ROOT="$HOME/TensorRT"
    fi
fi

# Check for required model files
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ Missing file: $1${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Found: $1${NC}"
        return 0
    fi
}

echo ""
echo "Checking required model files..."
MISSING_FILES=0

check_file "$CHECKPOINT_DECODER" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$CHECKPOINT_ENCODER" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$OBS_CONFIG" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$PLANNER" || MISSING_FILES=$((MISSING_FILES + 1))

if [ -d "$MOTION_DATA" ]; then
    echo -e "${GREEN}✅ Found: $MOTION_DATA${NC}"
else
    echo -e "${RED}❌ Missing directory: $MOTION_DATA${NC}"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some files are missing. Make sure you have pulled the model files.${NC}"
    echo "   You may need to run: git lfs pull"
fi

echo ""

# ============================================================================
# Step 2: Install Dependencies (if needed)
# ============================================================================

echo -e "${BLUE}[Step 2/4]${NC} Checking/Installing dependencies..."

# Check if just is installed
if ! command -v just &> /dev/null; then
    echo "Installing dependencies (just not found)..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}✅ just is already installed${NC}"
fi

# Check if other essential tools are available
DEPS_OK=true
for cmd in cmake clang git; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${YELLOW}⚠️  $cmd not found, will run install_deps.sh${NC}"
        DEPS_OK=false
        break
    fi
done

if [ "$DEPS_OK" = false ]; then
    echo "Installing missing dependencies..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}✅ All essential tools are installed${NC}"
fi

echo ""

# ============================================================================
# Step 3: Setup Environment & Build
# ============================================================================

echo -e "${BLUE}[Step 3/4]${NC} Setting up environment and building..."

# Source the environment setup script
echo "Sourcing environment setup..."
set +e  # Temporarily allow errors (for jetson_clocks on non-Jetson systems)
source scripts/setup_env.sh
set -e  # Re-enable exit on error

# Always build to ensure we have the latest version
echo "Building the project..."
just build

echo ""

# ============================================================================
# Step 4: Deploy
# ============================================================================

echo -e "${BLUE}[Step 4/4]${NC} Ready to deploy!"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                         DEPLOYMENT CONFIGURATION                       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Environment:        ${GREEN}$ENV_TYPE${NC}"
echo -e "  Network Interface:  ${GREEN}$TARGET${NC}"
echo -e "  Decoder Model:      ${GREEN}$CHECKPOINT_DECODER${NC}"
echo -e "  Encoder Model:      ${GREEN}$CHECKPOINT_ENCODER${NC}"
echo -e "  Motion Data:        ${GREEN}$MOTION_DATA${NC}"
echo -e "  Obs Config:         ${GREEN}$OBS_CONFIG${NC}"
echo -e "  Planner:            ${GREEN}$PLANNER${NC}"
echo -e "  Input Type:         ${GREEN}$INPUT_TYPE${NC}"
echo -e "  Output Type:        ${GREEN}$OUTPUT_TYPE${NC}"
echo -e "  ZMQ Host:           ${GREEN}$ZMQ_HOST${NC}"
if [[ -n "$EXTRA_ARGS" ]]; then
echo -e "  Extra Args:         ${GREEN}$EXTRA_ARGS${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The following command will be executed:${NC}"
echo ""
echo -e "${BLUE}just run g1_deploy_onnx_ref $TARGET $CHECKPOINT_DECODER $MOTION_DATA \\${NC}"
echo -e "${BLUE}    --obs-config $OBS_CONFIG \\${NC}"
echo -e "${BLUE}    --encoder-file $CHECKPOINT_ENCODER \\${NC}"
echo -e "${BLUE}    --planner-file $PLANNER \\${NC}"
echo -e "${BLUE}    --input-type $INPUT_TYPE \\${NC}"
echo -e "${BLUE}    --output-type $OUTPUT_TYPE \\${NC}"
echo -e "${BLUE}    --zmq-host $ZMQ_HOST${NC}"
if [[ -n "$EXTRA_ARGS" ]]; then
echo -e "${BLUE}    $EXTRA_ARGS${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Ask for confirmation
if [[ "$ENV_TYPE" == "real" ]]; then
    echo -e "${YELLOW}⚠️  WARNING: This will start the REAL robot control system!${NC}"
else
    echo -e "${YELLOW}📋 This will start the simulation control system.${NC}"
fi
echo ""
read -p "$(echo -e ${GREEN}Proceed with deployment? [Y/n]: ${NC})" confirm

if [[ "$confirm" =~ ^[Yy]$ ]] || [[ -z "$confirm" ]]; then
    echo ""
    echo -e "${GREEN}🚀 Starting deployment...${NC}"
    echo ""
    
    # Build the command with optional extra args
    if [[ -n "$EXTRA_ARGS" ]]; then
        just run g1_deploy_onnx_ref "$TARGET" "$CHECKPOINT_DECODER" "$MOTION_DATA" \
            --obs-config "$OBS_CONFIG" \
            --encoder-file "$CHECKPOINT_ENCODER" \
            --planner-file "$PLANNER" \
            --input-type "$INPUT_TYPE" \
            --output-type "$OUTPUT_TYPE" \
            --zmq-host "$ZMQ_HOST" \
            $EXTRA_ARGS
    else
        just run g1_deploy_onnx_ref "$TARGET" "$CHECKPOINT_DECODER" "$MOTION_DATA" \
            --obs-config "$OBS_CONFIG" \
            --encoder-file "$CHECKPOINT_ENCODER" \
            --planner-file "$PLANNER" \
            --input-type "$INPUT_TYPE" \
            --output-type "$OUTPUT_TYPE" \
            --zmq-host "$ZMQ_HOST"
    fi
else
    echo ""
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    exit 0
fi
