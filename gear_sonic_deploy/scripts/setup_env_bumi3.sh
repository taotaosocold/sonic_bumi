#!/usr/bin/env bash
# BUMI3 环境入口：控制端由 deploy.sh 自动 source；独立运行的 MuJoCo 端需在自身终端 source。
# 直接执行时编译内置 new_ws 中部署所需的 Noetix 包，source 时加载 ROS 2 与已编译的包。
# 编译和安装产物保存在工作空间内，路径从仓库位置解析，不要求个人工作空间或 colcon。
bumi3_ros2_environment() {
    local task_script_dir task_repo_root task_ros_setup task_nounset=0 task_path_id task_alias task_package
    task_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    task_repo_root="$(cd "$task_script_dir/../.." && pwd)"
    # Humble rosidl uses file(STRINGS) without UTF-8 when reading generated
    # paths. An ASCII alias also supports clones under directories like 桌面.
    task_path_id="$(printf '%s' "$task_repo_root" | cksum | cut -d ' ' -f 1)"
    task_alias="/tmp/sonic_bumi_ros2_${UID}_${task_path_id}"
    mkdir -p "$task_alias"
    if [[ -L "$task_alias/repo" ]]; then
        [[ "$(readlink "$task_alias/repo")" == "$task_repo_root" ]] || {
            echo "ROS build alias points to a different checkout: $task_alias/repo" >&2; return 2;
        }
    else
        ln -s "$task_repo_root" "$task_alias/repo" || return 2
    fi
    export BUMI_ROS_SOURCE="$task_alias/repo/gear_sonic_deploy"
    task_ros_setup="${BUMI_ROS_SETUP:-/opt/ros/${ROS_DISTRO:-humble}/setup.bash}"
    [[ -f "$task_ros_setup" ]] || {
        echo "ROS 2 setup not found: $task_ros_setup. Install ROS 2, or set BUMI_ROS_SETUP." >&2
        return 2
    }
    [[ "$-" == *u* ]] && task_nounset=1
    set +u
    source "$task_ros_setup"
    if [[ "${ROS_VERSION:-}" != 2 ]]; then
        [[ "$task_nounset" == 1 ]] && set -u
        echo "BUMI3 requires ROS 2. Select its setup.bash using BUMI_ROS_SETUP." >&2
        return 2
    fi
    export BUMI_ROS_INSTALL="${BUMI_ROS_INSTALL:-$BUMI_ROS_SOURCE/thirdparty/new_ws/install}"
    for task_package in noetix_msgs noetix_sdk2 noetix_robots noetix_bridge; do
        if [[ -f "$BUMI_ROS_INSTALL/share/$task_package/local_setup.bash" ]]; then
            source "$BUMI_ROS_INSTALL/share/$task_package/local_setup.bash"
        fi
    done
    if [[ ":${CMAKE_PREFIX_PATH:-}:" != *":$BUMI_ROS_INSTALL:"* ]]; then
        export CMAKE_PREFIX_PATH="$BUMI_ROS_INSTALL${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    fi
    [[ "$task_nounset" == 1 ]] && set -u
    return 0
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    bumi3_ros2_environment
    return $?
fi
set -eo pipefail
bumi3_ros2_environment
TASK_MODE="${1:-sim}"
[[ "$TASK_MODE" == sim || "$TASK_MODE" == real ]] || { echo "Usage: bash setup_env_bumi3.sh [sim|real]" >&2; exit 2; }
TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DEPLOY_ROOT="$BUMI_ROS_SOURCE"
TASK_WORKSPACE_ROOT="$TASK_DEPLOY_ROOT/thirdparty/new_ws"
TASK_BUILD_ROOT="$TASK_WORKSPACE_ROOT/build"
TASK_ROS_PYTHON="${BUMI_ROS_PYTHON:-/usr/bin/python3}"
TASK_VENDOR_ROOT="$TASK_WORKSPACE_ROOT/src"
build_package() {
    local name="$1" path="$2"
    cmake -S "$path" -B "$TASK_BUILD_ROOT/$name" \
        -DCMAKE_INSTALL_PREFIX="$BUMI_ROS_INSTALL" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF -DPython3_EXECUTABLE="$TASK_ROS_PYTHON" \
        -DPYTHON_EXECUTABLE="$TASK_ROS_PYTHON"
    cmake --build "$TASK_BUILD_ROOT/$name" --parallel "${BUMI_BUILD_JOBS:-2}"
    cmake --install "$TASK_BUILD_ROOT/$name"
    source "$BUMI_ROS_INSTALL/share/$name/local_setup.bash"
}
build_package noetix_msgs "$TASK_VENDOR_ROOT/noetix_ros2/noetix_msgs"
if [[ "$TASK_MODE" == real ]]; then
    build_package noetix_sdk2 "${BUMI3_SDK_ROOT:-$TASK_VENDOR_ROOT/noetix_sdk2}"
    build_package noetix_robots "$TASK_VENDOR_ROOT/noetix_robots"
    build_package noetix_bridge "$TASK_VENDOR_ROOT/noetix_ros2/noetix_bridge"
fi
echo "BUMI3 ROS 2 packages built. Load with: source gear_sonic_deploy/scripts/setup_env_bumi3.sh"
