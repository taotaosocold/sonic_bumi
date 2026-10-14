#!/usr/bin/env bash
# install_mujoco_sim.sh
# Minimal venv setup for running the MuJoCo simulator (run_sim_loop.py).
# Skips XRoboToolkit SDK and teleop dependencies that are NOT needed for sim.
# Based on install_pico.sh — see that script for the full teleop setup.
#
# Usage:  bash install_scripts/install_mujoco_sim.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# BUMI3 使用同一个安装入口准备 MuJoCo 环境与原生控制端，默认选择 GPU 推理 SDK。
if [[ "${1:-}" == "--robot-type" && "${2:-}" == "bumi3" ]]; then
    # ROS Python extension modules must match the environment's Python ABI.
    bash "$REPO_ROOT/gear_sonic_deploy/scripts/setup_bumi3_ros2.sh" sim
    VENV_PATH="${BUMI_DEPLOY_VENV:-$REPO_ROOT/.venv_sim}"
    PYTHON_COMMAND="${BUMI_INSTALL_PYTHON:-${BUMI_ROS_PYTHON:-/usr/bin/python3}}"
    # 解释器存在并不代表环境创建完整；中断的创建过程或不完整复制可能缺少配置文件、激活脚本。
    # 缺少任一必要文件时复用原目录补齐环境，不使用 --clear，保留已经安装的 Python 依赖。
    # pip 由下方的检查单独补装，修复激活脚本时不重新引导或升级已有 pip。
    if [[ ! -x "$VENV_PATH/bin/python" || ! -f "$VENV_PATH/pyvenv.cfg" || ! -r "$VENV_PATH/bin/activate" ]]; then
        "$PYTHON_COMMAND" -m venv --without-pip "$VENV_PATH"
    fi
    # 修复后再次核对启动所需文件，避免只有 pip 安装成功就把不完整环境报告为可用。
    [[ -x "$VENV_PATH/bin/python" && -f "$VENV_PATH/pyvenv.cfg" && -r "$VENV_PATH/bin/activate" ]] || {
        echo "Incomplete BUMI3 virtual environment: $VENV_PATH" >&2
        exit 2
    }
    "$VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1 || "$VENV_PATH/bin/python" -m ensurepip --upgrade
    "$VENV_PATH/bin/python" -m pip install 'numpy==1.26.4' 'scipy==1.15.3' \
        'mujoco==3.15.0' 'onnxruntime==1.23.2' 'tyro==0.8.14' 'pyyaml>=6,<7' 'joblib>=1.3,<2' 'pyzmq>=25,<28'
    # Python 包用于离线转换和 MuJoCo 环境端；策略默认由 C++ 控制端通过 TensorRT 在 GPU 上推理。
    # BUMI3 默认安装 GPU SDK，并保留 ONNX_VARIANT=cpu 的显式调试选择；两种 SDK 分目录保存。
    # CUDA、cuDNN 和 TensorRT 由目标部署机器配置，安装脚本不覆盖已有 SDK 或 GPU 运行库。
    BUMI_ORT_VARIANT="${ONNX_VARIANT:-gpu}"
    case "$BUMI_ORT_VARIANT" in
        gpu) BUMI_ORT_DEFAULT=/opt/onnxruntime-gpu/1.23.2 ;;
        cpu) BUMI_ORT_DEFAULT=/opt/onnxruntime/1.23.2 ;;
        *) echo "ONNX_VARIANT must be gpu or cpu" >&2; exit 2 ;;
    esac
    ONNX_VERSION=1.23.2 ONNX_VARIANT="$BUMI_ORT_VARIANT" \
        ONNX_INSTALL_PATH="${ONNX_INSTALL_PATH:-${onnxruntime_ROOT:-$BUMI_ORT_DEFAULT}}" \
        bash "$REPO_ROOT/gear_sonic_deploy/scripts/install_deps.sh" --onnxruntime-only
    echo "BUMI3 environment ready. Source gear_sonic_deploy/scripts/setup_bumi3_ros2.sh before the simulator."
    echo "After preparing PKLs, convert references before deployment:"
    echo "  source .venv_sim/bin/activate"
    echo "  python gear_sonic_deploy/reference/bumi3_convert_motion.py"
    exit 0
fi

# ── 0. Print detected architecture ───────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the uv env so it's available in this session
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Verify uv is now reachable
    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.10 (includes development headers) …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_sim (if present) …"
rm -rf .venv_sim

# ── 4. Create venv & install sim extra ────────────────────────────────────────
echo "[INFO] Creating .venv_sim with uv-managed Python 3.10 …"
uv venv .venv_sim --python "$MANAGED_PY" --prompt gear_sonic_sim
# shellcheck disable=SC1091
source .venv_sim/bin/activate
echo "[INFO] Installing gear_sonic[sim] …"
uv pip install -e "gear_sonic[sim]"

# ── 5. Install unitree_sdk2_python (needed by the sim ↔ WBC bridge) ──────────
echo "[INFO] Installing unitree_sdk2_python …"
uv pip install -e external_dependencies/unitree_sdk2_python

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_sim/bin/activate"
echo ""
echo "  You should see (gear_sonic_sim) in your prompt."
echo ""
echo "  Then run the MuJoCo simulator with:"
echo "    python gear_sonic/scripts/run_sim_loop.py"
echo "══════════════════════════════════════════════════════════════"
