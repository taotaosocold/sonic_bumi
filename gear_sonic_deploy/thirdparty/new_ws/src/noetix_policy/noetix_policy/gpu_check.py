from __future__ import annotations

from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "ONNX Runtime is not installed. Install a GPU-enabled build compatible with "
            "your CUDA/JetPack environment.") from exc

    providers = ort.get_available_providers()
    print(f"onnxruntime={ort.__version__}")
    print("available_providers=" + ",".join(providers))
    gpu = [p for p in providers if p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")]
    if not gpu:
        raise SystemExit(
            "No NVIDIA GPU execution provider found. Expected CUDAExecutionProvider and/or "
            "TensorrtExecutionProvider.")

    # get_available_providers() only says that the Python package was built with an EP.
    # Actually create a session so missing CUDA/cuDNN/TensorRT shared libraries are caught too.
    try:
        
        model = Path(get_package_share_directory("noetix_policy")) / "policy/bumi4340/walkrun.onnx"
        if not model.is_file():
            raise RuntimeError(f"packaged test model not found: {model}")

        selected = "CUDAExecutionProvider" if "CUDAExecutionProvider" in gpu else gpu[0]
        # 注意：不要设置 session.disable_cpu_ep_fallback，否则会与隐式添加的 CPU EP 冲突。
        # 直接传 providers=[selected] 即可让 ONNX Runtime 优先使用该 GPU EP，
        # 若 GPU 库加载失败，会自动回退到 CPU，随后通过 get_providers() 检查即可发现。
        session = ort.InferenceSession(str(model), providers=[selected])
        active = session.get_providers()
        if selected not in active:
            raise RuntimeError(f"requested {selected}, active providers={active}")
    except Exception as exc:
        raise SystemExit(f"GPU provider is listed but failed to create a real model session: {exc}") from exc

    print("gpu_ready=true")
    print("validated_provider=" + selected)


if __name__ == "__main__":
    main()