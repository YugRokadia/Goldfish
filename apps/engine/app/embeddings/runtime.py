from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


_openvino_dll_directory = None


def _prepare_openvino_dlls() -> None:
    """Make the OpenVINO native libraries discoverable on Windows."""
    global _openvino_dll_directory

    if os.name != "nt" or _openvino_dll_directory is not None:
        return

    try:
        import openvino
    except ImportError:
        return

    package_dir = Path(openvino.__file__).resolve().parent
    dll_dir = package_dir / "libs"
    if not dll_dir.is_dir():
        return

    _openvino_dll_directory = os.add_dll_directory(str(dll_dir))


@dataclass(frozen=True)
class RuntimeInfo:
    provider: str
    device: str


def detect_runtime() -> RuntimeInfo:
    """
    Detect the best available ONNX Runtime execution provider.

    Priority:
        1. Qualcomm QNN
        2. Intel OpenVINO
        3. CPU fallback

    The function only selects providers that are actually available
    in the current ONNX Runtime installation.
    """
    _prepare_openvino_dlls()

    try:
        import onnxruntime as ort
    except ImportError:
        return RuntimeInfo(
            provider="CPUExecutionProvider",
            device="CPU",
        )

    providers = ort.get_available_providers()

    # Qualcomm Snapdragon NPU
    if "QNNExecutionProvider" in providers:
        return RuntimeInfo(
            provider="QNNExecutionProvider",
            device="Qualcomm NPU",
        )

    # Intel OpenVINO / NPU
    if "OpenVINOExecutionProvider" in providers:
        return RuntimeInfo(
            provider="OpenVINOExecutionProvider",
            device="Intel NPU",
        )

    # Universal fallback
    return RuntimeInfo(
        provider="CPUExecutionProvider",
        device="CPU",
    )


def get_execution_provider() -> str:
    """
    Return the best available ONNX Runtime execution provider.
    """
    return detect_runtime().provider


def get_runtime_info() -> RuntimeInfo:
    """
    Return the selected provider and human-readable device.
    """
    return detect_runtime()