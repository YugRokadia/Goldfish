from __future__ import annotations

import os
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.embeddings.runtime import detect_runtime


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """
    Load the embedding model exactly once.

    Thread-safe so concurrent search requests cannot initialize
    multiple copies of the model.
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            runtime = detect_runtime()
            requested_backend = os.getenv(
                "RECALLX_EMBEDDING_BACKEND", "auto"
            ).strip().lower()

            if requested_backend == "torch":
                _model = SentenceTransformer(MODEL_NAME, backend="torch")
            else:
                backend = requested_backend
                if backend == "auto":
                    backend = {
                        "QNNExecutionProvider": "onnx",
                        "OpenVINOExecutionProvider": "onnx",
                    }.get(runtime.provider, "torch")
                elif backend in {"qnn", "onnxruntime"}:
                    backend = "onnx"
                elif backend in {"intel", "openvino-npu"}:
                    backend = "onnx"
                elif backend == "openvino":
                    backend = "onnx"

                try:
                    if backend == "onnx":
                        provider = (
                            "QNNExecutionProvider"
                            if requested_backend in {"qnn"}
                            else (
                                "OpenVINOExecutionProvider"
                                if requested_backend in {
                                    "openvino",
                                    "intel",
                                    "openvino-npu",
                                }
                                else runtime.provider
                            )
                        )
                        try:
                            import onnxruntime as ort
                        except ImportError as error:
                            raise RuntimeError(
                                "The ONNX embedding backend requires onnxruntime."
                            ) from error

                        available_providers = ort.get_available_providers()
                        if provider not in available_providers:
                            raise RuntimeError(
                                f"ONNX provider {provider!r} is unavailable. "
                                f"Available providers: {available_providers}. "
                                "Install the matching ONNX Runtime build for "
                                "this device."
                            )

                        model_kwargs = {
                            "provider": provider,
                            "file_name": "onnx/model.onnx",
                        }
                        qnn_backend_path = os.getenv(
                            "RECALLX_QNN_BACKEND_PATH", ""
                        ).strip()
                        if qnn_backend_path:
                            model_kwargs["provider_options"] = {
                                "backend_path": qnn_backend_path
                            }
                        elif provider == "OpenVINOExecutionProvider":
                            model_kwargs["provider_options"] = {
                                "device_type": os.getenv(
                                    "RECALLX_OPENVINO_DEVICE", "NPU"
                                ).strip()
                            }

                        _model = SentenceTransformer(
                            MODEL_NAME,
                            backend="onnx",
                            model_kwargs=model_kwargs,
                        )
                    else:
                        raise ValueError(
                            "RECALLX_EMBEDDING_BACKEND must be one of: "
                            "auto, torch, onnx, qnn, openvino, intel"
                        )
                except Exception:
                    if requested_backend != "auto":
                        raise
                    _model = SentenceTransformer(MODEL_NAME, backend="torch")

    return _model


def preload_model() -> None:
    """
    Load the embedding model during application startup.
    """
    get_model()


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single piece of text into a normalized embedding.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    model = get_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert multiple pieces of text into normalized embeddings.
    """
    if not texts:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32)