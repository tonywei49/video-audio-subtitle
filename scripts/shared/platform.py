from __future__ import annotations

import platform


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_linux() -> bool:
    return platform.system() == "Linux"


def _auto_detect_backend() -> str:
    return "mlx" if is_macos() else "cuda"


def get_backend() -> str:
    try:
        from .config import load_config

        configured = str(load_config().get("backend", "")).strip()
        if configured:
            return configured
    except Exception:
        pass

    return _auto_detect_backend()


def get_backend_label() -> str:
    backend = get_backend()
    if backend == "mlx":
        return "mlx (Apple Silicon)"
    if backend == "cuda":
        return "cuda (NVIDIA GPU)"
    return backend


def check_dependency_available(backend: str) -> bool:
    if backend == "mlx":
        try:
            import mlx_audio  # noqa: F401

            return True
        except ImportError:
            return False

    if backend == "cuda":
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    return False

