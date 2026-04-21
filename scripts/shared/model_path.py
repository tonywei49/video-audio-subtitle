from __future__ import annotations

import os

from .config import load_config


def get_model_source() -> str:
    return str(load_config().get("model_source", "modelscope"))


def get_model_cache_dir() -> str:
    return str(load_config().get("model_cache_dir", ""))


def _ensure_env(cache_dir: str) -> None:
    if cache_dir:
        os.environ["MODELSCOPE_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir


def resolve_model_path(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id

    cache_dir = get_model_cache_dir()
    _ensure_env(cache_dir)

    if get_model_source() == "huggingface":
        return _resolve_huggingface(model_id)
    return _resolve_modelscope(model_id)


def _resolve_modelscope(model_id: str) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise ImportError("modelscope is not installed.") from exc

    return snapshot_download(model_id, cache_dir=None)


def _resolve_huggingface(model_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is not installed.") from exc

    return snapshot_download(model_id, cache_dir=None)


def check_model_exists(model_id: str) -> bool:
    if os.path.isdir(model_id):
        return True

    cache_dir = get_model_cache_dir()
    if not cache_dir:
        return False

    if get_model_source() == "huggingface":
        safe_id = model_id.replace("/", "--")
        model_dir = os.path.join(cache_dir, "hub", f"models--{safe_id}")
    else:
        parts = model_id.split("/", 1)
        if len(parts) == 2:
            org, name = parts
            safe_name = name.replace(".", "___")
            model_dir = os.path.join(cache_dir, "models", org, safe_name)
        else:
            model_dir = os.path.join(cache_dir, "models", model_id)

    return os.path.isdir(model_dir)

