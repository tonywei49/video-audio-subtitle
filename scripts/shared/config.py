from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def get_config_dir() -> Path:
    configured = os.environ.get("VIDEO_AUDIO_SUBTITLE_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".video-audio-subtitle"


def get_config_file() -> Path:
    return get_config_dir() / "config.json"


DEFAULT_CONFIG = {
    "output_dir": os.environ.get("VIDEO_AUDIO_SUBTITLE_OUTPUT_DIR", tempfile.gettempdir()),
    "asr_model_size": "0.6B",
    "asr_language": "",
    # "" = auto detect
    "backend": "",
    "model_source": "modelscope",
    # empty = use library default cache location
    "model_cache_dir": os.environ.get("VIDEO_AUDIO_SUBTITLE_MODEL_CACHE", ""),
}


def load_config() -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    config_file = get_config_file()
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def save_config(key: str, value: object) -> None:
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    config[key] = value

    with get_config_file().open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

