from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from urllib import error as urllib_error
from urllib.parse import urlparse

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_ROOT.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from scripts.asr.pipeline import PipelineQualityError, _load_lines, _save_lines, run_pipeline, split_line_after
from scripts.shared.model_path import get_model_source
from scripts.shared.platform import check_dependency_available, get_backend

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SUBTITLE_EXTENSIONS = {".srt", ".ass"}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
BILIBILI_HOSTS = {
    "bilibili.com",
    "www.bilibili.com",
    "m.bilibili.com",
    "b23.tv",
}
TIKTOK_HOSTS = {
    "tiktok.com",
    "www.tiktok.com",
    "m.tiktok.com",
    "vm.tiktok.com",
}
DOUYIN_HOSTS = {
    "douyin.com",
    "www.douyin.com",
    "v.douyin.com",
}
REMOTE_VIDEO_KINDS = {"youtube_url", "bilibili_url", "tiktok_url", "douyin_url"}
PREFERRED_TOOL_PATHS = {
    "ffmpeg": [
        pathlib.Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"),
    ],
    "ffprobe": [
        pathlib.Path("/opt/homebrew/opt/ffmpeg-full/bin/ffprobe"),
    ],
}


class SkillError(RuntimeError):
    """Base error for this skill."""


class MissingDependencyError(SkillError):
    """Raised when runtime dependencies are missing."""


class UnsupportedPlatformError(SkillError):
    """Raised when current platform is unsupported."""


@dataclass
class SourceInfo:
    original: str
    kind: str
    resolved_path: str | None = None


@dataclass
class PlatformInfo:
    platform_name: str
    machine: str
    supported: bool
    message: str


@dataclass
class CheckResult:
    source: SourceInfo
    platform: PlatformInfo
    missing_tools: list[str]
    python_deps_ready: bool
    install_actions: list[str]


@dataclass
class SubtitleInfo:
    original: str
    kind: str
    resolved_path: str


@dataclass
class PreparedMedia:
    audio_path: pathlib.Path
    video_path: pathlib.Path | None = None


@dataclass(frozen=True)
class AudioPreprocessOptions:
    normalize_audio: bool = False
    trim_silence: bool = False


def _parse_srt_timestamp(timestamp: str) -> int:
    hours, minutes, seconds_millis = timestamp.split(":")
    seconds, millis = seconds_millis.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def seconds_to_srt_timestamp(seconds_value: float) -> str:
    total_ms = max(0, int(round(seconds_value * 1000)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    seconds = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def classify_source(source: str) -> SourceInfo:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower()
        if host in YOUTUBE_HOSTS:
            return SourceInfo(original=source, kind="youtube_url")
        if host in BILIBILI_HOSTS:
            return SourceInfo(original=source, kind="bilibili_url")
        if host in TIKTOK_HOSTS:
            return SourceInfo(original=source, kind="tiktok_url")
        if host in DOUYIN_HOSTS:
            return SourceInfo(original=source, kind="douyin_url")
        raise SkillError(
            f"当前只支持本地音视频文件，以及 YouTube / Bilibili / TikTok / Douyin 链接，不支持该 URL: {source}"
        )

    source_path = pathlib.Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return SourceInfo(original=source, kind="audio_file", resolved_path=str(source_path))
    if suffix in VIDEO_EXTENSIONS:
        return SourceInfo(original=source, kind="video_file", resolved_path=str(source_path))

    raise SkillError(f"不支持的文件类型: {source_path.suffix or '<无扩展名>'}")


def classify_subtitle(subtitle: str) -> SubtitleInfo:
    subtitle_path = pathlib.Path(subtitle).expanduser().resolve()
    if not subtitle_path.exists():
        raise FileNotFoundError(f"找不到字幕文件: {subtitle_path}")

    suffix = subtitle_path.suffix.lower()
    if suffix not in SUBTITLE_EXTENSIONS:
        raise SkillError(f"不支持的字幕文件类型: {subtitle_path.suffix or '<无扩展名>'}")

    return SubtitleInfo(
        original=subtitle,
        kind=suffix[1:],
        resolved_path=str(subtitle_path),
    )


def detect_platform(system: str | None = None, machine: str | None = None) -> PlatformInfo:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()

    if system == "Darwin":
        if machine != "arm64":
            return PlatformInfo(
                platform_name="macos",
                machine=machine,
                supported=False,
                message="当前 skill 只支持 Apple Silicon macOS；Intel Mac 未纳入支持范围。",
            )
        return PlatformInfo(
            platform_name="macos",
            machine=machine,
            supported=True,
            message="macOS Apple Silicon 已支持。",
        )

    if system == "Linux":
        return PlatformInfo(
            platform_name="linux",
            machine=machine,
            supported=False,
            message="Linux 版本暂未接入此 skill；后续可复用同一主流程扩展。",
        )

    if system == "Windows":
        return PlatformInfo(
            platform_name="windows",
            machine=machine,
            supported=False,
            message="Windows 原生未支持；后续建议单独走 WSL2 路线，不做隐式回退。",
        )

    return PlatformInfo(
        platform_name=system.lower(),
        machine=machine,
        supported=False,
        message=f"不支持的平台: {system}",
    )


def resolve_tool_path(command: str, *, required: bool = True) -> str | None:
    for candidate in PREFERRED_TOOL_PATHS.get(command, []):
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which(command)
    if resolved:
        return resolved

    if required:
        raise SkillError(f"找不到可执行文件: {command}")
    return None


def _command_exists(command: str) -> bool:
    return resolve_tool_path(command, required=False) is not None


def _skill_venv_python(*, required: bool = False) -> pathlib.Path | None:
    candidate = SKILL_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    if required:
        raise SkillError("当前 skill 还没有完成 `uv sync`，找不到 .venv/bin/python。")
    return None


def _python_deps_ready() -> bool:
    venv_python = _skill_venv_python(required=False)
    if venv_python is None:
        return False

    backend = get_backend()
    probe_backend = "import mlx_audio" if backend == "mlx" else "import torch"
    probe_modules = ["import soundfile", probe_backend]
    if get_model_source() == "huggingface":
        probe_modules.append("import huggingface_hub")
    else:
        probe_modules.append("import modelscope")

    probe = subprocess.run(
        [str(venv_python), "-c", "; ".join(probe_modules)],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def build_install_actions(
    *,
    missing_tools: list[str],
    python_deps_ready: bool,
    source_kind: str,
    platform_name: str,
    brew_available: bool,
) -> list[str]:
    actions: list[str] = []
    if platform_name == "macos":
        brew_packages: list[str] = []
        if "uv" in missing_tools:
            brew_packages.append("uv")
        if {"ffmpeg", "ffprobe"} & set(missing_tools):
            brew_packages.append("ffmpeg-full")
        if source_kind in REMOTE_VIDEO_KINDS and "yt-dlp" in missing_tools:
            brew_packages.append("yt-dlp")
        if brew_packages:
            if brew_available:
                actions.append(f"brew install {' '.join(dict.fromkeys(brew_packages))}")
            else:
                actions.append(f"缺少 Homebrew，无法自动安装: {', '.join(dict.fromkeys(brew_packages))}")
    else:
        if missing_tools:
            actions.append(f"当前平台未实现自动安装: {', '.join(missing_tools)}")

    if not python_deps_ready:
        actions.append("在当前 skill 目录执行 uv sync 安装 Python 依赖")

    return actions


def check_environment(source: str) -> CheckResult:
    source_info = classify_source(source)
    platform_info = detect_platform()

    missing_tools: list[str] = []
    for tool in ("python3", "uv"):
        if not _command_exists(tool):
            missing_tools.append(tool)

    if source_info.kind in {"video_file", *REMOTE_VIDEO_KINDS}:
        for tool in ("ffmpeg", "ffprobe"):
            if not _command_exists(tool):
                missing_tools.append(tool)

    if source_info.kind in REMOTE_VIDEO_KINDS and not _command_exists("yt-dlp"):
        missing_tools.append("yt-dlp")

    python_deps_ready = _python_deps_ready() if _command_exists("python3") else False
    actions = build_install_actions(
        missing_tools=missing_tools,
        python_deps_ready=python_deps_ready,
        source_kind=source_info.kind,
        platform_name=platform_info.platform_name,
        brew_available=_command_exists("brew"),
    )

    return CheckResult(
        source=source_info,
        platform=platform_info,
        missing_tools=missing_tools,
        python_deps_ready=python_deps_ready,
        install_actions=actions,
    )


def _run_command(command: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(command, cwd=cwd, env=env, text=True)
    if proc.returncode != 0:
        raise SkillError(f"命令执行失败: {' '.join(command)}")


def validate_max_chars(value: int | str) -> int:
    max_chars = int(value)
    if max_chars < 1 or max_chars > 80:
        raise ValueError("字幕每行长度必须在 1 到 80 之间。")
    return max_chars


def build_ytdlp_command(
    *,
    source_url: str,
    output_template: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
    extract_audio: bool,
) -> list[str]:
    command = ["yt-dlp", "-o", output_template]
    ffmpeg_bin = resolve_tool_path("ffmpeg", required=False)
    if ffmpeg_bin:
        command.extend(["--ffmpeg-location", str(pathlib.Path(ffmpeg_bin).parent)])
    if extract_audio:
        command.extend(["-x", "--audio-format", "wav"])
    else:
        command.extend(["--merge-output-format", "mp4"])
    if cookies_file:
        command.extend(["--cookies", cookies_file])
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.append(source_url)
    return command


def build_audio_preprocess_command(
    *,
    ffmpeg_bin: str,
    input_path: str,
    output_path: str,
    source_is_video: bool,
    options: AudioPreprocessOptions,
) -> list[str]:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
    ]
    if source_is_video:
        command.append("-vn")

    audio_filters: list[str] = []
    if options.trim_silence:
        audio_filters.append(
            "silenceremove="
            "start_periods=1:start_duration=0.3:start_threshold=-45dB:"
            "stop_periods=1:stop_duration=0.3:stop_threshold=-45dB"
        )
    if options.normalize_audio:
        audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if audio_filters:
        command.extend(["-af", ",".join(audio_filters)])

    command.extend(
        [
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
    )
    return command


def validate_audio_preprocess_options(options: AudioPreprocessOptions) -> None:
    if options.trim_silence:
        raise SkillError(
            "`--trim-silence` 当前未安全支持：它会改变字幕时间基准，"
            "而现有实现已在真实样本上出现过度裁切。为保证稳定性，当前版本直接阻断，不做隐式回退。"
        )


def parse_srt(content: str) -> list[dict[str, object]]:
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    entries: list[dict[str, object]] = []
    for block in blocks:
        lines = [line.rstrip("\n") for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError as exc:
            raise SkillError(f"SRT 序号格式错误: {lines[0]}") from exc
        if " --> " not in lines[1]:
            raise SkillError(f"SRT 时间轴格式错误: {lines[1]}")
        start, end = [part.strip() for part in lines[1].split(" --> ", 1)]
        text = "\n".join(lines[2:])
        entries.append({"index": index, "start": start, "end": end, "text": text})
    if not entries:
        raise SkillError("字幕文件里没有可解析的 SRT 内容。")
    return entries


def write_srt(entries: list[dict[str, object]]) -> str:
    blocks: list[str] = []
    for item in entries:
        blocks.append(
            f"{item['index']}\n{item['start']} --> {item['end']}\n{item['text']}"
        )
    return "\n\n".join(blocks) + "\n"


def build_srt_entries_from_line_entries(
    line_entries: list[dict[str, object]],
    *,
    min_duration: float = 0.8,
) -> list[dict[str, object]]:
    """从 lines.json 重建 SRT 条目。

    ASR 对快速语段经常产出多条 start_time 相同（或 end_time == start_time）的
    记录，直接转 SRT 后 ffmpeg 只会渲染第一条，导致字幕「只出第一句」。
    此函数在重建时维护一个游标 `cursor`，确保每条字幕至少有 `min_duration`
    秒的显示时长，且下一条的起始时间不早于上一条的结束时间。
    """
    entries: list[dict[str, object]] = []
    cursor: float = 0.0  # 当前已使用到的时间线位置
    for index, item in enumerate(line_entries, start=1):
        raw_start = float(item["start_time"])
        raw_end = float(item["end_time"])

        # 尊重 ASR 给的 start，但不能早于上一条结束（cursor）
        start = max(raw_start, cursor)
        # end 至少要比 start 多 min_duration 秒
        end = max(raw_end, start + min_duration)
        cursor = end

        entries.append(
            {
                "index": index,
                "start": seconds_to_srt_timestamp(start),
                "end": seconds_to_srt_timestamp(end),
                "text": str(item["text"]).strip(),
            }
        )
    return entries


def analyze_srt_timing(entries: list[dict[str, object]]) -> dict[str, int]:
    zero_duration = 0
    gaps_over_5s = 0
    previous_end_ms: int | None = None
    for item in entries:
        start_ms = _parse_srt_timestamp(str(item["start"]))
        end_ms = _parse_srt_timestamp(str(item["end"]))
        if end_ms <= start_ms:
            zero_duration += 1
        if previous_end_ms is not None and start_ms - previous_end_ms > 5000:
            gaps_over_5s += 1
        previous_end_ms = end_ms
    return {
        "total": len(entries),
        "zero_duration": zero_duration,
        "gaps_over_5s": gaps_over_5s,
    }


def assert_srt_timing_healthy(entries: list[dict[str, object]]) -> None:
    stats = analyze_srt_timing(entries)
    zero_duration_limit = max(3, stats["total"] // 20)
    if stats["zero_duration"] > zero_duration_limit:
        raise SkillError(
            "字幕时间轴异常：零时长字幕过多，当前结果不适合直接烧录。"
        )


def build_translation_output_path(subtitle_path: str, target_language: str) -> pathlib.Path:
    path = pathlib.Path(subtitle_path)
    return path.with_name(f"{path.stem}.{target_language}{path.suffix}")


def chunk_srt_entries_for_translation(
    entries: list[dict[str, object]],
    *,
    max_chars: int = 500,
) -> list[list[dict[str, object]]]:
    chunks: list[list[dict[str, object]]] = []
    current_chunk: list[dict[str, object]] = []
    current_size = 0
    for entry in entries:
        text = str(entry["text"])
        projected = current_size + len(text)
        if current_chunk and projected > max_chars:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(entry)
        current_size += len(text) + 24
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def split_translated_chunk(translated_text: str, *, marker: str) -> list[str]:
    return [part.strip() for part in translated_text.split(marker)]


def translate_text_google(text: str, *, source_language: str, target_language: str) -> str:
    query = urllib_parse.urlencode(
        {
            "client": "gtx",
            "sl": source_language,
            "tl": target_language,
            "dt": "t",
            "q": text,
        }
    )
    url = f"https://translate.googleapis.com/translate_a/single?{query}"
    request = urllib_request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib_request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    translated_chunks = [chunk[0] for chunk in payload[0] if chunk and chunk[0]]
    if not translated_chunks:
        raise SkillError("翻译服务返回空结果，无法继续。")
    return "".join(translated_chunks)


def translate_entries_google(
    entries: list[dict[str, object]],
    *,
    source_language: str,
    target_language: str,
) -> list[dict[str, object]]:
    translated_entries: list[dict[str, object]] = []
    pending_chunks = list(chunk_srt_entries_for_translation(entries))
    while pending_chunks:
        chunk = pending_chunks.pop(0)
        marker_token = "__CODEx_SEG_1__"
        joined_text = f"\n{marker_token}\n".join(str(item["text"]) for item in chunk)
        last_error: Exception | None = None
        translated_text: str | None = None
        for _ in range(3):
            try:
                translated_text = translate_text_google(
                    joined_text,
                    source_language=source_language,
                    target_language=target_language,
                )
                break
            except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
        if translated_text is None:
            if len(chunk) > 1:
                mid = len(chunk) // 2
                pending_chunks.insert(0, chunk[mid:])
                pending_chunks.insert(0, chunk[:mid])
                continue
            raise SkillError(f"翻译请求失败: {last_error}") from last_error
        translated_parts = split_translated_chunk(translated_text, marker=marker_token)
        if len(translated_parts) != len(chunk):
            if len(chunk) > 1:
                mid = len(chunk) // 2
                pending_chunks.insert(0, chunk[mid:])
                pending_chunks.insert(0, chunk[:mid])
                continue
            raise SkillError(
                "单条字幕翻译后分段数量异常，停止继续，避免错误覆盖字幕。"
            )
        for original, translated in zip(chunk, translated_parts, strict=True):
            translated_entries.append({**original, "text": translated})
    return translated_entries


def _process_llm_chunk(chunk: list[dict], api_key: str, source_language: str, target_language: str) -> list[dict]:
    if not chunk:
        return []
        
    payload_data = [{"id": idx, "text": str(item["text"])} for idx, item in enumerate(chunk)]
    
    prompt = (
        f"Please translate the following subtitle lines from {source_language} to {target_language}. "
        "They are continuous lines from a video. Translate contextually and fluently. "
        "You MUST keep the exact original JSON structure, returning a JSON array, where each element has 'id' and 'text'. "
        f"DO NOT merge lines. The number of elements in the output array MUST exactly be {len(chunk)}! "
        "Output ONLY valid JSON, without any markdown formatting block or additional text.\n\n"
        f"{json.dumps(payload_data, ensure_ascii=False)}"
    )
    
    data = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "system", "content": f"You are a professional contextual subtitle translator. Output exact JSON array matched length. Never merge items. You must output exactly {len(chunk)} items."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }
    
    req = urllib_request.Request("https://api.siliconflow.cn/v1/chat/completions", json.dumps(data).encode("utf-8"), headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    })
    
    last_error = None
    for attempt in range(3):
        try:
            with urllib_request.urlopen(req, timeout=90) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                
            content = resp_json["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            out_arr = json.loads(content)
            
            if len(out_arr) != len(chunk):
                raise ValueError(f"行数不匹配: 返回 {len(out_arr)}，要求 {len(chunk)}")
                
            translated = []
            for orig, trans in zip(chunk, out_arr):
                translated.append({**orig, "text": trans.get("text", "")})
            return translated
        except Exception as e:
            last_error = e
            # print(f"    [Retry] chunk size {len(chunk)} failed on attempt {attempt+1}: {e}")
            
    # If it fails 3 times, split in half and conquer
    if len(chunk) > 1:
        print(f"    [Fallback] Chunk of size {len(chunk)} failed multiple times, splitting in half...")
        mid = len(chunk) // 2
        return _process_llm_chunk(chunk[:mid], api_key, source_language, target_language) + \
               _process_llm_chunk(chunk[mid:], api_key, source_language, target_language)
    else:
        raise SkillError(f"大模型连单条字幕都无法正确返回格式: {last_error}")

def translate_entries_llm(
    entries: list[dict[str, object]],
    *,
    source_language: str,
    target_language: str,
) -> list[dict[str, object]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SkillError("请设置 DEEPSEEK_API_KEY 环境变量来进行大模型上下文翻译。")
    
    chunk_size = 40
    translated_entries: list[dict[str, object]] = []
    
    for i in range(0, len(entries), chunk_size):
        chunk = entries[i:i + chunk_size]
        translated_entries.extend(_process_llm_chunk(chunk, api_key, source_language, target_language))
            
    return translated_entries


def _escape_ffmpeg_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace("'", r"\'")


def ensure_subtitle_filter_available(subtitle_kind: str, filters_text: str) -> str:
    filter_name = "subtitles" if subtitle_kind == "srt" else "ass"
    marker = f" {filter_name} "
    if marker not in f" {filters_text} ":
        raise SkillError(
            f"当前 ffmpeg 不支持 `{filter_name}` 字幕滤镜，无法直接烧录 {subtitle_kind.upper()} 字幕。"
        )
    return filter_name


def _detect_subtitle_filter(subtitle_kind: str) -> str:
    ffmpeg_bin = resolve_tool_path("ffmpeg")
    probe = subprocess.run(
        [ffmpeg_bin, "-filters"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise SkillError("无法读取 ffmpeg 支持的滤镜列表，无法继续烧录字幕。")
    return ensure_subtitle_filter_available(subtitle_kind, probe.stdout)


def build_burn_command(*, ffmpeg_bin: str, video_path: str, subtitle_path: str, output_path: str) -> list[str]:
    suffix = pathlib.Path(subtitle_path).suffix.lower()
    if suffix == ".srt":
        filter_name = "subtitles"
    elif suffix == ".ass":
        filter_name = "ass"
    else:
        raise SkillError(f"不支持的字幕文件类型: {suffix or '<无扩展名>'}")

    escaped_subtitle = _escape_ffmpeg_filter_path(subtitle_path)
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"{filter_name}=filename='{escaped_subtitle}'",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]


def install_missing_dependencies(check: CheckResult) -> None:
    if check.platform.platform_name != "macos":
        raise UnsupportedPlatformError(check.platform.message)
    if not _command_exists("brew"):
        raise SkillError("缺少 Homebrew，当前无法自动安装 macOS 依赖。")

    brew_packages: list[str] = []
    if "uv" in check.missing_tools:
        brew_packages.append("uv")
    if {"ffmpeg", "ffprobe"} & set(check.missing_tools):
        brew_packages.append("ffmpeg-full")
    if check.source.kind in REMOTE_VIDEO_KINDS and "yt-dlp" in check.missing_tools:
        brew_packages.append("yt-dlp")

    if brew_packages:
        _run_command(["brew", "install", *dict.fromkeys(brew_packages)])

    if not check.python_deps_ready:
        _run_command(["uv", "sync"], cwd=SKILL_ROOT)


def _timestamped_run_dir(source: SourceInfo, output_root: pathlib.Path | None) -> pathlib.Path:
    base_root = output_root or (SKILL_ROOT / "runs")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_name = pathlib.Path(source.resolved_path or "youtube").stem
    run_dir = base_root / f"{stamp}-{source_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _prepare_audio_input(
    source: SourceInfo,
    run_dir: pathlib.Path,
    *,
    yt_dlp_cookies_file: str | None,
    yt_dlp_cookies_from_browser: str | None,
    keep_video: bool,
    audio_preprocess: AudioPreprocessOptions,
) -> PreparedMedia:
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_tool_path("ffmpeg")

    if source.kind == "audio_file":
        output_audio = work_dir / "source.wav"
        _run_command(
            build_audio_preprocess_command(
                ffmpeg_bin=ffmpeg_bin,
                input_path=str(source.resolved_path),
                output_path=str(output_audio),
                source_is_video=False,
                options=audio_preprocess,
            )
        )
        return PreparedMedia(audio_path=output_audio)

    if source.kind == "video_file":
        output_audio = work_dir / "source.wav"
        _run_command(
            build_audio_preprocess_command(
                ffmpeg_bin=ffmpeg_bin,
                input_path=str(source.resolved_path),
                output_path=str(output_audio),
                source_is_video=True,
                options=audio_preprocess,
            )
        )
        return PreparedMedia(
            audio_path=output_audio,
            video_path=pathlib.Path(source.resolved_path),
        )

    if source.kind in REMOTE_VIDEO_KINDS:
        downloaded_audio_template = work_dir / "source-download.%(ext)s"
        _run_command(
            build_ytdlp_command(
                source_url=source.original,
                output_template=str(downloaded_audio_template),
                cookies_file=yt_dlp_cookies_file,
                cookies_from_browser=yt_dlp_cookies_from_browser,
                extract_audio=True,
            )
        )
        downloaded_audio = work_dir / "source-download.wav"
        output_audio = work_dir / "source.wav"
        if not downloaded_audio.exists():
            raise SkillError("yt-dlp 执行后未产出 source-download.wav，无法继续 ASR。")
        _run_command(
            build_audio_preprocess_command(
                ffmpeg_bin=ffmpeg_bin,
                input_path=str(downloaded_audio),
                output_path=str(output_audio),
                source_is_video=False,
                options=audio_preprocess,
            )
        )
        video_path: pathlib.Path | None = None
        if keep_video:
            output_video_template = work_dir / "source-video.%(ext)s"
            _run_command(
                build_ytdlp_command(
                    source_url=source.original,
                    output_template=str(output_video_template),
                    cookies_file=yt_dlp_cookies_file,
                    cookies_from_browser=yt_dlp_cookies_from_browser,
                    extract_audio=False,
                )
            )
            video_candidates = sorted(work_dir.glob("source-video.*"))
            video_path = next((path for path in video_candidates if path.suffix.lower() in VIDEO_EXTENSIONS), None)
            if video_path is None:
                raise SkillError("已要求保留远程视频，但 yt-dlp 执行后没有产出可用视频文件。")
        return PreparedMedia(audio_path=output_audio, video_path=video_path)

    raise SkillError(f"未知输入类型: {source.kind}")


def _run_local_asr(
    audio_path: pathlib.Path,
    *,
    run_dir: pathlib.Path,
    language: str | None,
    model_size: str,
    max_chars: int,
    timing_mode: str,
) -> dict[str, str]:
    output_dir = run_dir / "output"
    workspace_dir = run_dir / "workspace"
    isolated_home = run_dir / "home"
    cache_dir = SKILL_ROOT / "cache" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    isolated_home.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_dir = isolated_home / ".video-audio-subtitle"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_payload = {
        "output_dir": str(output_dir),
        "model_source": "modelscope",
        "model_cache_dir": str(cache_dir),
        "asr_model_size": model_size,
        "asr_language": language or "",
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["VIDEO_AUDIO_SUBTITLE_CONFIG_DIR"] = str(config_dir)
    env["MODELSCOPE_CACHE"] = str(cache_dir)
    env["HF_HOME"] = str(cache_dir)

    command = [
        str(_skill_venv_python(required=True)),
        str(SCRIPT_ROOT / "media_subtitle.py"),
        "internal-asr",
        str(audio_path),
        "--output-dir",
        str(output_dir),
        "--model-size",
        model_size,
        "--max-chars",
        str(max_chars),
        "--timing-mode",
        timing_mode,
    ]
    if language:
        command.extend(["--language", language])
    _run_command(command, cwd=SKILL_ROOT, env=env)

    stem = audio_path.stem
    candidates = {
        "srt": output_dir / f"{stem}.srt",
        "ass": output_dir / f"{stem}.ass",
        "raw_json": output_dir / f"{stem}.raw.json",
        "lines_json": output_dir / f"{stem}.lines.json",
        "quality_json": output_dir / f"{stem}.quality.json",
    }

    missing_outputs = [name for name, path in candidates.items() if not path.exists()]
    if missing_outputs:
        raise SkillError(f"ASR 执行完成但缺少输出文件: {', '.join(missing_outputs)}")

    srt_entries = parse_srt(candidates["srt"].read_text(encoding="utf-8"))
    stats = analyze_srt_timing(srt_entries)
    zero_duration_limit = max(3, stats["total"] // 20)
    if stats["zero_duration"] > zero_duration_limit:
        line_entries = json.loads(candidates["lines_json"].read_text(encoding="utf-8"))
        rebuilt_entries = build_srt_entries_from_line_entries(line_entries)
        assert_srt_timing_healthy(rebuilt_entries)
        fixed_srt_path = output_dir / f"{stem}.fixed.srt"
        fixed_srt_path.write_text(write_srt(rebuilt_entries), encoding="utf-8")
        candidates["srt"] = fixed_srt_path

    quality_summary = json.loads(candidates["quality_json"].read_text(encoding="utf-8"))
    return {
        "paths": {name: str(path) for name, path in candidates.items()},
        "quality_summary": quality_summary,
    }


def execute_pipeline(
    source: str,
    *,
    output_root: str | pathlib.Path | None = None,
    language: str | None = None,
    model_size: str = "0.6B",
    max_chars: int | None = None,
    keep_video: bool = False,
    timing_mode: str = "stable",
    normalize_audio: bool = False,
    trim_silence: bool = False,
    install_missing: bool = False,
    yt_dlp_cookies_file: str | None = None,
    yt_dlp_cookies_from_browser: str | None = None,
) -> dict[str, object]:
    check = check_environment(source)
    if not check.platform.supported:
        raise UnsupportedPlatformError(check.platform.message)

    if max_chars is None:
        if language and language.lower() == "english":
            max_chars = 38
        else:
            max_chars = 14

    if check.missing_tools or not check.python_deps_ready:
        if not install_missing:
            action_text = "\n".join(f"- {action}" for action in check.install_actions) or "- 无自动安装动作"
            raise MissingDependencyError(
                "检测到依赖未就绪，请先获得用户同意再安装：\n"
                f"{action_text}"
            )
        install_missing_dependencies(check)
        check = check_environment(source)
        if check.missing_tools or not check.python_deps_ready:
            raise MissingDependencyError("依赖安装后仍未就绪，停止执行。")

    run_dir = _timestamped_run_dir(
        check.source,
        pathlib.Path(output_root).expanduser().resolve() if output_root else None,
    )
    audio_preprocess = AudioPreprocessOptions(
        normalize_audio=normalize_audio,
        trim_silence=trim_silence,
    )
    validate_audio_preprocess_options(audio_preprocess)
    prepared_media = _prepare_audio_input(
        check.source,
        run_dir,
        yt_dlp_cookies_file=yt_dlp_cookies_file,
        yt_dlp_cookies_from_browser=yt_dlp_cookies_from_browser,
        keep_video=keep_video,
        audio_preprocess=audio_preprocess,
    )
    outputs = _run_local_asr(
        prepared_media.audio_path,
        run_dir=run_dir,
        language=language,
        model_size=model_size,
        max_chars=validate_max_chars(max_chars),
        timing_mode=timing_mode,
    )

    result = {
        "source": asdict(check.source),
        "prepared_audio": str(prepared_media.audio_path),
        "prepared_video": str(prepared_media.video_path) if prepared_media.video_path else None,
        "run_dir": str(run_dir),
        "max_chars": max_chars,
        "timing_mode": timing_mode,
        "audio_preprocess": asdict(audio_preprocess),
        "outputs": outputs["paths"],
        "quality_summary": outputs["quality_summary"],
    }
    manifest = run_dir / "result.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def execute_burn(
    video: str,
    subtitle: str,
    *,
    output: str | None = None,
    output_root: str | pathlib.Path | None = None,
    install_missing: bool = False,
) -> dict[str, object]:
    video_info = classify_source(video)
    if video_info.kind != "video_file":
        raise SkillError("burn 只接受本地视频文件作为输入。")

    subtitle_info = classify_subtitle(subtitle)
    platform_info = detect_platform()
    if not platform_info.supported:
        raise UnsupportedPlatformError(platform_info.message)

    missing_tools = [tool for tool in ("ffmpeg",) if not _command_exists(tool)]
    install_actions = build_install_actions(
        missing_tools=missing_tools,
        python_deps_ready=True,
        source_kind=video_info.kind,
        platform_name=platform_info.platform_name,
        brew_available=_command_exists("brew"),
    )
    if missing_tools:
        if not install_missing:
            action_text = "\n".join(f"- {action}" for action in install_actions) or "- 无自动安装动作"
            raise MissingDependencyError(
                "检测到依赖未就绪，请先获得用户同意再安装：\n"
                f"{action_text}"
            )
        install_missing_dependencies(
            CheckResult(
                source=video_info,
                platform=platform_info,
                missing_tools=missing_tools,
                python_deps_ready=True,
                install_actions=install_actions,
            )
        )

    run_dir = _timestamped_run_dir(
        video_info,
        pathlib.Path(output_root).expanduser().resolve() if output_root else None,
    )
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output:
        output_path = pathlib.Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_stem = pathlib.Path(video_info.resolved_path).stem
        output_path = output_dir / f"{video_stem}.subtitled.mp4"

    _detect_subtitle_filter(subtitle_info.kind)
    ffmpeg_bin = resolve_tool_path("ffmpeg")
    _run_command(
        build_burn_command(
            ffmpeg_bin=ffmpeg_bin,
            video_path=str(video_info.resolved_path),
            subtitle_path=subtitle_info.resolved_path,
            output_path=str(output_path),
        )
    )

    result = {
        "video": asdict(video_info),
        "subtitle": asdict(subtitle_info),
        "run_dir": str(run_dir),
        "output_video": str(output_path),
    }
    manifest = run_dir / "result.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def execute_translate(
    subtitle: str,
    *,
    target_language: str = "zh-CN",
    source_language: str = "auto",
    output: str | None = None,
) -> dict[str, object]:
    subtitle_info = classify_subtitle(subtitle)
    if subtitle_info.kind != "srt":
        raise SkillError("translate 当前只支持 SRT；ASS 翻译暂未接入。")

    subtitle_path = pathlib.Path(subtitle_info.resolved_path)
    entries = parse_srt(subtitle_path.read_text(encoding="utf-8"))
    assert_srt_timing_healthy(entries)
    print(f"开始翻译字幕，共 {len(entries)} 条。")
    translated_entries = translate_entries_llm(
        entries,
        source_language=source_language,
        target_language=target_language,
    )

    output_path = pathlib.Path(output).expanduser().resolve() if output else build_translation_output_path(
        subtitle_info.resolved_path,
        target_language,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(write_srt(translated_entries), encoding="utf-8")

    result = {
        "subtitle": asdict(subtitle_info),
        "target_language": target_language,
        "source_language": source_language,
        "output_subtitle": str(output_path),
    }
    return result


def split_lines_json(
    lines_json: str,
    *,
    line: int,
    after: str,
    output: str | None = None,
) -> dict[str, object]:
    lines_path = pathlib.Path(lines_json).expanduser().resolve()
    if not lines_path.exists():
        raise FileNotFoundError(f"找不到 lines.json 文件: {lines_path}")
    if lines_path.suffix.lower() != ".json":
        raise SkillError(f"split 只接受 lines.json 文件，当前输入不是 json: {lines_path}")
    if not after:
        raise SkillError("`--after` 不能为空，必须明确提供拆分位置。")

    lines = _load_lines(str(lines_path))
    updated_lines = split_line_after(lines, line, after)

    output_path = pathlib.Path(output).expanduser().resolve() if output else lines_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_lines(updated_lines, str(output_path))

    return {
        "input_lines_json": str(lines_path),
        "output_lines_json": str(output_path),
        "line": line,
        "after": after,
    }


def _check_command(args: argparse.Namespace) -> int:
    result = check_environment(args.source)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


def _install_command(args: argparse.Namespace) -> int:
    result = check_environment(args.source)
    install_missing_dependencies(result)
    print("依赖安装完成。")
    return 0


def _run_command_entry(args: argparse.Namespace) -> int:
    result = execute_pipeline(
        args.source,
        output_root=args.output_root,
        language=args.language,
        model_size=args.model_size,
        max_chars=args.max_chars,
        keep_video=args.keep_video,
        timing_mode=args.timing_mode,
        normalize_audio=args.normalize_audio,
        trim_silence=args.trim_silence,
        install_missing=args.install_missing,
        yt_dlp_cookies_file=args.yt_dlp_cookies,
        yt_dlp_cookies_from_browser=args.yt_dlp_cookies_from_browser,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _burn_command_entry(args: argparse.Namespace) -> int:
    result = execute_burn(
        args.video,
        args.subtitle,
        output=args.output,
        output_root=args.output_root,
        install_missing=args.install_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _translate_command_entry(args: argparse.Namespace) -> int:
    result = execute_translate(
        args.subtitle,
        target_language=args.target_language,
        source_language=args.source_language,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _split_command_entry(args: argparse.Namespace) -> int:
    result = split_lines_json(
        args.lines_json,
        line=args.line,
        after=args.after,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _internal_asr_command_entry(args: argparse.Namespace) -> int:
    try:
        run_pipeline(
            args.audio,
            output_dir=args.output_dir,
            fmt="all",
            ass_style="default",
            fix_dir=None,
            language=args.language,
            model_size=args.model_size,
            max_chars=args.max_chars,
            timing_mode=args.timing_mode,
            resume_from=None,
        )
    except PipelineQualityError as exc:
        raise SkillError(str(exc)) from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视频/音频转字幕 skill 入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="检查输入与依赖")
    check_parser.add_argument("source", help="本地音频/视频路径或 YouTube 链接")
    check_parser.set_defaults(func=_check_command)

    install_parser = subparsers.add_parser("install", help="安装缺失依赖")
    install_parser.add_argument("source", help="本地音频/视频路径或 YouTube 链接")
    install_parser.set_defaults(func=_install_command)

    run_parser = subparsers.add_parser("run", help="执行完整字幕流程")
    run_parser.add_argument("source", help="本地音频/视频路径或 YouTube 链接")
    run_parser.add_argument("--output-root", default=None)
    run_parser.add_argument("--language", default=None)
    run_parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default="0.6B")
    run_parser.add_argument(
        "--timing-mode",
        choices=["stable", "experimental_segment_align"],
        default="stable",
        help="时间轴模式。默认 stable；experimental_segment_align 只用于显式实验。",
    )
    run_parser.add_argument("--max-chars", type=validate_max_chars, default=None, help="如果为 None 则按语种自动推断 (默认中日文14，纯英文38)")
    run_parser.add_argument(
        "--keep-video",
        action="store_true",
        help="远程视频源时额外保留下载后的视频文件，便于后续烧录字幕。",
    )
    run_parser.add_argument(
        "--normalize-audio",
        action="store_true",
        help="显式启用轻音量归一化（loudnorm）；默认只做格式规范化，不做增强。",
    )
    run_parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="显式裁掉首尾静音；会改变输入音频边界，默认关闭。",
    )
    run_parser.add_argument("--yt-dlp-cookies", default=None)
    run_parser.add_argument("--yt-dlp-cookies-from-browser", default=None)
    run_parser.add_argument(
        "--install-missing",
        action="store_true",
        help="已获用户同意时，自动安装缺失依赖后继续执行",
    )
    run_parser.set_defaults(func=_run_command_entry)

    burn_parser = subparsers.add_parser("burn", help="把本地字幕烧录进本地视频")
    burn_parser.add_argument("video", help="本地视频路径")
    burn_parser.add_argument("subtitle", help="本地字幕路径，仅支持 srt/ass")
    burn_parser.add_argument("--output", default=None, help="输出 mp4 绝对路径")
    burn_parser.add_argument("--output-root", default=None)
    burn_parser.add_argument(
        "--install-missing",
        action="store_true",
        help="已获用户同意时，自动安装缺失依赖后继续执行",
    )
    burn_parser.set_defaults(func=_burn_command_entry)

    translate_parser = subparsers.add_parser("translate", help="把 SRT 字幕翻译成目标语言")
    translate_parser.add_argument("subtitle", help="本地 srt 字幕路径")
    translate_parser.add_argument("--target-language", default="zh-CN")
    translate_parser.add_argument("--source-language", default="auto")
    translate_parser.add_argument("--output", default=None)
    translate_parser.set_defaults(func=_translate_command_entry)

    split_parser = subparsers.add_parser("split", help="手工拆分 lines.json 中的某一条字幕")
    split_parser.add_argument("lines_json", help="由 run 生成的 <name>.lines.json")
    split_parser.add_argument("--line", type=int, required=True, help="要拆分的字幕行号，从 1 开始")
    split_parser.add_argument("--after", required=True, help="在该文本之后拆开，必须与原行中的片段完全匹配")
    split_parser.add_argument("--output", default=None, help="可选输出 json 路径；默认覆盖原文件")
    split_parser.set_defaults(func=_split_command_entry)

    internal_asr_parser = subparsers.add_parser("internal-asr", help=argparse.SUPPRESS)
    internal_asr_parser.add_argument("audio")
    internal_asr_parser.add_argument("--output-dir", required=True)
    internal_asr_parser.add_argument("--language", default=None)
    internal_asr_parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default="0.6B")
    internal_asr_parser.add_argument("--max-chars", type=validate_max_chars, required=True)
    internal_asr_parser.add_argument(
        "--timing-mode",
        choices=["stable", "experimental_segment_align"],
        default="stable",
    )
    internal_asr_parser.set_defaults(func=_internal_asr_command_entry)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except SkillError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
