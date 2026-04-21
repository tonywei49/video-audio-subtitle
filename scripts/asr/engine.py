from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict

from scripts.shared.model_path import resolve_model_path
from scripts.shared.platform import get_backend

from .types import ASRResult, WordTimestamp

ASR_MODELS = {
    "cuda": {
        "1.7B": "Qwen/Qwen3-ASR-1.7B",
        "0.6B": "Qwen/Qwen3-ASR-0.6B",
    },
    "mlx": {
        "1.7B": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "0.6B": "mlx-community/Qwen3-ASR-0.6B-8bit",
    },
}

ALIGNER_MODELS = {
    "cuda": "Qwen/Qwen3-ForcedAligner-0.6B",
    "mlx": "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
}

_loaded_models: dict[str, object] = {}
_PUNCT = set('，。、！？；：""''《》【】（）,.!?;:\'"()[]{}·…—~ ')
_LONG_AUDIO_DURATION = 300.0
_MLX_SEGMENT_CHUNK_DURATION = 10.0


def _load_asr_cuda(model_id: str, with_aligner: bool = True):
    import torch
    from qwen_asr import Qwen3ASRModel

    asr_path = resolve_model_path(model_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if with_aligner:
        aligner_path = resolve_model_path(ALIGNER_MODELS["cuda"])
        return Qwen3ASRModel.from_pretrained(
            asr_path,
            dtype=torch.bfloat16,
            device_map=device,
            forced_aligner=aligner_path,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=device),
            max_inference_batch_size=8,
            max_new_tokens=4096,
        )

    return Qwen3ASRModel.from_pretrained(
        asr_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=8,
        max_new_tokens=4096,
    )


def _load_asr_mlx(model_id: str):
    from mlx_audio.stt.utils import load_model

    return load_model(resolve_model_path(model_id))


def _load_aligner_mlx():
    from mlx_audio.stt.utils import load_model

    return load_model(resolve_model_path(ALIGNER_MODELS["mlx"]))


def get_asr_model(model_id: str, with_aligner: bool = True):
    backend = get_backend()
    cache_key = f"{backend}_{model_id}_aligner_{with_aligner}"
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]

    if backend == "mlx":
        result = {"asr": _load_asr_mlx(model_id)}
        if with_aligner:
            result["aligner"] = _load_aligner_mlx()
        _loaded_models[cache_key] = result
        return result

    model = _load_asr_cuda(model_id, with_aligner=with_aligner)
    _loaded_models[cache_key] = model
    return model


def load_audio(audio_path: str):
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _restore_punctuation(words: list[WordTimestamp], full_text: str) -> list[WordTimestamp]:
    if not words or not full_text:
        return words

    result = [WordTimestamp(w.text, w.start_time, w.end_time) for w in words]
    pos = 0
    for wi in range(len(result)):
        expected = result[wi].text
        remaining = full_text[pos:]
        idx = remaining.find(expected)
        if idx < 0:
            continue
        if wi > 0:
            for ch in remaining[:idx]:
                if ch in _PUNCT:
                    result[wi - 1] = WordTimestamp(
                        result[wi - 1].text + ch,
                        result[wi - 1].start_time,
                        result[wi - 1].end_time,
                    )
        pos += idx + len(expected)
        while pos < len(full_text) and full_text[pos] in _PUNCT:
            result[wi] = WordTimestamp(
                result[wi].text + full_text[pos],
                result[wi].start_time,
                result[wi].end_time,
            )
            pos += 1
    return result


def asr_transcribe(audio_path: str, language: str | None = None, model_size: str = "1.7B") -> str:
    backend = get_backend()
    models = ASR_MODELS[backend]
    if model_size not in models:
        raise ValueError(f"Unknown model size '{model_size}'. Available: {', '.join(models.keys())}")

    model_id = models[model_size]
    model = get_asr_model(model_id, with_aligner=False)
    if backend == "mlx":
        return _asr_transcribe_mlx(model["asr"], audio_path, language)

    results = model.transcribe(audio=audio_path, language=language, return_time_stamps=False)
    return results[0].text


def _asr_transcribe_mlx(model, audio_path: str, language: str | None):
    return model.generate(audio_path, language=_language_to_code(language), verbose=True).text


def asr_align(audio_path: str, language: str | None = None, model_size: str = "1.7B") -> ASRResult:
    backend = get_backend()
    models = ASR_MODELS[backend]
    if model_size not in models:
        raise ValueError(f"Unknown model size '{model_size}'. Available: {', '.join(models.keys())}")

    model_id = models[model_size]
    wav, sr = load_audio(audio_path)
    duration = len(wav) / sr
    print(f"Audio duration: {duration:.2f}s", flush=True)

    if backend == "mlx" and duration > _LONG_AUDIO_DURATION:
        models_dict = get_asr_model(model_id, with_aligner=False)
        return _asr_align_mlx_segmented(
            models_dict["asr"],
            audio_path,
            duration,
            language,
        )

    models_dict = get_asr_model(model_id, with_aligner=True)
    if backend == "mlx":
        return _asr_align_mlx(models_dict["asr"], models_dict["aligner"], audio_path, wav, sr, duration, language)
    return _asr_align_cuda(models_dict, audio_path, wav, sr, duration, language)


def _asr_align_cuda(model, audio_path: str, wav, sr: int, duration: float, language: str | None) -> ASRResult:
    if duration > 300:
        chunk_duration = 30.0
        segment_samples = int(chunk_duration * sr)
        all_words = []
        detected_language = None
        for start in range(0, len(wav), segment_samples):
            end = min(start + segment_samples, len(wav))
            offset = start / sr
            results = model.transcribe(audio=(wav[start:end], sr), language=language, return_time_stamps=True)
            if detected_language is None:
                detected_language = results[0].language
            chunk_text = results[0].text
            chunk_words = []
            if results[0].time_stamps:
                for ts in results[0].time_stamps:
                    chunk_words.append(
                        WordTimestamp(text=ts.text, start_time=ts.start_time + offset, end_time=ts.end_time + offset)
                    )
            all_words.extend(_restore_punctuation(chunk_words, chunk_text))
        return ASRResult(
            language=detected_language or "unknown",
            text="".join(w.text for w in all_words),
            duration=duration,
            words=all_words,
        )

    results = model.transcribe(audio=audio_path, language=language, return_time_stamps=True)
    result = results[0]
    words = []
    if result.time_stamps:
        for ts in result.time_stamps:
            words.append(WordTimestamp(text=ts.text, start_time=ts.start_time, end_time=ts.end_time))
    words = _restore_punctuation(words, result.text)
    return ASRResult(language=result.language, text=result.text, duration=duration, words=words)


def _asr_align_mlx(asr_model, aligner_model, audio_path: str, wav, sr: int, duration: float, language: str | None):
    lang_code = _language_to_code(language) or "chinese"
    detected_language = lang_code

    print("Step 1/2: Transcribing (full audio)...", flush=True)
    asr_result = asr_model.generate(
        audio_path,
        language=lang_code,
        verbose=True,
    )
    full_text = asr_result.text
    if not full_text:
        return ASRResult(language=detected_language, text="", duration=duration, words=[])

    print(f"Transcribed text ({len(full_text)} chars): {full_text[:100]}...", flush=True)

    chunk_duration = 30.0
    chunk_samples = int(chunk_duration * sr)
    total_chunks = (len(wav) + chunk_samples - 1) // chunk_samples
    print(f"Step 2/2: Forced alignment ({chunk_duration:.0f}s chunks, {total_chunks} total)...", flush=True)

    all_words = []

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_samples
        chunk_end = min(chunk_start + chunk_samples, len(wav))
        chunk_wav = wav[chunk_start:chunk_end]
        offset = chunk_start / sr

        print(
            f"  Chunk {chunk_idx + 1}/{total_chunks} "
            f"(offset={offset:.1f}s, dur={len(chunk_wav) / sr:.1f}s)..."
        , flush=True)

        chunk_started_at = time.perf_counter()
        print(f"    Chunk ASR start", flush=True)
        try:
            chunk_asr = asr_model.generate(chunk_wav, language=lang_code, verbose=False)
            chunk_text = chunk_asr.text if (chunk_asr and chunk_asr.text) else ""
        except Exception as exc:
            print(f"    ASR failed: {exc}. Skipping chunk.", flush=True)
            continue
        chunk_asr_elapsed = time.perf_counter() - chunk_started_at

        if not chunk_text.strip():
            print("    No speech detected. Skipping chunk.", flush=True)
            continue
        print(f"    Chunk ASR done ({chunk_asr_elapsed:.1f}s, {len(chunk_text)} chars)", flush=True)

        align_started_at = time.perf_counter()
        print("    Chunk align start", flush=True)
        try:
            align_result = aligner_model.generate(audio=chunk_wav, text=chunk_text, language=lang_code)
        except Exception as exc:
            print(f"    Alignment failed: {exc}. Skipping chunk.", flush=True)
            continue
        align_elapsed = time.perf_counter() - align_started_at
        print(f"    Chunk align done ({align_elapsed:.1f}s, {len(align_result)} items)", flush=True)

        chunk_words = []
        for item in align_result:
            chunk_words.append(
                WordTimestamp(text=item.text, start_time=item.start_time + offset, end_time=item.end_time + offset)
            )
        all_words.extend(_restore_punctuation(chunk_words, chunk_text))
        total_elapsed = time.perf_counter() - chunk_started_at
        print(f"    Chunk merged ({len(chunk_words)} words, total {total_elapsed:.1f}s)", flush=True)

    merged_text = "".join(w.text for w in all_words) or full_text
    return ASRResult(language=detected_language, text=merged_text, duration=duration, words=all_words)


def _asr_align_mlx_segmented(asr_model, audio_path: str, duration: float, language: str | None) -> ASRResult:
    detected_language = _language_to_code(language) or "chinese"
    print(
        "MLX 长音频改走 segment 时间线模式："
        "不再执行每个 30 秒块的二次 ASR + ForcedAligner，"
        "改为直接使用 ASR 的 segment 时间并在段内按文本比例分配时间。",
        flush=True,
    )
    print(
        f"Step 1/1: Transcribing with chunk_duration={_MLX_SEGMENT_CHUNK_DURATION:.0f}s ...",
        flush=True,
    )
    asr_result = asr_model.generate(
        audio_path,
        language=detected_language,
        verbose=True,
        chunk_duration=_MLX_SEGMENT_CHUNK_DURATION,
    )
    full_text = asr_result.text
    segments = getattr(asr_result, "segments", None) or []
    if not full_text or not segments:
        raise RuntimeError("MLX 长音频 segment 模式没有返回可用 segments，无法继续生成字幕。")

    print(f"Segments: {len(segments)}", flush=True)
    all_words = _segments_to_word_timestamps(segments)
    if not all_words:
        raise RuntimeError("MLX 长音频 segment 模式未能生成任何可用时间片。")

    return ASRResult(
        language=detected_language,
        text=full_text,
        duration=duration,
        words=all_words,
    )


def _segments_to_word_timestamps(segments: list[dict]) -> list[WordTimestamp]:
    all_words: list[WordTimestamp] = []
    for segment in segments:
        text = str(segment.get("text", "") or "")
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if not text.strip() or end <= start:
            continue
        all_words.extend(_segment_text_to_words(text, start, end))
    return all_words


def _segment_text_to_words(text: str, start: float, end: float) -> list[WordTimestamp]:
    parts = _tokenize_segment_text(text)
    if not parts:
        return [WordTimestamp(text=text, start_time=start, end_time=end)]

    weights = [_token_weight(part) for part in parts]
    total_weight = sum(weights)
    if total_weight <= 0:
        return [WordTimestamp(text=text, start_time=start, end_time=end)]

    duration = end - start
    cursor = start
    result: list[WordTimestamp] = []
    for idx, (part, weight) in enumerate(zip(parts, weights, strict=True)):
        part_duration = duration * (weight / total_weight)
        part_end = end if idx == len(parts) - 1 else cursor + part_duration
        result.append(
            WordTimestamp(
                text=part,
                start_time=cursor,
                end_time=max(cursor, part_end),
            )
        )
        cursor = part_end
    return result


def _tokenize_segment_text(text: str) -> list[str]:
    parts = re.findall(r"\S+\s*", text)
    if len(parts) > 1:
        return parts

    stripped = text.strip()
    if not stripped:
        return []

    def _is_cjk_char(ch: str) -> bool:
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7AF)
        )

    tokens: list[str] = []
    latin_buffer = ""
    for ch in text:
        if _is_cjk_char(ch):
            if latin_buffer:
                tokens.append(latin_buffer)
                latin_buffer = ""
            tokens.append(ch)
            continue

        if ch.isspace() or ch in _PUNCT:
            if latin_buffer:
                latin_buffer += ch
                tokens.append(latin_buffer)
                latin_buffer = ""
            elif tokens:
                tokens[-1] += ch
            else:
                latin_buffer += ch
            continue

        latin_buffer += ch

    if latin_buffer:
        tokens.append(latin_buffer)
    return tokens or [text]


def _token_weight(token: str) -> float:
    weight = 0.0
    for ch in token:
        cp = ord(ch)
        is_cjk = (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7AF)
        )
        if is_cjk:
            weight += 1.0
        elif ch not in _PUNCT and not ch.isspace():
            weight += 0.5
    return weight if weight > 0 else 0.5


def _language_to_code(language: str | None):
    if not language:
        return None
    mapping = {
        "chinese": "Chinese",
        "english": "English",
        "japanese": "Japanese",
        "korean": "Korean",
        "german": "German",
        "french": "French",
        "russian": "Russian",
    }
    lower = language.lower().strip()
    if lower in mapping:
        return mapping[lower]
    for _, name in mapping.items():
        if name.lower() == lower:
            return name
    return language


def result_to_dict(result: ASRResult) -> dict:
    return {
        "language": result.language,
        "text": result.text,
        "duration": result.duration,
        "words": [asdict(word) for word in result.words],
    }
