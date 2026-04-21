from __future__ import annotations

import csv
import glob
import json
import os
from typing import Optional

from .engine import asr_align, result_to_dict, summarize_word_timing, validate_word_timing_summary
from .render import ASSSubtitleStyle, render_ass_from_lines, render_srt_from_lines
from .types import CheckError, Paragraph, SubtitleLine

_PUNCT = set('，。、！？；：\u201c\u201d\u2018\u2019《》【】（）,.!?;:\'"()[]{}·…—~ ')
_SENTENCE_END = set("。！？!?.")
_COMMA = set("，,;；：:")
_TRILING_STRIP = set("，。,.")
_LINE_MIN_DURATION = 0.4
_LINE_MAX_DURATION = 8.0
_LINE_OVERLAP_TOLERANCE = 0.05
_FRAGMENT_MERGE_GAP = 0.35
_ENGLISH_FRAGMENT_CONTENT = 4.0
_CJK_FRAGMENT_CONTENT = 2.0
_ENGLISH_SHORT_LINE_CONTENT = 5.0
_CJK_SHORT_LINE_CONTENT = 3.0


class PipelineQualityError(RuntimeError):
    """Raised when ASR or subtitle timing quality is not healthy enough to continue."""


def _word_cjk_len(word_text: str) -> float:
    count = 0.0
    for ch in word_text:
        cp = ord(ch)
        is_cjk = (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7AF)
        )
        if is_cjk:
            count += 1
        elif ch not in _PUNCT and ch != " ":
            count += 0.5
    return count


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0xAC00 <= cp <= 0xD7AF)
    )


def _word_ends_with(word_text: str, char_set: set) -> bool:
    for ch in reversed(word_text):
        if ch in char_set:
            return True
        if ch not in _PUNCT:
            break
    return False


def _text_of_words(words: list) -> str:
    return "".join(w.get("text", "") for w in words)


def _line_cjk_count(text: str) -> float:
    return _word_cjk_len(text)


def _line_visible_char_count(text: str) -> float:
    return float(sum(1 for ch in text if not ch.isspace() and ch not in _PUNCT))


def _detect_profile_from_words(words: list) -> str:
    ascii_alpha = 0
    cjk_count = 0
    for word in words:
        for ch in word.get("text", ""):
            if _is_cjk_char(ch):
                cjk_count += 1
            elif ch.isascii() and ch.isalpha():
                ascii_alpha += 1
    return "english" if ascii_alpha > cjk_count else "cjk"


def _line_unit_count(text: str, profile: str) -> float:
    if profile == "english":
        return _line_visible_char_count(text)
    return _line_cjk_count(text)


def stage1_asr(
    audio_path: str,
    output_dir: str,
    language=None,
    model_size: str = "1.7B",
    timing_mode: str = "stable",
) -> dict:
    result = asr_align(audio_path, language=language, model_size=model_size, timing_mode=timing_mode)
    result_dict = result_to_dict(result)
    result_dict["quality_summary"] = {
        "word_timing": summarize_word_timing(result.words, result.duration),
    }
    result_dict["source"] = os.path.basename(audio_path)
    raw_path = os.path.join(output_dir, _raw_json_name(audio_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(result_dict, handle, ensure_ascii=False, indent=2)
    issues = validate_word_timing_summary(result_dict["quality_summary"]["word_timing"])
    if issues:
        raise PipelineQualityError("ASR 词级时间轴检查失败：\n- " + "\n- ".join(issues))
    return result_dict


def stage2_break(result_dict: dict, output_dir: str, audio_path: str, max_chars: int = 14) -> list[SubtitleLine]:
    words = result_dict.get("words", [])
    if not words:
        return []
    paragraphs = _build_paragraphs(words)
    all_lines: list[SubtitleLine] = []
    for para in paragraphs:
        all_lines.extend(_break_paragraph(para, max_chars))
    for i in range(len(all_lines) - 1):
        gap = all_lines[i + 1].start_time - all_lines[i].end_time
        all_lines[i].pause_after = max(0, gap)
    lines_path = os.path.join(output_dir, _lines_json_name(audio_path))
    _save_lines(all_lines, lines_path)
    return all_lines


def _build_paragraphs(words: list) -> list[Paragraph]:
    paragraphs = []
    current = []
    for word in words:
        current.append(word)
        if _word_ends_with(word.get("text", ""), _SENTENCE_END):
            paragraphs.append(
                Paragraph(
                    text=_text_of_words(current),
                    start_time=current[0].get("start_time", 0),
                    end_time=current[-1].get("end_time", current[0].get("start_time", 0)),
                    words=[dict(w) for w in current],
                )
            )
            current = []
    if current:
        paragraphs.append(
            Paragraph(
                text=_text_of_words(current),
                start_time=current[0].get("start_time", 0),
                end_time=current[-1].get("end_time", current[0].get("start_time", 0)),
                words=[dict(w) for w in current],
            )
        )
    return paragraphs


def _break_paragraph(para: Paragraph, max_chars: int) -> list[SubtitleLine]:
    words = para.words
    if not words:
        return []
    profile = _detect_profile_from_words(words)
    segments = []
    current_words = []
    for word in words:
        current_words.append(word)
        if _word_ends_with(word.get("text", ""), _COMMA):
            segments.append(list(current_words))
            current_words = []
    if current_words:
        segments.append(list(current_words))

    lines = []
    for seg_words in segments:
        seg_len = sum(_line_unit_count(w.get("text", ""), profile) for w in seg_words)
        if seg_len <= max_chars:
            _emit_line(lines, seg_words)
        else:
            lines.extend(_smart_split(seg_words, max_chars, profile))
    return _compact_lines(lines, max_chars, profile)


def _smart_split(words: list, max_chars: int, profile: str) -> list[SubtitleLine]:
    if not words:
        return []
    line = _words_to_line(words)
    if line and _line_is_acceptable(line, max_chars, profile):
        return [line]

    valid_points = _find_valid_split_points(words, max_chars, profile)
    if valid_points:
        split_idx = max(valid_points, key=lambda idx: _split_score(words, idx, profile))
    else:
        split_idx = _find_best_force_split(words, max_chars, profile)
        if split_idx <= 0 or split_idx >= len(words):
            line = _words_to_line(words)
            return [line] if line else []

    left_words = words[:split_idx]
    right_words = words[split_idx:]
    return _smart_split(left_words, max_chars, profile) + _smart_split(right_words, max_chars, profile)


def _find_valid_split_points(words: list, max_chars: int, profile: str) -> list[int]:
    valid = []
    for idx in range(1, len(words)):
        left_len = sum(_line_unit_count(w.get("text", ""), profile) for w in words[:idx])
        right_len = sum(_line_unit_count(w.get("text", ""), profile) for w in words[idx:])
        if left_len <= max_chars and right_len <= max_chars:
            valid.append(idx)
    return valid


def _get_time_gap(words: list, split_idx: int) -> float:
    if split_idx <= 0 or split_idx >= len(words):
        return 0
    prev_end = words[split_idx - 1].get("end_time", 0)
    curr_start = words[split_idx].get("start_time", prev_end)
    return curr_start - prev_end


def _split_score(words: list, split_idx: int, profile: str) -> float:
    score = _get_time_gap(words, split_idx)
    prev_text = words[split_idx - 1].get("text", "")
    next_text = words[split_idx].get("text", "").strip().lower()
    if _word_ends_with(prev_text, _COMMA):
        score += 0.8
    if _word_ends_with(prev_text, _SENTENCE_END):
        score += 1.2
    if profile == "english" and next_text in {"and", "but", "or", "so", "because", "that", "which"}:
        score += 0.2
    return score


def _find_best_force_split(words: list, max_chars: int, profile: str) -> int:
    total_len = sum(_line_unit_count(w.get("text", ""), profile) for w in words)
    target = total_len / 2
    best_idx = 0
    best_score = float("inf")
    acc = 0.0
    for idx in range(1, len(words)):
        acc += _line_unit_count(words[idx - 1].get("text", ""), profile)
        if acc > max_chars:
            break
        score = abs(acc - target)
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _words_to_line(words: list) -> Optional[SubtitleLine]:
    if not words:
        return None
    return SubtitleLine(
        text=_text_of_words(words),
        start_time=words[0].get("start_time", 0),
        end_time=words[-1].get("end_time", words[0].get("start_time", 0)),
        words=[dict(w) for w in words],
    )


def _emit_line(lines: list[SubtitleLine], words: list) -> None:
    line = _words_to_line(words)
    if line:
        lines.append(line)


def _line_is_acceptable(line: SubtitleLine, max_chars: int, profile: str) -> bool:
    duration = float(line.end_time) - float(line.start_time)
    content_units = _line_unit_count(line.text, profile)
    if content_units > max_chars:
        return False
    if duration <= 0:
        return False
    if duration > _LINE_MAX_DURATION:
        return False
    cps = content_units / duration
    if cps > _line_cps_threshold(line.text):
        return False
    return True


def _is_fragment_line(line: SubtitleLine, profile: str) -> bool:
    duration = float(line.end_time) - float(line.start_time)
    content_units = _line_unit_count(line.text, profile)
    threshold = _ENGLISH_FRAGMENT_CONTENT if profile == "english" else _CJK_FRAGMENT_CONTENT
    return duration < 0.8 or content_units <= threshold


def _merge_two_lines(left: SubtitleLine, right: SubtitleLine) -> SubtitleLine:
    return SubtitleLine(
        text=left.text + right.text,
        start_time=left.start_time,
        end_time=right.end_time,
        words=[dict(word) for word in left.words] + [dict(word) for word in right.words],
        pause_after=right.pause_after,
    )


def _compact_lines(lines: list[SubtitleLine], max_chars: int, profile: str) -> list[SubtitleLine]:
    if len(lines) < 2:
        return lines
    current_lines = list(lines)
    while True:
        compacted: list[SubtitleLine] = []
        idx = 0
        merged_any = False
        while idx < len(current_lines):
            current = current_lines[idx]
            if idx + 1 < len(current_lines):
                next_line = current_lines[idx + 1]
                gap = max(0.0, next_line.start_time - current.end_time)
                merged = _merge_two_lines(current, next_line)
                if (
                    gap <= _FRAGMENT_MERGE_GAP
                    and (_is_fragment_line(current, profile) or _is_fragment_line(next_line, profile))
                    and _line_is_acceptable(merged, max_chars, profile)
                ):
                    compacted.append(merged)
                    idx += 2
                    merged_any = True
                    continue
            compacted.append(current)
            idx += 1
        if not merged_any:
            return compacted
        current_lines = compacted


def stage3_fix(lines: list[SubtitleLine], fix_dir: str) -> list[SubtitleLine]:
    if not fix_dir or not os.path.isdir(fix_dir):
        return lines
    fixed = [SubtitleLine(**_line_to_dict(line)) for line in lines]
    csv_paths = sorted(glob.glob(os.path.join(fix_dir, "fix_*.csv")))
    for csv_path in csv_paths:
        rules = _load_csv(csv_path)
        next_lines = []
        for line in fixed:
            replacement = rules.get(line.text)
            if replacement is None:
                next_lines.append(line)
            elif replacement.strip():
                line.text = replacement
                next_lines.append(line)
        fixed = next_lines
    return fixed


def _load_csv(csv_path: str) -> dict[str, str]:
    rules: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) >= 2:
                rules[row[0].strip()] = row[1].strip()
    return rules


def check_max_chars(lines: list[SubtitleLine], max_chars: int = 14) -> list[CheckError]:
    errors = []
    for idx, line in enumerate(lines, start=1):
        profile = _detect_profile_from_words(line.words) if line.words else "cjk"
        line_len = _line_unit_count(line.text, profile)
        if line_len > max_chars:
            errors.append(
                CheckError(
                    line_idx=idx,
                    checker="max_chars",
                    message=f'Line {idx} has {line_len:.1f} chars (max {max_chars}): "{line.text}"',
                )
            )
    return errors


def _line_cps_threshold(text: str) -> float:
    ascii_alpha = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    visible_chars = sum(1 for ch in text if not ch.isspace())
    non_ascii_visible = max(0, visible_chars - ascii_alpha)
    return 20.0 if ascii_alpha > non_ascii_visible else 8.0


def _short_line_content_threshold(profile: str) -> float:
    return _ENGLISH_SHORT_LINE_CONTENT if profile == "english" else _CJK_SHORT_LINE_CONTENT


def summarize_line_timing(lines: list[SubtitleLine]) -> dict[str, float | int]:
    total_lines = len(lines)
    min_duration = None
    max_duration = 0.0
    max_cps = 0.0
    too_short_lines = 0
    too_long_lines = 0
    high_cps_lines = 0
    overlap_lines = 0
    reversed_lines = 0

    for idx, line in enumerate(lines):
        duration = float(line.end_time) - float(line.start_time)
        profile = _detect_profile_from_words(line.words) if line.words else "cjk"
        cps_base = _line_unit_count(line.text, profile)
        cps = cps_base / duration if duration > 0 else float("inf")
        threshold = _line_cps_threshold(line.text)

        if min_duration is None or duration < min_duration:
            min_duration = duration
        if duration > max_duration:
            max_duration = duration
        if cps > max_cps:
            max_cps = cps
        if duration < _LINE_MIN_DURATION and cps_base > _short_line_content_threshold(profile):
            too_short_lines += 1
        if duration > _LINE_MAX_DURATION:
            too_long_lines += 1
        if cps > threshold:
            high_cps_lines += 1
        if line.end_time + _LINE_OVERLAP_TOLERANCE < line.start_time:
            reversed_lines += 1
        if idx > 0 and line.start_time + _LINE_OVERLAP_TOLERANCE < lines[idx - 1].end_time:
            overlap_lines += 1

    return {
        "total_lines": total_lines,
        "too_short_lines": too_short_lines,
        "too_long_lines": too_long_lines,
        "high_cps_lines": high_cps_lines,
        "overlap_lines": overlap_lines,
        "reversed_lines": reversed_lines,
        "min_line_duration": min_duration if min_duration is not None else 0.0,
        "max_line_duration": max_duration,
        "max_cps": max_cps,
    }


def check_line_timing(lines: list[SubtitleLine]) -> list[CheckError]:
    errors: list[CheckError] = []
    previous_line: SubtitleLine | None = None
    for idx, line in enumerate(lines, start=1):
        duration = float(line.end_time) - float(line.start_time)
        profile = _detect_profile_from_words(line.words) if line.words else "cjk"
        cps_base = _line_unit_count(line.text, profile)
        cps = cps_base / duration if duration > 0 else float("inf")
        cps_threshold = _line_cps_threshold(line.text)

        if line.end_time + _LINE_OVERLAP_TOLERANCE < line.start_time:
            errors.append(
                CheckError(
                    line_idx=idx,
                    checker="line_timing",
                    message=f"Line {idx} 的 end_time 早于 start_time。",
                )
            )
        if duration < _LINE_MIN_DURATION and cps_base > _short_line_content_threshold(profile):
            errors.append(
                CheckError(
                    line_idx=idx,
                    checker="line_timing",
                    message=f"Line {idx} 显示时长过短（{duration:.2f}s < {_LINE_MIN_DURATION:.1f}s）。",
                )
            )
        if duration > _LINE_MAX_DURATION:
            errors.append(
                CheckError(
                    line_idx=idx,
                    checker="line_timing",
                    message=f"Line {idx} 显示时长过长（{duration:.2f}s > {_LINE_MAX_DURATION:.1f}s）。",
                )
            )
        if cps > cps_threshold:
            errors.append(
                CheckError(
                    line_idx=idx,
                    checker="line_cps",
                    message=f"Line {idx} 阅读速度过高（{cps:.1f} > {cps_threshold:.1f} CPS）。",
                )
            )
        if previous_line and line.start_time + _LINE_OVERLAP_TOLERANCE < previous_line.end_time:
            errors.append(
                CheckError(
                    line_idx=idx,
                    checker="line_overlap",
                    message=(
                        f"Line {idx} 与上一行时间重叠（{line.start_time:.2f}s < "
                        f"{previous_line.end_time:.2f}s）。"
                    ),
                )
            )
        previous_line = line
    return errors


def _find_split_point(text: str, max_chars: int) -> int:
    count = 0.0
    for idx, ch in enumerate(text):
        count += _word_cjk_len(ch)
        if count >= max_chars:
            return idx
    return -1


def stage_check(lines: list[SubtitleLine], max_chars: int = 14) -> list[CheckError]:
    errors = check_max_chars(lines, max_chars=max_chars)
    errors.extend(check_line_timing(lines))
    for error in errors:
        error.fix_command = (
            "python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py "
            f'split <lines.json> --line {error.line_idx} --after "..."'
        )
    return errors


def _strip_trailing_punct(lines: list[SubtitleLine]) -> None:
    for line in lines:
        if not line.text:
            continue
        stripped = line.text.rstrip()
        while stripped and stripped[-1] in _TRILING_STRIP:
            stripped = stripped[:-1].rstrip()
        if stripped != line.text:
            line.text = stripped
            if line.words:
                last_word = line.words[-1]
                word_text = last_word.get("text", "")
                while word_text and word_text[-1] in _TRILING_STRIP:
                    word_text = word_text[:-1]
                last_word["text"] = word_text


def split_line_after(lines: list[SubtitleLine], line_idx: int, after_text: str) -> list[SubtitleLine]:
    if line_idx < 1 or line_idx > len(lines):
        raise ValueError(f"Line index {line_idx} out of range (1-{len(lines)})")

    target = lines[line_idx - 1]
    words = target.words
    if not words:
        raise ValueError(f"Line {line_idx} has no word-level data, cannot split")

    full_text = _text_of_words(words)
    count = full_text.count(after_text)
    if count == 0:
        raise ValueError(f'Text "{after_text}" not found in line {line_idx}')
    if count > 1:
        raise ValueError(f'Text "{after_text}" found {count} times in line {line_idx}, must be unique')

    pos = full_text.index(after_text)
    end_pos = pos + len(after_text)
    cum = 0
    split_word = None
    for i, word in enumerate(words):
        word_text = word.get("text", "")
        word_start = cum
        word_end = cum + len(word_text)
        cum = word_end
        if word_end <= end_pos:
            continue
        if word_start >= end_pos:
            split_word = i
            break
        if word_start < end_pos < word_end:
            split_offset = end_pos - word_start
            left_text = word_text[:split_offset]
            right_text = word_text[split_offset:]
            words[i] = dict(word)
            words[i]["text"] = left_text
            right_word = dict(word)
            right_word["text"] = right_text
            words.insert(i + 1, right_word)
            split_word = i + 1
            break

    if split_word is None:
        return lines
    if split_word == 0:
        raise ValueError("Split point is at the start of the line, cannot split")

    before_words = words[:split_word]
    after_words = words[split_word:]
    new_lines = []
    if before_words:
        new_lines.append(
            SubtitleLine(
                text=_text_of_words(before_words),
                start_time=before_words[0].get("start_time", target.start_time),
                end_time=before_words[-1].get("end_time", target.end_time),
                words=[dict(w) for w in before_words],
            )
        )
    if after_words:
        new_lines.append(
            SubtitleLine(
                text=_text_of_words(after_words),
                start_time=after_words[0].get("start_time", target.start_time),
                end_time=after_words[-1].get("end_time", target.end_time),
                words=[dict(w) for w in after_words],
            )
        )
    return lines[: line_idx - 1] + new_lines + lines[line_idx:]


def stage4_render(lines: list[SubtitleLine], output_dir: str, audio_path: str, fmt: str = "srt", ass_style: str = "default"):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    if fmt in ("srt", "all"):
        srt_path = os.path.join(output_dir, f"{base}.srt")
        render_srt_from_lines(lines, srt_path)
        paths["srt"] = srt_path
    if fmt in ("ass", "all"):
        ass_path = os.path.join(output_dir, f"{base}.ass")
        render_ass_from_lines(lines, ass_path, ASSSubtitleStyle.from_name(ass_style))
        paths["ass"] = ass_path
    return paths


def run_pipeline(
    audio_path: str,
    output_dir: Optional[str] = None,
    fmt: str = "srt",
    ass_style: str = "default",
    fix_dir: Optional[str] = None,
    language=None,
    model_size: str = "1.7B",
    max_chars: int = 14,
    timing_mode: str = "stable",
    resume_from: Optional[str] = None,
):
    audio_path = os.path.abspath(audio_path)
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_stages = ["asr", "break", "fix", "render"]
    start_idx = all_stages.index(resume_from) if resume_from in all_stages else 0

    if start_idx <= 0:
        result_dict = stage1_asr(audio_path, output_dir, language, model_size, timing_mode)
    else:
        with open(os.path.join(output_dir, _raw_json_name(audio_path)), "r", encoding="utf-8") as handle:
            result_dict = json.load(handle)

    if start_idx <= 1:
        lines = stage2_break(result_dict, output_dir, audio_path, max_chars)
    else:
        lines = _load_lines(os.path.join(output_dir, _lines_json_name(audio_path)))

    if fix_dir and start_idx <= 2:
        lines = stage3_fix(lines, fix_dir)
        _save_lines(lines, os.path.join(output_dir, _lines_json_name(audio_path)))
    elif fix_dir and start_idx > 2:
        lines = _load_lines(os.path.join(output_dir, _lines_json_name(audio_path)))

    quality_summary = {
        "word_timing": result_dict.get("quality_summary", {}).get("word_timing", {}),
        "line_timing": summarize_line_timing(lines),
    }
    quality_path = os.path.join(output_dir, _quality_json_name(audio_path))
    with open(quality_path, "w", encoding="utf-8") as handle:
        json.dump(quality_summary, handle, ensure_ascii=False, indent=2)

    errors = stage_check(lines, max_chars)
    if errors:
        details = "\n- ".join(error.message for error in errors[:10])
        raise PipelineQualityError(
            "字幕质量检查失败：\n- "
            + details
            + ("\n- 其余错误已省略。" if len(errors) > 10 else "")
        )

    _strip_trailing_punct(lines)
    render_paths = stage4_render(lines, output_dir, audio_path, fmt, ass_style)
    render_paths["quality_json"] = quality_path
    return render_paths


def _raw_json_name(audio_path: str) -> str:
    return f"{os.path.splitext(os.path.basename(audio_path))[0]}.raw.json"


def _lines_json_name(audio_path: str) -> str:
    return f"{os.path.splitext(os.path.basename(audio_path))[0]}.lines.json"


def _quality_json_name(audio_path: str) -> str:
    return f"{os.path.splitext(os.path.basename(audio_path))[0]}.quality.json"


def _save_lines(lines: list[SubtitleLine], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([_line_to_dict(line) for line in lines], handle, ensure_ascii=False, indent=2)


def _load_lines(path: str) -> list[SubtitleLine]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [SubtitleLine(**item) for item in data]


def _line_to_dict(line: SubtitleLine) -> dict:
    return {
        "text": line.text,
        "start_time": line.start_time,
        "end_time": line.end_time,
        "words": line.words,
        "pause_after": line.pause_after,
    }
