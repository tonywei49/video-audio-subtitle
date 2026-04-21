from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ASSSubtitleStyle:
    font: str = "Source Han Sans SC"
    bold: bool = True
    italic: bool = False
    size: int = 50
    color: str = "&H00FFFFFF"
    spacing: float = 0
    border_width: float = 3
    border_color: str = "&H00000000"
    shadow_enabled: bool = True
    shadow_color: str = "&H00000000"
    shadow_offset_x: float = 2
    shadow_offset_y: float = 2
    shadow_blur: float = 0
    shadow_opacity: int = 80
    alignment: int = 8
    margin_v: int = 150
    margin_l: int = 10
    margin_r: int = 10
    highlight_enabled: bool = True
    highlight_mode: str = "fill"
    highlight_color: str = "&H0000D7FF"
    highlight_scale: float = 108
    dim_color: str = "&H00FFFFFF"
    style_name: str = "Karaoke"

    @property
    def has_asymmetric_shadow(self) -> bool:
        return self.shadow_offset_x != self.shadow_offset_y

    def _shadow_alpha_byte(self) -> int:
        return round((100 - self.shadow_opacity) / 100 * 255)

    def _back_colour_hex(self) -> str:
        if not self.shadow_enabled:
            return "&H80000000"
        bgr = self.shadow_color[4:]
        return f"&H{self._shadow_alpha_byte():02X}{bgr}"

    @property
    def primary_colour(self) -> str:
        if not self.highlight_enabled:
            return self.color
        if self.highlight_mode == "fill":
            return self.highlight_color
        return self.dim_color

    @property
    def secondary_colour(self) -> str:
        return self.color

    def to_style_line(self) -> str:
        shadow_depth = self.shadow_offset_x if (self.shadow_enabled and not self.has_asymmetric_shadow) else 0
        return (
            f"Style: {self.style_name},{self.font},{self.size},"
            f"{self.primary_colour},{self.secondary_colour},"
            f"{self.border_color},{self._back_colour_hex()},"
            f"{'-1' if self.bold else '0'},{'-1' if self.italic else '0'},0,0,"
            f"100,100,{self.spacing:.0f},0,"
            f"1,{self.border_width:.0f},{shadow_depth:.0f},"
            f"{self.alignment},{self.margin_l},{self.margin_r},{self.margin_v},1"
        )

    def shadow_tags(self) -> str:
        if not self.shadow_enabled:
            return ""
        parts: list[str] = []
        if self.has_asymmetric_shadow:
            alpha_hex = f"&H{self._shadow_alpha_byte():02X}&"
            parts.append(f"\\4c{self.shadow_color}")
            parts.append(f"\\4a{alpha_hex}")
            parts.append(f"\\xshad{self.shadow_offset_x:.0f}")
            parts.append(f"\\yshad{self.shadow_offset_y:.0f}")
        if self.shadow_blur > 0:
            parts.append(f"\\be{self.shadow_blur:.0f}")
        return "{" + "".join(parts) + "}" if parts else ""

    def to_header(self) -> str:
        return (
            "[Script Info]\n"
            "Title: Karaoke Subtitles\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1920\n"
            "PlayResY: 1080\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"{self.to_style_line()}\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    @classmethod
    def from_name(cls, name: str) -> "ASSSubtitleStyle":
        if name in ("default", "yingshijf", "neon", "warm", "classic", "cyber", "ocean"):
            return cls()
        raise ValueError(f"Unknown style '{name}'. Available: default")


def format_ass_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        (0x4E00 <= codepoint <= 0x9FFF)
        or (0x3040 <= codepoint <= 0x309F)
        or (0x30A0 <= codepoint <= 0x30FF)
        or (0xAC00 <= codepoint <= 0xD7AF)
    )


def _is_punctuation(ch: str) -> bool:
    return ch in '，。、！？；：,.!?;:、）》」』】）)]}"\'”’ '


def _split_punctuation(text: str) -> tuple[list[str], str]:
    core = list(text)
    trail = []
    while core and _is_punctuation(core[-1]):
        trail.insert(0, core.pop())
    return core, "".join(trail)


def build_kf_tags(words: list[dict]) -> str:
    tags: list[str] = []
    for word in words:
        text = word.get("text", "")
        start = word.get("start_time", 0)
        end = word.get("end_time", start)
        core_chars, trail_punct = _split_punctuation(text)
        if not core_chars:
            if tags and trail_punct:
                tags[-1] += trail_punct
            continue
        duration_cs = int((end - start) * 100)
        if duration_cs <= 0:
            duration_cs = 10
        if is_cjk(core_chars[0]):
            char_cs = max(1, duration_cs // len(core_chars))
            for i, ch in enumerate(core_chars):
                suffix = trail_punct if i == len(core_chars) - 1 else ""
                tags.append(f"{{\\kf{char_cs}}}{ch}{suffix}")
        else:
            core_text = "".join(core_chars)
            tags.append(f"{{\\kf{duration_cs}}}{core_text}{trail_punct}")
    return "".join(tags)


def build_highlight_tags(words: list[dict], dialogue_start: float, style: ASSSubtitleStyle) -> str:
    del dialogue_start, style
    return build_kf_tags(words)


def render_srt_from_lines(lines, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    srt_lines = []
    idx = 1
    for line in lines:
        if not line.text:
            continue
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_srt_time(line.start_time)} --> {format_srt_time(line.end_time)}")
        srt_lines.append(line.text)
        srt_lines.append("")
        idx += 1

    output.write_text("\n".join(srt_lines), encoding="utf-8")


def render_ass_from_lines(lines, output_path: str, style: ASSSubtitleStyle) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ass_lines = [style.to_header()]
    shadow_prefix = style.shadow_tags()

    for line in lines:
        if not line.text:
            continue
        words = line.words
        if words:
            text = shadow_prefix + build_highlight_tags(words, line.start_time, style)
        else:
            text = shadow_prefix + line.text
        ass_lines.append(
            f"Dialogue: 0,{format_ass_time(line.start_time)},{format_ass_time(line.end_time)},"
            f"{style.style_name},,0,0,0,,{text}"
        )

    output.write_text("\n".join(ass_lines), encoding="utf-8")

