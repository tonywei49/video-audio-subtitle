from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WordTimestamp:
    text: str
    start_time: float
    end_time: float


@dataclass
class ASRResult:
    language: str
    text: str
    duration: float
    words: list[WordTimestamp]


@dataclass
class Paragraph:
    text: str
    start_time: float
    end_time: float
    words: list


@dataclass
class SubtitleLine:
    text: str
    start_time: float
    end_time: float
    words: list = field(default_factory=list)
    pause_after: float = 0.0


@dataclass
class CheckError:
    line_idx: int
    checker: str
    message: str
    fix_command: str = ""

