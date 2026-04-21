import json
import pathlib
from unittest import mock
import sys
import tempfile
import unittest
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared import config as shared_config
from scripts.shared import model_path as shared_model_path
from scripts.shared import platform as shared_platform
from scripts.asr import engine as standalone_engine
from scripts.asr.pipeline import (
    SubtitleLine as StandaloneSubtitleLine,
    _build_paragraphs as standalone_build_paragraphs,
    _smart_split as standalone_smart_split,
    split_line_after as standalone_split_line_after,
    stage_check as standalone_stage_check,
)
from scripts.asr.render import (
    ASSSubtitleStyle as StandaloneASSSubtitleStyle,
    format_ass_time as standalone_format_ass_time,
    format_srt_time as standalone_format_srt_time,
)
from scripts.media_subtitle import (
    analyze_srt_timing,
    assert_srt_timing_healthy,
    build_srt_entries_from_line_entries,
    build_burn_command,
    build_install_actions,
    build_parser,
    build_translation_output_path,
    build_ytdlp_command,
    classify_source,
    classify_subtitle,
    detect_platform,
    ensure_subtitle_filter_available,
    chunk_srt_entries_for_translation,
    parse_srt,
    resolve_tool_path,
    seconds_to_srt_timestamp,
    split_lines_json,
    split_translated_chunk,
    validate_max_chars,
    write_srt,
)


class ClassifySourceTests(unittest.TestCase):
    def test_classifies_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = pathlib.Path(tmpdir) / "sample.mp3"
            audio_path.write_bytes(b"fake")

            source = classify_source(str(audio_path))

        self.assertEqual(source.kind, "audio_file")

    def test_classifies_video_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / "clip.mp4"
            video_path.write_bytes(b"fake")

            source = classify_source(str(video_path))

        self.assertEqual(source.kind, "video_file")

    def test_classifies_youtube_url(self) -> None:
        source = classify_source("https://www.youtube.com/watch?v=k-71GnH2e0E")
        self.assertEqual(source.kind, "youtube_url")

    def test_classifies_bilibili_url(self) -> None:
        source = classify_source("https://www.bilibili.com/video/BV1xx411c7mD")
        self.assertEqual(source.kind, "bilibili_url")

    def test_classifies_tiktok_url(self) -> None:
        source = classify_source("https://www.tiktok.com/@user/video/1234567890")
        self.assertEqual(source.kind, "tiktok_url")

    def test_classifies_douyin_url(self) -> None:
        source = classify_source("https://www.douyin.com/video/1234567890")
        self.assertEqual(source.kind, "douyin_url")

    def test_rejects_missing_local_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            classify_source("/tmp/does-not-exist.wav")


class InstallActionTests(unittest.TestCase):
    def test_audio_input_does_not_require_ffmpeg_or_ytdlp(self) -> None:
        actions = build_install_actions(
            missing_tools=["uv"],
            python_deps_ready=False,
            source_kind="audio_file",
            platform_name="macos",
            brew_available=True,
        )

        self.assertEqual(
            actions,
            [
                "brew install uv",
                "在当前 skill 目录执行 uv sync 安装 Python 依赖",
            ],
        )

    def test_video_input_requires_ffmpeg(self) -> None:
        actions = build_install_actions(
            missing_tools=["ffmpeg", "ffprobe"],
            python_deps_ready=True,
            source_kind="video_file",
            platform_name="macos",
            brew_available=True,
        )

        self.assertEqual(actions, ["brew install ffmpeg-full"])

    def test_youtube_input_requires_ytdlp(self) -> None:
        actions = build_install_actions(
            missing_tools=["ffmpeg", "ffprobe", "yt-dlp"],
            python_deps_ready=True,
            source_kind="youtube_url",
            platform_name="macos",
            brew_available=True,
        )

        self.assertEqual(actions, ["brew install ffmpeg-full yt-dlp"])

    def test_bilibili_input_requires_ytdlp(self) -> None:
        actions = build_install_actions(
            missing_tools=["ffmpeg", "ffprobe", "yt-dlp"],
            python_deps_ready=True,
            source_kind="bilibili_url",
            platform_name="macos",
            brew_available=True,
        )

        self.assertEqual(actions, ["brew install ffmpeg-full yt-dlp"])

    def test_tiktok_input_requires_ytdlp(self) -> None:
        actions = build_install_actions(
            missing_tools=["ffmpeg", "ffprobe", "yt-dlp"],
            python_deps_ready=True,
            source_kind="tiktok_url",
            platform_name="macos",
            brew_available=True,
        )

        self.assertEqual(actions, ["brew install ffmpeg-full yt-dlp"])


class SubtitleTests(unittest.TestCase):
    def test_classifies_srt_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path = pathlib.Path(tmpdir) / "sample.srt"
            subtitle_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

            subtitle = classify_subtitle(str(subtitle_path))

        self.assertEqual(subtitle.kind, "srt")

    def test_classifies_ass_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path = pathlib.Path(tmpdir) / "sample.ass"
            subtitle_path.write_text("[Script Info]\n", encoding="utf-8")

            subtitle = classify_subtitle(str(subtitle_path))

        self.assertEqual(subtitle.kind, "ass")


class YtDlpCommandTests(unittest.TestCase):
    def test_builds_plain_command_without_cookies(self) -> None:
        command = build_ytdlp_command(
            source_url="https://www.youtube.com/watch?v=abc",
            output_template="/tmp/source.%(ext)s",
            cookies_file=None,
            cookies_from_browser=None,
            extract_audio=True,
        )

        self.assertEqual(command[0], "yt-dlp")
        self.assertIn("-x", command)
        self.assertIn("--audio-format", command)
        self.assertIn("wav", command)
        self.assertIn("-o", command)
        self.assertIn("/tmp/source.%(ext)s", command)
        self.assertIn("--ffmpeg-location", command)
        self.assertIn("https://www.youtube.com/watch?v=abc", command)

    def test_builds_command_with_browser_cookies(self) -> None:
        command = build_ytdlp_command(
            source_url="https://www.youtube.com/watch?v=abc",
            output_template="/tmp/source.%(ext)s",
            cookies_file=None,
            cookies_from_browser="chrome",
            extract_audio=True,
        )

        self.assertIn("--cookies-from-browser", command)
        self.assertIn("chrome", command)

    def test_builds_video_download_command_when_extract_audio_disabled(self) -> None:
        command = build_ytdlp_command(
            source_url="https://www.youtube.com/watch?v=abc",
            output_template="/tmp/source.%(ext)s",
            cookies_file=None,
            cookies_from_browser=None,
            extract_audio=False,
        )

        self.assertNotIn("-x", command)
        self.assertNotIn("--audio-format", command)
        self.assertIn("--merge-output-format", command)
        self.assertIn("mp4", command)


class SrtTranslationTests(unittest.TestCase):
    def test_parse_srt_reads_blocks(self) -> None:
        subtitle = parse_srt(
            "1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n2\n00:00:01,000 --> 00:00:02,000\nSecond line\n"
        )

        self.assertEqual(len(subtitle), 2)
        self.assertEqual(subtitle[0]["text"], "Hello world")
        self.assertEqual(subtitle[1]["end"], "00:00:02,000")

    def test_write_srt_preserves_timing(self) -> None:
        text = write_srt(
            [
                {
                    "index": 1,
                    "start": "00:00:00,000",
                    "end": "00:00:01,000",
                    "text": "你好，世界",
                }
            ]
        )

        self.assertIn("00:00:00,000 --> 00:00:01,000", text)
        self.assertIn("你好，世界", text)

    def test_build_translation_output_path_adds_language_suffix(self) -> None:
        path = build_translation_output_path("/tmp/source.srt", "zh-CN")
        self.assertEqual(path, pathlib.Path("/tmp/source.zh-CN.srt"))

    def test_chunk_translation_groups_entries(self) -> None:
        entries = [
            {"index": 1, "start": "00:00:00,000", "end": "00:00:01,000", "text": "Hello"},
            {"index": 2, "start": "00:00:01,000", "end": "00:00:02,000", "text": "World"},
        ]

        chunks = chunk_srt_entries_for_translation(entries, max_chars=64)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0]["index"], 1)
        self.assertEqual(chunks[0][1]["text"], "World")

    def test_split_translated_chunk_uses_markers(self) -> None:
        translated = "你好\n__CODEx_SEG_1__\n世界"

        parts = split_translated_chunk(translated, marker="__CODEx_SEG_1__")

        self.assertEqual(parts, ["你好", "世界"])


class SrtTimingTests(unittest.TestCase):
    def test_analyze_srt_timing_counts_zero_duration_and_large_gaps(self) -> None:
        stats = analyze_srt_timing(
            [
                {"index": 1, "start": "00:00:00,000", "end": "00:00:01,000", "text": "A"},
                {"index": 2, "start": "00:00:10,000", "end": "00:00:10,000", "text": "B"},
                {"index": 3, "start": "00:00:12,000", "end": "00:00:13,000", "text": "C"},
            ]
        )

        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["zero_duration"], 1)
        self.assertEqual(stats["gaps_over_5s"], 1)

    def test_assert_srt_timing_healthy_rejects_many_zero_duration_entries(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_srt_timing_healthy(
                [
                    {"index": 1, "start": "00:00:00,000", "end": "00:00:00,000", "text": "A"},
                    {"index": 2, "start": "00:00:01,000", "end": "00:00:01,000", "text": "B"},
                    {"index": 3, "start": "00:00:02,000", "end": "00:00:02,000", "text": "C"},
                    {"index": 4, "start": "00:00:03,000", "end": "00:00:03,000", "text": "D"},
                ]
            )

    def test_seconds_to_srt_timestamp_formats_fractional_seconds(self) -> None:
        self.assertEqual(seconds_to_srt_timestamp(142.72), "00:02:22,720")

    def test_build_srt_entries_from_line_entries_uses_line_timing(self) -> None:
        entries = build_srt_entries_from_line_entries(
            [
                {"text": "Hello", "start_time": 0.16, "end_time": 2.32},
                {"text": "World", "start_time": 2.50, "end_time": 3.10},
            ]
        )

        self.assertEqual(entries[0]["start"], "00:00:00,160")
        self.assertEqual(entries[0]["end"], "00:00:02,320")
        self.assertEqual(entries[1]["text"], "World")


class BurnCommandTests(unittest.TestCase):
    def test_builds_srt_burn_command(self) -> None:
        command = build_burn_command(
            ffmpeg_bin="/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
            video_path="/tmp/input.mp4",
            subtitle_path="/tmp/subtitle.srt",
            output_path="/tmp/output.mp4",
        )

        self.assertEqual(command[0:4], ["/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", "-y", "-i", "/tmp/input.mp4"])
        self.assertIn("-vf", command)
        self.assertIn("subtitles=filename='/tmp/subtitle.srt'", command)

    def test_builds_ass_burn_command(self) -> None:
        command = build_burn_command(
            ffmpeg_bin="/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
            video_path="/tmp/input.mp4",
            subtitle_path="/tmp/subtitle.ass",
            output_path="/tmp/output.mp4",
        )

        self.assertIn("ass=filename='/tmp/subtitle.ass'", command)

    def test_reports_missing_subtitle_filter(self) -> None:
        with self.assertRaises(RuntimeError):
            ensure_subtitle_filter_available("srt", " ... no matching filters ... ")


class MaxCharsTests(unittest.TestCase):
    def test_validate_max_chars_accepts_default_range(self) -> None:
        self.assertEqual(validate_max_chars(14), 14)
        self.assertEqual(validate_max_chars(38), 38)
        self.assertEqual(validate_max_chars(80), 80)

    def test_validate_max_chars_rejects_over_limit(self) -> None:
        with self.assertRaises(ValueError):
            validate_max_chars(81)


class DetectPlatformTests(unittest.TestCase):
    def test_windows_is_reported_as_unsupported(self) -> None:
        info = detect_platform(system="Windows", machine="AMD64")
        self.assertFalse(info.supported)
        self.assertEqual(info.platform_name, "windows")

    def test_macos_arm_is_supported(self) -> None:
        info = detect_platform(system="Darwin", machine="arm64")
        self.assertTrue(info.supported)
        self.assertEqual(info.platform_name, "macos")


class ToolResolutionTests(unittest.TestCase):
    def test_prefers_ffmpeg_full_binary(self) -> None:
        def fake_exists(path_obj: pathlib.Path) -> bool:
            return str(path_obj) == "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"

        with mock.patch("pathlib.Path.exists", autospec=True, side_effect=fake_exists), mock.patch(
            "scripts.media_subtitle.shutil.which",
            return_value="/opt/homebrew/bin/ffmpeg",
        ):
            self.assertEqual(resolve_tool_path("ffmpeg"), "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")

    def test_uses_system_binary_when_full_binary_missing(self) -> None:
        with mock.patch("scripts.media_subtitle.pathlib.Path.exists", return_value=False), mock.patch(
            "scripts.media_subtitle.shutil.which",
            return_value="/opt/homebrew/bin/ffmpeg",
        ):
            self.assertEqual(resolve_tool_path("ffmpeg"), "/opt/homebrew/bin/ffmpeg")


class SharedConfigTests(unittest.TestCase):
    def test_loads_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"VIDEO_AUDIO_SUBTITLE_CONFIG_DIR": tmpdir},
            clear=False,
        ):
            config = shared_config.load_config()

        self.assertEqual(config["asr_model_size"], "0.6B")
        self.assertEqual(config["model_source"], "modelscope")

    def test_save_and_reload_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"VIDEO_AUDIO_SUBTITLE_CONFIG_DIR": tmpdir},
            clear=False,
        ):
            shared_config.save_config("backend", "mlx")
            config = shared_config.load_config()

        self.assertEqual(config["backend"], "mlx")


class SharedPlatformTests(unittest.TestCase):
    def test_auto_detects_mlx_on_macos(self) -> None:
        with mock.patch("platform.system", return_value="Darwin"):
            self.assertEqual(shared_platform._auto_detect_backend(), "mlx")

    def test_get_backend_uses_config_override(self) -> None:
        with mock.patch("scripts.shared.config.load_config", return_value={"backend": "cuda"}):
            self.assertEqual(shared_platform.get_backend(), "cuda")


class SharedModelPathTests(unittest.TestCase):
    def test_prefers_local_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(shared_model_path.resolve_model_path(tmpdir), tmpdir)

    def test_modelscope_cache_path_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "scripts.shared.model_path.load_config",
            return_value={"model_source": "modelscope", "model_cache_dir": tmpdir},
        ):
            model_dir = pathlib.Path(tmpdir) / "models" / "Qwen" / "Qwen3-ASR-0___6B"
            model_dir.mkdir(parents=True)
            self.assertTrue(shared_model_path.check_model_exists("Qwen/Qwen3-ASR-0.6B"))

    def test_huggingface_cache_path_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "scripts.shared.model_path.load_config",
            return_value={"model_source": "huggingface", "model_cache_dir": tmpdir},
        ):
            model_dir = pathlib.Path(tmpdir) / "hub" / "models--mlx-community--Qwen3-ASR-0.6B-8bit"
            model_dir.mkdir(parents=True)
            self.assertTrue(shared_model_path.check_model_exists("mlx-community/Qwen3-ASR-0.6B-8bit"))


class StandaloneAsrPipelineTests(unittest.TestCase):
    def test_builds_paragraphs_by_sentence_end(self) -> None:
        paragraphs = standalone_build_paragraphs(
            [
                {"text": "Hello", "start_time": 0.0, "end_time": 0.5},
                {"text": " world.", "start_time": 0.5, "end_time": 1.0},
                {"text": "Next", "start_time": 1.2, "end_time": 1.6},
            ]
        )

        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0].text, "Hello world.")

    def test_smart_split_breaks_long_segment(self) -> None:
        lines = standalone_smart_split(
            [
                {"text": "This ", "start_time": 0.0, "end_time": 0.2},
                {"text": "is ", "start_time": 0.2, "end_time": 0.4},
                {"text": "a ", "start_time": 0.4, "end_time": 0.6},
                {"text": "long ", "start_time": 0.9, "end_time": 1.2},
                {"text": "line", "start_time": 1.2, "end_time": 1.5},
            ],
            max_chars=3,
        )

        self.assertGreaterEqual(len(lines), 2)

    def test_split_line_after_splits_unique_text(self) -> None:
        lines = [
            StandaloneSubtitleLine(
                text="Hello world",
                start_time=0.0,
                end_time=1.0,
                words=[
                    {"text": "Hello ", "start_time": 0.0, "end_time": 0.4},
                    {"text": "world", "start_time": 0.4, "end_time": 1.0},
                ],
            )
        ]

        result = standalone_split_line_after(lines, 1, "Hello ")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Hello ")
        self.assertEqual(result[1].text, "world")

    def test_stage_check_reports_long_line(self) -> None:
        errors = standalone_stage_check(
            [StandaloneSubtitleLine(text="這是一個非常非常長的字幕行", start_time=0.0, end_time=1.0)],
            max_chars=4,
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].checker, "max_chars")
        self.assertIn("media_subtitle.py split", errors[0].fix_command)


class StandaloneAsrRenderTests(unittest.TestCase):
    def test_formats_ass_time(self) -> None:
        self.assertEqual(standalone_format_ass_time(65.43), "0:01:05.43")

    def test_formats_srt_time(self) -> None:
        self.assertEqual(standalone_format_srt_time(65.432), "00:01:05,432")

    def test_ass_style_lookup(self) -> None:
        style = StandaloneASSSubtitleStyle.from_name("default")
        self.assertIn("Style:", style.to_header())


class StandaloneAsrEngineTests(unittest.TestCase):
    def test_mlx_align_matches_original_two_stage_flow(self) -> None:
        class FakeAsrResult:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeAlignItem:
            def __init__(self, text: str, start_time: float, end_time: float) -> None:
                self.text = text
                self.start_time = start_time
                self.end_time = end_time

        class FakeAsrModel:
            def __init__(self) -> None:
                self.calls = []

            def generate(self, audio, language=None, verbose=False):
                self.calls.append(audio)
                return FakeAsrResult("Hello ")

        class FakeAlignerModel:
            def generate(self, audio, text, language=None):
                return [FakeAlignItem(text, 0.0, 1.0)]

        wav = [0.0] * (31 * 16000)  # 31 秒，分成两块
        asr_model = FakeAsrModel()
        result = standalone_engine._asr_align_mlx(
            asr_model,
            FakeAlignerModel(),
            "/tmp/sample.wav",
            wav,
            16000,
            31.0,
            "English",
        )

        self.assertEqual(len(asr_model.calls), 3)
        self.assertIsInstance(asr_model.calls[0], str)
        self.assertTrue(all(not isinstance(call, str) for call in asr_model.calls[1:]))
        self.assertEqual(result.text, "Hello Hello ")

    def test_segment_text_to_words_preserves_text(self) -> None:
        words = standalone_engine._segment_text_to_words("Hello world. Next line.", 0.0, 4.0)
        self.assertEqual("".join(word.text for word in words), "Hello world. Next line.")
        self.assertEqual(words[0].start_time, 0.0)
        self.assertEqual(words[-1].end_time, 4.0)

    def test_segment_text_to_words_splits_pure_cjk_runs(self) -> None:
        words = standalone_engine._segment_text_to_words("哈哈哈哈", 0.0, 4.0)
        self.assertEqual("".join(word.text for word in words), "哈哈哈哈")
        self.assertEqual(len(words), 4)
        self.assertTrue(all(word.end_time > word.start_time for word in words))

    def test_segments_to_word_timestamps_expands_multiple_segments(self) -> None:
        words = standalone_engine._segments_to_word_timestamps(
            [
                {"text": "Hello world. ", "start": 0.0, "end": 2.0},
                {"text": "Second line.", "start": 2.0, "end": 4.0},
            ]
        )
        self.assertEqual("".join(word.text for word in words), "Hello world. Second line.")
        self.assertEqual(words[0].start_time, 0.0)
        self.assertEqual(words[-1].end_time, 4.0)

    def test_mlx_long_audio_path_aligns_segments_without_second_asr(self) -> None:
        class FakeResult:
            def __init__(self) -> None:
                self.text = "Hello world. Second line."
                self.segments = [
                    {"text": "Hello world. ", "start": 0.0, "end": 2.0},
                    {"text": "Second line.", "start": 2.0, "end": 4.0},
                ]

        class FakeAsrModel:
            def __init__(self) -> None:
                self.calls = []

            def generate(self, audio, **kwargs):
                self.calls.append((audio, kwargs))
                return FakeResult()

        class FakeAlignItem:
            def __init__(self, text: str, start_time: float, end_time: float) -> None:
                self.text = text
                self.start_time = start_time
                self.end_time = end_time

        class FakeAlignerModel:
            def __init__(self) -> None:
                self.calls = []

            def generate(self, audio, text, language=None):
                self.calls.append((audio, text, language))
                if text.startswith("Hello"):
                    return [
                        FakeAlignItem("Hello", 0.0, 0.8),
                        FakeAlignItem("world", 0.8, 1.8),
                    ]
                return [
                    FakeAlignItem("Second", 0.0, 0.7),
                    FakeAlignItem("line", 0.7, 1.5),
                ]

        fake_asr = FakeAsrModel()
        fake_aligner = FakeAlignerModel()

        with mock.patch("scripts.asr.engine.get_backend", return_value="mlx"), mock.patch(
            "scripts.asr.engine.get_asr_model",
            return_value={"asr": fake_asr, "aligner": fake_aligner},
        ) as get_model_mock, mock.patch(
            "scripts.asr.engine.load_audio",
            return_value=([0.0] * int(301 * 16000), 16000),
        ):
            result = standalone_engine.asr_align("/tmp/sample.wav", language="English", model_size="0.6B")

        self.assertTrue(result.words)
        self.assertEqual(result.text, "Hello world. Second line.")
        self.assertEqual(len(fake_asr.calls), 1)
        self.assertEqual(len(fake_aligner.calls), 2)
        self.assertEqual("".join(word.text for word in result.words), "Hello world. Second line.")
        get_model_mock.assert_called_once_with("mlx-community/Qwen3-ASR-0.6B-8bit", with_aligner=True)


class ParserTests(unittest.TestCase):
    def test_run_parser_supports_max_chars(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["run", "input.mp3", "--max-chars", "16"])

        self.assertEqual(args.max_chars, 16)

    def test_run_parser_language_defaults_to_none(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["run", "input.mp3"])

        self.assertIsNone(args.language)

    def test_run_parser_supports_keep_video(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["run", "https://www.youtube.com/watch?v=abc", "--keep-video"])

        self.assertTrue(args.keep_video)

    def test_burn_parser_accepts_video_and_subtitle(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["burn", "video.mp4", "subtitle.srt"])

        self.assertEqual(args.video, "video.mp4")
        self.assertEqual(args.subtitle, "subtitle.srt")

    def test_translate_parser_accepts_input_and_target_language(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["translate", "subtitle.srt", "--target-language", "zh-CN"])

        self.assertEqual(args.subtitle, "subtitle.srt")
        self.assertEqual(args.target_language, "zh-CN")

    def test_split_parser_accepts_lines_json_and_position(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["split", "sample.lines.json", "--line", "3", "--after", "Hello "])

        self.assertEqual(args.lines_json, "sample.lines.json")
        self.assertEqual(args.line, 3)
        self.assertEqual(args.after, "Hello ")


class SplitLinesJsonTests(unittest.TestCase):
    def test_split_lines_json_overwrites_input_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lines_path = pathlib.Path(tmpdir) / "sample.lines.json"
            lines_path.write_text(
                '[{"text":"Hello world","start_time":0.0,"end_time":1.0,"words":[{"text":"Hello ","start_time":0.0,"end_time":0.4},{"text":"world","start_time":0.4,"end_time":1.0}],"pause_after":0.0}]',
                encoding="utf-8",
            )

            result = split_lines_json(str(lines_path), line=1, after="Hello ")

            updated = json.loads(lines_path.read_text(encoding="utf-8"))

        self.assertEqual(pathlib.Path(result["output_lines_json"]).resolve(), lines_path.resolve())
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0]["text"], "Hello ")
        self.assertEqual(updated[1]["text"], "world")


if __name__ == "__main__":
    unittest.main()
