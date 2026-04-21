---
name: video-audio-subtitle
description: 用于把本地视频、本地音频或 YouTube 链接转换成带时间对齐的字幕文件。当前只正式支持 macOS Apple Silicon；调用时必须先检查依赖，若缺少 ffmpeg、yt-dlp 或当前 skill 自身的 Python 依赖，先明确告知并在用户同意后再安装，然后再执行字幕流程。
---

# Video Audio Subtitle

这个 skill 用来做一条明确链路：

- 本地视频 -> 抽音频 -> 本地 ASR 对齐 -> 生成 `srt/ass`
- 本地音频 -> 本地 ASR 对齐 -> 生成 `srt/ass`
- YouTube 链接 -> 下载音频 -> 本地 ASR 对齐 -> 生成 `srt/ass`
- Bilibili 链接 -> 下载音频 -> 本地 ASR 对齐 -> 生成 `srt/ass`
- TikTok / Douyin 链接 -> 下载音频 -> 本地 ASR 对齐 -> 生成 `srt/ass`
- 本地视频 + 本地 `srt/ass` -> 烧录字幕 -> 输出带字幕 `mp4`

当前约束：

- 只正式支持 `macOS Apple Silicon`
- 不承诺 `Windows` 原生可用
- 不承诺 `Linux` 当前可用
- 不做隐式回退到其他 ASR 引擎
- 缺依赖时不自动安装，必须先征得用户同意
- YouTube 若触发站点风控，必须明确报出，需要用户显式提供 cookies 才能继续
- macOS 长音频当前走 `单次 ASR + segment 时间线分配`，不会再执行原版 `每 30 秒二次 ASR` 的重路径

## 何时使用

满足以下任一情况时使用：

- 用户要给没有字幕的视频补字幕
- 用户给了 `mp3/wav/flac/m4a/aac/ogg`，要转成带时间轴字幕
- 用户给了 `mp4/mov/mkv/avi/webm/m4v`，要先抽音频再转字幕
- 用户给了 YouTube 链接，要直接产出字幕文件
- 用户给了 Bilibili / TikTok / Douyin 链接，要直接产出字幕文件

不适用情况：

- 用户要求 Windows 原生支持
- 用户要求自动改用 Whisper、云 ASR 或其他替代引擎
- 用户要的是翻译字幕或双语字幕，而不是原语言识别

## 固定执行顺序

### 1. 先检查，不要直接跑

先运行：

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py check "<source>"
```

这里的 `<source>` 可以是：

- 本地音频绝对路径
- 本地视频绝对路径
- YouTube / Bilibili / TikTok / Douyin 链接

检查项包括：

- 平台是否为受支持的 `macOS Apple Silicon`
- `uv` 是否存在
- 视频 / YouTube 输入是否需要 `ffmpeg`、`ffprobe`
- YouTube 输入是否需要 `yt-dlp`
- 当前 skill 的 `.venv/bin/python` 是否存在
- 当前 skill 是否需要执行 `uv sync`

对于 `ffmpeg` / `ffprobe`：

- skill 会优先查找 `/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`
- 查不到时才回退到系统 `PATH` 里的普通 `ffmpeg`
- 安装缺失依赖时，默认安装的是 `ffmpeg-full`，不是普通 `ffmpeg`

如果输出里 `missing_tools` 非空，或者 `python_deps_ready=false`，不要直接安装，先把将执行的安装动作告诉用户。

### 2. 只有在用户同意后，才安装

得到用户同意后再运行：

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py install "<source>"
```

当前安装策略：

- `macOS` 通过 `brew install ...` 安装系统工具
- 当前 skill 的 Python 依赖通过 `cd /Users/mac/Documents/skills/video-audio-subtitle && uv sync` 安装

其中媒体工具默认策略是：

- 安装：`brew install ffmpeg-full`
- 运行：优先使用 `ffmpeg-full` 的二进制绝对路径
- 只有在 `ffmpeg-full` 不存在时，才尝试系统默认 `ffmpeg`

如果缺少 `Homebrew`，直接报阻塞，不要偷偷换安装方式。

### 3. 依赖就绪后，再执行字幕流程

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<source>"
```

常用参数：

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<source>" --model-size 0.6B
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<source>" --language English
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<source>" --max-chars 12
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<source>" --install-missing
```

如果是 YouTube 且被站点风控拦截，只能在用户明确同意后追加：

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<youtube-url>" --yt-dlp-cookies-from-browser chrome
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py run "<youtube-url>" --yt-dlp-cookies /absolute/path/to/cookies.txt
```

其他远程站点说明：

- Bilibili / TikTok / Douyin 也走 `yt-dlp`
- 如果站点要求登录态、cookies 或临时风控，必须显式报出来
- 不要偷偷切换到其他第三方下载器

注意：

- `--install-missing` 只能在已经获得用户同意后使用
- 默认模型大小是 `0.6B`，优先降低首次测试成本
- 默认 `--language` 为 `None`，不再强行写死 `Chinese`
- 当 `--max-chars` 不传时，会按语种自动推断：
  - 纯英文：`38`
  - 其他语言：`14`
- `--max-chars` 当前允许 `1-80`
- 默认会先把输入规范化成 `16kHz / mono / wav`
- `--normalize-audio` 会显式启用轻音量归一化
- `--trim-silence` 当前保留为显式参数，但会直接阻断：
  - 原因是它会改变字幕时间基准
  - 现有实现已在真实样本上出现过度裁切
  - 当前版本不会偷偷执行，也不会隐式回退
- `--timing-mode` 默认是 `stable`
- `--timing-mode experimental_segment_align` 只用于显式实验：
  - 只对高风险 segment 尝试 ForcedAligner
  - 如果实验质量检查失败，会直接报错停止
- 输出目录在 `video-audio-subtitle/runs/<timestamp>-<name>/`

### 4. 需要把字幕烧录回视频时，使用 burn

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py burn "<video>" "<subtitle>"
```

常用参数：

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py burn "<video>" "<subtitle>" --output /absolute/path/to/output.mp4
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py burn "<video>" "<subtitle>" --install-missing
```

约束：

- `burn` 只接受本地视频文件
- 字幕文件只支持 `srt` 和 `ass`
- 当前依赖 `ffmpeg` 的字幕滤镜能力
- skill 会优先用 `ffmpeg-full` 做烧录
- 如果最终解析到的 `ffmpeg` 没有 `subtitles` 或 `ass` 滤镜，会直接报阻塞，不做隐式替代

### 5. 需要手工拆分超长字幕时，使用 split

```bash
python3 /Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py split "<lines.json>" --line 3 --after "Hello "
```

约束：

- `split` 只接受 `run` 产出的 `lines.json`
- `--line` 从 `1` 开始计数
- `--after` 必须与原字幕行中的文本片段完全匹配
- 默认直接覆盖原 `lines.json`
- 如果要保留原文件，显式传 `--output /absolute/path/to/new.lines.json`

## 输出内容

每次运行会产出：

- `result.json`
- `output/<name>.srt`
- `output/<name>.ass`
- `output/<name>.raw.json`
- `output/<name>.lines.json`
- `output/<name>.quality.json`

其中：

- `raw.json` 是 ASR + 强制对齐后的中间结果
- `lines.json` 是断句后的中间结果
- `quality.json` 是词级时间轴和行级显示健康摘要
- `srt/ass` 是最终字幕
- 模型缓存会落在 `video-audio-subtitle/cache/models/`，不会每次重新下载
- `burn` 模式会额外产出 `output/<video>.subtitled.mp4`，或写到 `--output` 指定位置

如果词级时间轴或行级显示健康检查失败：

- skill 会直接停止，不继续产出“假正常”的最终字幕
- 失败原因会明确打印出来，例如零时长词比例过高、时间轴倒挂、行显示时长异常
- `raw.json / lines.json / quality.json` 仍会保留，方便排查具体问题

如果导出的 `srt` 出现大量零时长字幕：

- skill 会把它视为时间轴异常
- 不再继续默默使用坏掉的 `srt`
- 会改用 `lines.json` 里的行级 `start_time / end_time` 重建一份正常 `srt`
- 这个修补只修复 `srt` 导出，不会伪装成 ASR 本身没有问题

## 平台边界

### macOS

当前唯一正式支持的平台。要求：

- Apple Silicon
- `Homebrew`
- `uv`

### Windows

当前不支持原生 Windows。若用户明确要求 Windows：

- 直接说明当前 skill 未支持
- 不要宣称“理论上应该可用”
- 建议后续单独做 `WSL2` 版本

### Linux

当前也不在本 skill 的正式支持范围内。后续如果扩展：

- 保留同一主流程
- 只替换平台检查、安装策略和后端依赖说明

## 关键脚本

- 主入口：
  `/Users/mac/Documents/skills/video-audio-subtitle/scripts/media_subtitle.py`
- 单元测试：
  `/Users/mac/Documents/skills/video-audio-subtitle/tests/test_media_subtitle.py`
- 小型手工样例：
  `/Users/mac/Documents/skills/video-audio-subtitle/fixtures/`

说明：

- `runs/`、`cache/`、`examples/`、`archive/` 都是运行时目录
- 干净仓库里不保留这些测试产物；第一次运行时会按需重新生成

## 明确问题点

- 本 skill 是独立的本地 ASR/字幕工具，不再依赖外部 `opc-cli` 目录
- 本 skill 自己补上了视频抽音频与 YouTube 下载这两层
- 本 skill 默认安装并优先使用 `ffmpeg-full`
- 本 skill 新增了 `burn`，但它是否可用仍取决于最终解析到的 `ffmpeg` 是否带字幕滤镜
- 本 skill 新增了 `split`，用于手工修正超长字幕行；它不会偷偷修改原句意，只按你指定的位置拆分
- `Windows` 没有被伪装成已支持
- 依赖缺失不会自动处理，必须先拿到用户同意
- YouTube 下载可能被源站要求登录或 cookies，这不是本 skill 的本地 ASR 能力问题
- Bilibili / TikTok / Douyin 也可能遇到站点风控、cookies 或地区限制；这属于源站下载问题，不是 ASR 问题
