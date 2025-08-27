## openlrc 项目文档

### 1. 简介

`openlrc` 是一个功能强大的 Python 工具，旨在自动化和优化音频/视频内容的字幕生成、翻译和后期处理流程。它利用先进的语音识别（ASR）技术和大型语言模型（LLM）的能力，为用户提供高质量、同步且经过优化的字幕文件。

**主要目标和功能：**

*   **自动化字幕生成**: 从音频或视频文件中自动生成时间戳精确的字幕。
*   **智能翻译**: 利用 LLM 将字幕翻译成多种目标语言，并支持上下文感知翻译和词汇表强制使用。
*   **字幕优化**: 对生成的字幕进行多种优化处理，包括合并短句、切割长句、标点符号规范化、重复内容处理等，以提高可读性和符合字幕规范。
*   **多语言支持**: 支持多种源语言和目标语言的转录和翻译。
*   **成本控制**: 内置 LLM API 费用估算和限制功能，帮助用户管理成本。
*   **用户友好界面**: 提供命令行界面（CLI）和基于 Streamlit 的图形用户界面（GUI），方便不同用户群体使用。

**核心能力：**

*   **高质量转录**: 基于 `faster-whisper`，提供高效准确的语音转文本服务。
*   **LLM 驱动翻译**: 集成 OpenAI、Anthropic、Google Gemini 等主流 LLM，实现智能、上下文感知的翻译。
*   **精细化字幕优化**: 提供一系列字幕后处理算法，确保最终字幕的专业性和可读性。
*   **鲁棒性与可恢复性**: 具备重试机制、中间结果保存和断点续传功能，应对 LLM 交互的不稳定性和长时间任务。

**技术栈概览：**

*   **核心语言**: Python
*   **语音识别**: `faster-whisper`
*   **大型语言模型**: OpenAI API (GPT系列), Anthropic API (Claude系列), Google Gemini API
*   **音频处理**: `ffmpeg`, `df.enhance` (DeepFilterNet), `ffmpeg_normalize`
*   **文本处理**: `pysbd`, `spacy`, `tiktoken`, `langcodes`, `lingua`, `zhconv`
*   **数据结构**: `Pydantic`, `dataclasses`
*   **并发处理**: `asyncio`, `concurrent.futures.ThreadPoolExecutor`, `Queue`
*   **用户界面**: `Streamlit`
*   **日志**: `colorlog`

### 2. 安装

本节将指导您如何设置 `openlrc` 项目并安装所有必要的依赖项。

#### 2.1 环境准备

*   **Python**: `openlrc` 需要 Python 3.9 或更高版本。建议使用虚拟环境来管理项目依赖。
    *   推荐使用 `uv` 或 `pip` 进行包管理。

#### 2.2 依赖安装

在项目根目录（`E:\MyProject\openlrc\`）下，您可以通过以下方式安装所有依赖：

1.  **使用 `uv` (推荐)**:
    如果您已安装 `uv`，可以直接同步项目依赖：
    ```bash
    uv sync
    ```
2.  **使用 `pip`**:
    如果您更喜欢 `pip`，可以先安装项目本身，然后安装开发依赖（如果存在 `requirements-dev.txt`）：
    ```bash
    pip install -e .
    # 如果存在开发依赖文件，请运行：
    # pip install -r requirements-dev.txt
    ```
    确保您也安装了 `pytest` 以运行测试：
    ```bash
    pip install pytest
    ```

#### 2.3 FFmpeg 安装

`openlrc` 依赖 `FFmpeg` 进行音频提取和视频字幕合并。请确保您的系统上已安装 `FFmpeg` 并将其添加到系统 PATH 中。

*   **Windows**:
    1.  从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载最新版本。
    2.  解压下载的压缩包到您选择的目录（例如 `C:\ffmpeg`）。
    3.  将 `C:\ffmpeg\bin` 添加到您的系统环境变量 `Path` 中。
*   **macOS**:
    ```bash
    brew install ffmpeg
    ```
*   **Linux**:
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

#### 2.4 API 密钥配置

`openlrc` 使用大型语言模型（LLM）进行翻译和上下文处理。您需要为所选的 LLM 服务提供 API 密钥。这些密钥通常通过环境变量进行配置。

*   **OpenAI (GPT Models)**:
    设置环境变量 `OPENAI_API_KEY`。
    ```bash
    # Windows (Command Prompt)
    set OPENAI_API_KEY="your_openai_api_key_here"

    # Linux/macOS
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
*   **Anthropic (Claude Models)**:
    设置环境变量 `ANTHROPIC_API_KEY`。
    ```bash
    # Windows (Command Prompt)
    set ANTHROPIC_API_KEY="your_anthropic_api_key_here"

    # Linux/macOS
    export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    ```
*   **Google (Gemini Models)**:
    设置环境变量 `GOOGLE_API_KEY`。
    ```bash
    # Windows (Command Prompt)
    set GOOGLE_API_KEY="your_google_api_key_here"

    # Linux/macOS
    export GOOGLE_API_KEY="your_google_api_key_here"
    ```
*   **Microsoft Translator (可选)**:
    如果您计划使用 `MSTranslator`（目前仅为示例实现），则需要设置 `MS_TRANSLATOR_KEY`。
    ```bash
    # Windows (Command Prompt)
    set MS_TRANSLATOR_KEY="your_ms_translator_key_here"

    # Linux/macOS
    export MS_TRANSLATOR_KEY="your_ms_translator_key_here"
    ```

**重要提示**: 建议将这些环境变量添加到您的 shell 配置文件（例如 `.bashrc`, `.zshrc`, `~/.profile` 或 Windows 的系统环境变量）中，以便它们在每次会话开始时自动加载。

### 3. 使用指南

本节将详细介绍如何使用 `openlrc` 的命令行界面（CLI）、图形用户界面（GUI）以及如何将其作为 Python 库集成到您的项目中。

#### 3.1 命令行界面 (CLI)

`openlrc` 的核心功能可以通过命令行进行操作。

**基本用法：**

使用 `openlrc` 命令后跟 `gui` 参数来启动图形界面。

```bash
openlrc gui
```

**注意**: 根据文件结构分析，`openlrc/cli.py` 中 `run_gui()` 函数尝试运行 `openlrc/gui/home.py`。然而，实际的 GUI 路径是 `openlrc/gui_streamlit/home.py`。这可能导致启动失败。如果遇到问题，请检查 `openlrc/cli.py` 中的路径是否正确。

**`LRCer` 参数说明：**

`LRCer` 类是 `openlrc` 的核心，它封装了整个转录、翻译和字幕生成流程。以下是其主要参数：

*   `paths` (Union[str, Path, List[Union[str, Path]]]):
    *   要处理的音频/视频文件路径。可以是单个路径字符串、`Path` 对象，或包含多个路径的列表。
*   `src_lang` (Optional[str]):
    *   源语言代码（例如 `'en'`, `'zh-cn'`, `'ja'`）。如果未指定，系统将尝试自动检测。
*   `target_lang` (str):
    *   目标语言代码。默认为 `'zh-cn'`（简体中文）。
*   `whisper_model` (str):
    *   用于转录的 Whisper 模型名称（例如 `'tiny'`, `'base'`, `'small'`, `'medium'`, `'large-v3'`）。默认为 `'large-v3'`。
*   `compute_type` (str):
    *   Whisper 模型的计算类型（例如 `'float16'`, `'int8'`, `'float32'`）。默认为 `'float16'`。
*   `device` (str):
    *   运行 Whisper 模型的设备（例如 `'cuda'` 用于 GPU，`'cpu'` 用于 CPU）。默认为 `'cuda'`。
*   `chatbot_model` (Union[str, ModelConfig]):
    *   用于翻译的 LLM 模型。可以是模型名称字符串（例如 `'gpt-4.1-nano'`, `'claude-3-5-sonnet-latest'`, `'gemini-1.5-pro'`），或 `ModelConfig` 对象以指定自定义 `base_url` 或 `proxy`。默认为 `'gpt-4.1-nano'`。
    *   您可以使用 `openlrc.list_chatbot_models()` 查看可用模型列表。
*   `fee_limit` (float):
    *   单次翻译调用允许的最大费用（美元）。如果估算费用超过此限制，将抛出异常。默认为 `0.8`。
*   `consumer_thread` (int):
    *   用于并行处理翻译任务的消费者线程数量。默认为 `4`。
*   `asr_options` (Optional[dict]):
    *   Whisper ASR 模型的额外参数。
*   `vad_options` (Optional[dict]):
    *   语音活动检测（VAD）模型的额外参数。
*   `preprocess_options` (Optional[dict]):
    *   音频预处理的额外参数。
*   `proxy` (Optional[str]):
    *   LLM API 请求的代理服务器地址（例如 `'http://127.0.0.1:7890'`）。
*   `base_url_config` (Optional[dict]):
    *   LLM API 的自定义基础 URL 配置。例如 `{'openai': 'https://openai.custom.com/', 'anthropic': 'https://api.another.com'}`。
*   `glossary` (Optional[Union[dict, str, Path]]):
    *   一个词汇表，用于强制特定词语的翻译一致性。可以是字典，或指向 JSON 词汇表文件的路径。
*   `retry_model` (Optional[Union[str, ModelConfig]]):
    *   当主翻译模型失败时，用于重试的备用 LLM 模型。
*   `is_force_glossary_used` (bool):
    *   是否强制在上下文中应用给定的词汇表。默认为 `False`。
*   `skip_trans` (bool):
    *   是否跳过翻译过程，仅进行转录和优化。默认为 `False`。
*   `noise_suppress` (bool):
    *   是否对音频进行噪音抑制预处理。默认为 `False`。
*   `bilingual_sub` (bool):
    *   是否生成双语字幕。默认为 `False`。
*   `clear_temp` (bool):
    *   是否清除所有临时文件（包括从视频中生成的 `.wav` 文件）。默认为 `False`。

**示例：**

由于 `openlrc` 的 CLI 目前主要用于启动 GUI，如果您想使用 `LRCer` 的核心功能，建议将其作为 Python 库使用。

#### 3.2 图形用户界面 (GUI)

`openlrc` 提供了一个基于 Streamlit 的图形用户界面，方便用户进行操作。

**如何启动 GUI：**

在项目根目录运行以下命令：

```bash
openlrc gui
```

成功启动后，您的浏览器将自动打开一个新标签页，显示 `openlrc` 的 GUI 界面。

**GUI 功能概述：**

GUI 界面通常会提供以下功能：

*   **文件选择**: 允许用户选择要处理的音频或视频文件。
*   **语言设置**: 选择源语言和目标语言。
*   **模型选择**: 选择用于转录和翻译的 LLM 模型。
*   **高级选项**: 配置噪音抑制、双语字幕、费用限制等。
*   **进度显示**: 显示转录和翻译的实时进度。
*   **结果下载**: 允许用户下载生成的字幕文件。

#### 3.3 作为库使用 (Using as a Library)

您可以将 `openlrc` 作为 Python 库导入到您的项目中，以编程方式控制字幕生成和翻译流程。

**基本代码示例：**

```python
from openlrc import LRCer
from pathlib import Path

# 假设您的音频/视频文件路径
audio_file = Path("path/to/your/audio.wav")
# 或者 video_file = Path("path/to/your/video.mp4")

# 初始化 LRCer
# 您可以根据需要配置参数
lrcer = LRCer(
    whisper_model='small',  # 使用较小的Whisper模型进行快速测试
    device='cpu',           # 如果没有GPU，请使用cpu
    chatbot_model='gpt-3.5-turbo', # 您的LLM模型
    target_lang='zh-cn',    # 目标语言
    fee_limit=0.5,          # 设置API费用限制
    noise_suppress=True,    # 启用噪音抑制
    bilingual_sub=True,     # 生成双语字幕
    clear_temp=True         # 处理完成后清除临时文件
)

try:
    # 运行转录和翻译流程
    # 返回生成字幕文件的路径列表
    output_subtitle_paths = lrcer.run(audio_file)

    print("字幕生成完成！")
    for p in output_subtitle_paths:
        print(f"生成文件: {p}")

except Exception as e:
    print(f"发生错误: {e}")

```

**更高级的用法：**

您可以根据需要，通过 `LRCer` 的初始化参数和 `run` 方法的参数，精细控制整个流程。例如，您可以：

*   提供自定义的 `asr_options` 或 `vad_options` 来调整转录行为。
*   使用 `ModelConfig` 对象来指定自定义的 LLM API 端点或代理。
*   提供 `glossary` 来强制翻译特定术语。

### 4. 架构设计

`openlrc` 项目采用模块化和分层设计，以实现高效、可扩展和鲁棒的字幕生成和翻译流程。其核心是一个生产者-消费者模式，将转录和翻译任务并行化。

#### 4.1 高层架构

`openlrc` 的高层架构可以概括为以下几个阶段和核心组件：

1.  **输入处理**: 接收音频/视频文件。
2.  **预处理**: 对音频进行降噪和响度归一化，并从视频中提取音频。
3.  **转录 (Producer)**: 将预处理后的音频转录为文本，并进行初步的句子分割。
4.  **翻译与优化 (Consumer)**: 并行地从转录队列中获取数据，进行 LLM 翻译、上下文审查、校对和字幕优化。
5.  **输出**: 生成最终的单语或双语字幕文件（LRC/SRT/JSON）。

**核心组件及其交互：**

*   **`LRCer` (openlrc.py)**:
    *   **角色**: 整个流程的中央协调器。它初始化所有子模块，管理生产者-消费者队列，并编排各个阶段的执行。
    *   **交互**: 调用 `Preprocessor` 进行预处理，`Transcriber` 进行转录，`LLMTranslator` 进行翻译，`SubtitleOptimizer` 进行优化，并使用 `Subtitle` 类进行文件操作。

*   **生产者-消费者模式**: 
    *   **生产者 (`produce_transcriptions` in `LRCer`)**: 负责将音频文件转录为 JSON 格式的文本，并将这些转录文件的路径放入一个 `Queue` 中。这个过程是顺序执行的。
    *   **消费者 (`consume_transcriptions` and `translation_worker` in `LRCer`)**: 启动多个工作线程（`consumer_thread`），这些线程并行地从队列中取出转录文件路径，执行翻译、优化和字幕文件生成。这种并行化显著提高了处理效率。

#### 4.2 模块分析

`openlrc/` 目录下的每个 Python 文件都代表了项目中的一个特定模块，负责一套内聚的功能。

*   **`openlrc.py`**:
    *   **功能**: 包含 `LRCer` 类，是整个字幕生成管道的入口和核心协调器。它整合了所有其他模块，实现了生产者-消费者模式，管理文件流、API 费用和临时文件清理。
    *   **关键类**: `LRCer`。

*   **`agents.py`**:
    *   **功能**: 定义了与 LLM 交互的各种“代理”类，每个代理负责一个特定的 LLM 任务（如分块翻译、上下文审查、校对、翻译评估）。这些代理抽象了 LLM API 的直接调用，专注于业务逻辑。
    *   **关键类**: `Agent` (基类), `ChunkedTranslatorAgent`, `ContextReviewerAgent`, `ProofreaderAgent`, `TranslationEvaluatorAgent`。

*   **`chatbot.py`**:
    *   **功能**: 提供了与不同 LLM API（OpenAI, Anthropic, Google Gemini）交互的统一抽象层。它处理 API 请求、响应解析、费用估算、重试机制和并发调用。
    *   **关键类**: `ChatBot` (基类), `GPTBot`, `ClaudeBot`, `GeminiBot`。

*   **`context.py`**:
    *   **功能**: 定义了用于在翻译流程中传递和管理上下文信息的 Pydantic 数据模型。这包括动态的翻译上下文（如摘要、场景）和静态的翻译任务信息（如标题、词汇表）。
    *   **关键类**: `TranslationContext`, `TranslateInfo`。

*   **`defaults.py`**:
    *   **功能**: 集中存储了项目各组件的默认配置选项，特别是 ASR (Whisper) 和 VAD (语音活动检测) 的参数，以及支持的语言列表。
    *   **关键变量**: `default_asr_options`, `default_vad_options`, `supported_languages`。

*   **`evaluate.py`**:
    *   **功能**: 定义了翻译质量评估的框架。目前主要实现了基于 LLM 的评估器，未来可扩展其他评估方法。
    *   **关键类**: `TranslationEvaluator` (基类), `LLMTranslationEvaluator`, `EmbeddingTranslationEvaluator` (占位符)。

*   **`exceptions.py`**:
    *   **功能**: 定义了项目特有的自定义异常类，用于更精确地指示错误类型，提高错误处理的粒度和可读性。
    *   **关键类**: `SameLanguageException`, `ChatBotException`, `LengthExceedException`, `FfmpegException` 等。

*   **`logger.py`**:
    *   **功能**: 配置了项目全局的日志系统，提供彩色输出，方便调试和监控。
    *   **关键对象**: `logger`。

*   **`models.py`**:
    *   **功能**: 作为 LLM 模型信息的中央注册表。它定义了模型提供商、模型配置和详细的模型信息（如价格、token 限制、上下文窗口），并提供了获取模型信息的方法。
    *   **关键类**: `ModelProvider`, `ModelConfig`, `ModelInfo`, `Models`。

*   **`opt.py`**:
    *   **功能**: 包含 `SubtitleOptimizer` 类，用于对字幕数据进行各种优化处理，以提高可读性、一致性和符合字幕规范。
    *   **关键类**: `SubtitleOptimizer`。

*   **`preprocess.py`**:
    *   **功能**: 提供了音频预处理功能，包括噪音抑制和响度归一化，以提高转录质量。
    *   **关键类**: `Preprocessor`。

*   **`prompter.py`**:
    *   **功能**: 定义了各种“提示词生成器”类，负责为 LLM 构建系统和用户提示词。这些提示词经过精心设计，以引导 LLM 的行为并确保输出符合特定格式。
    *   **关键类**: `Prompter` (基类), `ChunkedTranslatePrompter`, `ContextReviewPrompter`, `ProofreaderPrompter` 等。

*   **`subtitle.py`**:
    *   **功能**: 定义了处理字幕数据的核心数据结构和功能，支持单语和双语字幕，以及 JSON、LRC、SRT 等多种格式的加载和保存。
    *   **关键类**: `Element`, `Subtitle`, `BilingualElement`, `BilingualSubtitle`。

*   **`transcribe.py`**:
    *   **功能**: 负责使用 `faster-whisper` 进行音频转录，并对转录结果进行句子分割，以生成结构化的文本。
    *   **关键类**: `Transcriber`, `TranscriptionInfo`。

*   **`translate.py`**:
    *   **功能**: 实现了主要的翻译逻辑，特别是基于 LLM 的翻译。它处理文本分块、上下文感知翻译、重试机制和中间结果保存。
    *   **关键类**: `Translator` (基类), `LLMTranslator`, `MSTranslator` (占位符)。

*   **`utils.py`**:
    *   **功能**: 包含了项目各模块通用的实用函数集合，涵盖文件处理、音频处理、文本操作、计时、语言检测和 spaCy 模型加载等。
    *   **关键函数/类**: `extract_audio`, `get_file_type`, `Timer`, `parse_timestamp`, `format_timestamp`, `detect_lang`, `spacy_load` 等。

*   **`validators.py`**:
    *   **功能**: 定义了一组验证器类，负责检查 LLM 响应的格式和内容，确保 LLM 输出能够被下游组件可靠地解析和使用。
    *   **关键类**: `BaseValidator` (基类), `ChunkedTranslateValidator`, `AtomicTranslateValidator`, `ProofreaderValidator` 等。

### 5. 测试

`openlrc` 项目拥有一个全面且结构良好的测试套件，用于确保代码的质量、功能的正确性以及在不同场景下的鲁棒性。

#### 5.1 如何运行测试

`openlrc` 的测试是基于 Python 的 `unittest` 框架编写的，并且可以方便地通过 `pytest` 工具运行。

1.  **确保依赖已安装**: 
    在运行测试之前，请确保您已按照 [2. 安装](#2-安装) 中的说明安装了所有项目依赖，包括 `pytest`。

2.  **运行所有测试**: 
    在项目根目录（`E:\MyProject\openlrc\`）下，打开您的终端或命令提示符，然后运行以下命令：

    ```bash
    pytest
    ```
    `pytest` 会自动发现 `tests/` 目录下的所有测试文件并执行它们。

    如果您没有安装 `pytest`，也可以使用 Python 内置的 `unittest` 模块来运行测试：

    ```bash
    python -m unittest discover tests
    ```

3.  **运行特定测试文件**: 
    如果您只想运行某个特定的测试文件，例如 `tests/test_agents.py`：

    ```bash
    pytest tests/test_agents.py
    # 或者
    python -m unittest tests/test_agents.py
    ```

4.  **运行特定测试类或方法**: 
    *   运行 `tests/test_agents.py` 文件中的 `TestTranslatorAgent` 类：
        ```bash
        pytest tests/test_agents.py::TestTranslatorAgent
        # 或者
        python -m unittest tests.test_agents.TestTranslatorAgent
        ```
    *   运行 `TestTranslatorAgent` 类中的 `test_translate_chunk_success` 方法：
        ```bash
        pytest tests/test_agents.py::TestTranslatorAgent::test_translate_chunk_success
        # 或者
        python -m unittest tests.test_agents.TestTranslatorAgent.test_translate_chunk_success
        ```

运行测试后，您将在终端中看到详细的测试结果，包括通过的测试数量、失败的测试数量以及任何错误信息。

#### 5.2 测试覆盖范围概述

`openlrc` 的测试套件覆盖了项目的多个关键方面，确保了核心功能的稳定性和可靠性：

*   **核心工作流**: 涵盖了从音频/视频输入到最终字幕输出的端到端流程，包括转录、翻译和优化。
*   **模块化测试**: 每个主要模块（如 `agents`, `chatbot`, `opt`, `preprocess`, `subtitle`, `transcribe`, `translate`, `utils`, `validators`）都有独立的测试文件，确保了其功能的正确性。
*   **LLM 集成**: 对不同 LLM 提供商（OpenAI, Anthropic, Google Gemini）的 API 交互、响应解析、费用估算和重试机制进行了广泛测试。
*   **文件处理**: 验证了对各种音频、视频和字幕格式（JSON, LRC, SRT）的加载、保存和转换功能，以及对不支持文件类型的错误处理。
*   **数据转换与处理**: 测试了时间戳格式化、文本规范化、token 计数以及复杂的句子分割逻辑。
*   **上下文管理**: 验证了翻译过程中上下文（如摘要、场景、词汇表）的构建和使用。
*   **输出验证**: `validators` 模块的测试确保了 LLM 的输出符合预期的格式和内容要求，提高了下游处理的可靠性。
*   **鲁棒性**: 包含了对各种异常情况（如文件未找到、API 错误、LLM 响应不一致）的测试，确保了系统的健壮性。

这种全面的测试方法有助于在开发过程中及早发现问题，并确保 `openlrc` 能够提供高质量和可靠的字幕生成服务。

### 6. 贡献

`openlrc` 是一个开源项目，我们欢迎社区的贡献！无论您是想报告 Bug、提出新功能建议、改进文档，还是直接贡献代码，您的参与都将帮助 `openlrc` 变得更好。

**如何贡献：**

1.  **报告 Bug**: 
    如果您在使用 `openlrc` 过程中遇到任何问题或发现 Bug，请通过项目的 GitHub 仓库的 Issues 页面提交。请提供详细的复现步骤、错误信息和您的环境信息，以便我们能更快地定位和解决问题。

2.  **提出功能建议**: 
    如果您有任何希望 `openlrc` 实现的新功能或改进现有功能的想法，也请在 Issues 页面提出。详细描述您的想法、用例以及它将如何帮助用户。

3.  **贡献代码**: 
    *   **Fork 仓库**: 首先，将 `openlrc` 的 GitHub 仓库 Fork 到您自己的账户。
    *   **克隆到本地**: 将您 Fork 的仓库克隆到本地开发环境。
    *   **创建分支**: 为您的新功能或 Bug 修复创建一个新的 Git 分支（例如 `feature/your-new-feature` 或 `bugfix/fix-issue-123`）。
    *   **编写代码**: 在您的分支上进行开发。请遵循项目现有的代码风格和规范。
    *   **编写测试**: 为您的新功能或修复编写相应的测试用例，并确保所有现有测试都能通过。
    *   **提交更改**: 提交您的更改，并编写清晰、有意义的提交信息。
    *   **创建 Pull Request (PR)**: 将您的分支推送到您的 Fork 仓库，然后向 `openlrc` 的 `main` 分支创建一个 Pull Request。请在 PR 描述中详细说明您的更改内容、解决了什么问题或实现了什么功能。

**代码风格和规范**: 

*   请尽量遵循项目现有的代码风格（例如，使用 `black` 或 `ruff` 进行格式化）。
*   编写清晰的函数和类文档字符串。
*   确保您的代码是模块化的，并且易于理解和维护。

我们感谢您的时间和精力，并期待您的贡献！

### 7. 常见问题 (FAQ)

本节列出了在使用 `openlrc` 过程中可能遇到的一些常见问题及其解决方案。

#### 7.1 API 费用超限 (`FeeLimitExceededException`)

**问题**: 在运行翻译任务时，收到类似 `Approximated billing fee exceeds the limit` 的错误。

**原因**: 这表示 LLM API 的估算费用超过了您在 `LRCer` 初始化时设置的 `fee_limit` 参数。这是为了防止意外的高额费用。

**解决方案**: 
1.  **增加 `fee_limit`**: 如果您愿意支付更多费用，可以提高 `LRCer` 的 `fee_limit` 参数。
    ```python
    lrcer = LRCer(..., fee_limit=1.0) # 将限制提高到1美元
    ```
2.  **使用更便宜的模型**: 切换到成本更低的 LLM 模型。您可以在 `openlrc.models.Models` 中查看不同模型的 `input_price` 和 `output_price`。例如，从 `gpt-4` 切换到 `gpt-3.5-turbo` 或 `claude-3-haiku`。
    ```python
    lrcer = LRCer(..., chatbot_model='gpt-3.5-turbo')
    ```
3.  **减小 `chunk_size`**: 对于非常长的文本，减小 `LLMTranslator` 的 `chunk_size` 可能会减少每次 API 调用的 token 数量，从而降低单次调用的费用估算。但这可能会增加 API 调用次数。
    ```python
    # 在LLMTranslator初始化时设置，或者在LRCer中传递给LLMTranslator
    # LRCer目前没有直接暴露chunk_size参数，可能需要修改代码或在LLMTranslator中直接设置
    # 例如，如果您直接使用LLMTranslator:
    from openlrc.translate import LLMTranslator
    translator = LLMTranslator(..., chunk_size=15)
    ```

#### 7.2 LLM 响应格式不正确 (`Invalid response format`)

**问题**: 日志中出现 `Invalid response format` 警告，或者翻译结果不符合预期。

**原因**: LLM 的响应未能通过 `openlrc` 内部的验证器检查。这可能是由于 LLM 未能严格遵循提示词中指定的输出格式，或者其生成的内容不符合语言或结构要求。

**解决方案**: 
1.  **重试**: `openlrc` 内置了重试机制。通常，多次重试可能会解决问题。
2.  **检查提示词**: 如果您修改了 `prompter.py` 中的提示词，请确保它们清晰、明确，并包含所有必要的格式指令和示例。
3.  **使用更强大的模型**: 某些 LLM 模型在遵循指令方面表现更好。尝试使用更强大的模型（例如，从 `gpt-3.5-turbo` 升级到 `gpt-4` 系列）。
4.  **调整 `temperature`**: 降低 LLM 的 `temperature` 参数（使其更确定性）可能会改善格式遵循度，但可能会牺牲创造性。
    ```python
    # 在LRCer初始化时，通过chatbot_model的ModelConfig设置
    from openlrc.models import ModelConfig
    lrcer = LRCer(..., chatbot_model=ModelConfig(name='gpt-4', provider='openai', temperature=0.5))
    ```
5.  **检查输入内容**: 确保输入到 LLM 的文本（转录结果）没有异常字符或格式，这可能会干扰 LLM 的理解。

#### 7.3 翻译结果语言不正确 (`Translated language is X, not Y`)

**问题**: 翻译结果的语言与目标语言不符，日志中出现 `Translated language is X, not Y` 警告。

**原因**: LLM 生成的翻译内容被语言检测器识别为非目标语言。这可能是 LLM 的“幻觉”或语言混淆。

**解决方案**: 
1.  **重试**: 再次尝试翻译，LLM 可能会在重试时生成正确的语言。
2.  **使用更强大的模型**: 更强大的 LLM 模型通常在语言遵循方面表现更好。
3.  **明确提示**: 确保提示词中明确指定了目标语言，并且在示例中也使用了目标语言。
4.  **调整 `temperature`**: 降低 `temperature` 可能会减少 LLM 偏离指令的可能性。

#### 7.4 `FFmpeg` 未安装或未添加到 PATH

**问题**: 运行 `openlrc` 时，收到 `ffmpeg is not installed` 或类似错误。

**原因**: `FFmpeg` 是一个外部工具，`openlrc` 依赖它来处理音频/视频文件。它可能未安装，或者已安装但其可执行文件路径未添加到系统的环境变量 `Path` 中。

**解决方案**: 
1.  **安装 `FFmpeg`**: 按照 [2.3 FFmpeg 安装](#23-ffmpeg-安装) 中的说明安装 `FFmpeg`。
2.  **添加到 PATH**: 确保 `FFmpeg` 的 `bin` 目录已添加到您的系统 `Path` 环境变量中。安装后，您应该能够在终端中运行 `ffmpeg -version` 来验证其是否正确安装并可访问。

#### 7.5 `Spacy` 模型缺失 (`Spacy model X missed, downloading`)

**问题**: 首次运行或处理特定语言时，日志中出现 `Spacy model X missed, downloading` 警告，并可能伴随下载过程。

**原因**: `openlrc` 使用 `spaCy` 库进行句子分割和文本相似度计算，这需要下载特定语言的 `spaCy` 模型。如果模型不存在，`openlrc` 会尝试自动下载。

**解决方案**: 
*   **等待下载完成**: 通常，这是正常行为。请耐心等待 `spaCy` 模型下载完成。
*   **手动下载**: 如果自动下载失败或速度慢，您可以尝试手动下载 `spaCy` 模型。例如，对于中文模型：
    ```bash
    python -m spacy download zh_core_web_sm
    ```
    具体模型名称请参考日志中的警告信息。
