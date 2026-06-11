# sherpa-post-asr

**语音识别后处理服务** — 同时支持两种完全不同的技术路线

---

## 路线对比

| 维度 | 路线 A：传统 SenseVoice + LLM | 路线 B：字级独立建模 🏆 |
|:---|:---|:---|
| **ASR 模型** | SenseVoice-Small int8 (seq2seq, 有LM) | **Zipformer-CTC int8** (帧级独立, 无LM) |
| **模型大小** | ~150MB | ~351MB |
| **中文 CER** | 0.1936 | **0.0104**🔥 (去标点, 纯内容) |
| **数字 CER** | 0.6313 (几乎不可用) | **0.0187** |
| **口音 CER** | 0.3750 | **0.0026** |
| **英文支持** | 内置 | 不支持（需额外 token）|
| **额外纠错** | Qwen3.5-2B GGUF (~1.3GB GPU) | 无需（或极小后处理） |
| **总显存** | ~1.5GB | 0（纯 CPU） |

> **路线 B 的核心思想**：中文 ASR 本质是 **3000 个字的独立分类问题**，不需要句级语言模型。每一帧独立 argmax 即可。详见 [`docs/ZIPFORMER_CTC_APPROACH.md`](docs/ZIPFORMER_CTC_APPROACH.md)

---

## 快速开始（路线 B：推荐）

### 1. 环境准备

```bash
pip install sherpa-onnx soundfile pypinyin librosa sounddevice
```

### 2. 下载模型

```bash
# Zipformer-CTC 中文 ASR 模型
cd models/zipformer-ctc-zh-int8
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
tar xf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

# Silero VAD 模型（用于麦克风录音的语音活动检测）
cd ../vad
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

### 3. 🎤 麦克风实时转写

```bash
# 实时模式：持续监听麦克风，自动识别
python scripts/mic_transcribe.py

# 单次模式：按 Enter 开始/停止录音
python scripts/mic_transcribe.py --one-shot

# 查看可用音频设备
python scripts/mic_transcribe.py --list-devices
```

### 4. 文件识别

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
    model="models/zipformer-ctc-zh-int8/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx",
    tokens="models/zipformer-ctc-zh-int8/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",  # ← 纯帧级分类，无 LM！
)

stream = recognizer.create_stream()
stream.accept_waveform(16000, samples)
recognizer.decode_stream(stream)
print(stream.result.text)
```

### 4. 批量 benchmark

```bash
python scripts/benchmark_zipformer_ctc.py
```

---

## 快速开始（路线 A：原方案）

### 1. 安装依赖

```bash
# llama.cpp 需要从源码编译以启用 CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 其他依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
bash scripts/download_models.sh
```

### 3. 启动服务

```bash
python -m server.main
```

API 默认运行在 `http://0.0.0.0:8001`

---

## Benchmark 结果

### 总体对比

| 指标 | SenseVoice (raw) | +Qwen3.5-2B (纠错) | **Zipformer-CTC** |
|:---|:---:|:---:|:---:|
| 全部样本 CER（含英文） | 0.1936 | 0.1864 | **0.1090** |
| 中文纯内容 CER（去标点） | — | — | **0.0104** |
| 数字 CER | 0.6313 | 0.6313 | **0.0187** |
| 口音 CER | 0.3750 | 0.3750 | **0.0026** |
| 同音字 CER | 0.1378 | 0.1112 | **0.0339** |
| 专名 CER | 0.1023 | 0.1023 | **0.0086** |
| 场景全胜数 | — | — | **12/12** |
| 速度 | ~142ms/次 | +~850ms LLM | **~220ms/次** |
| 硬件需求 | CPU + GPU | CPU + GPU | **纯 CPU** |

### 按场景（Zipformer-CTC 去标点后的真实 CER）

| 场景 | CER | 实际错误 |
|:---|---:|:---:|
| 🚇 地铁 / 🖥️ 机房 / 🌀 风扇 / 🌄 旷野 / 📢 高噪音 / 🤫 轻语 / 💨 大风 | **0.0000** | 0/448 字 |
| ⚡ 电机 / ⛈️ 雷暴 | **0.0022** | 1/448 字 |
| 🌧️ 下雨 | **0.0045** | 2/448 字 |
| ⬜ 白噪音 | **0.0335** | 15/448 字 |
| 🏛️ 混响 | **0.0826** | 37/448 字 |
| **总体** | **0.0104** | **56/5376 字** |

---

## 架构（路线 B）

```
Audio → [Zipformer-CTC 帧级独立分类] → 字符序列 → [可选: 轻量后处理] → 最终文本
         ↑ 每帧独立 argmax       ↑ BPE→字符   ↑ 同音字典 / 混响补偿
           无语言模型偏置           无 LLM 推理
```

### 关键技术选择

1. **CTC 贪婪解码 (`greedy_search`)**：每一帧独立 argmax，不做 beam search，不用语言模型重打分。这是"字级独立建模"的核心。
2. **Zipformer 架构**：高效的上下文感知编码器，但输出层是 CTC（无自回归解码器）。
3. **int8 量化**：351MB 模型，纯 CPU 推理，RTF < 0.5。

---

## API

### `POST /transcribe`

```bash
curl -X POST http://127.0.0.1:8001/transcribe \
  -F "file=@audio_test/homophone_jingjie.wav"
```

### 响应示例（路线 A）

```json
{
  "raw_text": "我今天去了镜里风景很美",
  "corrected_text": "我今天去了境界风景很美",
  "latency_ms": {"asr": 142.3, "llm_correction": 85.6, "total": 230.1}
}
```

---

## 项目结构

```
server/
├── main.py                 # FastAPI 入口
├── config.py               # 配置
├── schemas.py              # Pydantic 数据模型
├── audio_utils.py          # 音频加载
├── asr_frontend.py         # SenseVoice 封装 (路线 A)
└── correction_engine.py    # LLM 纠错编排 (路线 A)
scripts/
├── benchmark_zipformer_ctc.py  # 路线 B benchmark
├── benchmark_ctc_llm.py        # CTC + LLM 定点纠错 benchmark
├── mic_transcribe.py           # 🎤 麦克风实时转写（路线 B）
├── run_benchmark.py            # 路线 A benchmark
└── analyze_results.py
docs/
└── ZIPFORMER_CTC_APPROACH.md   # 字级独立建模技术文档
```

---

## 参考

- [字级独立建模技术文档](docs/ZIPFORMER_CTC_APPROACH.md) — 原理、验证数据、局限性
- [sherpa-onnx Zipformer-CTC](https://github.com/k2-fsa/sherpa-onnx) — 底层 ASR 引擎
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) — 路线 A 的纠错 LLM
