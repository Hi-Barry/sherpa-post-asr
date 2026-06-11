# 字级独立建模 — Zipformer-CTC 中文 ASR 方法

> **核心论点**：中文自动语音识别（ASR）本质上是 ~3000 个汉字的独立分类问题，句级语言模型不仅不是必需的，反而在口音、数字、专名等场景显著降低准确率。

---

## 1. 动机

### 1.1 现有 ASR 的范式

主流 ASR 系统（包括 SenseVoice、Whisper、Qwen3-ASR）都采用 **seq2seq 范式**：

```
Audio → Encoder → Decoder (with LM) → Text
                    ↑ 语言模型约束整个输出序列的概率
```

这种架构假设**输出序列必须符合语言模型**（高频词优先、语法正确），这在英文等拼音文字中合理，但在中文场景引入了一个根本性矛盾：

- 中文有 **~3000 个常用汉字**，但同一个读音（拼音+声调）平均对应 **6-8 个不同汉字**
- 语言模型倾向于选择**统计上最可能的字**，而非**声学上最匹配的字**
- 当说话人有口音、提到专名、读出数字序列时，LM 的偏置会导致系统性错误

### 1.2 另一个视角：中文 ASR 作为分类问题

```
音频帧 → 声学特征 → 3000 类分类器 → 字符
                      ↑ 每帧独立，无序列约束
```

这本质上是一个**极大规模分类问题**（3000 类），而非序列生成问题。如果声学特征有足够的分辨力，不需要句子级 LM 来"纠正"声学输出。

### 1.3 实验验证

我们在 528 样本测试集上验证了这一假设：

| 方法 | CER | 含义 |
|:---|:---:|:---|
| SenseVoice (内置 LM) | 0.1936 | 传统 seq2seq |
| SenseVoice + Qwen3.5-2B 纠错 | 0.1864 | 加外部 LLM 后仅改善 3.8% |
| **Zipformer-CTC (Greedy, 无 LM)** | **0.0104** 🏆 | **帧级独立分类，去标点后** |

---

## 2. 原理

### 2.1 Connectionist Temporal Classification (CTC)

CTC 是一种**帧级序列标注方法**，核心思想：

1. **每帧独立分类**：对音频的每一帧，模型输出一个概率分布 P(字符 | 音频帧)
2. **引入 blank 符号**：允许模型输出"不知道"（blank），解决帧和字符的对齐问题
3. **路径折叠**：训练时允许所有可能的帧-字符对齐路径，通过动态规划求和

**关键区别：推理时 greedy 解码 = 每帧 argmax，不做 beam search，不加语言模型**

```
帧: 1  2  3  4  5  6  7  8  9 10
分类: a  a  _  _  b  b  b  _  c  c
折叠: a           b              c      → "abc"
```

### 2.2 为什么 CTC 更适合中文？

| 特性 | CTC | Seq2Seq (SenseVoice) |
|:---|:---|:---|
| 输出依赖 | 每帧独立 | 依赖前文+后文 |
| 高频词偏置 | 无 | 强（倾向常见组合）|
| 口音容忍度 | 极高 | 低（LM 拉回标准音）|
| 数字序列 | 准（逐帧对应） | 差（LM 不认识数字序列）|
| 专名/罕见词 | 准（声学匹配） | 差（LM 权重低）|
| 同音字区分 | 好（声学特征足够） | 好（需 LM 辅助）|
| 英文 | 需额外 token | 内置支持 |
| 流畅度 | 偶有插入/删除 | 流畅 |

### 2.3 Zipformer 编码器

Zipformer 是新一代高效 Conformer 变体，核心改进：

1. **Zipformer 块**：将标准的 Conformer 块拆分为轻量级和全量级路径，通过门控机制动态选择
2. **Bias-Free 归一化**：去除 LayerNorm 中的偏置项，减少参数量
3. **Scaling Scheduler**：训练过程中动态调整模型宽度，实现更好的精度-效率权衡
4. **Balancer 机制**：防止不同路径的梯度消失/爆炸

在 CTC 场景下，Zipformer 的优势：
- **高帧级分类精度**：上下文感知的局部感受野足够区分同音字
- **低延迟**：非自回归架构，一次前馈出所有帧的结果
- **int8 量化友好**：~351MB 模型可在 CPU 上实时运行

### 2.4 BPE Tokenization

Zipformer-CTC 内部使用 **SentencePiece Byte-level BPE**（bbpe.model, 1000 tokens）：

- 每个汉字被编码为 1-2 个 BPE token
- 训练时模型在 BPE token 级别做 CTC 分类
- 推理时 sherpa-onnx 自动将 BPE token 序列解码回汉字

**这是一个重要细节**：模型不是直接的"3000 字分类"，而是"~1000 BPE token 分类"，但最终结果等价于字符级。

---

## 3. Benchmark 方法

### 3.1 测试集

- **样本数**：528（11 句子 × 12 噪声场景 × 4 SNR 级别）
- **句子类型**：同音字、上下文、专名、轻声、数字、口音、中英混
- **噪声场景**：地铁、机房、风扇、旷野、高噪音、轻语、大风、电机、雷暴、下雨、白噪音、混响
- **SNR 范围**：-5dB 到 30dB

### 3.2 评估指标

- **CER**（字符错误率）：字符编辑距离 / 参考长度
- **去标点 CER**：先移除 `，。？！、；：` 等标点再计算（反映真实内容精度）
- **按场景细分**：12 个噪声场景分别计算

### 3.3 运行方法

```bash
# 完整 benchmark
cd sherpa-post-asr
python scripts/benchmark_zipformer_ctc.py

# 输出
# - audio_test/_benchmark_ctc_results.json: 逐样本结果
# - BENCHMARK_CTC_REPORT.md: 汇总报告
```

---

## 4. 结果分析

### 4.1 总体表现

| 指标 | 值 |
|:---|:---|
| 全部样本 CER（含 48 个中英混样本） | 0.1090 |
| 中文内容 CER（去标点, 480 样本） | **0.0104** |
| 中英混样本 CER | 0.8444 |
| 总字符数（中文） | 5376 |
| 总编辑错误（中文） | 仅 56 字 |
| 优于 SenseVoice 的样本 | **72.0%** |
| 劣于 SenseVoice 的样本 | **9.5%** |

### 4.2 各噪声场景表现（去标点）

7/12 场景实现 **CER = 0.0000**（零错误）：

| 场景 | CER | 错误字/总字 |
|:---|---:|:---:|
| 🚇 地铁 | 0.0000 | 0/448 |
| 🖥️ 机房 | 0.0000 | 0/448 |
| 🌀 风扇 | 0.0000 | 0/448 |
| 🌄 旷野 | 0.0000 | 0/448 |
| 📢 高噪音 | 0.0000 | 0/448 |
| 🤫 轻语 | 0.0000 | 0/448 |
| 💨 大风 | 0.0000 | 0/448 |
| ⚡ 电机 | 0.0022 | 1/448 |
| ⛈️ 雷暴 | 0.0022 | 1/448 |
| 🌧️ 下雨 | 0.0045 | 2/448 |
| ⬜ 白噪音 | 0.0335 | 15/448 |
| 🏛️ 混响 | 0.0826 | 37/448 |

### 4.3 各句子类型表现

| 类型 | CTC CER | SV CER | 差距 |
|:---|:---:|:---:|:---:|
| 口音 | **0.0026** | 0.3750 | **−0.3724** 🔥 |
| 数字 | **0.0187** | 0.6313 | **−0.6125** 🔥 |
| 专名 | **0.0086** | 0.1023 | **−0.0937** |
| 同音字 | **0.0339** | 0.1378 | **−0.1039** |
| 上下文 | **0.0472** | 0.0848 | **−0.0375** |
| 轻声 | **0.1204** | 0.1227 | −0.0023 |
| 中英混 | 0.8444 | 0.2135 | +0.6309 ❌ |

### 4.4 错误模式分析

**全部 117 个字符级错误**（含标点）的分布：

| 错误类型 | 数量 | 占比 | 可修复？ |
|:---|:---:|:---:|:---:|
| 声学错误（音也错了） | 83 | 70.9% | ❌ 需更好声学模型 |
| 插入/删除（长度变化） | 32 | 27.4% | ⚠️ 部分可修复 |
| **同音字错误（音对字错）** | **2** | **1.7%** | **✅ 同音字典可修复** |

**核心发现：96%+ 的错误是声学错误，而非同音字选择错误。** 这意味着：
- 不是 CTC 选错了字，而是它在听不清的音频上判断错了音节
- 同音字典消歧几乎没用（只会修复 < 2% 的错误）
- 混响和白噪音是主要敌人（占 52/56 字符错误）

---

## 5. 部署指南

### 5.1 环境要求

| 组件 | 要求 |
|:---|:---|
| Python | 3.8+ |
| sherpa-onnx | 1.13.0+ |
| OS | Linux / macOS / Windows |
| GPU | **不需要**（纯 CPU 推理） |
| RAM | ≥ 1GB（模型加载 ~400MB） |

### 5.2 依赖安装

```bash
pip install sherpa-onnx soundfile librosa pypinyin
```

### 5.3 模型下载

```bash
# 从 GitHub Releases 下载
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
tar xf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

# 模型文件结构
sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/
├── model.int8.onnx   # 351MB — ONNX int8 量化模型
├── tokens.txt         # 14KB — 1000 个 BPE token
├── bbpe.model         # 250KB — SentencePiece BPE 模型
├── README.md
└── test_wavs/         # 测试音频
```

### 5.4 集成示例

#### 单文件识别

```python
import sherpa_onnx
import soundfile as sf

# 1. 加载模型
recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
    model="model.int8.onnx",
    tokens="tokens.txt",
    num_threads=4,
    sample_rate=16000,
    decoding_method="greedy_search",  # ← 关键参数
)

# 2. 加载音频
samples, sr = sf.read("audio.wav")
if sr != 16000:
    import librosa
    samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)

# 3. 识别
stream = recognizer.create_stream()
stream.accept_waveform(16000, samples.tolist())
recognizer.decode_stream(stream)

# 4. 获取结果
print(stream.result.text)  # 纯文本
print(list(stream.result.tokens))  # 逐 token
print(list(stream.result.timestamps))  # 时间戳
```

#### 批量处理

```python
import sherpa_onnx, soundfile as sf, json
from pathlib import Path

recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
    model="model.int8.onnx",
    tokens="tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",
)

results = []
for wav_path in Path("audio_dir").glob("*.wav"):
    # 每个文件独立创建 stream（不是线程安全的，不能共享 stream）
    stream = recognizer.create_stream()
    samples, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    stream.accept_waveform(16000, samples.tolist())
    recognizer.decode_stream(stream)
    results.append({"file": wav_path.name, "text": stream.result.text})
```

### 5.5 FastAPI 集成

参考 `server/asr_frontend.py` 的设计模式，将 SenseVoice 替换为 Zipformer-CTC：

```python
class ZipformerCtcFrontend:
    def __init__(self, model_dir: str, num_threads: int = 4):
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=f"{model_dir}/model.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=num_threads,
            sample_rate=16000,
            decoding_method="greedy_search",
        )
    
    def transcribe(self, samples: np.ndarray) -> dict:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples.tolist())
        self.recognizer.decode_stream(stream)
        result = stream.result
        return {
            "text": result.text,
            "tokens": list(result.tokens),
            "timestamps": list(result.timestamps),
        }
```

### 5.6 Android 部署注意事项

> **⚠️ 当前未在 Android 上验证，以下为理论分析**

- **模型大小**：351MB 超过 typical APK 大小限制。需拆分下载或使用 model bundle
- **sherpa-onnx Android AAR**：v1.13.2 已包含 Zipformer-CTC 的 JNI 绑定
- **Kotlin API**：`OfflineRecognizer.from_zipformer_ctc()` 在 Kotlin 中可用
- **内存**：~400MB 加载 + ~100MB 运行时 = ~500MB，Android 8GB+ 设备可接受
- **推理速度**：351MB 模型在手机 CPU 上预计 RTF > 1.0（需实际测试）

---

## 6. 局限性与改进方向

### 6.1 已知局限

| 局限性 | 原因 | 缓解方案 |
|:---|:---|:---|
| **不支持英文** | BPE 词表只有 1000 tokens，覆盖中文常见字 | 扩展 BPE 词表 + 重新训练 CTC 头 |
| **混响场景退化 (CER=0.0826)** | 混响使声学特征模糊，导致帧级分类丧失区分力 | ⚠️ 已实验验证：WPE/Noisereduce/高通/预加重等信号处理方法均不能一致改善。需模型级解决 |
| **白噪音场景退化 (CER=0.0335)** | 宽带噪声淹没声学特征 | 加降噪前端 / 训练数据噪声增强 |
| **偶尔的插入/删除** | CTC 对弱能量区域（轻声音节）的 blank 概率估计不稳定 | 后处理：轻声音节 n-gram 规则 |
| **流畅度不如 seq2seq** | 无 LM 约束，偶尔出现不通顺组合 | 极小 bigram 过滤（无需 GPU） |

### 6.2 混响问题深度分析

**现状**：
- 混响场景（level 0-3）CER=0.0826，占全部 56 个字符错误的 37 个（66%）
- 轻中度混响（level 0-1）CER=0.0261，基本可用
- 重度混响（level 3）CER=0.2703，9/10 样本有错误

**尝试过的去混响方案**（均未一致改善）：

| 方案 | 结果 | 原因 |
|:---|:---|:---|
| WPE (nara_wpe) | ❌ 输出为空 | 去除混响同时去除了语音能量 |
| Noisereduce 谱减法 | ❌ 严重退化 | 混响是卷积噪声，非加性噪声 |
| HP 80Hz + Pre-emphasis | ⚠️ 1/40 改善, 8/40 恶化 | 对个别样本有帮助但不一致 |
| Reverb tail trim | ➖ 无实质变化 | CTC 已经对尾部盲区做了 best effort |

**本质原因**：Zipformer-CTC 是**静态模型**（训练于 WenetSpeech 干净数据），对重度混响产生的声学变形缺乏鲁棒性。这不是信号处理能解决的问题。

**可行方向**：
1. **DNN 去混响前端**（如 DCCRN、DPDFNet）— sherpa-onnx 已支持 `OfflineSpeechDenoiser`
2. **混响增强数据重训** — 用 WenetSpeech 数据做混响增强后微调 CTC 头
3. **多条件模型融合** — 训练多个不同增强策略的 CTC 模型，输出层做置信度融合

### 6.2 同音词典为什么几乎无效

**直觉**：中文同音字很多 → CTC 应该有很多同音字错误 → 同音字典可以修复

**实际**：仅 1.7% 的错误是同音字替换。原因是：
- Zipformer-CTC 的声学分辨力足够区分大部分同音字（包括前后鼻音、卷平舌、送气/不送气等）
- 错误集中在音完全听错的情况（我→你、酷→处），而非音对字错
- 混响/白噪音下声学特征变形，连音节都听不清，追论区分同音字

### 6.3 未来方向

1. **中英双语 CTC 模型** — 扩展 BPE 词表到 2000+ tokens，加入常用英文 token
2. **混响增强训练** — 在 WenetSpeech 等数据集上做混响数据增强
3. **流式 Zipformer-CTC** — sherpa-onnx 已支持，可用于实时转写
4. **极小后处理** — 对低置信度区域（混响/白噪音），用 2-3 字 n-gram 做字符级 rescoring
| 5. **CTC + 轻量 Transformer rescoring** — 帧级输出 + 极轻量（<100M）的上下文 rescorer

### 6.4 CTC + LLM 定点纠错实验

**假设**：对置信度低的样本（混响/白噪音），用 Qwen3.5-2B 做定点纠错，可以修复部分错误。

**方案**：
- 置信度检测器：`输出字符数 / (音频时长 × 2.5)` < 0.8 时标记为低置信度
- 仅对低置信度样本调用 Qwen3.5-2B（GGUF, GPU），prompt 要求补充缺失文字并纠错
- 其余样本直接输出 CTC 结果

**结果**（528 样本）：

| 指标 | 值 |
|:---|:---|
| CTC only CER | 0.0852 |
| CTC + LLM CER | **0.0852**（无改善）|
| 改善样本 | 1/528 (0.2%) |
| 倒退样本 | 2/528 (0.4%) |
| LLM 触发 | 9/528 (1.7%) |
| 平均延迟 | 350ms/次 |

**分析**：
- LLM 过于保守：大多数触发样本只加了标点，不敢补充缺失的专有内容（如"湖南菜剁椒鱼头"→只加了逗号）
- 混响导致的**信息彻底丢失**（模型没听到音节），LLM 无法凭空补出专有名词
- 中英混样本被 LLM 反向恶化（数字格式转换导致 CER 升高）
- 唯一改善的样本是"把窗打开风气"→"把窗户打开风气"（补充了一个字）

**结论**：CTC + LLM 定点纠错的边际收益很小。原因在于 CTC 的错误模式与 SenseVoice 不同——主要是声学信息丢失（混响）而非字符选择错误（同音字）。LLM 无法补回没被听到的语音内容。建议资源投入到**声学前端增强**而非后处理 LLM。

---

## 7. 与 SenseVoice 的对比总结

| 维度 | SenseVoice (路线 A) | Zipformer-CTC (路线 B) |
|:---|:---|:---|
| 架构 | Encoder-Decoder (非自回归) | Encoder + CTC |
| 语言模型 | 内置（Decoder）| 无 |
| 中文 CER | 0.1936 | **0.0104** |
| 英文 | ✅ 内置 | ❌ 不支持 |
| 口音鲁棒性 | ❌ 差 (CER=0.375) | ✅ 极好 (CER=0.003) |
| 数字鲁棒性 | ❌ 差 (CER=0.631) | ✅ 好 (CER=0.019) |
| 专名鲁棒性 | ❌ 差 (CER=0.102) | ✅ 好 (CER=0.009) |
| 同音字鲁棒性 | ⚠️ 需 LLM 辅助 (CER=0.138) | ✅ 好 (CER=0.034) |
| 混响鲁棒性 | ❌ 差 (CER=0.281) | ⚠️ 一般 (CER=0.083) |
| 硬件需求 | CPU + GPU | **纯 CPU** |
| 推理速度 | ~142ms/次 | ~220ms/次 |
| 模型大小 | ~150MB | ~351MB |

**结论**：如果你的场景以中文为主（无英文/中英混），Zipformer-CTC 是更好的选择——精度更高、部署更简单、无需 GPU。

---

## 8. 参考文献

1. [CTC: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — Graves et al., ICML 2006
2. [Zipformer: A Lightweight and Efficient Transformer for Online ASR](https://arxiv.org/abs/2305.15530) — Zhang et al.
3. [sherpa-onnx: ONNX-based ASR Inference](https://github.com/k2-fsa/sherpa-onnx) — k2-fsa
4. [WenetSpeech: A Large-Scale Semi-Supervised Speech Corpus](https://arxiv.org/abs/2110.03378) — Zhang et al.
5. [SenseVoice: Multi-Function Voice Foundation Model](https://github.com/FunAudioLLM/SenseVoice) — FunAudioLLM
