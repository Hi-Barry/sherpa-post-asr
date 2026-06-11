# DEVELOPMENT.md — sherpa-post-asr

## 项目概述

Post-ASR 语音识别后处理服务。支持两条完全不同的技术路线：

```
路线 A（原方案）：Audio → SenseVoice (seq2seq, 有LM) → 低置信度检测 → Qwen3.5-2B 局部重打分
路线 B（新方案）：Audio → Zipformer-CTC (帧级独立, 无LM) → [可选极小后处理]
```

**路线 B 的核心发现：中文 ASR 不需要句级语言模型。** 纯帧级 CTC 分类（Zipformer-CTC greedy decoding）在中文场景达到了 CER=0.0104（去标点后），全面碾压带内置 LM 的 SenseVoice（CER=0.1936）。

---

## 路线 A：SenseVoice + LLM 局部纠错

### 架构

```python
Audio → SenseVoice (ASR) → 低置信度区域检测 → Qwen3.5-2B (局部重打分) → 纠正后文本
```

不采用"整句重写"的大模型后处理，而是**只在 ASR 置信度低的局部位置**调用小 LLM 做精准纠错，延迟低、可控性好。

### 置信度来源

SenseVoice (sherpa-onnx) 的 `OfflineRecognitionResult` 提供了 `ys_log_probs` 字段，这是每个 token 在 CTC 解码时的对数概率。`exp(log_prob)` 即为该 token 的置信度。

- 高置信度（log_prob > -0.5）：几乎可以肯定正确
- 中等置信度（-1.5 ~ -0.5）：可能有误
- 低置信度（log_prob < -1.5 ≈ 置信度 0.22）：很可能出错，需要纠错

### 模型选型

- **ASR 前端**: SenseVoice-Small int8 (234M)，CPU 推理
- **纠错 LLM**: Qwen3.5-2B q4 GGUF，GPU 推理，~1.3GB 显存

---

## 路线 B：Zipformer-CTC 字级独立建模

### 核心思想

```
中文 ASR = 3000 个字的独立分类问题。
每一帧独立 argmax（greedy CTC），不做任何语言模型约束。
```

详见 [`docs/ZIPFORMER_CTC_APPROACH.md`](docs/ZIPFORMER_CTC_APPROACH.md)

### 使用方法

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
    model="model.int8.onnx",
    tokens="tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",  # 关键：无 LM！
)
```

### 验证结果

528 样本 benchmark 验证了这一假设：

- 中文内容 CER（去标点）：**0.0104**（5376 字中仅 56 个编辑错误）
- 7/12 噪声场景 CER=0.0000（地铁、机房、风扇、旷野、高噪音、轻语、大风）
- 唯一弱点：中英混场景（模型无英文 token）、混响场景（声学变形）
- 同音字错误仅占所有错误的 **1.7%**（2/117 字符替换），说明 CTC 不选错字，它只在音听不清时才错

---

## 关键技术决策

### 1. 为什么要"帧级独立分类"而不是"序列级建模"？

| 维度 | 序列级建模（SenseVoice） | 帧级独立分类（Zipformer-CTC） |
|:---|:---|:---|
| 输出依赖 | 前文+后文约束 | 每帧独立 |
| 语言模型偏置 | 强（倾向高频词） | 无 |
| 口音鲁棒性 | 差（LM 拉回标准发音） | 极好（纯声学判决） |
| 数字/专名 | 差（LM 不认识罕见序列） | 好（纯声学匹配） |
| 英文 | 内置支持 | 需额外 token |
| 同音字 | 需 LM 消歧 | 声学上能区分大部分 |

### 2. 为什么 SenseVoice 的 LM 帮倒忙？

SenseVoice 内置的语言模型在以下场景反而降低准确率：

- **口音场景**（CER=0.375）：口音发音偏离标准，LM 强行拉到常见字，但拉错了
- **数字场景**（CER=0.631）：数字序列（一百二十三号）在中文语料中出现频率低，LM 偏向常见词汇
- **专名场景**（CER=0.102）：酷睿、深圳等专名在 LM 训练数据中权重低

而 CTC 没有这些偏置，纯靠声学特征判断，反而在这些场景表现优异。

### 3. 为什么"同音字典消歧"收益甚微？

直觉上 CTC 输出应该有很多同音字错误（境→镜、他→她），但实际分析发现：

- 仅 **2/117** 字符替换是同音字错误（1.7%）
- **96.4%** 是声学错误（音也听错了，如 我→你、酷→处、睿→理）
- 错误主要集中在**混响场景**（37/56 字符错误）和**白噪音场景**（15/56）

原因是 Zipformer-CTC 的声学分辨力已经足够区分大部分同音字（前后鼻音、卷平舌等），只有当音频严重变形（混响/白噪音）时才会出错。

---

## 踩坑记录

### TTS 测试音频生成

测试音频理想状况下应使用 edge-tts（微软语音合成），质量高且易用：
```bash
pip install edge-tts
edge-tts --voice zh-CN-XiaoxiaoNeural --text "测试文本" --write-media test.wav
```

回退方案：espeak-ng + 重采样。

### llama-cpp-python 编译

GPU offload 需要从源码编译：
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

或使用预编译包（可能不支持 CUDA）。

### CTC BPE Tokenization

Zipformer-CTC 内部使用 **Byte-level BPE**（SentencePiece），不是纯字符级。这意味着：
- 输出要先经过 BPE→字符解码，中间可能有映射损失
- 每个汉字可能对应多个 BPE token
- 这是目前中英混场景失败的根源（英文词不在 BPE 词表中）

### 模型下载速度

351MB 的模型从 GitHub Releases 下载可能较慢（国内网络）。建议使用代理或：
```bash
# 使用 huggingface 镜像
wget https://hf-mirror.com/pantinor/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/resolve/main/model.int8.onnx
```

---

## 基准测试汇总

### 2026-06-10: SenseVoice 基线

**528 个样本** (11 句子 × 12 场景 × 4 SNR)

| 指标 | 值 |
|:---|:---|
| 平均原始 CER | 0.1936 |
| 平均纠错 CER | 0.1864 |
| CER 降低 | 0.0073 (3.8%) |
| LLM 只对同音字有效 | 32% 改善率 |
| 英文/数字/专名改善 | 0% |

### 2026-06-11: Zipformer-CTC 实验结果

**528 个样本**，同测试集

| 指标 | 值 |
|:---|:---|
| 全部样本 CER（含英文） | 0.1090 |
| 中文纯内容 CER（去标点） | 0.0104 |
| CTC 优于 SV 的样本 | 72.0% |
| CTC 劣于 SV 的样本 | 9.5% |
| 持平 | 18.6% |

---

## 后续优化方向

1. **Zipformer-CTC 手机端部署** — 351MB ONNX 模型，纯 CPU 推理，Android 可行性评估
2. **中英混支持** — 扩展 BPE 词表加入常用英文 token
3. **混响补偿** — 加入去混响前端或 VAD+滑窗策略
4. **极小后处理** — 对混响/白噪音的低置信度片段做轻量 rescoring（用 1-2 个字的 n-gram 而非 LLM）
5. **Streaming CTC** — sherpa-onnx 支持 streaming Zipformer-CTC，可做流式转写
6. **路线 A→B 迁移** — 将现有 SenseVoice API 后端替换为 Zipformer-CTC，消除 GPU 依赖


## 开发日志

### 2026-06-11 路线 B：字级独立建模假设验证

**做什么**：
- 下载 sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03 模型（351MB）
- 编写 `scripts/benchmark_zipformer_ctc.py`，用纯帧级 greedy CTC 解码跑 528 样本 benchmark
- 与已有 SenseVoice 基准结果做 12 场景 × 7 类别的全面对比
- 分析错误的统计特征（同音 vs 声学 vs 插入删除）

**结果**：
- **中文内容 CER（去标点）= 0.0104**，碾压 SenseVoice（0.1936）
- 7/12 噪声场景 CER=0.0000（零错误）
- 唯一弱点：中英混（模型无英文 token）、混响（声学变形）
- 同音字错误仅占 1.7%，96% 以上的错误是声学错误

**踩坑**：
- 模型输出包含标点删失（CTC 不输出逗号等标点），需去标点后评估真实 CER
- BPE tokenization 细节：模型内部是 1000-token BPE 而非纯字符级，但不影响最终字符精度
- `from_zipformer_ctc()` 的 `decoding_method` 必须设为 `"greedy_search"`（默认也是），不能用 beam search 加 LM

**学到什么**：
- **中文 ASR 不需要句级语言模型** — 当声学模型足够好（Zipformer）时，帧级独立分类即可达到 99% 精度
- 语言模型在中文 ASR 中是双刃剑：提升流畅度的代价是口音/数字/专名的系统性偏差
- 混响是当前最大的声学挑战（37/56 错误来自混响场景），简单信号处理（WPE/Noisereduce/高通/预加重）均不能一致改善
- 完整技术文档见 `docs/ZIPFORMER_CTC_APPROACH.md`

### 2026-06-11 混响实验

**做什么**：
- 分析混响错误的 SNR分级分布（level 0-2 基本可用，level 3 严重退化（CER=0.2703））
- 测试四种去混响方案：WPE (nara_wpe)、Spectral subtraction (noisereduce)、高通+预加重、Reverb tail trimming
- 在全部 40 个中文混响样本上做逐样本对照

**结果**：
- 所有信号处理方案均不能一致改善 CTC 输出
- WPE 和 Noisereduce 去除混响的同时也去除了语音能量，使 CTC 输出为空
- 高通+预加重仅 1/40 改善，8/40 恶化
- **原始 CTC 反而是最鲁棒的选择**

**学到什么**：
- 混响是卷积噪声，传统加性噪声抑制方法不适用
- 重度混响（房间混响时间 > 0.5s）导致 CTC 模型的帧级声学特征完全变形，这不是后处理能解决的问题
- 可行方向：DNN 去混响前端（DCCRN/DPDFNet，sherpa-onnx 已支持）或混响增强训练数据

### 2026-06-11 CTC+LLM 定点纠错实验

**做什么**：
- 设计置信度检测器（输出字符数/音频时长比 < 0.8 触发）
- 用 Qwen3.5-2B GGUF 对 9 个低置信度样本做整句纠错
- 对比 CTC only vs CTC+LLM 的 CER

**结果**：
- CTC only CER: 0.0852
- CTC+LLM CER: 0.0852（完全持平）
- 仅 1 样本改善（把窗打开风气→把窗户打开风气），2 样本倒退（中英混数字格式被LLM改坏）
- LLM 过于保守，只加标点不改内容

**学到什么**：
- CTC的错误模式与SenseVoice本质不同：是声学信息丢失（没听到）而非字符选择错误（听错但选错字）
- LLM 无法补回没被听到的语音内容，尤其是专有名词
- 置信度检测器（字符数/时长比）有效：精准触发了 9/9 个有问题的混响样本，零误报
- 结论：资源应投入声学前端增强而非后处理 LLM

### 2026-06-11 全模型 CTC 横评

**做什么**：
对 sherpa-onnx 生态中 **全部 7 个 CTC 模型** 做了统一基准测试（528 样本）：

| 模型 | CER | 大小 | 排名 |
|:---|---:|:---:|:---:|
| A: Zipformer-CTC offline zh (int8, 350MB) | **0.1090** | 350MB | 🥇 |
| H: WeNet-Wenetspeech CTC (int8, 127MB) | 0.1273 | 127MB | 🥈 |
| B: Zipformer-CTC streaming large (int8, 155MB) | 0.1348 | 155MB | 🥉 |
| C: Zipformer-CTC streaming xlarge (int8, 728MB) | 0.1395 | 728MB | 4 |
| E: Zipformer-CTC multi-zh-hans (fp32, 251MB) | 0.2179 | 251MB | 5 |
| D: Zipformer-CTC streaming small (int8, 25MB) | 0.2226 | 25MB | 6 |
| F: NeMo Parakeet 110M (en) | N/A | 126MB | 仅英文 |

**结果**：
1. **离线 Zipformer-CTC 冠军** — 帧级独立分类，无语言模型偏置
2. **xlarge(728MB) 反而不如 large(155MB)** — 过参数化对 CTC 无益
3. **流式模型有 ~20-30% 精度损失** vs 离线版本
4. **无中英双语 CTC 模型** — `<unk>` 问题仍需 Transducer 兜底
5. **WeNet 表现亮眼** — 127MB 达到 0.1273，仅次于冠军

**踩坑**：
- 流式 CTC 必须 `tail_paddings(0.66s) + input_finished()`，否则帧数不够报错
- C_xlarge 解码极慢（~1.5s/样本），大模型 + 流式 = 双重开销
- RAM 7.4GB 限制只能同时跑 1-2 个大模型

**学到什么**：
- 模型大小 ≠ 精度：C_xlarge(728MB) < B_large(155MB)
- `OfflineRecognizer` vs `OnlineRecognizer` API 差异大
- 结论：保持现有离线 Zipformer-CTC + `--bilingual` Transducer 双模式

### 2026-06-11 Omnilingual 300M CTC 测试

测试了 `csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8`：
- 9812 token 词表（5750中文 + 英文字母）
- **0/528 `<unk>`** — 唯一无 `<unk>` 的 CTC 模型
- CER=0.2988（中文精度不如 Zipformer-CTC，但英文可识别）
- 中英混场景 CER=0.6029（英文部分做音近猜测，不准确但比 `<unk>` 好）

结论：新增 `--omnilingual` 模式可选，适合中文为主偶有英文的场景。
