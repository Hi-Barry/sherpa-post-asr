# 🏆 全模型 CTC 横评报告

**测试日期**: 2026-06-11 16:31
**测试样本**: 528 条
**音频来源**: 12 场景 × 7 类别 × 多 SNR

## 📊 总排名（CER 升序）

| 排名 | 模型 | CER | 模型大小 | 备注 |
|:---:|:---|---:|:---:|:---|
| **1** | A: Zipformer-CTC offline zh (int8) | **0.1090** | 350MB | 🏆 冠军，离线非流式 |
| **2** | H: WeNet-Wenetspeech CTC (int8) | **0.1273** | 127MB | WeNet 架构，流式+非流式双用 |
| **3** | B: Zipformer-CTC streaming large (int8) | **0.1348** | 155MB | 流式模型，速度中等 |
| **4** | C: Zipformer-CTC streaming xlarge (int8) | **0.1395** | 728MB | 流式最大但精度反而不如 Large |
| **5** | E: Zipformer-CTC multi-zh-hans (fp32) | **0.2179** | 251MB | 2023年老模型，fp32非量化 |
| **6** | D: Zipformer-CTC streaming small (int8) | **0.2226** | 25MB | 仅 25MB，适合嵌入式/移动端 |

## 📋 按类别对比

| 类别 | A: Zipformer-CTC offline zh (int8) | H: WeNet-Wenetspeech CTC (int8) | B: Zipformer-CTC streaming large (int8) |
|:---|:---:|:---:|:---:|
| 轻声 | 0.1204 | 0.1551 | 0.2500 |
| 同音字 | 0.0339 | 0.0527 | 0.0417 |
| 数字 | 0.0187 | 0.0333 | 0.0521 |
| 上下文 | 0.0472 | 0.0792 | 0.0847 |
| 专名 | 0.0086 | 0.0768 | 0.0823 |
| 口音 | 0.0026 | 0.0260 | 0.0703 |
| 中英混 | 0.8444 | 0.7155 | 0.6510 |

## 📋 按场景对比

| 场景 | A: Zipformer-CTC offline zh (int8) | H: WeNet-Wenetspeech CTC (int8) | B: Zipformer-CTC streaming large (int8) |
|:---|:---:|:---:|:---:|
| 🤫 轻语 | 0.1090 | 0.0960 | 0.0758 |
| 🖥️ 机房 | 0.1019 | 0.0953 | 0.0623 |
| 🌀 风扇 | 0.1090 | 0.0960 | 0.0680 |
| 🌄 旷野 | 0.0948 | 0.0953 | 0.0758 |
| ⚡ 电机 | 0.1034 | 0.0990 | 0.0967 |
| ⬜ 白噪音 | 0.1190 | 0.2453 | 0.2614 |
| 💨 大风 | 0.1054 | 0.0953 | 0.0772 |
| 🌧️ 下雨 | 0.1013 | 0.1303 | 0.1513 |
| 🚇 地铁 | 0.0983 | 0.0960 | 0.0637 |
| 📢 高噪音 | 0.1054 | 0.0953 | 0.0758 |
| ⛈️ 雷暴 | 0.1002 | 0.1084 | 0.1273 |
| 🏛️ 混响 | 0.1610 | 0.2749 | 0.4820 |

## 🌐 英文模型（F: NeMo Parakeet 110M）

该模型仅支持纯英文，在含英文的中文音频上识别结果不佳，属于预期行为。

**识别英文样本数**: 48/528

**示例输出**:
- `Would go how Sinina? Yang Si admin at ample dot com`
- `Would go how S mein? Y ang Si atm me at ample dot com`
- `Would G how S meina? Y angXi admits at ample dot com`
- `Would Ghouse she means that? YangXi atmed at isample dot com`
- `Would G H ing Lon? Yu ang Si admin at examample dot com`


## 🆕 G: Omnilingual 300M CTC 多语言模型

| 指标 | 值 |
|:---|---:|
| **CER** | 0.2988 |
| **`<unk>` 出现** | 0/528 ✅ |
| **识别出英文** | 69/528 |
| **词表** | 9812 tokens（含5750中文 + 英文字母） |
| **模型大小** | 348MB (int8) |

### 与冠军 Zipformer-CTC 对比

| 维度 | Zipformer-CTC offline | Omnilingual 300M |
|:---|---:|---:|
| CER | **0.1090** 🏆 | 0.2988 |
| `<unk>` | 有 ❌ | **0/528** ✅ |
| 中英混 CER | ~1.0（英文全错） | **0.6029** |
| 中文精度 | **极高** | 中等 |

### 各类别 CER

| 类别 | CER |
|:---|---:|
| 口音 | 0.1484 ✅ |
| 数字 | 0.2604 |
| 同音字 | 0.2613 |
| 专名 | 0.2682 |
| 上下文 | 0.3132 |
| 轻声 | 0.3287 |
| **中英混** | 0.6029 ❌ |

### 英文样本示例

```
GT: 我的工号是零零三，邮箱是 admin at example dot com
HY(Omni): 我的公号是零零三 尤像是 admin at iampcom
HY(Zipformer): 我的工号是零零三，邮箱是 <unk> at <unk> dot <unk>
```

### 结论

Omnilingual 300M 是 **唯一能同时处理中英文的 CTC 模型**，CER 虽不如中文专用模型，但彻底解决了 `<unk>` 问题。适合中文为主但偶尔需要识别英文的场景。
## 📈 各模型详细分析

### A: Zipformer-CTC offline zh (int8)

- **平均 CER**: 0.1090
- **模型大小**: 350MB
- **测试样本**: 528
- **说明**: 离线非流式 Zipformer-CTC（int8 量化，350MB）。帧级独立分类，无 LM 偏置。

**最佳样本**:
  - `homophone_cengjing__subway__15dB.wav` → CER=0.0000
  - `homophone_cengjing__subway__10dB.wav` → CER=0.0000
**最差样本**:
  - `mixed_english__subway__15dB.wav` → CER=0.9375
    GT: `我的工号是零零三，邮箱是 admin at example dot com`
    HY: `我的工号是零零三邮箱是<unk> <unk> <unk> <unk> <unk> <unk>`
  - `mixed_english__server_room__15dB.wav` → CER=0.9375
    GT: `我的工号是零零三，邮箱是 admin at example dot com`
    HY: `我的工号是零零三邮箱是<unk> <unk> <unk><unk> <unk> <unk>`

### B: Zipformer-CTC streaming large (int8)

- **平均 CER**: 0.1348
- **模型大小**: 155MB
- **测试样本**: 528
- **说明**: 流式 Zipformer-CTC large（int8，155MB）。实时流式解码。

**最佳样本**:
  - `homophone_jingjie__subway__15dB.wav` → CER=0.0000
  - `homophone_jingjie__subway__10dB.wav` → CER=0.0000
**最差样本**:
  - `homophone_cengjing__reverb__l3.wav` → CER=1.0000
    GT: `曾经有一份真挚的爱情`
    HY: ``
  - `homophone_yijing__reverb__l3.wav` → CER=1.0000
    GT: `事情已经处理完了`
    HY: ``

### C: Zipformer-CTC streaming xlarge (int8)

- **平均 CER**: 0.1395
- **模型大小**: 728MB
- **测试样本**: 528
- **说明**: 流式 Zipformer-CTC xlarge（int8，728MB）。参数最多但精度不如 B。

**最佳样本**:
  - `homophone_cengjing__subway__15dB.wav` → CER=0.0000
  - `homophone_cengjing__subway__10dB.wav` → CER=0.0000
**最差样本**:
  - `proper_place__reverb__l2.wav` → CER=1.0000
    GT: `明天要去深圳出差`
    HY: ``
  - `proper_place__reverb__l3.wav` → CER=1.0000
    GT: `明天要去深圳出差`
    HY: ``

### D: Zipformer-CTC streaming small (int8)

- **平均 CER**: 0.2226
- **模型大小**: 25MB
- **测试样本**: 528
- **说明**: 流式 Zipformer-CTC small（int8，仅 25MB）。极致轻量，适合嵌入式。

**最佳样本**:
  - `homophone_cengjing__subway__15dB.wav` → CER=0.0000
  - `homophone_cengjing__subway__10dB.wav` → CER=0.0000
**最差样本**:
  - `homophone_jingjie__white_noise__-5dB.wav` → CER=1.0000
    GT: `我今天去了境界，风景很美`
    HY: `目`
  - `homophone_cengjing__reverb__l3.wav` → CER=1.0000
    GT: `曾经有一份真挚的爱情`
    HY: `资是`

### E: Zipformer-CTC multi-zh-hans (fp32)

- **平均 CER**: 0.2179
- **模型大小**: 251MB
- **测试样本**: 528
- **说明**: 流式 Zipformer-CTC multi-zh-hans（fp32，251MB）。2023年训练，14k小时数据，多方言。

**最佳样本**:
  - `homophone_cengjing__subway__15dB.wav` → CER=0.0000
  - `homophone_cengjing__subway__10dB.wav` → CER=0.0000
**最差样本**:
  - `homophone_cengjing__reverb__l2.wav` → CER=1.0000
    GT: `曾经有一份真挚的爱情`
    HY: `此`
  - `homophone_cengjing__reverb__l3.wav` → CER=1.0000
    GT: `曾经有一份真挚的爱情`
    HY: `此`

### H: WeNet-Wenetspeech CTC (int8)

- **平均 CER**: 0.1273
- **模型大小**: 127MB
- **测试样本**: 528
- **说明**: WeNet-Wenetspeech CTC（int8，127MB）。U2++ 架构，含 CTC + Attention。

**最佳样本**:
  - `homophone_cengjing__subway__15dB.wav` → CER=0.0000
  - `homophone_cengjing__subway__10dB.wav` → CER=0.0000
**最差样本**:
  - `mixed_english__reverb__l3.wav` → CER=0.9375
    GT: `我的工号是零零三，邮箱是 admin at example dot com`
    HY: `我好云森留下是言泪怎么都好`
  - `mixed_english__white_noise__-5dB.wav` → CER=0.9062
    GT: `我的工号是零零三，邮箱是 admin at example dot com`
    HY: `捂的多后是农民根提香是的米花是天出烧块子`

## ⏱️ 速度对比

| 模型 | 平均延迟 | 相对速度 |
|:---|---:|:---:|
| A: Zipformer-CTC offline zh (int8) | ~220ms | 0.7x |
| B: Zipformer-CTC streaming large (int8) | ~320ms | 0.5x |
| C: Zipformer-CTC streaming xlarge (int8) | ~1500ms | 0.1x |
| D: Zipformer-CTC streaming small (int8) | ~145ms | 1.0x |
| E: Zipformer-CTC multi-zh-hans (fp32) | ~500ms | 0.3x |
| H: WeNet-Wenetspeech CTC (int8) | ~200ms | 0.7x |

## 💡 结论与建议

1. **最佳中文 CTC 模型**: `A: Zipformer-CTC offline zh (int8)` (CER=0.1090)
2. **亚军**: `H: WeNet-Wenetspeech CTC (int8)` (CER=0.1273)
3. **流式最佳**: `B: Zipformer-CTC streaming large (int8)` — 兼顾实时和精度
4. **轻量冠军**: `E: Zipformer-CTC multi-zh-hans (fp32)` — 仅 25MB

5. **&#x3C;unk&#x3E;问题**：所有中文 CTC 模型均有此问题（词表不含英文）。
   中英混合场景推荐使用 `--bilingual` Transducer 模式。