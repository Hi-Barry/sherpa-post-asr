"""
置信度分析与混淆区检测

支持两种模式：
1. 基于 log_probs 的局部检测（当数据可用时）
2. 基于时间戳的启发式分割（作为回退）
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfusionRegion:
    """一个需要纠错的区域"""
    start: int            # 在原始 tokens 中的起始位置
    end: int              # 结束位置（不包含）
    text: str             # 该区间的原始识别文字
    avg_confidence: float # 平均置信度
    context_before: str   # 上文
    context_after: str    # 下文
    is_whole: bool = False  # 是否为整句模式


class ConfusionDetector:
    """
    混淆区检测器。

    降级策略：如果 SenseVoice 未提供置信度（ys_log_probs 为空），
    则根据标点符号分割句子，逐句送入 LLM 纠错。
    """

    def __init__(
        self,
        log_prob_threshold: float = -1.5,
        min_low_len: int = 1,
        context_window: int = 8,
        max_correct_len: int = 4,
        max_sentence_len: int = 30,  # 单句最大长度（超过则再分割）
    ):
        self.log_prob_threshold = log_prob_threshold
        self.min_low_len = min_low_len
        self.context_window = context_window
        self.max_correct_len = max_correct_len
        self.max_sentence_len = max_sentence_len

    def detect(
        self, tokens: list[str],
        log_probs: Optional[list[float]] = None,
        full_text: Optional[str] = None,
    ) -> list[ConfusionRegion]:
        """
        检测需要纠错的区域。

        如果有 log_probs，做精确的置信度检测。
        如果没有，则按标点分句，逐句纠错。
        """
        if full_text is None:
            full_text = "".join(tokens)

        # 如果有 log_probs 且有数据，使用精确模式
        if log_probs and len(log_probs) > 0:
            return self._detect_by_confidence(tokens, log_probs)

        # 回退：基于标点分句
        return self._detect_by_sentence(tokens, full_text)

    def _detect_by_confidence(
        self, tokens: list[str], log_probs: list[float]
    ) -> list[ConfusionRegion]:
        """基于置信度的精确检测"""
        if len(tokens) != len(log_probs):
            min_len = min(len(tokens), len(log_probs))
            tokens = tokens[:min_len]
            log_probs = log_probs[:min_len]

        low_mask = [lp < self.log_prob_threshold for lp in log_probs]

        regions = []
        in_low = False
        seg_start = 0

        for i, is_low in enumerate(low_mask):
            if is_low and not in_low:
                seg_start = i
                in_low = True
            elif not is_low and in_low:
                self._add_region(regions, tokens, log_probs, seg_start, i)
                in_low = False

        if in_low:
            self._add_region(regions, tokens, log_probs, seg_start, len(tokens))

        regions.sort(key=lambda r: r.avg_confidence)
        return regions

    def _detect_by_sentence(
        self, tokens: list[str], full_text: str
    ) -> list[ConfusionRegion]:
        """基于标点分句的回退检测"""
        # 标点分割
        split_chars = set("，。！？；：、\n")
        sentences = []
        current = []
        for i, t in enumerate(tokens):
            current.append(t)
            if t in split_chars and len(current) > 1:
                sent_text = "".join(current[:-1])  # 不包括标点
                if sent_text.strip():
                    sentences.append({
                        "start": i - len(current) + 1,
                        "end": i,  # 不包括标点
                        "text": sent_text,
                    })
                # 标点单独放...
                sentences.append({
                    "start": i,
                    "end": i + 1,
                    "text": t,
                    "is_punct": True,
                })
                current = []

        # 剩余部分
        if current:
            sent_text = "".join(current)
            if sent_text.strip():
                sentences.append({
                    "start": len(tokens) - len(current),
                    "end": len(tokens),
                    "text": sent_text,
                })

        # 合并短句
        merged = []
        buffer = []
        buf_start = 0
        for s in sentences:
            if s.get("is_punct"):
                if buffer:
                    merged.append(buffer)
                    buffer = []
                continue
            text_len = len(s["text"])
            if text_len > self.max_sentence_len:
                if buffer:
                    merged.append(buffer)
                    buffer = []
                merged.append([s])
            elif sum(len(x["text"]) for x in buffer) + text_len <= self.max_sentence_len:
                if not buffer:
                    buf_start = s["start"]
                buffer.append(s)
            else:
                if buffer:
                    merged.append(buffer)
                buf_start = s["start"]
                buffer = [s]

        if buffer:
            merged.append(buffer)

        # 转换为 ConfusionRegion
        regions = []
        for group in merged:
            text = "".join(s["text"] for s in group)
            start = group[0]["start"]
            end = group[-1]["end"]
            if not text.strip():
                continue
            regions.append(ConfusionRegion(
                start=start,
                end=end,
                text=text,
                avg_confidence=0.5,    # 默认中等置信度
                context_before="",
                context_after="",
                is_whole=True,
            ))

        return regions

    def _add_region(self, regions, tokens, log_probs, seg_start, seg_end):
        length = seg_end - seg_start
        if length < self.min_low_len or length > self.max_correct_len:
            return
        seg_text = "".join(tokens[seg_start:seg_end])
        seg_lps = log_probs[seg_start:seg_end]
        avg_conf = float(np.mean([np.exp(lp) for lp in seg_lps]))

        ctx_before = "".join(tokens[max(0, seg_start - self.context_window):seg_start])
        ctx_after = "".join(tokens[seg_end:min(len(tokens), seg_end + self.context_window)])

        if not seg_text.strip() or all(c in "，。、；：？！""''【】（）「」—…·" for c in seg_text):
            return

        regions.append(ConfusionRegion(
            start=seg_start, end=seg_end, text=seg_text,
            avg_confidence=avg_conf, min_confidence=float(np.min([np.exp(lp) for lp in seg_lps])),
            context_before=ctx_before, context_after=ctx_after,
        ))

    def build_correction_prompt(self, region: ConfusionRegion) -> str:
        """
        构造纠错 prompt，严格控制只改 ASR 误识，不改其他。
        """
        if region.is_whole:
            return (
                f"你是 ASR 纠错助手。只改确认有错的地方，不改其他。\n\n"
                f"示例：\n"
                f"  原文：我今天去了静界，风景很美。\n"
                f"  输出：我今天去了境界，风景很美。\n"
                f"  (只改了「静界→境界」，其余完全不变)\n\n"
                f"  原文：我的工号是003，邮箱是edmin at example dotcom\n"
                f"  输出：我的工号是003，邮箱是admin at example dotcom\n"
                f"  (只改了「edmin→admin」，数字格式保留)\n\n"
                f"现在校对以下文本——**必须逐字保留原文**，只改明显错误的字：\n"
                f"原文：{region.text}\n"
                f"输出："
            )
        else:
            placeholder = "＿" * len(region.text)
            return (
                f"请纠正以下语音识别结果中的潜在错误。\n\n"
                f"上下文：{region.context_before}{placeholder}{region.context_after}\n"
                f"原始识别：「{region.text}」\n"
                f"置信度：{region.avg_confidence:.2f}\n\n"
                f"要求：**只输出纠正后的 {len(region.text)} 个字，不要解释**"
            )
