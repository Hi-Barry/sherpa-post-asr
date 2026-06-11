"""
纠错引擎：编排 ASR → 混淆检测 → LLM 局部纠错 → 回填
"""

import time
import logging
import numpy as np
from typing import Optional

from .schemas import DiffItem, CorrectionSegment, TranscribeResponse
from .confusion_detector import ConfusionDetector

logger = logging.getLogger(__name__)


class CorrectionEngine:
    """
    Post-ASR 纠错引擎。

    流程：
    1. ASR 前端识别 → text + tokens + (可选) log_probs
    2. 混淆检测 → 需要纠错的段列表
       - 如有置信度：只在低置信度位置纠错
       - 如无置信度：按标点分句，逐句纠错
    3. 逐段 LLM 纠错
    4. 回填 → 最终输出
    """

    def __init__(
        self,
        asr_frontend,
        correction_llm,
        confusion_detector: ConfusionDetector = None,
    ):
        self.asr = asr_frontend
        self.llm = correction_llm
        self.detector = confusion_detector or ConfusionDetector()

    def process(self, samples: np.ndarray) -> TranscribeResponse:
        t_start = time.perf_counter()
        timings = {}

        # ── Step 1: ASR 识别 ──────────────────────
        asr_result = self.asr.transcribe(samples)
        timings["asr"] = asr_result["latency_ms"]

        raw_text = asr_result["text"]
        tokens = asr_result["tokens"]

        logger.info(f"ASR: {raw_text} (tokens={len(tokens)})")

        if not tokens or not raw_text.strip():
            return TranscribeResponse(
                raw_text=raw_text, corrected_text=raw_text, latency_ms=timings,
            )

        # ── Step 2: 检测纠错区 ────────────────────
        # 如果有 log_probs 则用精确模式，否则用分句回退
        log_probs = asr_result.get("log_probs", [])
        regions = self.detector.detect(tokens, log_probs=log_probs, full_text=raw_text)
        timings["detection"] = 0.0

        if not regions:
            logger.info("未检测到需纠错区")
            timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
            return TranscribeResponse(
                raw_text=raw_text, corrected_text=raw_text, latency_ms=timings,
            )

        has_confidence = bool(log_probs)
        mode = "局部(置信度)" if has_confidence else f"分句({len(regions)}段)"
        logger.info(f"检测到 {len(regions)} 个纠错区 [{mode}]: "
                     f"{[r.text[:20] for r in regions]}")

        # ── Step 3: 如果没有 LLM，跳过纠错 ────────
        if self.llm is None or not self.llm.is_loaded:
            logger.warning("LLM 未加载，跳过纠错")
            timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
            return TranscribeResponse(
                raw_text=raw_text, corrected_text=raw_text, latency_ms=timings,
            )

        # ── Step 4: LLM 逐段纠错 ──────────────────
        diffs: list[DiffItem] = []
        corrections: list[tuple[int, int, str, str]] = []
        total_llm_ms = 0.0

        for region in regions:
            prompt = self.detector.build_correction_prompt(region)
            corrected, llm_ms = self.llm.correct_local(prompt)
            total_llm_ms += llm_ms

            corrected = corrected.strip()
            if not corrected or corrected == region.text:
                continue  # LLM 认为无需修改

            logger.info(f"  纠错: '{region.text}' → '{corrected}' ({llm_ms:.1f}ms)")

            diffs.append(DiffItem(
                position=region.start,
                original=region.text,
                corrected=corrected,
                confidence=round(region.avg_confidence, 3),
                context_before=region.context_before,
                context_after=region.context_after,
            ))
            corrections.append((region.start, region.end, region.text, corrected))

        timings["llm_correction"] = round(total_llm_ms, 1)

        # ── Step 5: 回填（带安全校验）─────────────
        if corrections:
            accepted = 0
            rejected = 0
            # 在字符串层面操作，避免 BPE token 对齐问题
            result_text = raw_text

            for start, end, original, corrected in reversed(corrections):
                # 智能校验
                def clean(s):
                    return s.replace(" ", "").replace("　", "")

                clean_orig = clean(original)
                clean_corr = clean(corrected)

                # 计算改动量
                changes = sum(1 for a, b in zip(clean_orig, clean_corr) if a != b)
                pad = abs(len(clean_orig) - len(clean_corr))
                total_changes = changes + pad
                max_len = max(len(clean_orig), len(clean_corr))

                if (len(clean_orig) == len(clean_corr)
                        and total_changes > 0
                        and total_changes <= max_len * 0.5
                        and clean_orig != clean_corr):

                    if original in result_text:
                        result_text = result_text.replace(original, corrected, 1)
                        accepted += 1
                        logger.info(f"  ✅ 接受纠错: '{clean_orig}' → '{clean_corr}'")
                    else:
                        # 尝试 clean 版本匹配
                        if clean_orig in result_text:
                            # 只替换不同字符的部分
                            chars = list(result_text)
                            idx = result_text.find(clean_orig)
                            for j, (oc, cc) in enumerate(zip(clean_orig, clean_corr)):
                                if oc != cc and idx + j < len(chars):
                                    chars[idx + j] = cc
                            result_text = "".join(chars)
                            accepted += 1
                            logger.info(f"  ✅ 接受纠错(二级匹配): '{clean_orig}' → '{clean_corr}'")
                        else:
                            rejected += 1
                            logger.warning(f"  ❌ 拒绝（原文不可用）: '{original}'")
                else:
                    rejected += 1
                    logger.info(f"  ⏭️ 跳过（安全校验未通过）: '{clean_orig}' → '{clean_corr}' (changes={total_changes}/{max_len})")

            corrected_text = result_text
            if accepted == 0 and rejected > 0:
                logger.info("所有纠错均被拒绝，保留原始 ASR 结果")
                corrected_text = raw_text
                diffs = []
                segments = []
        else:
            corrected_text = raw_text

        timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)

        segments = [
            CorrectionSegment(
                start=r.start, end=r.end,
                original=r.text, corrected=c,
                avg_confidence=round(r.avg_confidence, 3),
                reason=f"LLM 纠正 '{r.text}' → '{c}'",
            )
            for r, (_, _, _, c) in zip(regions, corrections)
        ]

        return TranscribeResponse(
            raw_text=raw_text,
            corrected_text=corrected_text,
            diffs=diffs,
            segments=segments,
            latency_ms=timings,
        )
