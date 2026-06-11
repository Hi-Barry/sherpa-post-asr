"""
ASR 前端：SenseVoice 封装

使用 sherpa-onnx 离线识别器。
输出文本 + 逐 token 信息（可能不含置信度）。
"""

import numpy as np
import sherpa_onnx as sherpa
from pathlib import Path
import time


class AsrFrontend:
    """SenseVoice ASR 前端（CPU）"""

    def __init__(
        self,
        model_path: str | Path,
        tokens_path: str | Path,
        language: str = "zh",
        use_itn: bool = True,
        num_threads: int = 4,
        provider: str = "cpu",
    ):
        self.sample_rate = 16000
        self._init_recognizer(
            model_path, tokens_path, language, use_itn, num_threads, provider
        )

    def _init_recognizer(self, model_path, tokens_path, language, use_itn, num_threads, provider):
        model_path = str(model_path)
        tokens_path = str(tokens_path)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        if not Path(tokens_path).exists():
            raise FileNotFoundError(f"tokens 不存在: {tokens_path}")

        self.recognizer = sherpa.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider=provider,
            language=language,
            use_itn=use_itn,
        )

    def transcribe(self, samples: np.ndarray) -> dict:
        """
        识别音频。

        Returns:
            text: str           识别文本
            tokens: [str]       逐 token
            timestamps: [float] 逐 token 时间戳
            log_probs: [float]  逐 token 对数概率（可能为空）
            confidence: [float]  置信度（可能为空）
            latency_ms: float
        """
        start = time.perf_counter()

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples.tolist())
        self.recognizer.decode_stream(stream)

        result = stream.result
        elapsed = (time.perf_counter() - start) * 1000

        text = result.text
        tokens = list(result.tokens) if result.tokens else []
        timestamps = list(result.timestamps) if result.timestamps else []

        # 尝试获取 log_probs（SenseVoice 通常不提供此信息）
        log_probs = []
        confidence = []
        if hasattr(result, "ys_log_probs") and result.ys_log_probs is not None:
            lp_list = result.ys_log_probs
            if len(lp_list) > 0:
                for i in range(len(lp_list)):
                    lp = float(lp_list[i])
                    log_probs.append(lp)
                    confidence.append(float(np.exp(lp)))

        # 对齐长度
        min_len = min(len(tokens), len(log_probs)) if log_probs else len(tokens)
        tokens = tokens[:min_len]
        log_probs = log_probs[:min_len]
        confidence = confidence[:min_len]
        timestamps = timestamps[:min_len] if len(timestamps) > min_len else timestamps

        return {
            "text": text,
            "tokens": tokens,
            "timestamps": timestamps,
            "log_probs": log_probs,
            "confidence": confidence,
            "latency_ms": round(elapsed, 1),
        }

    def __repr__(self):
        return f"AsrFrontend(model=...)"

    @property
    def has_confidence(self) -> bool:
        """SenseVoice 是否提供了置信度信息"""
        return False  # 实测 SenseVoice 不输出 ys_log_probs
