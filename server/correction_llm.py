"""
纠错 LLM 引擎：Qwen3.5-2B GGUF 局部重打分

使用 llama-cpp-python 加载 GGUF 量化模型，进行局部纠错推理。
"""

import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CorrectionLLM:
    """Qwen3.5-2B q4 局部纠错引擎"""

    def __init__(
        self,
        model_path: str | Path,
        n_gpu_layers: int = -1,   # -1 = 全部 offload 到 GPU
        n_ctx: int = 2048,
        batch_size: int = 512,
        verbose: bool = False,
    ):
        self.model_path = str(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.batch_size = batch_size
        self.verbose = verbose
        self._llm = None

    def load(self):
        """加载 GGUF 模型（延迟加载）"""
        if self._llm is not None:
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"GGUF 模型不存在: {self.model_path}")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "需要 llama-cpp-python: pip install llama-cpp-python"
            )

        logger.info(
            f"加载 Qwen3.5-2B GGUF: {self.model_path}, "
            f"n_gpu_layers={self.n_gpu_layers}"
        )

        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.batch_size,
            verbose=self.verbose,
        )
        logger.info("模型加载完成")

    def unload(self):
        """释放模型"""
        self._llm = None
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def correct_local(self, prompt: str, max_tokens: int = 8) -> tuple[str, float]:
        """
        单次局部纠错推理（chat completion API）。

        Args:
            prompt: 纠错 prompt（见 confusion_detector）
            max_tokens: 最大输出 token 数（对局部纠错 4-8 就够了）

        Returns:
            (corrected_text, latency_ms)
        """
        self.load()

        start = time.perf_counter()

        response = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个语音识别纠错助手。根据上下文和语义，纠正识别错误的文字。输出简洁，不要解释。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,       # 低温度 = 确定性输出
            top_p=0.9,
            stop=["\n"],           # 遇到换行就停
        )

        elapsed = (time.perf_counter() - start) * 1000

        answer = response["choices"][0]["message"]["content"].strip()
        return answer, round(elapsed, 1)

    def create_completion(self, prompt: str, max_tokens: int = 256,
                          temperature: float = 0.1, stop=None) -> dict:
        """
        文本补全接口（用于 Mode B 整句纠错）。

        Qwen3.5 是 chat 模型，底层仍走 chat 模板。
        这里兼容 engine 的调用方式。
        """
        self.load()
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            top_p=0.9,
        )
        # 包装成类似 completion API 的格式
        return {
            "choices": [{
                "text": response["choices"][0]["message"]["content"]
            }]
        }

    def rescore_candidates(
        self, prompt: str, candidates: list[str], max_tokens: int = 4
    ) -> tuple[str, float]:
        """
        候选重打分模式。调用 LLM 从候选中选择最佳项。

        Returns:
            (chosen_candidate, latency_ms)
        """
        self.load()
        start = time.perf_counter()

        response = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "从候选中选择最合适的一项。只输出编号。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            stop=["\n"],
        )

        elapsed = (time.perf_counter() - start) * 1000
        answer = response["choices"][0]["message"]["content"].strip()

        # 尝试解析编号
        chosen_idx = None
        for c in candidates:
            if c in answer:
                chosen_idx = c
                break

        if chosen_idx is None:
            # 尝试解析数字
            import re
            nums = re.findall(r"\d+", answer)
            if nums:
                idx = int(nums[0]) - 1
                if 0 <= idx < len(candidates):
                    chosen_idx = candidates[idx]

        return chosen_idx or candidates[0], round(elapsed, 1)
