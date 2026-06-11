"""
sherpa-post-asr 配置
"""

from pathlib import Path
import os

# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 模型存放目录
MODELS_DIR = PROJECT_DIR / "models"

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# ── ASR 前端 (SenseVoice) ──────────────────────────────
SENSEVOICE_DIR = MODELS_DIR / "sense-voice-encoder-int8"
SENSEVOICE_MODEL = SENSEVOICE_DIR / "model.int8.onnx"
SENSEVOICE_TOKENS = SENSEVOICE_DIR / "tokens.txt"
SENSEVOICE_LANGUAGE = "zh"   # 中文优先
SENSEVOICE_USE_ITN = True    # 逆文本正则化
ASR_NUM_THREADS = 4
ASR_PROVIDER = "cpu"         # SenseVoice 放 CPU

# ── 纠错 LLM (Qwen3.5-2B) ──────────────────────────────
# 实际下载的文件名（download_models.sh 中可能因源变化而不同）
import glob
_qwen_candidates = list(MODELS_DIR.glob("Qwen3.5-2B*.gguf"))
if _qwen_candidates:
    QWEN_GGUF_PATH = _qwen_candidates[0]
    print(f"[config] 找到 GGUF: {QWEN_GGUF_PATH.name}")
else:
    QWEN_GGUF_PATH = MODELS_DIR / "Qwen3.5-2B-Instruct-Q4_K_M.gguf"
QWEN_N_GPU_LAYERS = 20       # GPU 层数（2B 可以全 offload）
QWEN_N_CTX = 2048            # 上下文长度
QWEN_BATCH_SIZE = 512

# ── 置信度 & 纠错参数 ──────────────────────────────────
# 字符级 log_prob < LOG_PROB_THRESHOLD 视为低置信度
LOG_PROB_THRESHOLD = -1.5     # ≈ 置信度 0.22
# 连续低置信度至少 MIN_LOW_LEN 个字符才触发纠错
MIN_LOW_LEN = 1
# 低置信度区域两侧的上下文窗口（字符数）
CONTEXT_WINDOW = 8
# 最大纠错段长度（超过此长度跳过，避免 LLM 瞎猜）
MAX_CORRECT_LEN = 4

# ── 测试音频 ────────────────────────────────────────────
TEST_AUDIO_DIR = PROJECT_DIR / "audio_test"

# ── 可用 GPU 检测 ──────────────────────────────────────
def detect_gpu_layers(model_total_layers: int) -> int:
    """根据 VRAM 估算可 offload 的层数"""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        free_mib = int(result.stdout.strip().split("\n")[0])
        # 每层约 50-80MB; 保守估计
        per_layer_mb = 60
        max_gpu_layers = min(model_total_layers, int((free_mib - 300) / per_layer_mb))
        return max(0, max_gpu_layers)
    except Exception:
        return 0
