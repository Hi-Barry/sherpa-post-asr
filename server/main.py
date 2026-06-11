"""
sherpa-post-asr: 语音识别后纠错服务

FastAPI 服务，提供音频转写 + 局部纠错 API。
"""

import sys
import os
import logging
from pathlib import Path

# 确保 server 包可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse

from server.config import (
    HOST, PORT,
    SENSEVOICE_MODEL, SENSEVOICE_TOKENS,
    SENSEVOICE_LANGUAGE, SENSEVOICE_USE_ITN,
    ASR_NUM_THREADS, ASR_PROVIDER,
    QWEN_GGUF_PATH, QWEN_N_GPU_LAYERS, QWEN_N_CTX, QWEN_BATCH_SIZE,
    TEST_AUDIO_DIR,
    LOG_PROB_THRESHOLD, MIN_LOW_LEN, CONTEXT_WINDOW, MAX_CORRECT_LEN,
)
from server.schemas import TranscribeResponse
from server.audio_utils import load_audio
from server.asr_frontend import AsrFrontend
from server.correction_llm import CorrectionLLM
from server.correction_engine import CorrectionEngine

# ── 日志 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sherpa-post-asr")

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(
    title="sherpa-post-asr",
    description="语音识别后纠错服务 — SenseVoice + Qwen3.5-2B 局部重打分",
    version="0.1.0",
)

# ── 全局引擎（延迟初始化）────────────────────────────
_engine: CorrectionEngine = None


def get_engine() -> CorrectionEngine:
    global _engine
    if _engine is not None:
        return _engine

    logger.info("初始化 ASR 前端 (SenseVoice)...")
    if not SENSEVOICE_MODEL.exists():
        raise RuntimeError(
            f"SenseVoice 模型未找到: {SENSEVOICE_MODEL}\n"
            f"请先运行: bash scripts/download_models.sh"
        )
    asr = AsrFrontend(
        model_path=SENSEVOICE_MODEL,
        tokens_path=SENSEVOICE_TOKENS,
        language=SENSEVOICE_LANGUAGE,
        use_itn=SENSEVOICE_USE_ITN,
        num_threads=ASR_NUM_THREADS,
        provider=ASR_PROVIDER,
    )

    logger.info("初始化 LLM 纠错引擎 (Qwen3.5-2B)...")
    if not QWEN_GGUF_PATH.exists():
        logger.warning(
            f"GGUF 模型未找到: {QWEN_GGUF_PATH}\n"
            f"纠错功能将不可用，仅返回原始 ASR 结果。\n"
            f"请运行: bash scripts/download_models.sh"
        )
        llm = None
    else:
        llm = CorrectionLLM(
            model_path=QWEN_GGUF_PATH,
            n_gpu_layers=QWEN_N_GPU_LAYERS,
            n_ctx=QWEN_N_CTX,
            batch_size=QWEN_BATCH_SIZE,
        )
        # 预加载 LLM
        try:
            llm.load()
            logger.info("LLM 模型加载完成")
        except Exception as e:
            logger.error(f"LLM 加载失败: {e}")
            llm = None

    _engine = CorrectionEngine(asr_frontend=asr, correction_llm=llm)
    logger.info("引擎初始化完成")
    return _engine


# ── API 端点 ──────────────────────────────────────────


@app.get("/health")
async def health():
    """健康检查"""
    engine = get_engine()
    return {
        "status": "ok",
        "asr_loaded": True,
        "llm_loaded": engine.llm.is_loaded if engine.llm else False,
        "asr_model": str(SENSEVOICE_MODEL.name),
        "llm_model": str(QWEN_GGUF_PATH.name) if QWEN_GGUF_PATH.exists() else "not found",
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
):
    """
    上传音频文件，返回原始 ASR 结果 + 局部纠错结果。

    支持格式: WAV, MP3, FLAC, OGG, M4A
    """
    engine = get_engine()

    # 保存上传的文件到临时文件
    import tempfile
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        samples = load_audio(tmp_path)
        response = engine.process(samples)
        return response
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe_file", response_model=TranscribeResponse)
async def transcribe_file(
    path: str = Body(..., embed=True, description="服务端本地音频文件路径"),
):
    """
    对服务端本地的音频文件进行转写 + 纠错。

    用于测试（避免每次上传大文件）。
    """
    engine = get_engine()
    audio_path = Path(path)

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")

    try:
        samples = load_audio(audio_path)
        response = engine.process(samples)
        return response
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── 启动入口 ──────────────────────────────────────────


def main():
    """启动 Uvicorn 服务器"""
    logger.info(f"启动 sherpa-post-asr 服务: {HOST}:{PORT}")
    uvicorn.run(
        "server.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
