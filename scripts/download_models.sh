#!/usr/bin/env bash
# ── 模型下载脚本 ──────────────────────────────────────
# 下载 SenseVoice int8 + Qwen3.5-2B q4 GGUF
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "=== 下载目录: $MODELS_DIR ==="

# ── 1. SenseVoice int8 (sherpa-onnx 格式) ────────────
SENSEVOICE_MODEL="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSEVOICE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SENSEVOICE_MODEL}.tar.bz2"

if [ ! -d "sense-voice-encoder-int8" ]; then
    echo "[1/2] 下载 SenseVoice int8..."
    if command -v curl &>/dev/null; then
        curl -L -o "${SENSEVOICE_MODEL}.tar.bz2" "$SENSEVOICE_URL"
    else
        wget -O "${SENSEVOICE_MODEL}.tar.bz2" "$SENSEVOICE_URL"
    fi
    tar xjf "${SENSEVOICE_MODEL}.tar.bz2"
    mv "${SENSEVOICE_MODEL}" "sense-voice-encoder-int8"
    rm -f "${SENSEVOICE_MODEL}.tar.bz2"
    echo "   SenseVoice 下载完成"
else
    echo "[1/2] SenseVoice 已存在，跳过"
fi

# ── 2. Qwen3.5-2B q4 GGUF (HuggingFace) ─────────────
QWEN_GGUF="Qwen3.5-2B-Q4_K_M.gguf"
QWEN_URL="https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf"

if [ ! -f "$QWEN_GGUF" ]; then
    echo "[2/2] 下载 Qwen3.5-2B q4 GGUF (~1.3GB)..."
    if command -v curl &>/dev/null; then
        curl -L -o "$QWEN_GGUF" "$QWEN_URL"
    else
        wget -O "$QWEN_GGUF" "$QWEN_URL"
    fi
    echo "   Qwen3.5-2B 下载完成"
else
    echo "[2/2] Qwen3.5-2B GGUF 已存在，跳过"
fi

echo ""
echo "=== 全部下载完成 ==="
echo "模型位置:"
ls -lh "$MODELS_DIR"/
