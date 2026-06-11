#!/usr/bin/env bash
# ── 启动服务脚本 ──────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 激活 venv（如果有）
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== 启动 sherpa-post-asr 服务 ==="
echo "端口: ${PORT:-8001}"
echo ""

python -m server.main
