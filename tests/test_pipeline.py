#!/usr/bin/env python3
"""
测试客户端 — 快速验证本地 post-asr 服务

用法:
    python tests/test_pipeline.py                      # 测试默认音频
    python tests/test_pipeline.py <audio_path>          # 测试指定音频
    python tests/test_pipeline.py --all                 # 批量测试所有样本
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import argparse
import requests
from server.config import PORT, TEST_AUDIO_DIR

BASE_URL = f"http://127.0.0.1:{PORT}"


def _get(url, **kwargs):
    """请求包装：自动绕过代理"""
    import subprocess
    result = subprocess.run(
        ["curl", "--noproxy", "*", "-s", url],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout


def _post_json(url, json_data):
    """POST JSON 请求包装：自动绕过代理"""
    import subprocess, json as _json
    result = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "-X", "POST",
         url,
         "-H", "Content-Type: application/json",
         "-d", _json.dumps(json_data)],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout


def check_health():
    """检查服务健康状态"""
    try:
        import json
        return json.loads(_get(f"{BASE_URL}/health"))
    except Exception as e:
        return {"status": "error", "error": str(e)}


def transcribe_file(audio_path: str) -> dict:
    """通过文件路径调用转写"""
    import json
    raw = _post_json(f"{BASE_URL}/transcribe_file", {"path": str(audio_path)})
    return json.loads(raw)


def transcribe_upload(audio_path: str) -> dict:
    """上传文件调用转写"""
    import subprocess, json
    result = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "-X", "POST",
         f"{BASE_URL}/transcribe",
         "-F", f"file=@{audio_path}"],
        capture_output=True, text=True, timeout=120,
    )
    return json.loads(result.stdout)


def print_result(result: dict, show_diff: bool = True):
    """格式化打印结果"""
    raw = result["raw_text"]
    corr = result["corrected_text"]
    lat = result.get("latency_ms", {})
    diffs = result.get("diffs", [])

    print(f"\n{'='*60}")
    print(f"原始: 「{raw}」")
    print(f"纠错: 「{corr}」")
    print(f"{'='*60}")
    print(f"延迟: ASR={lat.get('asr','?'):>6}ms  LLM={lat.get('llm_correction','?'):>6}ms  总计={lat.get('total','?'):>6}ms")

    if diffs and show_diff:
        print(f"\n差异 ({len(diffs)} 处):")
        for d in diffs:
            old, new = d["original"], d["corrected"]
            conf = d["confidence"]
            mark = "✅" if old != new else "⏭️"
            print(f"  {mark} 「{old}」→「{new}」 (置信度 {conf:.3f})  "
                  f"上下文: ..{d.get('context_before','')}|{d.get('context_after','')}..")

    return raw != corr  # True 表示有修改


def run_all_tests():
    """批量运行所有测试音频"""
    wav_files = sorted(TEST_AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        print("❌ 未找到测试音频，请先运行 scripts/generate_test_audio.py")
        return

    print(f"批量测试 {len(wav_files)} 个音频...\n")
    stats = {"total": 0, "changed": 0, "errors": 0}

    for wav in wav_files:
        name = wav.stem
        try:
            result = transcribe_file(str(wav))
            changed = print_result(result)
            stats["total"] += 1
            if changed:
                stats["changed"] += 1
        except Exception as e:
            print(f"\n❌ {name}: {e}")
            stats["errors"] += 1

    print(f"\n{'='*60}")
    print(f"统计: 共 {stats['total']} 个 | 有修改 {stats['changed']} 个 | 错误 {stats['errors']} 个")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="sherpa-post-asr 测试客户端")
    parser.add_argument("audio", nargs="?", help="音频文件路径")
    parser.add_argument("--all", action="store_true", help="批量测试所有样本")
    parser.add_argument("--upload", action="store_true", help="使用文件上传模式")
    args = parser.parse_args()

    # 1. 健康检查
    health = check_health()
    if health.get("status") != "ok":
        print(f"❌ 服务未运行: {health}")
        print(f"   请先启动: python -m server.main")
        sys.exit(1)
    llm_status = health.get('llm_loaded')
    if llm_status:
        print(f"服务正常 | LLM=已加载")
    else:
        print(f"服务正常 | LLM=未加载(仅ASR)")

    # 2. 批量测试
    if args.all:
        run_all_tests()
        return

    # 3. 单文件测试
    audio_path = args.audio
    if not audio_path:
        # 找第一个测试音频
        wavs = sorted(TEST_AUDIO_DIR.glob("*.wav"))
        if wavs:
            audio_path = str(wavs[0])
            print(f"未指定音频，使用默认: {audio_path}")
        else:
            print("请指定音频文件路径")
            sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        sys.exit(1)

    # 调用
    if args.upload:
        result = transcribe_upload(audio_path)
    else:
        result = transcribe_file(audio_path)

    print_result(result)


if __name__ == "__main__":
    main()
