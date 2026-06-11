"""
批量基准测试 — 对所有带噪样本跑 ASR + 纠错，收集结果
"""
import sys, json, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8001/transcribe_file"
BASE_DIR = Path(__file__).resolve().parent.parent
NOISY_DIR = BASE_DIR / "audio_test" / "noisy"
MANIFEST_PATH = BASE_DIR / "audio_test" / "_benchmark_manifest.json"
RESULT_PATH = BASE_DIR / "audio_test" / "_benchmark_results.json"
PROGRESS_PATH = BASE_DIR / "audio_test" / "_benchmark_progress.json"


def call_api(audio_path: str) -> dict | None:
    """调用纠错 API"""
    import http.client
    payload = json.dumps({"path": audio_path}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=120)
    try:
        conn.request("POST", "/transcribe_file", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status == 200:
            data = json.loads(resp.read().decode("utf-8"))
            return data
        else:
            return {"error": f"HTTP {resp.status}", "raw_text": "", "corrected_text": ""}
    except Exception as e:
        return {"error": str(e), "raw_text": "", "corrected_text": ""}
    finally:
        conn.close()


def compute_cer(ref: str, hyp: str) -> float:
    """简单 CER：字符编辑距离 / 参考长度"""
    ref = ref.replace(" ", "").replace("　", "")
    hyp = hyp.replace(" ", "").replace("　", "")
    if not ref:
        return 0.0
    # Levenshtein
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n] / m


def normalize_text(text: str) -> str:
    """标准化文本用于对比（去空格、统一标点）"""
    text = text.replace(" ", "").replace("　", "")
    # 英文转小写
    import re
    def lower_en(m):
        return m.group(0).lower()
    text = re.sub(r'[a-zA-Z]+', lower_en, text)
    return text


def run_benchmark(resume: bool = True):
    """执行批量测试"""
    # 加载 manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 恢复进度
    results = []
    processed_files = set()
    if resume and PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                progress = json.load(f)
            results = progress.get("results", [])
            processed_files = set(r["file"] for r in results)
            print(f"恢复进度: 已完成 {len(processed_files)}/{len(manifest)}")
        except:
            pass

    total = len(manifest)
    start_time = time.time()

    for idx, item in enumerate(manifest):
        fname = item["file"]
        if fname in processed_files:
            continue

        audio_path = str(NOISY_DIR / fname)
        if not os.path.exists(audio_path):
            print(f"[{idx+1}/{total}] ⚠️ 文件不存在: {fname}")
            continue

        # 调用 API
        api_result = call_api(audio_path)

        if api_result and "error" not in api_result:
            raw = api_result.get("raw_text", "")
            corrected = api_result.get("corrected_text", "")
            latency = api_result.get("latency_ms", {})

            gt = item["ground_truth"]
            raw_cer = compute_cer(normalize_text(gt), normalize_text(raw))
            cor_cer = compute_cer(normalize_text(gt), normalize_text(corrected))
            improved = raw_cer - cor_cer
            llm_changed = (normalize_text(raw) != normalize_text(corrected))
        else:
            raw = corrected = ""
            raw_cer = cor_cer = 1.0
            improved = 0.0
            llm_changed = False
            latency = {}

        # 记录结果
        result = {
            **item,
            "raw_text": raw,
            "corrected_text": corrected,
            "raw_cer": round(raw_cer, 4),
            "corrected_cer": round(cor_cer, 4),
            "cer_improvement": round(improved, 4),
            "llm_changed": llm_changed,
            "latency_ms": latency,
        }
        results.append(result)

        # 反馈
        bar = "█" * int((idx + 1) / total * 30) + "░" * (30 - int((idx + 1) / total * 30))
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total - idx - 1) / rate if rate > 0 else 0
        print(f"[{idx+1}/{total}] {bar} {fname[:40]:40s} "
              f"rawCER={raw_cer:.3f} corCER={cor_cer:.3f} "
              f"{'✅' if improved > 0 else ('❌' if improved < 0 else '➖')} "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        # 每 30 个保存一次进度
        if (idx + 1) % 30 == 0:
            with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": results, "timestamp": time.time()},
                          f, ensure_ascii=False)

        # 避免把服务器打爆
        time.sleep(0.1)

    # 保存最终结果
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 清理进度文件
    PROGRESS_PATH.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    avg_cer_raw = sum(r["raw_cer"] for r in results) / len(results)
    avg_cer_cor = sum(r["corrected_cer"] for r in results) / len(results)
    improved_count = sum(1 for r in results if r["cer_improvement"] > 0)
    worsened_count = sum(1 for r in results if r["cer_improvement"] < 0)

    print(f"\n{'='*60}")
    print(f"基准测试完成!")
    print(f"总样本: {len(results)}")
    print(f"耗时: {elapsed:.0f}s")
    print(f"平均原始 CER: {avg_cer_raw:.4f}")
    print(f"平均纠错 CER: {avg_cer_cor:.4f}")
    print(f"改善: {improved_count} 个  | 变差: {worsened_count} 个")
    print(f"结果保存: {RESULT_PATH}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    run_benchmark(resume=True)
