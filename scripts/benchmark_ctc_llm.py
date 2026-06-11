"""
Benchmark: Zipformer-CTC + Qwen3.5-2B 定点纠错

流程：
1. Zipformer-CTC 帧级识别
2. 置信度检测器：如果输出字符数/音频时长 < 阈值，标记为低置信度
3. 仅对低置信度样本调用 Qwen3.5-2B 做整句重写纠错
4. 对比纠错前后的 CER
"""

import json, time, os, re
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sherpa_onnx
import soundfile as sf

# ── 路径 ──
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "zipformer-ctc-zh-int8" / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
NOISY_DIR = BASE_DIR / "audio_test" / "noisy"
MANIFEST_PATH = BASE_DIR / "audio_test" / "_benchmark_manifest.json"
RESULT_PATH = BASE_DIR / "audio_test" / "_benchmark_ctc_llm_results.json"
GGUF_PATH = BASE_DIR / "models" / "Qwen3.5-2B-Q4_K_M.gguf"


def compute_cer(ref: str, hyp: str) -> float:
    """字符编辑距离 / 参考长度"""
    ref = ref.replace(" ", "").replace("　", "")
    hyp = hyp.replace(" ", "").replace("　", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def strip_punc(s):
    return re.sub(r'[，。？！、；：""''【】《》（）\-\s]', '', s)


def is_low_confidence(text: str, duration_s: float) -> bool:
    """置信度检测器：基于输出字符数/时长比"""
    chars = len(strip_punc(text))
    expected = max(1, int(duration_s * 2.5))  # ~2.5 chars/sec
    ratio = chars / expected
    # 阈值：chars/sec < 2.0 或 chars < expected * 0.8
    return ratio < 0.8 or (chars / max(duration_s, 0.1)) < 2.0


def main():
    print("=" * 60)
    print("Zipformer-CTC + Qwen3.5-2B 定点纠错 Benchmark")
    print("=" * 60)

    # ── 加载 CTC 模型 ──
    t0 = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=str(MODEL_DIR / "model.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=4,
        sample_rate=16000,
        decoding_method="greedy_search",
    )
    print(f"CTC 模型加载: {time.time() - t0:.1f}s")

    # ── 加载 LLM ──
    print("加载 Qwen3.5-2B GGUF...")
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(
        model_path=str(GGUF_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=512,
        verbose=False,
    )
    print(f"LLM 加载: {time.time() - t0:.1f}s")

    # ── 加载 manifest ──
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"测试样本: {len(manifest)}")

    # ── 逐样本处理 ──
    results = []
    start_time = time.time()
    total = len(manifest)
    llm_calls = 0
    llm_total_ms = 0

    for idx, item in enumerate(manifest):
        fname = item["file"]
        audio_path = str(NOISY_DIR / fname)

        if not os.path.exists(audio_path):
            print(f"[{idx + 1}/{total}] ⚠️ 文件不存在: {fname}")
            continue

        # 加载音频
        try:
            samples, sr = sf.read(audio_path)
            if sr != 16000:
                import librosa
                samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        except Exception as e:
            print(f"[{idx + 1}/{total}] ❌ 读取失败: {fname} ({e})")
            continue

        duration = item.get("duration_s", len(samples) / 16000)

        # ── Step 1: CTC 识别 ──
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples.tolist())
        recognizer.decode_stream(stream)
        ctc_text = stream.result.text or ""

        gt = item["ground_truth"]
        ctc_cer = compute_cer(strip_punc(gt), strip_punc(ctc_text))

        # ── Step 2: 置信度检测 ──
        low_conf = is_low_confidence(ctc_text, duration)
        llm_text = ctc_text
        llm_latency = 0

        if low_conf:
            # ── Step 3: LLM 纠错 ──
            llm_calls += 1
            t_start = time.perf_counter()

            prompt = f"""你是一个语音识别纠错助手。由于环境噪声，以下识别结果可能存在错字或缺失文字。

识别结果：{ctc_text}

请纠正错字、补充缺失的文字，使其成为通顺的语句。
- 保持原意不变
- 仅做必要的修正，不要改写正确的部分
- 直接输出修改后的文本，不要解释"""

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "你是一个精准的语音识别纠错工具。只修正错误，不改写正确内容。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=64,
                temperature=0.1,
                top_p=0.9,
                stop=["\n"],
            )

            llm_latency = (time.perf_counter() - t_start) * 1000
            llm_total_ms += llm_latency
            llm_text = response["choices"][0]["message"]["content"].strip()
            
            # 如果 LLM 返回空，回退到 CTC 结果
            if not llm_text:
                llm_text = ctc_text

        llm_cer = compute_cer(strip_punc(gt), strip_punc(llm_text))

        result = {
            "file": fname,
            "scenario": item["scenario"],
            "scenario_label": item.get("scenario_label", ""),
            "category": item["category"],
            "snr": item.get("snr", ""),
            "level": item.get("level", 0),
            "duration_s": duration,
            "ground_truth": gt,
            "ctc_text": ctc_text,
            "llm_text": llm_text,
            "ctc_cer": round(ctc_cer, 4),
            "llm_cer": round(llm_cer, 4),
            "improvement": round(ctc_cer - llm_cer, 4),
            "low_confidence": low_conf,
            "llm_latency_ms": round(llm_latency, 1),
        }
        results.append(result)

        # 进度
        elapsed_total = time.time() - start_time
        rate = (idx + 1) / elapsed_total if elapsed_total > 0 else 0
        eta = (total - idx - 1) / rate if rate > 0 else 0

        indicator = ""
        diff = ctc_cer - llm_cer
        if diff > 0.01:
            indicator = " ✅LLM改善"
        elif diff < -0.01:
            indicator = " ❌LLM倒退"
        elif low_conf:
            indicator = " ⚠️低置信-LLM未改善"

        print(f"[{idx + 1}/{total}] {fname[:45]:45s} "
              f"CTC={ctc_cer:.3f}→LLM={llm_cer:.3f} {indicator} "
              f"({elapsed_total:.0f}s, ETA {eta:.0f}s)")

    # ── 保存 ──
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    elapsed = time.time() - start_time
    total_ctc_cer = sum(r["ctc_cer"] for r in results) / len(results)
    total_llm_cer = sum(r["llm_cer"] for r in results) / len(results)
    improved = sum(1 for r in results if r["improvement"] > 0.01)
    worsened = sum(1 for r in results if r["improvement"] < -0.01)
    triggered = sum(1 for r in results if r["low_confidence"])

    print(f"\n{'=' * 60}")
    print(f"Benchmark 完成! {len(results)} 样本, {elapsed:.0f}s")
    print(f"{'=' * 60}")
    print(f"CTC only CER:         {total_ctc_cer:.4f}")
    print(f"CTC + LLM CER:        {total_llm_cer:.4f}")
    print(f"改善:                 {improved} ({improved/len(results)*100:.1f}%)")
    print(f"倒退:                 {worsened} ({worsened/len(results)*100:.1f}%)")
    print(f"LLM 触发次数:         {triggered} ({triggered/len(results)*100:.1f}%)")
    print(f"平均 LLM 延迟:        {llm_total_ms/max(llm_calls,1):.0f}ms")
    print(f"{'=' * 60}")

    # 按场景分析
    print(f"\n按场景: ")
    by_scene = defaultdict(list)
    for r in results:
        by_scene[r["scenario"]].append(r)
    for scene, items in sorted(by_scene.items()):
        c = sum(i["ctc_cer"] for i in items) / len(items)
        l = sum(i["llm_cer"] for i in items) / len(items)
        t = sum(1 for i in items if i["low_confidence"])
        print(f"  {scene:<12} CTC={c:.4f} → LLM={l:.4f} (触发{t}/{len(items)})")

    return results


if __name__ == "__main__":
    main()
