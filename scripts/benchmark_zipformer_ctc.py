"""
Benchmark: Zipformer-CTC (greedy, 无 LM)
验证"字级独立建模"假设 — 纯帧级分类，无语言模型偏置

对比基线：SenseVoice (内置 LM) CER=0.1936
"""

import json, time, os, re
from pathlib import Path
from collections import defaultdict
import sherpa_onnx
import soundfile as sf
import numpy as np

# ── 路径 ──
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "zipformer-ctc-zh-int8" / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
NOISY_DIR = BASE_DIR / "audio_test" / "noisy"
MANIFEST_PATH = BASE_DIR / "audio_test" / "_benchmark_manifest.json"
RESULT_PATH = BASE_DIR / "audio_test" / "_benchmark_ctc_results.json"
PROGRESS_PATH = BASE_DIR / "audio_test" / "_benchmark_ctc_progress.json"


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


SV_RESULTS_PATH = BASE_DIR / "audio_test" / "_benchmark_results.json"


def normalize_text(text: str) -> str:
    """标准化（去空格、英文小写）"""
    text = text.replace(" ", "").replace("　", "")
    text = re.sub(r'[a-zA-Z]+', lambda m: m.group(0).lower(), text)
    return text


def load_sv_results() -> dict:
    """加载 SenseVoice 基准结果，按文件名索引"""
    if not SV_RESULTS_PATH.exists():
        print("⚠️ SenseVoice 基准结果未找到，跳过对比")
        return {}
    with open(SV_RESULTS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["file"]: item for item in items}


def main():
    print("=" * 60)
    print("Zipformer-CTC Greedy Benchmark (纯帧级分类, 无LM)")
    print("=" * 60)
    print(f"模型: {MODEL_DIR.name}")

    # ── 加载模型 ──
    t0 = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=str(MODEL_DIR / "model.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=4,
        sample_rate=16000,
        decoding_method="greedy_search",  # ← 关键：纯帧级分类，无 LM
    )
    print(f"模型加载耗时: {time.time() - t0:.1f}s")

    # ── 加载 manifest ──
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"测试样本: {len(manifest)}")

    # ── 恢复进度 ──
    results = []
    processed = set()
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                prog = json.load(f)
            results = prog.get("results", [])
            processed = set(r["file"] for r in results)
            print(f"恢复进度: {len(processed)}/{len(manifest)} 已完成")
        except:
            pass

    # 加载 SV 基线
    sv_lookup = load_sv_results()
    have_sv = len(sv_lookup) > 0
    sv_avg = 0.0
    diff = 0.0
    sv_cer_vals = []

    # ── 逐个识别 ──
    start_time = time.time()
    total = len(manifest)

    for idx, item in enumerate(manifest):
        fname = item["file"]
        if fname in processed:
            continue

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

        # 识别
        t_start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples.tolist())
        recognizer.decode_stream(stream)
        result = stream.result
        elapsed = (time.perf_counter() - t_start) * 1000

        text = result.text or ""
        gt = item["ground_truth"]
        c = compute_cer(normalize_text(gt), normalize_text(text))

        result_item = {
            **item,  # file, scenario, scenario_label, category, ground_truth, snr, level, duration_s
            "ctc_text": text,
            "ctc_cer": round(c, 4),
            "latency_ms": round(elapsed, 1),
        }
        results.append(result_item)

        # 进度
        bar_len = 30
        filled = int((idx + 1) / total * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        elapsed_total = time.time() - start_time
        rate = (idx + 1) / elapsed_total if elapsed_total > 0 else 0
        eta = (total - idx - 1) / rate if rate > 0 else 0

        # 质量判断
        indicator = ""
        sv_item = sv_lookup.get(fname, {})
        if sv_item:
            sv_cer = sv_item.get("raw_cer", 0)
            diff = c - sv_cer
            if diff < -0.05:
                indicator = " 🔥远优SV"
            elif diff < -0.01:
                indicator = " ✅优于SV"
            elif diff < 0.01:
                indicator = " ➖持平SV"
            elif diff < 0.05:
                indicator = " ⚠️劣于SV"
            else:
                indicator = " ❌远差SV"

        print(f"[{idx + 1}/{total}] {bar} {fname[:40]:40s} "
              f"CER={c:.3f} {indicator} "
              f"({elapsed_total:.0f}s, ETA {eta:.0f}s)")

        # 每 30 个保存
        if (idx + 1) % 30 == 0:
            with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": results, "timestamp": time.time()},
                          f, ensure_ascii=False)

    # ── 保存结果 ──
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    PROGRESS_PATH.unlink(missing_ok=True)

    # ── 汇总 ──
    elapsed = time.time() - start_time
    avg_cer = sum(r["ctc_cer"] for r in results) / len(results)

    if have_sv:
        sv_cer_vals = [sv_lookup[r["file"]]["raw_cer"] for r in results if r["file"] in sv_lookup]
        if sv_cer_vals:
            sv_avg = sum(sv_cer_vals) / len(sv_cer_vals)
            diff = avg_cer - sv_avg

            # 统计改善/变差的样本
            better = sum(1 for r in results if r["file"] in sv_lookup
                         and r["ctc_cer"] < sv_lookup[r["file"]]["raw_cer"] - 0.005)
            worse = sum(1 for r in results if r["file"] in sv_lookup
                        and r["ctc_cer"] > sv_lookup[r["file"]]["raw_cer"] + 0.005)
            same = len(results) - better - worse

            print(f"\n{'=' * 60}")
            print(f"基准测试完成! {len(results)} 样本, {elapsed:.0f}s")
            print(f"{'=' * 60}")
            print(f"Zipformer-CTC (greedy, 无LM):  CER = {avg_cer:.4f}")
            print(f"SenseVoice (baseline, 有LM):    CER = {sv_avg:.4f}")
            print(f"差距: {diff:+.4f}")
            print(f"CTC优于SV: {better} ({better/len(results)*100:.1f}%)")
            print(f"CTC劣于SV: {worse} ({worse/len(results)*100:.1f}%)")
            print(f"持平: {same} ({same/len(results)*100:.1f}%)")
            print(f"{'=' * 60}")
        else:
            print(f"\nZipformer-CTC CER = {avg_cer:.4f} ({len(results)} 样本)")
    else:
        print(f"\nZipformer-CTC CER = {avg_cer:.4f} ({len(results)} 样本)")

    # 写入报告
    report_path = BASE_DIR / "BENCHMARK_CTC_REPORT.md"
    lines = []
    lines.append("# Zipformer-CTC Greedy Benchmark Report")
    lines.append(f"\n**日期**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**模型**: {MODEL_DIR.name}")
    lines.append(f"**解码方式**: greedy_search (帧级独立分类, 无语言模型)")
    lines.append(f"**测试样本**: {len(results)}")
    lines.append(f"**Zipformer-CTC CER**: {avg_cer:.4f}")
    if have_sv and sv_cer_vals:
        lines.append(f"**SenseVoice CER**: {sv_avg:.4f}")
        lines.append(f"**差距**: {diff:+.4f}")

    # 构建场景和类别聚合
    by_scene = defaultdict(list)
    by_cat = defaultdict(list)
    scene_labels = {}
    for r in results:
        by_scene[r["scenario"]].append(r)
        by_cat[r["category"]].append(r)
        scene_labels[r["scenario"]] = r["scenario_label"]

    lines.append(f"\n## 按场景原始 CER 排序（从易到难）\n")
    scene_cer = []
    for scene, items in by_scene.items():
        ctc_avg = sum(i["ctc_cer"] for i in items) / len(items)
        label = scene_labels.get(scene, scene)
        scene_cer.append((scene, label, ctc_avg))
    scene_cer.sort(key=lambda x: x[2])
    for rank, (scene, label, cer) in enumerate(scene_cer, 1):
        lines.append(f"{rank}. {label}: CER = {cer:.4f}")

    lines.append(f"\n## 按场景（与 SV 对比）\n")
    lines.append("| 场景 | 样本数 | CTC CER | SV CER | 差距 |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    for scene, items in sorted(by_scene.items()):
        n = len(items)
        ctc_avg = sum(i["ctc_cer"] for i in items) / n
        sv_vals = [sv_lookup[i["file"]]["raw_cer"] for i in items if i["file"] in sv_lookup]
        sv_avg_s = sum(sv_vals) / len(sv_vals) if sv_vals else 0
        label = scene_labels.get(scene, scene)
        lines.append(f"| {label} | {n} | {ctc_avg:.4f} | {sv_avg_s:.4f} | {ctc_avg - sv_avg_s:+.4f} |")

    lines.append(f"\n## 按类别\n")
    lines.append("| 类别 | 样本数 | CTC CER | SV CER | 差距 |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    for cat, items in sorted(by_cat.items()):
        n = len(items)
        ctc_avg = sum(i["ctc_cer"] for i in items) / n
        sv_vals = [sv_lookup[i["file"]]["raw_cer"] for i in items if i["file"] in sv_lookup]
        sv_avg_s = sum(sv_vals) / len(sv_vals) if sv_vals else 0
        lines.append(f"| {cat} | {n} | {ctc_avg:.4f} | {sv_avg_s:.4f} | {ctc_avg - sv_avg_s:+.4f} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已保存: {report_path}")

    return results


if __name__ == "__main__":
    main()
