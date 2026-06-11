#!/usr/bin/env python3
"""
CTC Benchmark: all models, 528 samples.
"""

import json, time, os, re, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import sherpa_onnx
import soundfile as sf

BASE_DIR = Path("/home/barry/Projects/sherpa-post-asr")
NOISY_DIR = BASE_DIR / "audio_test" / "noisy"
MANIFEST_PATH = BASE_DIR / "audio_test" / "_benchmark_manifest.json"
RESULT_DIR = BASE_DIR / "audio_test" / "ctc_all"
RESULT_DIR.mkdir(exist_ok=True)

MODELS = {
    "A_offline_zh": {
        "label": "A: Zipformer-CTC offline zh (int8, 350MB)",
        "type": "offline_zipformer_ctc",
        "path": BASE_DIR / "models" / "zipformer-ctc-zh-int8" / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03" / "model.int8.onnx",
        "tokens": BASE_DIR / "models" / "zipformer-ctc-zh-int8" / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03" / "tokens.txt",
    },
    "B_large_zh": {
        "label": "B: Zipformer-CTC streaming large (int8, 155MB)",
        "type": "online_zipformer_ctc",
        "path": BASE_DIR / "models" / "zipformer-ctc-large" / "model.int8.onnx",
        "tokens": BASE_DIR / "models" / "zipformer-ctc-large" / "tokens.txt",
    },
    "C_xlarge_zh": {
        "label": "C: Zipformer-CTC streaming xlarge (int8, 728MB)",
        "type": "online_zipformer_ctc",
        "path": BASE_DIR / "models" / "zipformer-ctc-xlarge" / "model.int8.onnx",
        "tokens": BASE_DIR / "models" / "zipformer-ctc-xlarge" / "tokens.txt",
    },
    "D_small_zh": {
        "label": "D: Zipformer-CTC streaming small (int8, 25MB)",
        "type": "online_zipformer_ctc",
        "path": BASE_DIR / "models" / "zipformer-ctc-small" / "model.int8.onnx",
        "tokens": BASE_DIR / "models" / "zipformer-ctc-small" / "tokens.txt",
    },
    "E_multi_zh": {
        "label": "E: Zipformer-CTC multi-zh-hans (fp32, 251MB)",
        "type": "online_zipformer_ctc",
        "path": BASE_DIR / "models" / "zipformer-ctc-multi" / "ctc-epoch-20-avg-1-chunk-16-left-128.onnx",
        "tokens": BASE_DIR / "models" / "zipformer-ctc-multi" / "tokens.txt",
    },
    "F_parakeet_en": {
        "label": "F: NeMo Parakeet TDT CTC 110M (en, int8)",
        "type": "offline_nemo_ctc",
        "path": BASE_DIR / "models" / "nemo-parakeet" / "model.int8.onnx",
        "tokens": BASE_DIR / "models" / "nemo-parakeet" / "tokens.txt",
    },
    "H_wenet_zh": {
        "label": "H: WeNet-Wenetspeech CTC (zh, int8, 127MB)",
        "type": "offline_wenet_ctc",
        "path": BASE_DIR / "models" / "wenet-ctc" / "model.int8.onnx",
        "tokens": BASE_DIR / "models" / "wenet-ctc" / "tokens.txt",
    },
}

ENGLISH_ONLY = {"F_parakeet_en"}


def compute_cer(ref, hyp):
    ref = ref.replace(" ", "").replace("\u3000", "")
    hyp = hyp.replace(" ", "").replace("\u3000", "")
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


def normalize_text(text):
    text = re.sub(r"[^\u4e00-\u9fff\w]", "", text)
    return text.lower().replace(" ", "")


def create_recognizer(mid, cfg):
    mp = str(cfg["path"])
    tp = str(cfg["tokens"])
    t = cfg["type"]
    if t == "offline_zipformer_ctc":
        return sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=mp, tokens=tp, num_threads=4, sample_rate=16000, decoding_method="greedy_search")
    elif t == "offline_nemo_ctc":
        return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=mp, tokens=tp, num_threads=4, sample_rate=16000, decoding_method="greedy_search")
    elif t == "offline_wenet_ctc":
        return sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
            model=mp, tokens=tp, num_threads=4, sample_rate=16000, decoding_method="greedy_search")
    elif t == "online_zipformer_ctc":
        return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=tp, model=mp, num_threads=4, sample_rate=16000, decoding_method="greedy_search")


def decode_offline(rec, samples):
    s = rec.create_stream()
    s.accept_waveform(16000, samples.tolist())
    rec.decode_stream(s)
    return s.result.text or ""


def decode_online(rec, samples):
    s = rec.create_stream()
    s.accept_waveform(16000, samples.tolist())
    tail = np.zeros(int(0.66 * 16000), dtype=np.float32)
    s.accept_waveform(16000, tail.tolist())
    s.input_finished()
    while True:
        if rec.is_ready(s):
            rec.decode_stream(s)
        else:
            break
    return rec.get_result(s) or ""


def test_model(mid, cfg, manifest):
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")
    is_en = mid in ENGLISH_ONLY
    rf = RESULT_DIR / f"{mid}.json"
    pf = RESULT_DIR / f"{mid}_progress.json"

    results = []
    processed = set()
    if pf.exists():
        with open(pf) as f:
            prog = json.load(f)
            results = prog.get("results", [])
            processed = set(r["file"] for r in results)
        print(f"  恢复进度: {len(processed)}/{len(manifest)}")
    elif rf.exists():
        with open(rf) as f:
            results = json.load(f)
            processed = set(r["file"] for r in results)
        print(f"  已有结果: {len(processed)}/{len(manifest)}")
        return results

    if len(processed) >= len(manifest):
        print("  OK")
        return results

    t0 = time.time()
    rec = create_recognizer(mid, cfg)
    print(f"  加载: {time.time()-t0:.1f}s")
    is_online = cfg["type"].startswith("online_")

    total = len(manifest)
    start_time = time.time()
    batch_save = 50

    for idx, item in enumerate(manifest):
        fname = item["file"]
        if fname in processed:
            continue
        gt = item["ground_truth"]

        if is_en and not re.search(r"[a-zA-Z]", gt):
            results.append({**item, f"{mid}_text": "(skip)", f"{mid}_cer": -1.0, "latency_ms": 0})
            continue

        ap = str(NOISY_DIR / fname)
        if not os.path.exists(ap):
            print(f"  ? 文件缺失: {fname}")
            continue
        try:
            samples, sr = sf.read(ap)
            if sr != 16000:
                import librosa
                samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        except Exception as e:
            print(f"  X 读取失败: {fname} ({e})")
            continue

        ts = time.perf_counter()
        try:
            text = decode_online(rec, samples) if is_online else decode_offline(rec, samples)
        except Exception as e:
            print(f"  X 解码失败: {fname} ({e})")
            text = ""
        el = (time.perf_counter() - ts) * 1000

        cer = compute_cer(normalize_text(gt), normalize_text(text)) if not is_en else -1
        results.append({**item, f"{mid}_text": text, f"{mid}_cer": round(cer, 4), "latency_ms": round(el, 1)})

        done = len(results)
        et = time.time() - start_time
        rate2 = done / et if et > 0 else 0
        eta2 = (total - done) / rate2 if rate2 > 0 else 0
        bar = "\u2588" * int(done / total * 20) + "\u2591" * (20 - int(done / total * 20))
        status = " OK" if cer >= 0 and cer <= 0.05 else " WARN" if cer > 0.05 and cer <= 0.15 else " BAD" if cer > 0.15 else ""
        print(f"  [{done}/{total}] {fname[:36]:36s} CER={cer:.3f}{status} ({et:.0f}s/{eta2:.0f}s)")

        if len(results) % batch_save == 0 and len(results) > 0:
            with open(pf, "w") as f:
                json.dump({"results": results, "timestamp": time.time()}, f, ensure_ascii=False)
            if len(results) % (batch_save * 2) == 0:
                gc.collect()

    with open(rf, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pf.unlink(missing_ok=True)

    scored = [r for r in results if r[f"{mid}_cer"] >= 0]
    if scored:
        avg = sum(r[f"{mid}_cer"] for r in scored) / len(scored)
        lats = [r["latency_ms"] for r in results if r.get("latency_ms", 0) > 0]
        avg_lat = sum(lats) / len(lats) if lats else 0
        print(f"\n  CER={avg:.4f}  Lat={avg_lat:.0f}ms  ({len(scored)}/{len(results)} samples)")
    return results


def main():
    print("=" * 60)
    print("  Full CTC Benchmark")
    print("=" * 60)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"  Samples: {len(manifest)}")

    all_results = {}
    for mid, cfg in MODELS.items():
        try:
            all_results[mid] = test_model(mid, cfg, manifest)
        except Exception as e:
            print(f"\n  FAILED {mid}: {e}")
            import traceback
            traceback.print_exc()
            all_results[mid] = []
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    rankings = []
    for mid, results in all_results.items():
        scored = [r for r in results if r.get(f"{mid}_cer", -1) >= 0]
        if scored:
            avg = sum(r[f"{mid}_cer"] for r in scored) / len(scored)
            sz = os.path.getsize(MODELS[mid]["path"]) / (1024 * 1024) if os.path.exists(MODELS[mid]["path"]) else 0
            rankings.append((avg, mid, MODELS[mid]["label"], sz))
            lats = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]
            avg_lat = sum(lats) / len(lats) if lats else 0
            print(f"  {MODELS[mid]['label']}")
            print(f"    CER={avg:.4f}  Lat={avg_lat:.0f}ms  Size={sz:.0f}MB")
        else:
            en_texts = [r.get(f"{mid}_text", "") for r in results if r.get(f"{mid}_text") and r.get(f"{mid}_text") != "(skip)"]
            print(f"  {MODELS[mid]['label']}")
            print(f"    English samples: {len(en_texts)}")

    if rankings:
        rankings.sort(key=lambda x: x[0])
        print(f"\n  BEST: {rankings[0][2]} (CER={rankings[0][0]:.4f})")


if __name__ == "__main__":
    main()
