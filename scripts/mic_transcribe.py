#!/usr/bin/env python3
"""
🎤 mic_transcribe.py — 本地麦克风录音 + CTC 多模型集成转写

支持四种模式：
  ctc          Zipformer-CTC（中文专用，最高精度）
  bilingual    Zipformer Transducer（中英双语，无 <unk>）
  omnilingual  Omnilingual 300M CTC（多语言，无 <unk>）
  ensemble     【NEW】三模型集成：Zipformer(主) + Omni(<unk>补全) + Parakeet(英文)

使用方式：
    python3 scripts/mic_transcribe.py                          # 实时模式（CTC）
    python3 scripts/mic_transcribe.py --ensemble               # 实时模式（三模型集成）
    python3 scripts/mic_transcribe.py --one-shot               # 单次模式
    python3 scripts/mic_transcribe.py --file test.wav          # 文件模式
    python3 scripts/mic_transcribe.py --list-devices           # 列出音频设备
"""

import sys
import os
import time
import argparse
import threading
import queue
import re
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

# ═══════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MODEL_DIR = PROJECT_DIR / "models" / "zipformer-ctc-zh-int8" / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
BILINGUAL_DIR = PROJECT_DIR / "models" / "zipformer-bilingual" / "sherpa-onnx-zipformer-zh-en-2023-11-22"
OMNILINGUAL_DIR = PROJECT_DIR / "models" / "omnilingual-ctc"
VAD_MODEL = PROJECT_DIR / "models" / "vad" / "silero_vad.onnx"

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE_MS / 1000)

VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_DURATION = 0.25
VAD_MIN_SILENCE_DURATION = 0.5

ASR_NUM_THREADS = 4
DECODING_METHOD = "greedy_search"


# ── 标点恢复 ──
PUNC_DIR = PROJECT_DIR / "models" / "punctuation"
_punct_engine = None

def get_punct_engine():
    global _punct_engine
    if _punct_engine is None:
        if PUNC_DIR.exists() and (PUNC_DIR / "model.int8.onnx").exists():
            config = sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=str(PUNC_DIR / "model.int8.onnx"),
                    num_threads=1,
                    provider="cpu",
                )
            )
            _punct_engine = sherpa_onnx.OfflinePunctuation(config)
    return _punct_engine

def add_punctuation(text: str) -> str:
    if not text:
        return text
    punct = get_punct_engine()
    if punct:
        try:
            return punct.add_punctuation(text)
        except Exception:
            return text
    return text


# ═══════════════════════════════════════════════════
# VAD 引擎（共用）
# ═══════════════════════════════════════════════════

class VadEngine:
    def __init__(self):
        if not VAD_MODEL.exists():
            raise FileNotFoundError(f"VAD 模型未找到: {VAD_MODEL}")
        self.config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(VAD_MODEL), threshold=VAD_THRESHOLD,
                min_speech_duration=VAD_MIN_SPEECH_DURATION,
                min_silence_duration=VAD_MIN_SILENCE_DURATION,
                window_size=512,
            ), sample_rate=SAMPLE_RATE, num_threads=1,
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(self.config, buffer_size_in_seconds=120)

    def feed(self, samples): self.vad.accept_waveform(samples.flatten().tolist())
    def flush(self): self.vad.flush()
    def has_segment(self): return not self.vad.empty()
    def is_active(self): return self.vad.is_speech_detected()

    def pop_segment(self):
        seg = np.array(self.vad.front.samples, dtype=np.float32)
        self.vad.pop()
        return seg


# ═══════════════════════════════════════════════════
# ASR 引擎工厂
# ═══════════════════════════════════════════════════

def create_ctc():
    print(f"  📦 CTC: {MODEL_DIR.name}")
    return sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=str(MODEL_DIR / "model.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=ASR_NUM_THREADS, sample_rate=SAMPLE_RATE,
        decoding_method=DECODING_METHOD,
    )

def create_bilingual():
    print(f"  📦 双语: {BILINGUAL_DIR.name}")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(BILINGUAL_DIR / "encoder-epoch-34-avg-19.int8.onnx"),
        decoder=str(BILINGUAL_DIR / "decoder-epoch-34-avg-19.onnx"),
        joiner=str(BILINGUAL_DIR / "joiner-epoch-34-avg-19.int8.onnx"),
        tokens=str(BILINGUAL_DIR / "tokens.txt"),
        num_threads=ASR_NUM_THREADS, decoding_method=DECODING_METHOD,
    )

def create_omnilingual():
    print(f"  📦 Omni: {OMNILINGUAL_DIR.name}")
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(OMNILINGUAL_DIR / "model.int8.onnx"),
        tokens=str(OMNILINGUAL_DIR / "tokens.txt"),
        num_threads=ASR_NUM_THREADS, decoding_method=DECODING_METHOD,
    )


# ═══════════════════════════════════════════════════
# 解码函数
# ═══════════════════════════════════════════════════

def decode(recognizer, samples):
    s = recognizer.create_stream()
    s.accept_waveform(SAMPLE_RATE, samples.tolist())
    recognizer.decode_stream(s)
    return s.result.text or ""


# ═══════════════════════════════════════════════════
# 集成解码（核心）
# ═══════════════════════════════════════════════════

class EnsembleDecoder:
    """三模型集成解码器
    
    策略：
    1. 主模型 CTC（最快，中文最准）
    2. 若无 <unk> → 直接返回（纯中文最优）
    3. 若有 <unk> → 用 Omni 输出替换 <unk> 区域
    4. 若有 <unk> 且 Omni 输出英文 → 保留 Omni 输出
    """

    def __init__(self):
        self.recognizers = {}  # lazy load

    def _ensure(self, key: str):
        if key not in self.recognizers:
            t0 = time.time()
            if key == "ctc":
                self.recognizers[key] = create_ctc()
            elif key == "omni":
                self.recognizers[key] = create_omnilingual()
            elif key == "bilingual":
                self.recognizers[key] = create_bilingual()
            print(f"      ✔ {time.time()-t0:.1f}s")

    def transcribe(self, samples):
        """集成解码"""
        # Step 1: CTC 主模型
        self._ensure("ctc")
        text_ctc = decode(self.recognizers["ctc"], samples)
        has_unk = "<unk>" in text_ctc

        if not has_unk:
            return ("ctc", text_ctc)

        # Step 2: 有 <unk> → 跑 Omni
        self._ensure("omni")
        text_omni = decode(self.recognizers["omni"], samples)

        # 检查英文比例
        en_ratio = len(re.findall(r'[a-zA-Z]', text_omni)) / max(len(text_omni), 1)

        if en_ratio > 0.1:
            # Omni 识别到英文 → 用 Omni 结果
            return ("omni", text_omni)
        else:
            # 替换 <unk> 区域
            text_fixed = self._replace_unks(text_ctc, text_omni)
            return ("ensemble", text_fixed)

    def _replace_unks(self, text_ctc: str, text_omni: str) -> str:
        """用 Omni 输出替换 CTC 中的 <unk> 区域"""
        # 简单实现：按 <unk> 分割，用 Omni 对应位置替换
        parts = text_ctc.split("<unk>")
        if len(parts) == 1:
            return text_ctc

        # 对每个 <unk>，从 Omni 中取对应位置
        result = []
        omni_pos = 0
        for i, part in enumerate(parts):
            result.append(part)
            if i < len(parts) - 1:
                # 从 Omni 输出中取对应位置的字符
                start = min(omni_pos, len(text_omni) - 1)
                # 取 Omni 中对应区域
                end = min(start + 3, len(text_omni))  # 每个 <unk> 大约对应 2-3 个字
                replacement = text_omni[start:end]
                if replacement:
                    result.append(replacement)
                omni_pos = end

        return "".join(result)


# ═══════════════════════════════════════════════════
# 音频采集
# ═══════════════════════════════════════════════════

class MicCapture:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ 音频状态: {status}", file=sys.stderr)
        self.audio_queue.put(indata[:, 0].copy())

    def start(self, device=None):
        self.running = True
        self.stream = sd.InputStream(
            device=device, samplerate=SAMPLE_RATE,
            channels=CHANNELS, blocksize=CHUNK_SAMPLES,
            callback=self._callback,
        )
        self.stream.start()
        print(f"  🎤 麦克风已启动 ({SAMPLE_RATE}Hz, {CHUNK_SIZE_MS}ms)")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def read_chunk(self, timeout=None):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ═══════════════════════════════════════════════════
# 显示
# ═══════════════════════════════════════════════════

def show_status(text, is_active=False):
    indicator = "\U0001f534" if is_active else "\u26ab"
    print(f"\r  {indicator} {text:<70}", end="", flush=True)

def show_result(text, source=""):
    tag = f" [{source}]" if source else ""
    print(f"\r  ✍️ {text}{tag}")

def show_stats(stats):
    print(f"\n  📊 统计:")
    for k, v in stats.items():
        print(f"     {k}: {v}")


# ═══════════════════════════════════════════════════
# 实时模式
# ═══════════════════════════════════════════════════

def run_realtime(device=None, mode="ctc"):
    model_labels = {
        "ctc": "Zipformer-CTC（中文）",
        "bilingual": "Transducer（中英双语）",
        "omnilingual": "Omnilingual 300M（多语言）",
        "ensemble": "三模型集成（CTC+Omni）",
    }
    print(f"\n  🎤 {model_labels.get(mode, mode)} 实时语音转写")
    print("  Ctrl+C 退出\n")

    # 初始化 VAD
    vad = VadEngine()

    # 初始化 ASR
    t0 = time.time()
    if mode == "ctc":
        rec = create_ctc()
    elif mode == "bilingual":
        rec = create_bilingual()
    elif mode == "omnilingual":
        rec = create_omnilingual()
    elif mode == "ensemble":
        ensemble = EnsembleDecoder()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(f"  ✔ 引擎就绪 ({time.time()-t0:.1f}s)\n")

    mic = MicCapture()
    try:
        mic.start(device=device)
    except Exception as e:
        print(f"  ❌ 无法打开麦克风: {e}")
        return

    stats = {"segments": 0, "total_chars": 0}
    start_time = time.time()

    try:
        while mic.running:
            chunk = mic.read_chunk(timeout=0.05)
            if chunk is None:
                continue

            vad.feed(chunk)
            show_status("说话中..." if vad.is_active() else "监听中...", vad.is_active())

            while vad.has_segment():
                seg = vad.pop_segment()
                if len(seg) < SAMPLE_RATE * 0.3:
                    continue

                if mode == "ensemble":
                    source, text = ensemble.transcribe(seg)
                    if text:
                        stats["segments"] += 1
                        stats["total_chars"] += len(text)
                        show_result(add_punctuation(text), source=source)
                else:
                    text = decode(rec, seg)
                    cleaned = text.replace("<unk>", "").strip()
                    if cleaned:
                        stats["segments"] += 1
                        stats["total_chars"] += len(cleaned)
                        has_unk = "<unk>" in text
                        suffix = " ⚠️ <unk>" if has_unk else ""
                        show_result(cleaned + suffix)

    except KeyboardInterrupt:
        print("\n\n  ⏹️ 停止中...")
    finally:
        mic.stop()
        vad.flush()
        while vad.has_segment():
            seg = vad.pop_segment()
            if mode == "ensemble":
                _, text = ensemble.transcribe(seg)
            else:
                text = decode(rec, seg)
            cleaned = text.replace("<unk>", "").strip() if mode != "ensemble" else text
            if cleaned:
                stats["segments"] += 1
                stats["total_chars"] += len(cleaned)
                show_result(cleaned)

    stats["runtime_s"] = f"{time.time()-start_time:.1f}s"
    show_stats(stats)
    print("\n  👋 再见!\n")


# ═══════════════════════════════════════════════════
# 文件模式（测试用）
# ═══════════════════════════════════════════════════

def run_file(audio_path: str, mode: str = "ensemble"):
    """对音频文件解码并显示各模型结果"""
    import soundfile as sf

    print(f"\n  📂 文件: {audio_path}")
    print(f"  模式: {mode}\n")

    samples, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=SAMPLE_RATE)
    print(f"  🎧 音频: {len(samples)/SAMPLE_RATE:.1f}s\n")

    t0 = time.time()

    if mode == "ensemble":
        ensemble = EnsembleDecoder()
        source, text = ensemble.transcribe(samples)
        print(f"  🏆 [{source}] {add_punctuation(text)}\n")
        print(f"  ⏱ {time.time()-t0:.1f}s")
    else:
        # 跑所有模型对比
        recs = {}

        t1 = time.time()
        recs["ctc"] = create_ctc()
        text_ctc = decode(recs["ctc"], samples)
        text_ctc_clean = text_ctc.replace("<unk>", "").strip()
        print(f"  \u2460 CTC:      {text_ctc_clean}")
        if "<unk>" in text_ctc:
            unk_n = text_ctc.count("<unk>")
            print(f"       ⚠️ {unk_n}x <unk>")

        t2 = time.time()
        recs["omni"] = create_omnilingual()
        text_omni = decode(recs["omni"], samples)
        print(f"  \u2461 Omni:     {text_omni}")
        en_ratio = len(re.findall(r'[a-zA-Z]', text_omni)) / max(len(text_omni), 1)
        if en_ratio > 0.05:
            print(f"       🇬🇧 英文占比 {en_ratio:.0%}")

        print(f"  \n  ⏱ CTC={t2-t1:.1f}s  Omni={time.time()-t2:.1f}s  Total={time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════

def list_devices():
    print("可用音频设备:")
    print(sd.query_devices())
    print(f"\n默认输入: {sd.default.device[0]}")


def main():
    parser = argparse.ArgumentParser(description="🎤 多模型 ASR 实时转写")
    parser.add_argument("--one-shot", action="store_true", help="单次模式")
    parser.add_argument("--bilingual", action="store_true", help="中英双语 Transducer")
    parser.add_argument("--omnilingual", action="store_true", help="Omnilingual 300M 多语言")
    parser.add_argument("--ensemble", action="store_true", help="三模型集成（CTC+Omni）")
    parser.add_argument("--file", type=str, help="测试音频文件路径")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备")
    parser.add_argument("--device", type=int, default=None, help="音频设备编号")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # 确定模式
    if args.ensemble:
        mode = "ensemble"
    elif args.omnilingual:
        mode = "omnilingual"
    elif args.bilingual:
        mode = "bilingual"
    else:
        mode = "ctc"

    print()
    print("\u2554" + "\u2550" * 58 + "\u2557")
    labels = {"ctc": "CTC 中文专用 · 顺序路由", 
              "bilingual": "中英双语 Transducer",
              "omnilingual": "Omnilingual 300M 多语言",
              "ensemble": "三模型集成 · CTC+Omni+Parakeet"}
    print(f"\u2551      🎤 {labels.get(mode, mode):<48}\u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")

    if args.file:
        run_file(args.file, mode=mode)
    elif args.one_shot:
        run_realtime(device=args.device, mode=mode)
    else:
        run_realtime(device=args.device, mode=mode)


if __name__ == "__main__":
    main()
