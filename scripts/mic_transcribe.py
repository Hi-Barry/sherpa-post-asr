#!/usr/bin/env python3
"""
🎤 mic_transcribe.py — 本地麦克风录音 + Zipformer-CTC 实时转写

基于"字级独立建模"（帧级 CTC 分类，无语言模型偏置），
使用本地麦克风实时捕获语音并通过 Zipformer-CTC 转写为文字。

使用方式：
    python3 scripts/mic_transcribe.py              # 实时模式（持续监听）
    python3 scripts/mic_transcribe.py --one-shot   # 单次模式（回车开始/结束）
    python3 scripts/mic_transcribe.py --list-devices  # 列出音频设备

依赖：
    pip install sounddevice sherpa-onnx numpy
"""

import sys
import os
import time
import argparse
import threading
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

# ═══════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════

# 项目路径（假设脚本在 scripts/ 目录下）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 模型路径
MODEL_DIR = PROJECT_DIR / "models" / "zipformer-ctc-zh-int8" / "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
VAD_MODEL = PROJECT_DIR / "models" / "vad" / "silero_vad.onnx"

# 音频参数
SAMPLE_RATE = 16000          # sherpa-onnx 要求的采样率
CHANNELS = 1                 # 单声道
CHUNK_SIZE_MS = 100          # 每次读取的音频长度（毫秒），与 Android 版一致
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE_MS / 1000)  # 每次读取的采样点数

# VAD 参数
VAD_THRESHOLD = 0.5          # Silero VAD 阈值（0.0-1.0），越高越严格
VAD_MIN_SPEECH_DURATION = 0.25   # 最短语音段（秒），过滤短促噪声
VAD_MIN_SILENCE_DURATION = 0.5   # 最长静音（秒），超过此长度认为语音结束

# ASR 参数
ASR_NUM_THREADS = 4
DECODING_METHOD = "greedy_search"  # ← 关键：纯帧级分类，无 LM


# ═══════════════════════════════════════════════════
# 引擎初始化
# ═══════════════════════════════════════════════════

class AsrEngine:
    """Zipformer-CTC ASR + Silero VAD 引擎"""

    def __init__(self):
        # 加载 VAD
        if not VAD_MODEL.exists():
            raise FileNotFoundError(
                f"VAD 模型未找到: {VAD_MODEL}\n"
                f"请下载: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
            )
        self.vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(VAD_MODEL),
                threshold=VAD_THRESHOLD,
                min_speech_duration=VAD_MIN_SPEECH_DURATION,
                min_silence_duration=VAD_MIN_SILENCE_DURATION,
                window_size=512,
            ),
            sample_rate=SAMPLE_RATE,
            num_threads=1,
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=120)

        # 加载 Zipformer-CTC
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"CTC 模型目录未找到: {MODEL_DIR}\n"
                f"请下载: cd {MODEL_DIR.parent} && "
                f"wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                f"sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2 && "
                f"tar xf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2"
            )

        print(f"  📦 加载 CTC 模型: {MODEL_DIR.name}")
        t0 = time.time()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=str(MODEL_DIR / "model.int8.onnx"),
            tokens=str(MODEL_DIR / "tokens.txt"),
            num_threads=ASR_NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            decoding_method=DECODING_METHOD,  # ← 纯帧级分类
        )
        print(f"  ✅ 模型加载完成 ({time.time() - t0:.1f}s)")

    def feed_audio(self, samples: np.ndarray):
        """喂入音频到 VAD"""
        samples = samples.flatten()  # 确保一维
        self.vad.accept_waveform(samples.tolist())

    def flush_vad(self):
        """刷新 VAD 缓冲区"""
        self.vad.flush()

    def has_segment(self) -> bool:
        """是否有完成的语音段"""
        return not self.vad.empty()

    def pop_segment(self) -> np.ndarray:
        """取出下一个语音段"""
        seg_obj = self.vad.front
        seg = np.array(seg_obj.samples, dtype=np.float32)
        self.vad.pop()
        return seg

    def transcribe(self, samples: np.ndarray) -> str:
        """对音频段做 CTC 识别（帧级独立分类）"""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples.tolist())
        self.recognizer.decode_stream(stream)
        return stream.result.text

    @staticmethod
    def clean_text(text: str) -> tuple[str, float]:
        """清理输出文本：移除 <unk>，返回 (清理后文本, <unk>占比)"""
        if not text:
            return "", 0.0
        total = len(text.split())
        unk_count = text.count("<unk>")
        cleaned = text.replace("<unk>", "").strip()
        # 合并多余空格
        import re
        cleaned = re.sub(r'\s+', '', cleaned)
        unk_ratio = unk_count / max(total, 1)
        return cleaned, unk_ratio

    def is_speech_active(self) -> bool:
        """当前是否处于语音活动状态"""
        return self.vad.is_speech_detected()


# ═══════════════════════════════════════════════════
# 音频采集
# ═══════════════════════════════════════════════════

class MicCapture:
    """麦克风音频采集"""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        """sounddevice 回调：每次 CHUNK_SAMPLES 个采样点"""
        if status:
            print(f"⚠️ 音频状态: {status}", file=sys.stderr)
        # indata shape: (frames, channels)，取单声道
        self.audio_queue.put(indata[:, 0].copy())

    def start(self, device=None):
        """启动麦克风采集"""
        self.running = True
        self.stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=CHUNK_SAMPLES,
            callback=self._callback,
        )
        self.stream.start()
        print(f"  🎤 麦克风已启动 (采样率={SAMPLE_RATE}Hz, 块大小={CHUNK_SIZE_MS}ms)")

    def stop(self):
        """停止麦克风采集"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def read_chunk(self, timeout: float = None) -> np.ndarray | None:
        """读取一个音频块（阻塞）"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ═══════════════════════════════════════════════════
# 显示工具
# ═══════════════════════════════════════════════════

class Display:
    """终端显示"""

    @staticmethod
    def clear_line():
        """清除当前行"""
        print("\r" + " " * 80 + "\r", end="", flush=True)

    @staticmethod
    def show_status(text: str, is_active: bool = False):
        """显示状态"""
        indicator = "🔴" if is_active else "⚫"
        print(f"\r  {indicator} {text:<70}", end="", flush=True)

    @staticmethod
    def show_result(text: str):
        """显示转写结果"""
        print(f"\r  ✍️ {text}")

    @staticmethod
    def show_stats(stats: dict):
        """显示统计信息"""
        print(f"\n  📊 会话统计:")
        print(f"     转写段数: {stats.get('segments', 0)}")
        print(f"     总字符数: {stats.get('total_chars', 0)}")
        print(f"     运行时间: {stats.get('runtime_s', 0):.1f}s")


# ═══════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════

def list_devices():
    """列出可用音频设备"""
    print("可用音频设备:")
    print(sd.query_devices())
    print(f"\n默认输入设备: {sd.default.device[0]}")


def run_realtime(device=None):
    """实时模式：持续监听麦克风，语音段自动转写"""
    print("\n" + "=" * 60)
    print("  🎤 Zipformer-CTC 实时语音转写")
    print("  " + "=" * 40)
    print("  模式: 实时监听 (按 Ctrl+C 退出)")
    print("  " + "=" * 40)

    # 初始化引擎
    try:
        engine = AsrEngine()
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return

    # 启动麦克风
    mic = MicCapture()
    try:
        mic.start(device=device)
    except Exception as e:
        print(f"  ❌ 无法打开麦克风: {e}")
        print("  请用 --list-devices 查看可用设备")
        return

    # 统计
    stats = {"segments": 0, "total_chars": 0, "runtime_s": 0}
    start_time = time.time()

    print("\n  🟢 开始监听...")
    print()

    try:
        while mic.running:
            # 读取音频块
            chunk = mic.read_chunk(timeout=0.05)
            if chunk is None:
                continue

            # 喂入 VAD
            engine.feed_audio(chunk)

            # 显示 VAD 状态
            is_active = engine.is_speech_active()
            status_text = "说话中..." if is_active else "静音监听中..."
            Display.show_status(status_text, is_active=is_active)

            # 处理完成的语音段
            while engine.has_segment():
                segment = engine.pop_segment()
                if len(segment) < SAMPLE_RATE * 0.3:  # 小于 0.3 秒的段跳过
                    continue

                # 转写 + 清理
                text = engine.transcribe(segment)
                text, unk_ratio = engine.clean_text(text)

                if text:
                    stats["segments"] += 1
                    stats["total_chars"] += len(text)
                    suffix = f" ⚠️ <unk>占比 {unk_ratio:.0%}" if unk_ratio > 0.3 else ""
                    Display.show_result(text + suffix)

    except KeyboardInterrupt:
        print("\n\n  ⏹️  正在停止...")
    finally:
        mic.stop()
        engine.flush_vad()

        # 处理剩余的语音段
        while engine.has_segment():
            segment = engine.pop_segment()
            if len(segment) >= SAMPLE_RATE * 0.3:
                text, _ = engine.clean_text(engine.transcribe(segment))
                if text:
                    stats["segments"] += 1
                    stats["total_chars"] += len(text)
                    Display.show_result(text)

    stats["runtime_s"] = time.time() - start_time
    Display.show_stats(stats)
    print("\n  👋 再见!\n")


def run_one_shot(device=None):
    """单次模式：回车开始录音，再回车结束并转写"""
    print("\n" + "=" * 60)
    print("  🎤 Zipformer-CTC 单次语音转写")
    print("  " + "=" * 40)
    print("  模式: 按 Enter 开始/停止录音")
    print("  " + "=" * 40)

    # 初始化引擎
    try:
        engine = AsrEngine()
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return

    # 启动麦克风
    mic = MicCapture()
    try:
        mic.start(device=device)
    except Exception as e:
        print(f"  ❌ 无法打开麦克风: {e}")
        print("  请用 --list-devices 查看可用设备")
        return

    input("\n  ⏸️  按 Enter 开始录音...")
    print("\n  🟢 录音中... (按 Enter 停止)\n")

    # 采集音频到缓冲区
    buffer = []
    input()  # 等待再次回车
    print("  ⏹️  停止录音\n")

    # 收集剩余音频
    time.sleep(0.3)  # 等待最后的音频块
    while not mic.audio_queue.empty():
        try:
            chunk = mic.audio_queue.get_nowait()
            buffer.append(chunk)
        except queue.Empty:
            break

    mic.stop()

    if not buffer:
        print("  ⚠️  未采集到音频")
        return

    # 拼接音频
    audio = np.concatenate(buffer) if len(buffer) > 1 else buffer[0]

    # 用 VAD 分段
    print("  🔄 正在分析语音段...")
    engine.feed_audio(audio)
    engine.flush_vad()

    segments = []
    while engine.has_segment():
        segment = engine.pop_segment()
        if len(segment) >= SAMPLE_RATE * 0.3:
            segments.append(segment)

    if not segments:
        print("  ⚠️  未检测到有效语音（请检查麦克风或说话声音）")
        return

    print(f"\n  检测到 {len(segments)} 个语音段，正在转写...\n")

    # 逐段转写
    total_chars = 0
    for i, seg in enumerate(segments):
        raw_text = engine.transcribe(seg)
        text, unk_ratio = engine.clean_text(raw_text)
        if text:
            total_chars += len(text)
            suffix = f" ⚠️ <unk>占比 {unk_ratio:.0%}" if unk_ratio > 0.3 else ""
            print(f"  ✍️ [{i + 1}] {text}{suffix}")

    print(f"\n  📊 总计: {total_chars} 字符 / {len(segments)} 段\n")


def main():
    parser = argparse.ArgumentParser(
        description="🎤 Zipformer-CTC 本地麦克风实时转写",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    实时模式（持续监听）
  %(prog)s --one-shot         单次模式（回车开始/结束）
  %(prog)s --list-devices     列出音频设备
  %(prog)s --device 2         指定音频设备编号
        """,
    )
    parser.add_argument(
        "--one-shot", action="store_true",
        help="单次模式（回车开始/结束录音）"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="列出可用音频设备"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="音频设备编号（默认: 系统默认输入设备）"
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    print()
    print("╔════════════════════════════════════════════════╗")
    print("║      🎤 Zipformer-CTC 本地语音转写工具         ║")
    print("║      帧级独立分类 · 无语言模型偏置              ║")
    print("╚════════════════════════════════════════════════╝")

    if args.one_shot:
        run_one_shot(device=args.device)
    else:
        run_realtime(device=args.device)


if __name__ == "__main__":
    main()
