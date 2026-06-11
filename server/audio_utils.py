"""
音频加载与 VAD 工具
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """
    加载音频文件，自动重采样到 target_sr。
    返回 shape=(samples,) 的 float32 数组，范围 [-1, 1]。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    data, sr = sf.read(path, dtype="float32")
    # 多声道取平均
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 重采样
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")

    return data


def energy_vad_detect(
    samples: np.ndarray, sr: int = 16000,
    frame_ms: int = 30, hop_ms: int = 10,
    energy_threshold: float = 0.002,
    min_speech_ms: int = 200,
    min_silence_ms: int = 400,
) -> list[tuple[int, int]]:
    """
    简单能量 VAD，返回 [(start_sample, end_sample), ...] 语音段。
    用于分句。
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    min_speech = int(sr * min_speech_ms / 1000)
    min_silence = int(sr * min_silence_ms / 1000)

    # 计算每帧能量
    energies = []
    for start in range(0, len(samples) - frame_len + 1, hop_len):
        frame = samples[start:start + frame_len]
        energy = np.sqrt(np.mean(frame ** 2))
        energies.append(energy)

    # 二值化
    is_speech = np.array(energies) > energy_threshold

    # 找语音段
    segments = []
    in_speech = False
    seg_start = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            seg_start = i * hop_len
            in_speech = True
        elif not speech and in_speech:
            seg_end = i * hop_len
            if seg_end - seg_start >= min_speech:
                segments.append((seg_start, seg_end))
            in_speech = False

    if in_speech:
        seg_end = len(is_speech) * hop_len
        if seg_end - seg_start >= min_speech:
            segments.append((seg_start, seg_end))

    # 合并间隔 < min_silence 的段
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[0] - merged[-1][1] < min_silence:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    return merged
