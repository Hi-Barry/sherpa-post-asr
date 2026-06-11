"""
噪音合成器 — 12 种场景噪声生成

每个场景实现为函数，返回 (samples, sr=16000) 的 numpy float32 数组。
"""
import numpy as np
from scipy import signal
import soundfile as sf
from pathlib import Path

SR = 16000  # 统一采样率


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)))


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def _apply_envelope(samples: np.ndarray, attack_ms=20, release_ms=50):
    """起落包络，避免首尾咔嗒声"""
    a = int(SR * attack_ms / 1000)
    r = int(SR * release_ms / 1000)
    if a > 0 and len(samples) > a:
        samples[:a] *= np.linspace(0, 1, a)
    if r > 0 and len(samples) > r:
        samples[-r:] *= np.linspace(1, 0, r)
    return samples


def noise_white(duration_sec: float) -> np.ndarray:
    """白噪声 — 均匀频谱"""
    n = int(SR * duration_sec)
    noise = np.random.randn(n).astype(np.float32)
    return _apply_envelope(noise)


def noise_pink(duration_sec: float) -> np.ndarray:
    """粉红噪声 — -3dB/oct (Voss-McCartney 算法)"""
    n = int(SR * duration_sec)
    # 用滤波器实现
    b, a = signal.butter(4, 200 / (SR / 2), btype="low")
    white = np.random.randn(n + 1000).astype(np.float32)
    pink = signal.filtfilt(b, a, white)
    # 再叠加 -3dB/oct 趋势
    fft = np.fft.rfft(pink[:n])
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = 1  # 避免除零
    fft *= 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n).astype(np.float32)
    return _apply_envelope(pink[:n])


def noise_brown(duration_sec: float) -> np.ndarray:
    """布朗噪声（红噪声） — -6dB/oct"""
    n = int(SR * duration_sec)
    white = np.random.randn(n).astype(np.float32)
    brown = np.cumsum(white).astype(np.float32)
    brown /= np.max(np.abs(brown)) * 0.8
    return _apply_envelope(brown)


def noise_subway(duration_sec: float) -> np.ndarray:
    """
    地铁噪声：低频轰鸣 + 中频人潮 + 高频摩擦
    """
    n = int(SR * duration_sec)
    # 低频轰鸣 (40-150Hz)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    hum = (np.sin(2 * np.pi * 55 * t) * 0.4
           + np.sin(2 * np.pi * 85 * t) * 0.3
           + np.sin(2 * np.pi * 120 * t) * 0.2)
    hum += np.random.randn(n).astype(np.float32) * 0.05  # 随机调制
    mod = noise_brown(0.5)[:int(SR * 0.5)] if SR * 0.5 <= n else np.ones(n)
    # 拉伸 mod 到 n 长度
    mod = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(mod)), mod)
    hum *= mod * 0.5 + 0.5

    # 轨道摩擦高频 (2-6kHz 宽带)
    friction = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(4, [2000 / (SR / 2), 6000 / (SR / 2)], btype="band")
    friction = signal.filtfilt(b, a, friction) * 0.3

    # 人潮中频 (300-1000Hz)
    crowd = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(4, [300 / (SR / 2), 1000 / (SR / 2)], btype="band")
    crowd = signal.filtfilt(b, a, crowd) * 0.15

    # 风噪低频
    wind = noise_brown(duration_sec) * 0.1

    noise = (hum * 0.5 + friction * 0.25 + crowd * 0.15 + wind * 0.1)
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.7)


def noise_server_room(duration_sec: float) -> np.ndarray:
    """
    机房噪声：多风扇谐波
    """
    n = int(SR * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # 多个风扇的谐波组合
    fans = sum(
        np.sin(2 * np.pi * base * t + np.random.uniform(0, 2 * np.pi))
        for base in [55, 62, 70, 120, 240, 360]
    )
    # 加上湍流噪声
    turbulent = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(4, 500 / (SR / 2), btype="low")
    turbulent = signal.filtfilt(b, a, turbulent) * 0.2

    noise = fans / 6 + turbulent
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.6)


def noise_fan(duration_sec: float) -> np.ndarray:
    """
    大功率风扇噪声：窄带低频 + 湍流
    """
    n = int(SR * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # 叶片通过频率 (50-200Hz)
    blade = (
        np.sin(2 * np.pi * 75 * t) * 0.5
        + np.sin(2 * np.pi * 150 * t) * 0.3
        + np.sin(2 * np.pi * 225 * t) * 0.15
    )
    # 空气湍流
    turbo = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(4, 800 / (SR / 2), btype="low")
    turbo = signal.filtfilt(b, a, turbo) * 0.4

    noise = blade * 0.5 + turbo * 0.5
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.7)


def noise_motor(duration_sec: float) -> np.ndarray:
    """
    电机噪音：电磁谐波 + 机械振动
    """
    n = int(SR * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # 基频 50Hz + 谐波
    harmonics = sum(
        np.sin(2 * np.pi * 50 * h * t + np.random.uniform(0, 0.1))
        * (0.6 / h)
        for h in range(1, 13)
    )
    # 机械冲击 (随机脉冲)
    impulses = np.zeros(n)
    for _ in range(int(duration_sec * 15)):  # ~15次/秒
        pos = np.random.randint(0, n)
        width = int(SR * 0.003)  # 3ms 冲击
        if pos + width < n:
            impulses[pos:pos + width] = np.random.randn() * 0.3

    noise = harmonics * 0.6 + impulses * 0.25 + np.random.randn(n).astype(np.float32) * 0.15
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.65)


def noise_reverb(duration_sec: float, rt60: float = 0.5) -> np.ndarray:
    """
    混响：用随机 IR 模拟室内混响
    输出经过混响处理的 speech-like noise（测试混响场景下 ASR 的表现）
    实际测试时，这个噪声会与干净语音混合
    """
    # 生成 IR (随机反射)
    ir_len = int(SR * rt60)
    ir = np.random.randn(ir_len).astype(np.float32)
    decay = np.exp(-3 * np.linspace(0, 1, ir_len) * np.log(10) / rt60)
    ir *= decay
    ir = _apply_envelope(ir, attack_ms=1, release_ms=rt60 * 1000)
    # 对白噪声卷积得到混响噪声
    white = noise_white(duration_sec + rt60)
    rev = np.convolve(white, ir, mode="same")[:int(SR * duration_sec)]
    return _apply_envelope(rev / np.max(np.abs(rev)) * 0.5)


def noise_open_field(duration_sec: float) -> np.ndarray:
    """
    旷野：极微弱的低频风噪
    """
    n = int(SR * duration_sec)
    wind = noise_brown(duration_sec)
    b, a = signal.butter(4, 200 / (SR / 2), btype="low")
    wind = signal.filtfilt(b, a, wind)
    return wind * 0.05  # 非常微弱


def noise_high_noise(duration_sec: float) -> np.ndarray:
    """
    高噪音：粉红噪声，但幅度很大
    """
    return noise_pink(duration_sec) * 1.5


def noise_whisper_noise(duration_sec: float) -> np.ndarray:
    """
    轻语场景的背景噪声：极低电平的粉红噪声
    主要用于混合时控制信噪比
    """
    return noise_pink(duration_sec) * 0.3


def noise_rain(duration_sec: float) -> np.ndarray:
    """
    下雨声：滤波噪声模拟雨滴
    """
    n = int(SR * duration_sec)
    # 主雨声：300-3000Hz 噪声音乐
    rain = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(4, [300 / (SR / 2), 3000 / (SR / 2)], btype="band")
    rain = signal.filtfilt(b, a, rain)

    # 雨滴冲击 (稀疏冲击)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    drops = np.zeros(n)
    n_drops = int(duration_sec * 80)  # 80滴/秒
    for _ in range(n_drops):
        pos = np.random.randint(0, n)
        width = int(SR * 0.002)  # 2ms
        amp = np.random.uniform(0.1, 1.0)
        if pos + width < n:
            envelope = np.exp(-np.linspace(0, 5, width))
            drops[pos:pos + width] += amp * envelope

    noise = rain * 0.4 + drops * 0.6
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.7)


def noise_thunderstorm(duration_sec: float) -> np.ndarray:
    """
    雷暴：雨声 + 低频雷击冲击
    """
    n = int(SR * duration_sec)
    # 基础雨声
    rain_noise = noise_rain(duration_sec)

    # 雷击 (强低频冲击)
    thunder = np.zeros(n)
    n_strikes = int(duration_sec / 8) + 1  # 每 8 秒一次雷
    for _ in range(n_strikes):
        pos = np.random.randint(0, n)
        width = int(SR * np.random.uniform(0.3, 1.0))
        if pos + width < n:
            # 雷声 = 低频正弦 + 爆破噪声
            t_local = np.linspace(0, width / SR, width, endpoint=False)
            strike = (
                np.sin(2 * np.pi * 50 * t_local) * np.exp(-t_local * 3) * 0.5
                + np.random.randn(width).astype(np.float32) * np.exp(-t_local * 2) * 0.5
            )
            # 低频强调
            b, a = signal.butter(4, 150 / (SR / 2), btype="low")
            strike = signal.filtfilt(b, a, strike)
            thunder[pos:pos + width] += strike * 2.0

    noise = rain_noise * 0.5 + thunder * 0.5
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.75)


def noise_wind(duration_sec: float) -> np.ndarray:
    """
    大风：低频湍流 + 风噪
    """
    n = int(SR * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # 低频风 (0-200Hz)
    wind = noise_brown(duration_sec)
    b, a = signal.butter(4, 300 / (SR / 2), btype="low")
    wind = signal.filtfilt(b, a, wind)
    # 频率调制 (模拟风的不稳定性)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    wind *= mod
    # 高频风声
    gust = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(4, [200 / (SR / 2), 2000 / (SR / 2)], btype="band")
    gust = signal.filtfilt(b, a, gust) * 0.2

    noise = wind * 0.7 + gust * 0.3
    return _apply_envelope(noise / np.max(np.abs(noise)) * 0.8)


def apply_reverb_to_speech(speech: np.ndarray, rt60: float = 0.5) -> np.ndarray:
    """
    对实际语音施加混响效果（不混合噪声）。
    用于混响场景的单独测试。
    """
    # 生成 IR
    ir_len = int(SR * rt60)
    ir = np.random.randn(ir_len).astype(np.float32)
    decay = np.exp(-3 * np.linspace(0, 1, ir_len) / (rt60 / 0.5))
    ir *= decay
    ir /= np.max(np.abs(ir))
    ir = _apply_envelope(ir, attack_ms=1, release_ms=rt60 * 1000)

    # 卷积
    reverb = np.convolve(speech, ir, mode="full")[:len(speech)]
    # 干湿混合
    mixed = speech * 0.6 + reverb * 0.4
    return _apply_envelope(mixed / np.max(np.abs(mixed)))


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    以指定 SNR 混合语音和噪声。

    SNR = 10 * log10(P_speech / P_noise)
    """
    # 确保长度相同
    min_len = min(len(speech), len(noise))
    speech = speech[:min_len]
    noise = noise[:min_len]

    speech_rms = _rms(speech)
    noise_rms = _rms(noise)

    if noise_rms == 0:
        return speech.copy()

    target_noise_rms = speech_rms / _db_to_linear(snr_db)
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = speech + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed /= peak * 1.05

    return mixed.astype(np.float32)


# ── 工厂函数 ──────────────────────────────────────────

NOISE_FUNCTIONS = {
    "subway": noise_subway,
    "server_room": noise_server_room,
    "fan": noise_fan,
    "motor": noise_motor,
    "reverb": noise_reverb,
    "open_field": noise_open_field,
    "high_noise": noise_high_noise,
    "white_noise": noise_white,
    "whisper": noise_whisper_noise,
    "rain": noise_rain,
    "thunderstorm": noise_thunderstorm,
    "wind": noise_wind,
}

NOISE_LABELS = {
    "subway": "🚇 地铁",
    "server_room": "🖥️ 机房",
    "fan": "🌀 风扇",
    "motor": "⚡ 电机",
    "reverb": "🏛️ 混响",
    "open_field": "🌄 旷野",
    "high_noise": "📢 高噪音",
    "white_noise": "⬜ 白噪音",
    "whisper": "🤫 轻语",
    "rain": "🌧️ 下雨",
    "thunderstorm": "⛈️ 雷暴",
    "wind": "💨 大风",
}
