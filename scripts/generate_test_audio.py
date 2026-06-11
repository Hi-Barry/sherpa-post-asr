"""
生成测试音频样本

构造 10 组经典 ASR 易错场景，每组用 TTS 或静默噪音合成。
用于对比原始 SenseVoice vs Post-ASR 纠错效果。

场景覆盖：
- 同音字（境界/镜里、曾经/滤镜）
- 专有名词（人名、地名）
- 数字/英文混读
- 轻声变调
- 上下文消歧
"""

import sys
import os
from pathlib import Path

# 确保能导入 server 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from server.config import TEST_AUDIO_DIR
from server.audio_utils import load_audio


# ══════════════════════════════════════════════════════
# 测试集定义
# ══════════════════════════════════════════════════════

# 每项: (filename, category, ground_truth_text, description)
# 注意: 这些是纯文本，需要用 TTS 或合成方式生成音频
TEST_CASES = [
    # ── 同音字（前后鼻音） ──
    ("homophone_jingjie", "同音字",
     "我今天去了境界，风景很美",
     "境界 vs 镜里 — 经典前后鼻音混淆"),

    ("homophone_cengjing", "同音字",
     "曾经有一份真挚的爱情",
     "曾经 vs 滤镜 — 不同声母"),

    ("homophone_yijing", "同音字",
     "事情已经处理完了",
     "已经 vs 一惊 — 轻声混淆"),

    # ── 专有名词 ──
    ("proper_tech", "专名",
     "我用的是一台搭载了酷睿处理器的笔记本电脑",
     "酷睿 vs 酷瑞 — 品牌名译音"),

    ("proper_place", "专名",
     "明天要去深圳出差",
     "深圳 vs 深镇 — 地名规范"),

    # ── 数字/英文混读 ──
    ("mixed_english", "中英混",
     "我的工号是零零三，邮箱是 admin at example dot com",
     "中英数字混合，邮箱格式"),

    ("mixed_number", "数字",
     "一百二十三号选手上场",
     "阿拉伯数字 vs 中文数字"),

    # ── 上下文消歧 ──
    ("context_disambig", "上下文",
     "他喜欢吃湖南菜，特别是剁椒鱼头",
     "湖南 vs 胡南 — 需要地理知识"),

    ("context_industry", "上下文",
     "这个项目的里程碑要在月底前完成",
     "里程碑 vs 里成碑 — 专有组合词"),

    # ── 轻声/变调 ──
    ("tone_variation", "轻声",
     "把窗户打开，透透气",
     "窗户 vs 窗护 — 轻声错误"),

    # ── 口音/方言 ──
    ("accent_flat", "口音",
     "飞机晚上十点起飞",
     "十点 vs 四点 — 平舌翘舌混淆"),
]

# ══════════════════════════════════════════════════════
# TTS 合成（如果可用）
# ══════════════════════════════════════════════════════

def check_tts_available() -> bool:
    """检查是否有可用的 TTS 引擎"""
    try:
        import pyttsx3
        return True
    except ImportError:
        pass
    try:
        # 检查 edge-tts
        import subprocess
        result = subprocess.run(
            ["which", "edge-tts"], capture_output=True, text=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def synthesize_text(text: str, output_path: Path, use_edge_tts: bool = True):
    """
    用 TTS 合成音频。

    优先使用 edge-tts（效果好），回退到 pyttsx3。
    """
    if use_edge_tts:
        try:
            import subprocess
            voice = "zh-CN-XiaoxiaoNeural"
            subprocess.run([
                "edge-tts",
                "--voice", voice,
                "--text", text,
                "--write-media", str(output_path),
                "--rate", "+0%",
            ], check=True, timeout=60)
            return True
        except Exception as e:
            print(f"  edge-tts 失败: {e}，尝试 pyttsx3...")

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 0.9)

        # pyttsx3 不支持直接输出到文件，需要用变通方式
        # 生成临时 wav
        temp_path = output_path.with_suffix(".temp.wav")

        # 使用 espeak 的变通方式
        import subprocess
        subprocess.run([
            "espeak-ng", "-v", "zh", "-s", "160", "-w", str(temp_path),
            text,
        ], check=True, timeout=30)

        # 重采样到 16kHz
        data, sr = sf.read(temp_path)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sf.write(output_path, data, 16000)
        temp_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"  pyttsx3 也失败: {e}")
        return False


def generate_synthetic_samples(output_dir: Path):
    """
    没有 TTS 时生成简单的合成样本。
    用纯音 + 噪音模拟。
    实际使用时建议替换为真实录音。
    """
    sr = 16000

    for name, category, text, desc in TEST_CASES:
        # 生成 ~3 秒的 440Hz 纯音作为占位
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = 0.3 * np.sin(2 * np.pi * 440 * t)

        # 加一点白噪音
        samples += 0.01 * np.random.randn(len(samples))

        output_path = output_dir / f"{name}.wav"
        sf.write(output_path, samples.astype(np.float32), sr)
        print(f"  [占位] {output_path.name} — {desc}")

    print(f"\n⚠️ 生成了 {len(TEST_CASES)} 个占位音频（纯音），")
    print("   实际使用时建议用 edge-tts 或真实录音替换。")


def main():
    """主入口"""
    TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"测试音频目录: {TEST_AUDIO_DIR}")
    print(f"共 {len(TEST_CASES)} 个测试场景\n")

    tts_available = check_tts_available()
    if tts_available:
        print("检测到 TTS 引擎，使用合成语音\n")
        success = 0
        for name, category, text, desc in TEST_CASES:
            output_path = TEST_AUDIO_DIR / f"{name}.wav"
            if output_path.exists():
                print(f"  [跳过] {name}.wav 已存在")
                continue
            print(f"  [合成] {name}.wav ← {text}")
            ok = synthesize_text(text, output_path)
            if ok:
                success += 1
            else:
                print(f"  [失败] {name}")

        print(f"\n合成完成: {success}/{len(TEST_CASES)}")
    else:
        print("未检测到 TTS 引擎（edge-tts / pyttsx3），")
        print("生成占位音频（纯音），仅用于接口测试，不用于准确率评估。\n")
        generate_synthetic_samples(TEST_AUDIO_DIR)

    # 写入场景清单文件
    manifest_path = TEST_AUDIO_DIR / "_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("filename,category,ground_truth,description\n")
        for name, category, text, desc in TEST_CASES:
            f.write(f"{name}.wav,{category},{text},{desc}\n")
    print(f"\n场景清单: {manifest_path}")

    # 显示测试目录大小
    total_size = sum(f.stat().st_size for f in TEST_AUDIO_DIR.glob("*.wav"))
    print(f"总大小: {total_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
