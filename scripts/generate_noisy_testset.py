"""
生成带噪测试音频集

流程: TTS 合成干净语音 → 12 场景噪音 → 混合 (4 个 SNR) → 保存
"""

import sys, os, json, time, subprocess, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from scripts.noise_synthesizer import (
    NOISE_FUNCTIONS, NOISE_LABELS, SR,
    mix_at_snr, apply_reverb_to_speech, _rms,
)

# 测试句子 (filename, category, ground_truth)
TEST_SENTENCES = [
    ("homophone_jingjie", "同音字", "我今天去了境界，风景很美"),
    ("homophone_cengjing", "同音字", "曾经有一份真挚的爱情"),
    ("homophone_yijing", "同音字", "事情已经处理完了"),
    ("proper_tech", "专名", "我用的是一台搭载了酷睿处理器的笔记本电脑"),
    ("proper_place", "专名", "明天要去深圳出差"),
    ("mixed_english", "中英混", "我的工号是零零三，邮箱是 admin at example dot com"),
    ("mixed_number", "数字", "一百二十三号选手上场"),
    ("context_disambig", "上下文", "他喜欢吃湖南菜，特别是剁椒鱼头"),
    ("context_industry", "上下文", "这个项目的里程碑要在月底前完成"),
    ("tone_variation", "轻声", "把窗户打开，透透气"),
    ("accent_flat", "口音", "飞机晚上十点起飞"),
]

# 每个场景的 SNR 配置
# (场景名, SNR列表(dB), 特殊处理)
SCENARIO_CONFIG = [
    ("subway",      [15, 10, 5, 0],    "mix"),
    ("server_room", [15, 10, 5, 0],    "mix"),
    ("fan",         [15, 10, 5, 0],    "mix"),
    ("motor",       [15, 10, 5, 0],    "mix"),
    ("reverb",      [0, 0, 0, 0],      "reverb"),  # rt60: 0.3, 0.5, 0.7, 1.0
    ("open_field",  [30, 25, 20, 15],  "mix"),     # SNR 较高
    ("high_noise",  [10, 5, 0, -5],    "mix"),
    ("white_noise", [10, 5, 0, -5],    "mix"),
    ("whisper",     ["q0", "q1", "q2", "q3"], "whisper"),  # speech 降幅
    ("rain",        [15, 10, 5, 0],    "mix"),
    ("thunderstorm", [15, 10, 5, 0],  "mix"),
    ("wind",        [15, 10, 5, 0],    "mix"),
]


def synthesize_speech(text: str, output_path: Path) -> bool:
    """用 edge-tts 合成语音"""
    cmd = [
        "edge-tts", "--voice", "zh-CN-XiaoxiaoNeural",
        "--text", text,
        "--write-media", str(output_path),
        "--rate", "+0%",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60, capture_output=True)
        return True
    except Exception as e:
        print(f"  [TTS 失败] {e}")
        return False


def generate_noisy_dataset(
    clean_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    force_tts: bool = False,
):
    """生成完整带噪测试集"""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = 0

    for name, category, gt_text in TEST_SENTENCES:
        # Step 1: 合成干净语音
        clean_wav = clean_dir / f"{name}.wav"
        if not clean_wav.exists() or force_tts:
            print(f"[TTS] {name}: {gt_text[:30]}...")
            if not synthesize_speech(gt_text, clean_wav):
                print(f"  ⚠️ 跳过 {name}")
                continue
            time.sleep(0.5)  # edge-tts 限流

        # 读取干净语音
        speech, _ = sf.read(clean_wav, dtype="float32")
        if len(speech) == 0:
            continue

        # Step 2: 对每个场景生成带噪版本
        for scenario_name, snr_list, mode in SCENARIO_CONFIG:
            noise_fn = NOISE_FUNCTIONS.get(scenario_name)
            if noise_fn is None:
                continue

            label = NOISE_LABELS.get(scenario_name, scenario_name)

            for level_idx, snr in enumerate(snr_list):
                # 生成带噪音频
                if mode == "mix":
                    noise = noise_fn(len(speech) / SR + 0.5)
                    noisy = mix_at_snr(speech, noise, snr_db=snr)
                elif mode == "reverb":
                    rt60_vals = [0.3, 0.5, 0.7, 1.0]
                    rt60 = rt60_vals[level_idx]
                    noisy = apply_reverb_to_speech(speech, rt60=rt60)
                    snr_label = f"rt60={rt60}"
                elif mode == "whisper":
                    # 降低语音幅度
                    levels = [0, -6, -12, -20]
                    amp_db = levels[level_idx]
                    amp_linear = 10 ** (amp_db / 20)
                    noisy = speech * amp_linear
                    # 加一点底噪
                    noise = noise_fn(len(speech) / SR + 0.5) * 0.02
                    noisy = mix_at_snr(noisy, noise, snr_db=20)
                    snr_label = f"{amp_db}dB"
                else:
                    continue

                # 保存
                suffix = f"l{level_idx}" if mode in ("reverb", "whisper") else f"{snr}dB"
                out_name = f"{name}__{scenario_name}__{suffix}.wav"
                out_path = output_dir / out_name
                sf.write(out_path, noisy.astype(np.float32), SR)

                # 记录 manifest
                record = {
                    "file": out_name,
                    "scenario": scenario_name,
                    "scenario_label": label,
                    "category": category,
                    "ground_truth": gt_text,
                    "snr": str(snr) if isinstance(snr, (int, float)) else snr,
                    "level": level_idx,
                    "duration_s": round(len(speech) / SR, 2),
                }
                manifest.append(record)
                total += 1

    # 写入 manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"生成完成: {total} 个测试样本")
    print(f"场景: {len(SCENARIO_CONFIG)} 个")
    print(f"句子: {len(TEST_SENTENCES)} 个")
    print(f"干净语音: {clean_dir}")
    print(f"带噪音频: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*50}")

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-tts", action="store_true", help="强制重新合成 TTS")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    clean_dir = project_root / "audio_test" / "clean"
    noisy_dir = project_root / "audio_test" / "noisy"
    manifest_path = project_root / "audio_test" / "_benchmark_manifest.json"

    generate_noisy_dataset(clean_dir, noisy_dir, manifest_path, force_tts=args.force_tts)


if __name__ == "__main__":
    main()
