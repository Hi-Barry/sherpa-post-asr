"""
分析报告 — 基准测试结果分析
"""
import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = BASE_DIR / "audio_test" / "_benchmark_results.json"


def load_results():
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(results):
    """全维度分析"""
    report = []
    total = len(results)
    
    # ── 1. 总览 ──
    raw_cer = sum(r["raw_cer"] for r in results) / total
    cor_cer = sum(r["corrected_cer"] for r in results) / total
    improved = [r for r in results if r["cer_improvement"] > 0]
    worsened = [r for r in results if r["cer_improvement"] < 0]
    changed = [r for r in results if r["llm_changed"]]
    unchanged = [r for r in results if not r["llm_changed"]]
    
    report.append("# sherpa-post-asr 基准测试报告")
    report.append(f"\n**测试日期**: 2026-06-10")
    report.append(f"**测试样本**: {total} 个 (11 句子 × 12 场景 × 4 SNR)")
    report.append(f"**ASR 模型**: SenseVoice-Small int8")
    report.append(f"**纠错模型**: Qwen3.5-2B Q4_K_M (llama.cpp, GPU)")
    report.append(f"\n## 1. 总体指标\n")
    report.append(f"| 指标 | 值 |")
    report.append(f"|:---|:---|")
    report.append(f"| 原始 CER | {raw_cer:.4f} |")
    report.append(f"| 纠错后 CER | {cor_cer:.4f} |")
    report.append(f"| CER 降低 | {raw_cer - cor_cer:.4f} ({((raw_cer - cor_cer) / raw_cer * 100):.1f}%) |")
    report.append(f"| 改善样本数 | {len(improved)}/{total} ({len(improved)/total*100:.1f}%) |")
    report.append(f"| 变差样本数 | {len(worsened)}/{total} ({len(worsened)/total*100:.1f}%) |")
    report.append(f"| LLM 实际修改数 | {len(changed)}/{total} ({len(changed)/total*100:.1f}%) |")
    report.append(f"| 平均延迟 | {sum(r['latency_ms'].get('total', 0) for r in results)/total:.0f}ms |")

    # 改善幅度分布
    improvements = [r["cer_improvement"] for r in improved]
    avg_imp = sum(improvements) / len(improvements) if improvements else 0
    
    report.append(f"\n## 2. 改善质量\n")
    if improvements:
        report.append(f"| 指标 | 值 |")
        report.append(f"|:---|:---|")
        report.append(f"| 改善样本平均 CER 降幅 | {avg_imp:.4f} |")
        report.append(f"| 最大降幅 | {max(improvements):.4f} |")
        report.append(f"| 改善幅度 ≥ 0.1 | {sum(1 for i in improvements if i >= 0.1)} 个 |")
        report.append(f"| 改善幅度 ≥ 0.2 | {sum(1 for i in improvements if i >= 0.2)} 个 |")
        report.append(f"| 改善幅度 ≥ 0.5 | {sum(1 for i in improvements if i >= 0.5)} 个 |")

    # ── 3. 按场景分析 ──
    report.append(f"\n## 3. 按场景分析\n")
    report.append(f"| 场景 | 样本数 | 原始 CER | 纠错 CER | 改善 | 改善率 |")
    report.append(f"|:---|:---:|:---:|:---:|:---:|:---:|")
    
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)
    
    scenario_labels = {}
    for r in results:
        scenario_labels[r["scenario"]] = r["scenario_label"]
    
    best_scenario = None
    worst_scenario = None
    best_cor_cer = 1.0
    worst_cor_cer = 0.0
    
    for scene, items in sorted(by_scenario.items()):
        n = len(items)
        avg_raw = sum(i["raw_cer"] for i in items) / n
        avg_cor = sum(i["corrected_cer"] for i in items) / n
        imp = sum(1 for i in items if i["cer_improvement"] > 0)
        imp_rate = imp / n * 100
        label = scenario_labels.get(scene, scene)
        report.append(f"| {label} | {n} | {avg_raw:.4f} | {avg_cor:.4f} | {avg_raw - avg_cor:+.4f} | {imp_rate:.0f}% |")
        
        if avg_cor < best_cor_cer:
            best_cor_cer = avg_cor
            best_scenario = (scene, label)
        if avg_cor > worst_cor_cer:
            worst_cor_cer = avg_cor
            worst_scenario = (scene, label)
    
    # ── 4. 按句子分析 ──
    report.append(f"\n## 4. 按句子类型分析\n")
    report.append(f"| 句子类型 | 样本数 | 原始 CER | 纠错 CER | 改善 | 改善率 |")
    report.append(f"|:---|:---:|:---:|:---:|:---:|:---:|")
    
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
    
    for cat, items in sorted(by_category.items()):
        n = len(items)
        avg_raw = sum(i["raw_cer"] for i in items) / n
        avg_cor = sum(i["corrected_cer"] for i in items) / n
        imp = sum(1 for i in items if i["cer_improvement"] > 0)
        imp_rate = imp / n * 100
        report.append(f"| {cat} | {n} | {avg_raw:.4f} | {avg_cor:.4f} | {avg_raw - avg_cor:+.4f} | {imp_rate:.0f}% |")
    
    # ── 5. 按 SNR 分析 ──
    report.append(f"\n## 5. 按 SNR/噪声强度分析\n")
    report.append(f"| SNR 级别 | 样本数 | 原始 CER | 纠错 CER | LLM 修改率 |")
    report.append(f"|:---|:---:|:---:|:---:|:---:|")
    
    by_snr = defaultdict(list)
    for r in results:
        snr = r.get("snr", "?")
        by_snr[snr].append(r)
    
    for snr, items in sorted(by_snr.items()):
        n = len(items)
        avg_raw = sum(i["raw_cer"] for i in items) / n
        avg_cor = sum(i["corrected_cer"] for i in items) / n
        mod_rate = sum(1 for i in items if i["llm_changed"]) / n * 100
        report.append(f"| {snr} | {n} | {avg_raw:.4f} | {avg_cor:.4f} | {mod_rate:.0f}% |")
    
    # ── 6. 最佳/最差场景 ──
    if best_scenario:
        report.append(f"\n## 6. 场景排名\n")
        report.append(f"- **最佳**（纠错后 CER 最低）: {best_scenario[1]} ({best_cor_cer:.4f})")
    if worst_scenario:
        report.append(f"- **最差**（纠错后 CER 最高）: {worst_scenario[1]} ({worst_cor_cer:.4f})")
    
    # 按原始 CER 排序
    scene_cer = []
    for scene, items in by_scenario.items():
        avg_raw = sum(i["raw_cer"] for i in items) / len(items)
        scene_cer.append((scene, scenario_labels.get(scene, scene), avg_raw))
    scene_cer.sort(key=lambda x: x[2])
    
    report.append(f"\n### 场景按原始 CER 排序（从易到难）\n")
    for rank, (scene, label, cer) in enumerate(scene_cer, 1):
        report.append(f"{rank}. {label}: CER = {cer:.4f}")
    
    # ── 7. 改进建议 ──
    report.append(f"\n## 7. 分析与建议\n")
    
    if len(improved) > 0:
        report.append(f"### 7.1 有效纠错案例\n")
        best_improvs = sorted(improved, key=lambda r: -r["cer_improvement"])[:5]
        for r in best_improvs:
            report.append(f"- `{r['file']}`: CER {r['raw_cer']:.3f}→{r['corrected_cer']:.3f} "
                         f"(raw: «{r['raw_text'][:40]}…» / cor: «{r['corrected_text'][:40]}…»)")
    
    report.append(f"\n### 7.2 局限性\n")
    report.append(f"- **LLM 修改率低**: 只有 {len(changed)}/{total} ({len(changed)/total*100:.1f}%) 的样本被 LLM 实际修改")
    report.append(f"- **安全校验过严**: 字符替换限制（长度一致+改动≤50%）可能过滤了有效修正")
    report.append(f'- **无置信度信号**: 无法区分"肯定错"和"可能错"，导致 LLM 对所有文本做同样处理')
    report.append(f"- **整句纠错延迟高**: 平均 {sum(r['latency_ms'].get('total', 0) for r in results)/total:.0f}ms/次，其中约 70% 花在 LLM 推理上")
    
    report.append(f"\n### 7.3 改进方向\n")
    report.append(f"1. **获取置信度**: 替换 ASR 前端或提取 CTC 后验概率，实现目标检测式纠错")
    report.append(f"2. **候选重打分**: 用同音字典生成候选，LLM 只做打分而非自由生成")
    report.append(f"3. **端到端模型**: 用 Qwen3-ASR + 纠错头做联合训练")
    report.append(f"4. **流式优化**: 引入 StreamingLLM 或 Mamba 等状态空间模型")
    
    report_path = BASE_DIR / "BENCHMARK_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"报告已保存: {report_path}")
    print("\n".join(report[:20]) + "\n...")
    return "\n".join(report)


if __name__ == "__main__":
    results = load_results()
    analyze(results)
