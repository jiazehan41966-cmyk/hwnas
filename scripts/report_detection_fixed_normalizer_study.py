"""Six-run deliverables for the fixed-normalizer no_object_weight study.

Reads the fetched study root (runs/, diagnostics/) and emits the completion
table, config/checkpoint index, training curves and the one-page comparison.
Validation split only; the promotion rule is exploratory and n=3.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics

ARMS = {"baseline": 0.1, "noobj02": 0.2}
SEEDS = (42, 43, 44)
CLASSES = ("cylider", "manta")
FP_LIMITS = ("0.1", "0.5", "1.0")
BUDGET = 80
# 0.2 minus 0.1; primary is the registered validation legacy map(.50,.75).
DELTA_KEYS = ("map", "map50", "map75", "ap50_cylider", "ap50_manta",
              "ap75_cylider", "ap75_manta", "op_f1",
              "recall_at_fp0.1", "recall_at_fp0.5", "recall_at_fp1.0")
T95_N3 = 4.302652729911275


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def collect(root):
    runs, manifests, diags, failures = {}, {}, {}, []
    for arm in ARMS:
        for seed in SEEDS:
            key = (arm, seed)
            name = f"{arm}_seed{seed}"
            folder = root / "runs" / name
            if not folder.is_dir():
                failures.append({"run": name, "reason": "run directory absent; job never started"})
                continue
            manifest = folder / "run_manifest.json"
            status = folder / "status.json"
            if not status.is_file():
                epochs = 0
                log = folder / "epoch_metrics.jsonl"
                if log.is_file():
                    epochs = sum(1 for _ in log.open(encoding="utf-8"))
                failures.append({"run": name, "reason": f"status.json absent; run did not reach COMPLETE "
                                                        f"(epochs logged: {epochs}/{BUDGET})"})
                continue
            record = load(status)
            if record.get("status") != "COMPLETE":
                failures.append({"run": name, "reason": f"status={record.get('status')}"})
                continue
            runs[key] = record
            manifests[key] = load(manifest)
            diagnostic = root / "diagnostics" / name / "diagnostics" / "diagnostics.json"
            if diagnostic.is_file():
                diags[key] = load(diagnostic)
    return runs, manifests, diags, failures


def metrics(status, diagnostic):
    best = status["best_validation"]
    row = {"map": best["map"], "map50": best["map50"], "map75": best["map75"]}
    for index, name in enumerate(CLASSES):
        row[f"ap50_{name}"] = best["ap_by_iou"]["0.50"][index]
        row[f"ap75_{name}"] = best["ap_by_iou"]["0.75"][index]
    point = status["validation_operating_point"]
    row["op_threshold"] = point["score_threshold"]
    row["op_f1"] = point["f1"]
    row["op_precision"] = point["precision"]
    row["op_recall"] = point["recall"]
    for limit in FP_LIMITS:
        selected = (diagnostic or {}).get("validation_recall_at_fp_per_image", {}).get(limit)
        row[f"recall_at_fp{limit}"] = selected["recall"] if selected else None
        row[f"threshold_at_fp{limit}"] = selected["threshold"] if selected else None
        row[f"observed_fp_per_image{limit}"] = selected["fp_per_image"] if selected else None
    return row


def check_integrity(runs, manifests, root):
    complete = len(runs) == len(ARMS) * len(SEEDS)
    same_initial = {}
    for seed in SEEDS:
        pair = [manifests.get((arm, seed), {}).get("initial_state_sha256") for arm in ARMS]
        same_initial[str(seed)] = bool(pair[0]) and pair[0] == pair[1]
    data_signatures = {json.dumps(m["data_bindings"], sort_keys=True) for m in manifests.values()}
    architectures = {json.dumps(m["architecture"], sort_keys=True) for m in manifests.values()}
    config_hashes = {arm: {manifests[(arm, s)]["config_sha256"] for s in SEEDS if (arm, s) in manifests}
                     for arm in ARMS}
    loss_sections = {}
    for arm in ARMS:
        for seed in SEEDS:
            resolved = root / "runs" / f"{arm}_seed{seed}" / "resolved_config.json"
            if resolved.is_file():
                loss_sections[f"{arm}_seed{seed}"] = load(resolved)["training"]["detection_loss"]
    normalization = {v.get("objectness_normalization") for v in loss_sections.values()}
    weights = {name: section["no_object_weight"] for name, section in loss_sections.items()}
    differing = set()
    if len(loss_sections) > 1:
        reference = next(iter(loss_sections.values()))
        for section in loss_sections.values():
            differing |= {k for k in reference if reference[k] != section.get(k)}
    gate = {
        "all_six_runs_complete": complete,
        "same_initial_state_each_seed": same_initial,
        "identical_data_bindings": len(data_signatures) == 1,
        "identical_architecture": len(architectures) == 1,
        "one_config_hash_per_arm": all(len(v) == 1 for v in config_hashes.values()),
        "arm_config_hashes_differ": len({next(iter(v)) for v in config_hashes.values() if v}) == len(ARMS),
        "all_runs_at_budget": all(r["epochs"] == BUDGET for r in runs.values()),
        "objectness_normalization_fixed_weights": normalization == {"fixed_weights"},
        "loss_keys_differing_across_runs": sorted(differing),
        "only_no_object_weight_differs": differing <= {"no_object_weight"},
        "no_object_weight_by_run": weights,
        "test_never_accessed": all(m.get("test_accessed") is False for m in manifests.values())
                               and all(r.get("test_accessed") is False for r in runs.values()),
    }
    gate["comparison_permitted"] = bool(
        complete and all(same_initial.values()) and gate["identical_data_bindings"]
        and gate["identical_architecture"] and gate["all_runs_at_budget"]
        and gate["objectness_normalization_fixed_weights"] and gate["only_no_object_weight_differs"]
        and gate["test_never_accessed"])
    return gate


def paired_stats(rows):
    stats = {}
    for key in DELTA_KEYS:
        deltas, seeds_used = [], []
        for seed in SEEDS:
            high = rows.get(("noobj02", seed), {}).get(key)
            low = rows.get(("baseline", seed), {}).get(key)
            if high is None or low is None:
                continue
            deltas.append(high - low)
            seeds_used.append(seed)
        if not deltas:
            continue
        entry = {"seeds": seeds_used, "seed_deltas": deltas, "mean": statistics.mean(deltas),
                 "positive_seeds": sum(d > 0 for d in deltas),
                 "arm_means": {arm: statistics.mean([rows[(arm, s)][key] for s in seeds_used])
                               for arm in ARMS}}
        if len(deltas) > 1:
            entry["std"] = statistics.stdev(deltas)
        if len(deltas) == 3:
            half = T95_N3 * entry["std"] / 3 ** .5
            entry["seed_t95_interval"] = [entry["mean"] - half, entry["mean"] + half]
            flips = [abs(sum(d * s for d, s in zip(deltas, signs)) / 3)
                     for signs in itertools.product((-1, 1), repeat=3)]
            entry["exact_sign_flip_two_sided_p"] = sum(f >= abs(entry["mean"]) - 1e-12 for f in flips) / len(flips)
        stats[key] = entry
    return stats


def promotion(stats):
    primary = stats.get("map")
    manta = stats.get("ap50_manta")
    if not primary or not manta:
        return {"evaluated": False, "reason": "primary or manta AP50 deltas unavailable"}
    checks = {
        "mean_primary_delta_positive": primary["mean"] > 0,
        "positive_in_at_least_two_seeds": primary["positive_seeds"] >= 2,
        "mean_manta_ap50_not_declining": manta["arm_means"]["noobj02"] >= manta["arm_means"]["baseline"],
    }
    return {"evaluated": True, "rule": "registered exploratory candidate rule", "checks": checks,
            "enters_candidate_configuration": all(checks.values()),
            "mean_primary_delta": primary["mean"], "primary_positive_seeds": primary["positive_seeds"],
            "mean_manta_ap50": manta["arm_means"],
            "non_claim": "n=3 seeds on one fixed validation split; this does not establish "
                         "statistical significance or generalization."}


def curves(root, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"baseline": "#1f77b4", "noobj02": "#d62728"}
    styles = {42: "-", 43: "--", 44: ":"}
    history = {}
    for arm in ARMS:
        for seed in SEEDS:
            log = root / "runs" / f"{arm}_seed{seed}" / "epoch_metrics.jsonl"
            if log.is_file():
                history[(arm, seed)] = [json.loads(line) for line in log.open(encoding="utf-8") if line.strip()]
    if not history:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for (arm, seed), rows in sorted(history.items()):
        label = f"no_obj={ARMS[arm]} seed{seed}"
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["train"]["loss"] for r in rows], color=colors[arm],
                     linestyle=styles[seed], linewidth=1.1, label=label)
        axes[1].plot(epochs, [r["val"]["map"] for r in rows], color=colors[arm],
                     linestyle=styles[seed], linewidth=1.1, label=label)
        axes[2].plot(epochs, [r["val"]["ap_by_iou"]["0.50"][1] for r in rows], color=colors[arm],
                     linestyle=styles[seed], linewidth=1.1, label=label)
    axes[0].set_yscale("log")
    for ax, title, ylabel in zip(axes,
            ("Train loss", "Validation map(.50,.75)", "Validation manta AP50"),
            ("Train loss (log)", "map(.50,.75)", "AP50 (manta)")):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    axes[1].legend(fontsize=7, ncol=2)
    path = figures / "training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def delta_figure(stats, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in DELTA_KEYS if k in stats]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for offset, seed in zip((-.26, 0, .26), SEEDS):
        values = [next((d for s, d in zip(stats[k]["seeds"], stats[k]["seed_deltas"]) if s == seed), 0)
                  for k in keys]
        ax.bar([i + offset for i in range(len(keys))], values, width=.24, label=f"seed{seed}")
    ax.plot(range(len(keys)), [stats[k]["mean"] for k in keys], "k_", markersize=22, label="mean")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Paired delta (no_obj 0.2 minus 0.1)")
    ax.set_title("Per-seed paired validation deltas at the selected checkpoint")
    ax.legend(fontsize=8)
    path = figures / "paired_metric_deltas.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def completion_table(rows, runs, failures):
    lines = ["# 六次运行完成表", "",
             "验证集选择检查点（最大 validation legacy map(.50,.75)）；未构建、未读取 test 划分。",
             "工作点阈值由验证集 F1 选出，属于工作点诊断，不是独立泛化评估。", "",
             "| 组别 | no_object_weight | seed | 状态 | epoch | 最佳 epoch | map(.50,.75) | AP50 | AP75 |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            key = (arm, seed)
            if key not in rows:
                lines.append(f"| {arm} | {ARMS[arm]} | {seed} | 失败/未完成 | — | — | — | — | — |")
                continue
            row, status = rows[key], runs[key]
            lines.append(f"| {arm} | {ARMS[arm]} | {seed} | {status['status']} | {status['epochs']} | "
                         f"{status['best_epoch']} | {fmt(row['map'])} | {fmt(row['map50'])} | {fmt(row['map75'])} |")
    lines += ["", "## 逐类 AP50 / AP75（选定检查点，验证集）", "",
              "| 组别 | seed | cylider AP50 | manta AP50 | cylider AP75 | manta AP75 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            row = rows.get((arm, seed))
            if not row:
                continue
            lines.append(f"| {arm} | {seed} | {fmt(row['ap50_cylider'])} | {fmt(row['ap50_manta'])} | "
                         f"{fmt(row['ap75_cylider'])} | {fmt(row['ap75_manta'])} |")
    lines += ["", "## 固定误检水平下的验证集召回率", "",
              "阈值在验证集上按“实际 FP/图不超过上限”挑选；FP 计入背景、重复、分类混淆与定位误检。", "",
              "| 组别 | seed | FP/图上限 | 选取阈值 | 实际 FP/图 | Recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            row = rows.get((arm, seed))
            if not row:
                continue
            for limit in FP_LIMITS:
                if row[f"recall_at_fp{limit}"] is None:
                    lines.append(f"| {arm} | {seed} | {limit} | — | — | 诊断未生成 |")
                    continue
                lines.append(f"| {arm} | {seed} | {limit} | {row[f'threshold_at_fp{limit}']:.2f} | "
                             f"{row[f'observed_fp_per_image{limit}']:.4f} | {fmt(row[f'recall_at_fp{limit}'], 4)} |")
    lines += ["", "## 工作点（验证集 F1 选择）", "",
              "| 组别 | seed | 阈值 | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in SEEDS:
            row = rows.get((arm, seed))
            if not row:
                continue
            lines.append(f"| {arm} | {seed} | {row['op_threshold']:.2f} | {fmt(row['op_precision'], 4)} | "
                         f"{fmt(row['op_recall'], 4)} | {fmt(row['op_f1'], 4)} |")
    lines += ["", "## 失败或未完成的运行", ""]
    lines += [f"- {f['run']}：{f['reason']}" for f in failures] or ["- 无。六次运行全部完成。"]
    return lines


def index_lines(runs, manifests, root):
    lines = ["# 配置与检查点索引", "",
             "远端执行平面 ssh jiajia@10.70.23.121:7889，研究根目录 "
             "`/home/jiajia/experiments/detection_fixed_normalizer_20260912`。", "",
             "| 运行 | 配置 | config sha256 | 初始权重 sha256 | 检查点 sha256 | 完成时间 |",
             "|---|---|---|---|---|---|"]
    for arm in ARMS:
        for seed in SEEDS:
            key = (arm, seed)
            if key not in runs:
                continue
            manifest, status = manifests[key], runs[key]
            lines.append(f"| {arm}_seed{seed} | configs/{arm}.yaml | `{manifest['config_sha256']}` | "
                         f"`{manifest['initial_state_sha256']}` | `{status['checkpoint_sha256']}` | "
                         f"{status['completed_at'][:19]} |")
    binding = next(iter(manifests.values()))["data_bindings"] if manifests else {}
    lines += ["", "## 数据绑定（六次运行一致）", ""]
    for split, value in binding.items():
        lines.append(f"- {split}：{value['images']} 张，sha256 `{value['sha256_images_and_labels']}`，"
                     f"类别 {value['class_names']}")
    lines += ["- test：本轮从未构建，也从未读取。", "",
              "## 检查点与产物路径（相对研究根目录）", "",
              "- `runs/<arm>_seed<seed>/checkpoints/best_model.pt`：验证集 map 最优权重。",
              "- `runs/<arm>_seed<seed>/epoch_metrics.jsonl`：80 epoch 逐轮训练与验证指标。",
              "- `runs/<arm>_seed<seed>/status.json`：完成状态、最佳验证指标、阈值扫描、检查点哈希。",
              "- `runs/<arm>_seed<seed>/run_manifest.json`：源码、配置、数据、初始权重哈希与种子控制。",
              "- `runs/<arm>_seed<seed>/valid_predictions.jsonl`：验证集逐图预测。",
              "- `diagnostics/<arm>_seed<seed>/diagnostics/`：逐类错误分解与固定误检召回率。",
              "- `snapshot/`：本轮源码快照；`source_snapshot_manifest.json` 记录 99 个文件哈希。",
              "- `study_manifest.json`、`launch.json`、`remote_preflight.json`、`queue_status.json`：队列与启动证据。",
              "- `takeover_20260913.json`：本次接管记录（沿用已有队列，未重复启动）。"]
    return lines


def conclusion_lines(rows, stats, gate, promo, failures):
    primary = stats.get("map", {})
    manta = stats.get("ap50_manta", {})
    lines = ["# 一页比较结论：修正归一化后 no_object_weight 0.1 与 0.2", "",
             "## 结论", ""]
    if not gate["comparison_permitted"]:
        lines += ["配对完整性未通过，不给出比较结论。未通过项见下方完整性表。", ""]
    elif promo.get("enters_candidate_configuration"):
        lines += [f"在修正归一化公式后，no_object_weight=0.2 相对 0.1 的验证主指标平均差值为 "
                  f"{primary['mean']:+.6f}（配对差值标准差 {primary.get('std', float('nan')):.6f}，"
                  f"{primary['positive_seeds']}/3 个种子为正），平均 manta AP50 由 "
                  f"{manta['arm_means']['baseline']:.6f} 变为 {manta['arm_means']['noobj02']:.6f}，"
                  "三项已登记条件全部满足，因此 0.2 进入候选配置。", "",
                  "这只是候选资格。三个种子共用同一验证集，不能据此宣称统计显著提升，也不能宣称泛化成立。", ""]
    else:
        failed = [k for k, v in promo["checks"].items() if not v]
        lines += [f"no_object_weight=0.2 相对 0.1 的验证主指标平均差值为 {primary['mean']:+.6f}"
                  f"（{primary['positive_seeds']}/3 个种子为正），未满足已登记候选规则中的：{', '.join(failed)}。"
                  "因此 0.2 不进入候选配置。", "",
                  "同样地，这不构成 0.2 更差的显著性结论；三个种子只描述优化随机性。", ""]
    lines += ["## 主比较（0.2 − 0.1，配对同种子）", "",
              "| 指标 | seed42 | seed43 | seed44 | 均值 | 标准差 | 正向种子 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key in DELTA_KEYS:
        entry = stats.get(key)
        if not entry:
            continue
        per_seed = dict(zip(entry["seeds"], entry["seed_deltas"]))
        cells = " | ".join(f"{per_seed[s]:+.6f}" if s in per_seed else "—" for s in SEEDS)
        lines.append(f"| {key} | {cells} | {entry['mean']:+.6f} | "
                     f"{entry.get('std', float('nan')):.6f} | {entry['positive_seeds']}/{len(entry['seeds'])} |")
    lines += ["", "## 两组均值", "", "| 指标 | no_obj=0.1 均值 | no_obj=0.2 均值 |", "|---|---:|---:|"]
    for key in DELTA_KEYS:
        entry = stats.get(key)
        if not entry:
            continue
        lines.append(f"| {key} | {entry['arm_means']['baseline']:.6f} | {entry['arm_means']['noobj02']:.6f} |")
    if primary.get("seed_t95_interval"):
        lines += ["", f"主指标按种子计算的 t 95% 区间 = [{primary['seed_t95_interval'][0]:+.6f}, "
                      f"{primary['seed_t95_interval'][1]:+.6f}]；精确双侧符号翻转 p = "
                      f"{primary['exact_sign_flip_two_sided_p']:.3f}（8 种符号组合，n=3 时最小可达 0.25）。",
                  "该区间只描述优化随机性，正态假设无法用 3 个样本检验。"]
    lines += ["", "## 已登记候选规则判定", ""]
    if promo.get("evaluated"):
        for name, passed in promo["checks"].items():
            lines.append(f"- {name}：{'满足' if passed else '不满足'}")
        lines.append(f"- 进入候选配置：{'是' if promo['enters_candidate_configuration'] else '否'}")
    lines += ["", "## 配对完整性", ""]
    for name in ("all_six_runs_complete", "identical_data_bindings", "identical_architecture",
                 "all_runs_at_budget", "objectness_normalization_fixed_weights",
                 "only_no_object_weight_differs", "test_never_accessed"):
        lines.append(f"- {name}：{gate[name]}")
    lines.append(f"- same_initial_state_each_seed：{gate['same_initial_state_each_seed']}")
    lines.append(f"- 运行间除 no_object_weight 外的损失差异项：{gate['loss_keys_differing_across_runs'] or '无'}")
    lines += ["", "## 边界与不声明", "",
              "- 仅验证集；test 划分本轮从未构建或读取。",
              "- 3 个种子、同一验证集：不宣称统计显著提升，也不宣称泛化成立。",
              "- 旧公式（legacy_dynamic）的队列与 150 epoch 运行只作历史诊断，不并入本表统计。",
              "- 绝对 map 量级仍很低（约 0.02–0.04），本轮只比较相同预算下的相对差异，未声明可用检测精度。",
              "- 未改动骨干、固定 Neck/Head、锚框、数据划分与标注；未开展算子搜索、HLS 布线、板上运行或功耗测量。"]
    if failures:
        lines += ["", "## 失败运行", ""] + [f"- {f['run']}：{f['reason']}" for f in failures]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs, manifests, diags, failures = collect(root)
    rows = {key: metrics(runs[key], diags.get(key)) for key in runs}
    gate = check_integrity(runs, manifests, root)
    stats = paired_stats(rows) if gate["comparison_permitted"] else {}
    promo = promotion(stats) if stats else {"evaluated": False, "reason": "comparison not permitted"}
    (root / "six_run_completion_table.md").write_text(
        "\n".join(completion_table(rows, runs, failures)) + "\n", encoding="utf-8")
    (root / "config_checkpoint_index.md").write_text(
        "\n".join(index_lines(runs, manifests, root)) + "\n", encoding="utf-8")
    (root / "comparison_conclusion.md").write_text(
        "\n".join(conclusion_lines(rows, stats, gate, promo, failures)) + "\n", encoding="utf-8")
    figure_paths = []
    if not args.skip_figures:
        figure_paths = [str(p) for p in (curves(root, figures), delta_figure(stats, figures)) if p]
    payload = {"artifact_type": "detection_fixed_normalizer_six_run_deliverables", "schema_version": 1,
               "study": "detection_fixed_normalizer_20260912", "arms": ARMS, "seeds": list(SEEDS),
               "epochs_per_run": BUDGET, "completed_runs": len(runs), "expected_runs": len(ARMS) * len(SEEDS),
               "selection": "maximum validation legacy map(.50,.75)", "test_accessed": False,
               "per_run_metrics": {f"{a}_seed{s}": rows[(a, s)] for a, s in rows},
               "per_run_status": {f"{a}_seed{s}": {k: runs[(a, s)][k] for k in
                                                   ("status", "epochs", "best_epoch", "checkpoint_sha256", "completed_at")}
                                  for a, s in runs},
               "paired_integrity": gate, "paired_deltas": stats, "promotion_rule": promo,
               "failures": failures, "figures": figure_paths,
               "claim_boundary": "Validation-split comparison of one loss hyperparameter at fixed budget. "
                                 "No test split, no significance or generalization claim from n=3 seeds, "
                                 "no HLS route, board runtime or measured power."}
    (root / "six_run_deliverables.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"completed_runs": len(runs), "comparison_permitted": gate["comparison_permitted"],
                      "candidate": promo.get("enters_candidate_configuration"),
                      "mean_primary_delta": stats.get("map", {}).get("mean"), "failures": failures},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
