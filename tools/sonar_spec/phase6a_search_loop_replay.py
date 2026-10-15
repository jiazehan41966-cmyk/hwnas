"""Phase 6A: warm-start surrogate search-loop replay for X_V0.1.

No training is launched. The script replays architecture selection using existing
Phase2E 80-epoch standalone results as ground truth labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hwnas_fpga.models.sonar_supernet import enumerate_architectures, architecture_id  # noqa: E402

REPORT = ROOT / "reports" / "phase6a_search_loop_v0"
FINAL = ROOT / "reports" / "phase2e_multifidelity" / "final_results.csv"
STAGE1 = ROOT / "reports" / "phase2e_multifidelity" / "stage1_results.csv"
PHASE2F = ROOT / "reports" / "phase2f_multiseed" / "multiseed_results.csv"
H4_COST = ROOT / "reports" / "fpga_h4_tooling_v2" / "NETWORK_COST_SUMMARY.csv"
SEARCH_SPACE = ROOT / "configs" / "search_space_sonar_v0_1.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def parse_int(row: dict[str, str], key: str, default: int = 0) -> int:
    v = row.get(key, "")
    return int(float(v)) if v not in ("", None) else default


def arch_features(arch: dict[str, Any], params: float, macs: float) -> dict[str, float]:
    combo = arch["output_combination"]
    kernels = arch["kernels"]
    stages = [1, 2, 4, 5, 6, 7]
    feats: dict[str, float] = {}
    for name in ("P3_P4_P5", "P2_P3_P4", "P2_P3_P4_P5"):
        feats[f"combo_{name}"] = 1.0 if combo == name else 0.0
    feats["has_P2"] = 1.0 if combo in ("P2_P3_P4", "P2_P3_P4_P5") else 0.0
    feats["has_P5"] = 1.0 if combo in ("P3_P4_P5", "P2_P3_P4_P5") else 0.0
    feats["num_heads"] = {"P3_P4_P5": 3, "P2_P3_P4": 3, "P2_P3_P4_P5": 4}[combo]
    feats["last_stage"] = {"P3_P4_P5": 7, "P2_P3_P4": 5, "P2_P3_P4_P5": 7}[combo]
    active = set(int(k.replace("stage", "")) for k in kernels)
    k5_count = 0
    early_k5 = 0
    deep_k5 = 0
    for s in stages:
        k = int(kernels.get(f"stage{s}", 3)) if s in active else 0
        feats[f"stage{s}_active"] = 1.0 if s in active else 0.0
        feats[f"stage{s}_k5"] = 1.0 if k == 5 else 0.0
        feats[f"stage{s}_kernel_value"] = float(k)
        if k == 5:
            k5_count += 1
            if s in (1, 2): early_k5 += 1
            if s in (5, 6, 7): deep_k5 += 1
    feats["num_k5"] = float(k5_count)
    feats["early_k5_count"] = float(early_k5)
    feats["deep_k5_count"] = float(deep_k5)
    feats["Params"] = float(params)
    feats["MACs"] = float(macs)
    return feats


def pareto_ids(rows: list[dict[str, Any]], acc_key="mAP50_95", cost_key="MACs") -> set[str]:
    ids = set()
    for r in rows:
        dominated = False
        for q in rows:
            if q["architecture_id"] == r["architecture_id"]: continue
            if float(q[acc_key]) >= float(r[acc_key]) and float(q[cost_key]) <= float(r[cost_key]) and (
                float(q[acc_key]) > float(r[acc_key]) or float(q[cost_key]) < float(r[cost_key])
            ):
                dominated = True; break
        if not dominated:
            ids.add(r["architecture_id"])
    return ids


def build_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archs = {architecture_id(a): a for a in enumerate_architectures()}
    if len(archs) != 144:
        raise ValueError(f"Expected 144 architectures, got {len(archs)}")
    stage1 = {r["architecture_id"]: r for r in read_csv(STAGE1)}
    final = {r["architecture_id"]: r for r in read_csv(FINAL)}
    rows: list[dict[str, Any]] = []
    for aid, arch in sorted(archs.items()):
        s1 = stage1.get(aid, {})
        params = parse_int(s1, "Params", 0)
        macs = parse_int(s1, "MACs", 0)
        f = final.get(aid)
        row = {
            "architecture_id": aid,
            "output_combination": arch["output_combination"],
            "kernels": json.dumps(arch["kernels"], sort_keys=True),
            "Params": params,
            "MACs": macs,
            "has_final80": f is not None,
            "mAP50_95": parse_float(f, "mAP50_95") if f else math.nan,
            "AP50": parse_float(f, "AP50") if f else math.nan,
            "AP75": parse_float(f, "AP75") if f else math.nan,
            "Recall": parse_float(f, "Recall") if f else math.nan,
            "best_epoch": parse_int(f, "best_epoch", 0) if f else 0,
        }
        row.update(arch_features(arch, params, macs))
        rows.append(row)
    meta = {
        "n_architectures": len(rows),
        "n_stage1_cost_rows": len(stage1),
        "n_final80_ground_truth": sum(1 for r in rows if r["has_final80"]),
        "h4_network_cost_rows": len(read_csv(H4_COST)) if H4_COST.exists() else 0,
        "h4_direct_144_coverage": False,
        "cost_for_replay": "Params/MACs from Phase2E stage1/final rows; H4 FPGA cost is shortlist-level only, not 144 architecture-level",
    }
    return rows, meta


def matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(r.get(k, 0.0)) for k in feature_names] for r in rows], dtype=float)


def fit_predict_ensemble(train_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], feature_names: list[str], seed: int, n_models: int = 96) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = matrix(train_rows, feature_names)
    y = np.asarray([float(r["mAP50_95"]) for r in train_rows], dtype=float)
    Xc = matrix(cand_rows, feature_names)
    # Normalize from train only.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xn = (X - mu) / sd
    Xcn = (Xc - mu) / sd
    Xn = np.concatenate([np.ones((len(Xn), 1)), Xn], axis=1)
    Xcn = np.concatenate([np.ones((len(Xcn), 1)), Xcn], axis=1)
    preds = []
    n = len(train_rows)
    # Bootstrap ridge ensemble. Feature bagging gives non-zero uncertainty for tiny n.
    for i in range(n_models):
        idx = rng.integers(0, n, size=n)
        feat_mask = np.ones(Xn.shape[1], dtype=bool)
        if Xn.shape[1] > 8:
            keep = rng.random(Xn.shape[1]) < 0.75
            keep[0] = True
            # Always keep cost and macro features sometimes lost? bootstrap handles.
            feat_mask = keep
        lam = 10 ** rng.uniform(-3, 0.7)
        A = Xn[idx][:, feat_mask]
        b = y[idx]
        reg = lam * np.eye(A.shape[1]); reg[0, 0] = 0.0
        try:
            coef = np.linalg.solve(A.T @ A + reg, A.T @ b)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(A.T @ A + reg) @ A.T @ b
        full = np.zeros(Xn.shape[1])
        full[feat_mask] = coef
        preds.append(Xcn @ full)
    P = np.vstack(preds)
    return P.mean(axis=0), P.std(axis=0)


def normalize_values(vals: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals))
    if hi - lo < 1e-12:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


def acquisition(mu: np.ndarray, sigma: np.ndarray, cand_rows: list[dict[str, Any]], observed_rows: list[dict[str, Any]], beta: float = 0.35) -> np.ndarray:
    macs = np.asarray([float(r["MACs"]) for r in cand_rows])
    acc_n = normalize_values(mu)
    unc_n = normalize_values(sigma)
    cost_good = 1.0 - normalize_values(macs)
    # Pareto-potential proxy: high predicted acc, lower cost than many observed points, plus uncertainty.
    obs_acc = np.asarray([float(r["mAP50_95"]) for r in observed_rows])
    obs_macs = np.asarray([float(r["MACs"]) for r in observed_rows])
    best_obs = float(obs_acc.max())
    improvement = np.maximum(mu - best_obs, 0.0)
    imp_n = normalize_values(improvement)
    cheaper_fraction = np.asarray([(obs_macs >= m).mean() for m in macs])
    return 0.45 * acc_n + 0.20 * cost_good + 0.20 * imp_n + 0.15 * cheaper_fraction + beta * unc_n


def deterministic_initial_ids(gt_rows: list[dict[str, Any]], n: int = 20, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in gt_rows:
        groups.setdefault(r["output_combination"], []).append(r)
    selected: list[str] = []
    # stratify macro and mAP quantiles, but deterministic and not model-informed at replay time.
    quotas = {"P3_P4_P5": 7, "P2_P3_P4": 3, "P2_P3_P4_P5": 10}
    for combo, quota in quotas.items():
        rows = sorted(groups.get(combo, []), key=lambda r: (float(r["MACs"]), r["architecture_id"]))
        if not rows: continue
        bins = np.array_split(rows, quota)
        for b in bins:
            if len(b):
                selected.append(rng.choice(list(b))["architecture_id"])
    # Fill if exact quotas impossible.
    remaining = [r["architecture_id"] for r in gt_rows if r["architecture_id"] not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, n - len(selected))])
    return selected[:n]


def run_replay(rows: list[dict[str, Any]], batch: int = 6, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gt_rows = [r for r in rows if r["has_final80"]]
    feature_names = [k for k in rows[0] if k.startswith("combo_") or k.startswith("stage") or k in ("has_P2","has_P5","num_heads","last_stage","num_k5","early_k5_count","deep_k5_count","Params","MACs")]
    true_top = sorted(gt_rows, key=lambda r: float(r["mAP50_95"]), reverse=True)
    top3 = {r["architecture_id"] for r in true_top[:3]}
    top5 = {r["architecture_id"] for r in true_top[:5]}
    true_pareto = pareto_ids(gt_rows)
    observed_ids = set(deterministic_initial_ids(gt_rows, 20, seed))
    replay_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    step = 0
    while True:
        observed = [r for r in gt_rows if r["architecture_id"] in observed_ids]
        best = max(float(r["mAP50_95"]) for r in observed)
        best_id = max(observed, key=lambda r: float(r["mAP50_95"]))["architecture_id"]
        budget = len(observed)
        selected_rows.extend([])
        replay_rows.append({
            "step": step,
            "evaluated": budget,
            "best_mAP50_95": best,
            "best_architecture_id": best_id,
            "regret_to_true_best": float(true_top[0]["mAP50_95"]) - best,
            "top3_recall": len(top3 & observed_ids) / len(top3),
            "top5_recall": len(top5 & observed_ids) / len(top5),
            "pareto_recall": len(true_pareto & observed_ids) / len(true_pareto),
        })
        if budget >= len(gt_rows):
            break
        cand = [r for r in gt_rows if r["architecture_id"] not in observed_ids]
        mu, sig = fit_predict_ensemble(observed, cand, feature_names, seed + step)
        score = acquisition(mu, sig, cand, observed)
        order = np.argsort(-score)
        chosen = [cand[i] for i in order[: min(batch, len(cand))]]
        for rank, r in enumerate(chosen, 1):
            selected_rows.append({
                "step": step + 1,
                "selection_rank": rank,
                "architecture_id": r["architecture_id"],
                "pred_mu": float(mu[order[rank-1]]),
                "pred_sigma": float(sig[order[rank-1]]),
                "acquisition": float(score[order[rank-1]]),
                "true_mAP50_95": float(r["mAP50_95"]),
                "MACs": int(r["MACs"]),
                "Params": int(r["Params"]),
                "output_combination": r["output_combination"],
            })
            observed_ids.add(r["architecture_id"])
        step += 1
    meta = {"feature_names": feature_names, "true_top3": list(top3), "true_top5": list(top5), "true_pareto": sorted(true_pareto)}
    return replay_rows, selected_rows, meta


def random_baseline(rows: list[dict[str, Any]], batch: int = 6, n_trials: int = 500, seed: int = 123) -> list[dict[str, Any]]:
    gt_rows = [r for r in rows if r["has_final80"]]
    true_top = sorted(gt_rows, key=lambda r: float(r["mAP50_95"]), reverse=True)
    top3 = {r["architecture_id"] for r in true_top[:3]}
    top5 = {r["architecture_id"] for r in true_top[:5]}
    true_pareto = pareto_ids(gt_rows)
    budgets = list(range(20, len(gt_rows)+1, batch))
    if budgets[-1] != len(gt_rows): budgets.append(len(gt_rows))
    rng = random.Random(seed)
    out = []
    for trial in range(n_trials):
        ids = [r["architecture_id"] for r in gt_rows]
        rng.shuffle(ids)
        for b in budgets:
            obs = set(ids[:b])
            observed = [r for r in gt_rows if r["architecture_id"] in obs]
            best = max(float(r["mAP50_95"]) for r in observed)
            out.append({
                "trial": trial,
                "evaluated": b,
                "best_mAP50_95": best,
                "regret_to_true_best": float(true_top[0]["mAP50_95"]) - best,
                "top3_recall": len(top3 & obs) / len(top3),
                "top5_recall": len(top5 & obs) / len(top5),
                "pareto_recall": len(true_pareto & obs) / len(true_pareto),
            })
    return out


def summarize_random(random_rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    budgets = sorted({int(r["evaluated"]) for r in random_rows})
    summary = {}
    for b in budgets:
        group = [r for r in random_rows if int(r["evaluated"]) == b]
        entry = {"n_trials": len(group)}
        for key in ("best_mAP50_95", "regret_to_true_best", "top3_recall", "top5_recall", "pareto_recall"):
            vals = np.asarray([float(r[key]) for r in group])
            entry[f"{key}_median"] = float(np.median(vals))
            entry[f"{key}_p25"] = float(np.quantile(vals, .25))
            entry[f"{key}_p75"] = float(np.quantile(vals, .75))
        summary[b] = entry
    return summary


def recommend_next(rows: list[dict[str, Any]], batch: int = 6, seed: int = 42) -> list[dict[str, Any]]:
    observed = [r for r in rows if r["has_final80"]]
    candidates = [r for r in rows if not r["has_final80"]]
    feature_names = [k for k in rows[0] if k.startswith("combo_") or k.startswith("stage") or k in ("has_P2","has_P5","num_heads","last_stage","num_k5","early_k5_count","deep_k5_count","Params","MACs")]
    mu, sig = fit_predict_ensemble(observed, candidates, feature_names, seed + 9000)
    score = acquisition(mu, sig, candidates, observed)
    order = np.argsort(-score)
    recs = []
    for rank, idx in enumerate(order[:batch], 1):
        r = candidates[idx]
        recs.append({
            "rank": rank,
            "architecture_id": r["architecture_id"],
            "output_combination": r["output_combination"],
            "pred_mu_mAP50_95": float(mu[idx]),
            "pred_sigma": float(sig[idx]),
            "acquisition": float(score[idx]),
            "Params": int(r["Params"]),
            "MACs": int(r["MACs"]),
            "kernels": r["kernels"],
            "recommendation_status": "CANDIDATE_ONLY_REQUIRES_REPLAY_ADMISSION",
        })
    return recs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--initial-size", type=int, default=20)
    p.add_argument("--random-trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, data_meta = build_dataset()
    write_csv(REPORT / "SEARCH_LOOP_V0_DATASET.csv", rows)
    protocol = {
        "phase": "6A",
        "search_loop_version": "Search Loop V0",
        "search_space": "X_V0.1 = 144 architectures",
        "search_space_sha256": sha256(SEARCH_SPACE) if SEARCH_SPACE.exists() else None,
        "accuracy_evidence": {
            "phase2e_final80": str(FINAL.relative_to(ROOT)),
            "phase2e_final80_sha256": sha256(FINAL),
            "phase2f_multiseed": str(PHASE2F.relative_to(ROOT)) if PHASE2F.exists() else None,
            "phase2f_multiseed_sha256": sha256(PHASE2F) if PHASE2F.exists() else None,
        },
        "cost_evidence": {
            "direct_replay_cost": "Params and MACs for all 144 from Phase2E stage1/stage2/final metadata",
            "stage1_sha256": sha256(STAGE1),
            "h4_network_cost_summary": str(H4_COST.relative_to(ROOT)) if H4_COST.exists() else None,
            "h4_network_cost_sha256": sha256(H4_COST) if H4_COST.exists() else None,
            "h4_144_direct_coverage": False,
            "h4_use_in_phase6a": "audit only; not acquisition input because it covers shortlist networks/mapping policies, not all 144 architectures",
        },
        "surrogate": "bootstrap ridge ensemble over architecture encoding + Params/MACs",
        "replay": {"ground_truth": "72 architectures with 80-epoch standalone mAP50:95", "initial_size": args.initial_size, "batch_size": args.batch_size, "random_trials": args.random_trials, "seed": args.seed},
        "admission_rule": "Admit Phase6B only if replay improves best/regret/top-k/Pareto discovery over random baseline; otherwise keep offline only.",
        "no_training_launched": True,
        "test_evaluated": False,
    }
    (REPORT / "SEARCH_LOOP_V0_PROTOCOL.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8")
    replay, selected, replay_meta = run_replay(rows, batch=args.batch_size, seed=args.seed)
    random_rows = random_baseline(rows, batch=args.batch_size, n_trials=args.random_trials, seed=args.seed + 1)
    random_summary = summarize_random(random_rows)
    for r in replay:
        b = int(r["evaluated"])
        if b in random_summary:
            r.update({f"random_{k}": v for k, v in random_summary[b].items()})
            r["beats_random_median_regret"] = float(r["regret_to_true_best"]) <= random_summary[b]["regret_to_true_best_median"]
            r["beats_random_p75_best"] = float(r["best_mAP50_95"]) >= random_summary[b]["best_mAP50_95_p75"]
    write_csv(REPORT / "REPLAY_TRAJECTORY.csv", replay)
    write_csv(REPORT / "REPLAY_SELECTED_BATCHES.csv", selected)
    write_csv(REPORT / "RANDOM_BASELINE_SUMMARY.csv", [{"evaluated": b, **v} for b, v in random_summary.items()])
    recs = recommend_next(rows, batch=args.batch_size, seed=args.seed)
    write_csv(REPORT / "NEXT_BATCH_RECOMMENDATIONS.csv", recs)
    # Admission summary.
    checkpoints = [r for r in replay if int(r["evaluated"]) in (20, 32, 44, 56, 68, 72)]
    comparable = [r for r in checkpoints if "random_regret_to_true_best_median" in r]
    beat_regret = sum(bool(r["beats_random_median_regret"]) for r in comparable)
    beat_top5 = sum(float(r["top5_recall"]) >= float(r.get("random_top5_recall_median", 999)) for r in comparable)
    final = replay[-1]
    # Keep conservative: require consistent regret improvement and top5 not worse at most checkpoints.
    admitted = beat_regret >= math.ceil(0.67 * len(comparable)) and beat_top5 >= math.ceil(0.67 * len(comparable))
    admission = {
        "status": "PASS" if admitted else "FAIL",
        "phase6b_training_allowed": bool(admitted),
        "n_final80_ground_truth": data_meta["n_final80_ground_truth"],
        "batch_size": args.batch_size,
        "checkpoints_considered": len(comparable),
        "beat_random_median_regret_checkpoints": beat_regret,
        "beat_random_top5_recall_checkpoints": beat_top5,
        "final_replay": final,
        "true_top3": replay_meta["true_top3"],
        "true_top5": replay_meta["true_top5"],
        "true_pareto_count": len(replay_meta["true_pareto"]),
        "cost_scope": data_meta["cost_for_replay"],
        "test_evaluated": False,
    }
    dump_json(REPORT / "SEARCH_LOOP_V0_ADMISSION.json", admission)
    dump_json(REPORT / "PHASE6A_MANIFEST.json", {"protocol": protocol, "data_meta": data_meta, "admission": admission})
    # Markdown report.
    best_rows = sorted([r for r in rows if r["has_final80"]], key=lambda r: float(r["mAP50_95"]), reverse=True)[:10]
    report = []
    report.append("# Phase 6A — Warm-start Surrogate Search Loop Replay\n")
    report.append("No new training was launched. Test remains sealed (`test_evaluated=false`).\n")
    report.append(f"- X_V0.1 architectures: {data_meta['n_architectures']}\n")
    report.append(f"- 80-epoch ground truth for replay: {data_meta['n_final80_ground_truth']}\n")
    report.append(f"- Cost used in replay: {data_meta['cost_for_replay']}\n")
    report.append(f"- H4 network-cost rows found: {data_meta['h4_network_cost_rows']}; direct 144 coverage: false\n")
    report.append("\n## Replay admission\n")
    report.append(f"Status: **{admission['status']}**\n")
    report.append(f"Beat random median regret checkpoints: {beat_regret}/{len(comparable)}\n")
    report.append(f"Beat random top5-recall checkpoints: {beat_top5}/{len(comparable)}\n")
    report.append("\n## Replay trajectory\n\n")
    report.append("| evaluated | best mAP50:95 | regret | top3 recall | top5 recall | pareto recall | random median regret |\n")
    report.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in replay:
        if int(r["evaluated"]) in (20, 26, 32, 38, 44, 50, 56, 62, 68, 72):
            report.append(f"| {r['evaluated']} | {float(r['best_mAP50_95']):.6f} | {float(r['regret_to_true_best']):.6f} | {float(r['top3_recall']):.2f} | {float(r['top5_recall']):.2f} | {float(r['pareto_recall']):.2f} | {float(r.get('random_regret_to_true_best_median', float('nan'))):.6f} |\n")
    report.append("\n## True top 10 among the 72 final80 architectures\n\n")
    report.append("| rank | architecture_id | mAP50:95 | MACs | Params |\n|---:|---|---:|---:|---:|\n")
    for i, r in enumerate(best_rows, 1):
        report.append(f"| {i} | `{r['architecture_id']}` | {float(r['mAP50_95']):.6f} | {int(r['MACs'])} | {int(r['Params'])} |\n")
    report.append("\n## Next-batch recommendations if admission is accepted\n\n")
    report.append("These are recommendations only. If admission is FAIL, do not train them as a closed-loop claim.\n\n")
    report.append("| rank | architecture_id | pred μ | pred σ | MACs |\n|---:|---|---:|---:|---:|\n")
    for r in recs:
        report.append(f"| {r['rank']} | `{r['architecture_id']}` | {r['pred_mu_mAP50_95']:.6f} | {r['pred_sigma']:.6f} | {r['MACs']} |\n")
    (REPORT / "PHASE6A_REPLAY_REPORT.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps({"status": admission["status"], "phase6b_training_allowed": admitted, "report": str((REPORT / "PHASE6A_REPLAY_REPORT.md").relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
