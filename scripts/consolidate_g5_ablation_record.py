"""Consolidate the G5 sonar-operator ablations (v1 + v2) into a durable,
version-controlled record under artifacts/ (results/ is gitignored).

Records per-run outer macro_f1 by (fold, seed), paired deltas vs the matched
control, fold-cluster sign-flip p-values, and Holm-adjusted p-values.
"""
import glob, json, math, statistics, random, sys
from pathlib import Path

ROOT = Path(r"E:\1\hwnas\hwnas")
sys.path.insert(0, str(ROOT / "src"))
from hwnas_fpga.hardware.sonar_operator_gate import holm_adjust

CAMPAIGNS = {
    "v1_original_operators": {
        "prefix": "g5_ablation_",
        "control": "mbconv_control",
        "variants": ["denoise", "edge", "denoise_edge"],
        "note": "v1 operators: DenoiseBlock (fixed Gaussian smoothing prior) and "
                "EdgeAwareBlock (4-direction Sobel, discards intensity/DC).",
    },
    "v2_redesigned_operators": {
        "prefix": "g5v2_",
        "control": "mbconv_control_v2",
        "variants": ["adaptive_denoise", "edge_v2", "adaptive_denoise_edge_v2"],
        "note": "v2 redesign: AdaptiveDenoiseBlock (Lee-style learnable gate on "
                "pooled edge evidence) and EdgeAugmentBlock (intensity main path "
                "plus additive edge branch, gamma initialised to 0).",
    },
}


def load_runs(run_dir):
    out = {}
    for f in sorted(glob.glob(str(run_dir / "run_fold*_seed*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        out[(int(r["fold"]), int(r["seed"]))] = {
            "outer_macro_f1": r["outer_val"]["macro_f1"],
            "outer_top1": r["outer_val"]["top1"],
            "inner_macro_f1": r["inner_val"]["macro_f1"],
            "best_epoch": r.get("best_epoch"),
        }
    return out


def sign_flip_p(folds, observed, iters=20000, seed=42):
    random.seed(seed)
    keys = sorted(folds)
    null = []
    for _ in range(iters):
        s = []
        for f in keys:
            sg = random.choice([-1, 1])
            s.extend(sg * x for x in folds[f])
        null.append(statistics.fmean(s))
    return (sum(1 for x in null if abs(x) >= abs(observed)) + 1) / (iters + 1)


payload = {
    "schema_version": 1,
    "record": "G5 sonar operator ablations (complete)",
    "protocol": "nksid_outer5fold_inner_contiguous_v1",
    "design": "2x2 factorial on stage-3 slots A/B of sonar_ablation_backbone_v1 "
              "(rl_arch_135 encoding, stage-3 depth 2)",
    "statistics": "paired per (fold,seed); fold-cluster sign-flip permutation "
                  "(20000 iters, seed 42); Holm across the 3 comparisons",
    "campaigns": {},
}

for cname, cfg in CAMPAIGNS.items():
    base = ROOT / "results/protocol"
    ctrl = load_runs(base / (cfg["prefix"] + cfg["control"]))
    if not ctrl:
        print(f"skip {cname}: no control records")
        continue
    entry = {
        "note": cfg["note"],
        "control_run": cfg["prefix"] + cfg["control"],
        "control": {
            "n": len(ctrl),
            "outer_macro_f1_mean": statistics.fmean(v["outer_macro_f1"] for v in ctrl.values()),
            "outer_macro_f1_std": statistics.stdev(v["outer_macro_f1"] for v in ctrl.values()),
            "per_run": {f"fold{f}_seed{s}": round(v["outer_macro_f1"], 6) for (f, s), v in sorted(ctrl.items())},
        },
        "comparisons": {},
    }
    raw = {}
    for v in cfg["variants"]:
        runs = load_runs(base / (cfg["prefix"] + v))
        if not runs:
            continue
        folds = {}
        for (f, s), val in runs.items():
            if (f, s) in ctrl:
                folds.setdefault(f, []).append(val["outer_macro_f1"] - ctrl[(f, s)]["outer_macro_f1"])
        deltas = [x for lst in folds.values() for x in lst]
        mean = statistics.fmean(deltas)
        p = sign_flip_p(folds, mean)
        raw[v] = p
        entry["comparisons"][v] = {
            "run": cfg["prefix"] + v,
            "n_paired": len(deltas),
            "outer_macro_f1_mean": statistics.fmean(x["outer_macro_f1"] for x in runs.values()),
            "outer_macro_f1_std": statistics.stdev(x["outer_macro_f1"] for x in runs.values()),
            "paired_delta_mean": mean,
            "paired_delta_std": statistics.stdev(deltas),
            "wins": sum(1 for d in deltas if d > 0),
            "per_fold_mean_delta": {f"fold{f}": round(statistics.fmean(folds[f]), 6) for f in sorted(folds)},
            "p_value_sign_flip": p,
            "per_run": {f"fold{f}_seed{s}": round(val["outer_macro_f1"], 6) for (f, s), val in sorted(runs.items())},
        }
    if raw:
        names = list(raw)
        for n, adj in zip(names, holm_adjust([raw[n] for n in names])):
            entry["comparisons"][n]["p_value_holm"] = adj
            entry["comparisons"][n]["holm_significant"] = adj < 0.05
            entry["comparisons"][n]["passes_g5_gain_gate"] = (
                adj < 0.05 and entry["comparisons"][n]["paired_delta_mean"] > 0
            )
    payload["campaigns"][cname] = entry

payload["verdict"] = {
    "denoise_family": "No classification benefit. v1 -0.012 (slight harm) -> v2 +0.0014 "
                      "(exactly neutral, p=0.81, 8/15 wins). The redesign removed the harm, "
                      "confirming the fixed-Gaussian diagnosis, but the prior is redundant "
                      "with what mbconv already learns.",
    "edge_family": "No classification benefit. v1 -0.092 (clearly harmful) -> v2 -0.040 "
                   "(harm halved, 1/15 wins, all 5 folds negative, Holm p=0.19). The additive "
                   "intensity-preserving redesign fixed the DC-loss mechanism but did not turn "
                   "the operator positive. Confounded: edge_v2 also has 29% fewer stage-3 MACs "
                   "than its control.",
    "gate_status": "Neither operator meets the G5 admission bar (Holm-significant AND delta>0). "
                   "Both remain excluded from the formal search space.",
    "dataset_context": "NKSID chips are pre-cropped and resampled 0.78x-4.88x per class, so real "
                       "speckle is largely destroyed (ENL 5.25). This explains why an image-domain "
                       "denoise/edge prior cannot help here, and means the null result must NOT be "
                       "extrapolated to full-scene side-scan data (Figshare ENL 3.51, Roboflow 0.35).",
}

out = ROOT / "artifacts/sonar_operator_gate/g5_ablation_complete_record.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"written: {out}")
for c, e in payload["campaigns"].items():
    print(f"\n[{c}] control mean={e['control']['outer_macro_f1_mean']:.4f} (n={e['control']['n']})")
    for v, d in e["comparisons"].items():
        print(f"  {v:26s} delta={d['paired_delta_mean']:+.4f} wins={d['wins']}/{d['n_paired']} "
              f"p={d['p_value_sign_flip']:.4f} holm={d.get('p_value_holm', float('nan')):.4f}")
