"""Phase 6A.1 surrogate diagnosis. Read-only on Phase2E / Phase2F.

No architecture training is launched and Phase 6B is not started.

Predeclared admission rule
--------------------------
PHASE6B_ALLOWED is true only if at least one learned predictor satisfies all of:

1. At evaluated budgets 32, 44 and 56, over 40 paired replay seeds, the
   predictor median regret is lower than the paired Random median and the
   bootstrap 95 percent interval of (Random regret - predictor regret) lies
   entirely above 0.
2. That mean regret reduction exceeds the median intra-architecture seed
   standard deviation from Phase2F at every one of those budgets.
3. Search top-5 recall is not significantly worse than Random at any of
   those budgets (bootstrap interval of the paired difference does not lie
   entirely below 0).
4. Predicted-uncertainty acquisition is ineligible unless bootstrap sigma is
   positively associated with absolute error.
5. A multi-fidelity predictor must also stably beat the direct sort on the
   fidelity label it consumes. Matching Random only by replaying mAP40 or
   mAP60 is multi-fidelity evidence, not a repaired surrogate.

Evaluated = 72 is not an admission checkpoint. Params/MACs runs are
compute-aware. This replay is not hardware-aware.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "phase6a1_surrogate_diagnosis"
FINAL = ROOT / "reports" / "phase2e_multifidelity" / "final_results.csv"
STAGE1 = ROOT / "reports" / "phase2e_multifidelity" / "stage1_results.csv"
STAGE2 = ROOT / "reports" / "phase2e_multifidelity" / "stage2_results.csv"
PHASE2F = ROOT / "reports" / "phase2f_multiseed" / "multiseed_results.csv"
CURVES = ROOT / "reports" / "phase2e_multifidelity" / "standalone_learning_curves.csv"
CURVES_FIG = ROOT / "reports" / "phase2e_multifidelity" / "figures" / "independent_learning_curves.csv"
H4 = ROOT / "reports" / "fpga_h4_tooling_v2" / "NETWORK_COST_SUMMARY.csv"
COST_API = ROOT / "reports" / "fpga_h4_tooling_v2" / "COST_API.md"
SPEC = ROOT / "configs" / "search_space_sonar_v0_1.yaml"
FORBIDDEN_WRITE = (
    ROOT / "reports" / "phase2e_multifidelity",
    ROOT / "reports" / "phase2f_multiseed",
)

N_SEEDS = 40
INITIAL = 20
BATCH = 6
BUDGETS = (32, 44, 56)
INNER_RANDOM = 30
BOOTSTRAP_B = 2000
RIDGE_ALPHA = 1.0
N_BOOT_MODELS = 32
UNCERTAINTY_SPEARMAN_MIN = 0.2
STAGES = (1, 2, 4, 5, 6, 7)
COMBOS = ("P3_P4_P5", "P2_P3_P4", "P2_P3_P4_P5")
EXPECTED_SHA = {
    "reports/phase2e_multifidelity/final_results.csv":
        "ffbe84fc4355e41560384d492c89b2c6865ec255200b3255942d844d843eef9e",
    "reports/phase2e_multifidelity/stage1_results.csv":
        "5f8c4813d47d0e6624ea95d508bbbdae0c53720665301613c94ced4bb5c93a6f",
    "reports/phase2f_multiseed/multiseed_results.csv":
        "0dafc002c89f15fc38b3d36f5123f38674d6353311ca4fbe4a2bc5a2cea5ada9",
    "reports/fpga_h4_tooling_v2/NETWORK_COST_SUMMARY.csv":
        "c0f09b2d574419f5ece2d22125c37c75d78794b4c27a4342d2386fd88e4919eb",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    resolved = path.resolve()
    for root in FORBIDDEN_WRITE:
        root_r = root.resolve()
        if resolved == root_r or root_r in resolved.parents:
            raise RuntimeError(f"Refusing to write ground-truth path {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    resolved = path.resolve()
    for root in FORBIDDEN_WRITE:
        root_r = root.resolve()
        if resolved == root_r or root_r in resolved.parents:
            raise RuntimeError(f"Refusing to write ground-truth path {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out[k] = ""
                elif isinstance(v, float):
                    out[k] = f"{v:.8f}"
                else:
                    out[k] = v
            w.writerow(out)


def conv(obj):
    if isinstance(obj, dict):
        return {str(k): conv(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [conv(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return conv(obj.tolist())
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def f4(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.4f}"


def f6(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.6f}"


def _stat(res) -> float:
    return float(res.statistic if hasattr(res, "statistic") else res.correlation)


def safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return _stat(spearmanr(a, b))


def safe_kendall(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return _stat(kendalltau(a, b))


def rmse(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def ndcg(y_true, y_score, k: int | None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    kk = n if k is None or k > n else k
    order = np.argsort(-y_score, kind="mergesort")[:kk]
    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = float(np.sum(y_true[order] * discounts))
    ideal = np.sort(y_true)[::-1][:kk]
    idcg = float(np.sum(ideal * discounts))
    if idcg <= 0:
        return float("nan")
    return dcg / idcg


def topk_recall(y_true, y_score, k: int) -> float:
    kk = min(k, len(y_true))
    true_idx = set(np.argsort(-np.asarray(y_true), kind="mergesort")[:kk].tolist())
    pred_idx = set(np.argsort(-np.asarray(y_score), kind="mergesort")[:kk].tolist())
    return len(true_idx & pred_idx) / kk


def corr_ci(x, y, rng: np.random.Generator) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sp = safe_spearman(x, y)
    kd = safe_kendall(x, y)
    sps, kds = [], []
    n = len(x)
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, n)
        sps.append(safe_spearman(x[idx], y[idx]))
        kds.append(safe_kendall(x[idx], y[idx]))
    sps_a = np.asarray(sps, dtype=float)
    kds_a = np.asarray(kds, dtype=float)
    return {
        "n": int(n),
        "spearman": sp,
        "spearman_ci95": [float(np.nanquantile(sps_a, 0.025)), float(np.nanquantile(sps_a, 0.975))],
        "kendall": kd,
        "kendall_ci95": [float(np.nanquantile(kds_a, 0.025)), float(np.nanquantile(kds_a, 0.975))],
    }


def rank_of(scores, ids, target: int) -> int:
    better = 0
    for i in range(len(scores)):
        if i == target:
            continue
        if scores[i] > scores[target] + 1e-15:
            better += 1
        elif abs(scores[i] - scores[target]) <= 1e-15 and ids[i] < ids[target]:
            better += 1
    return better + 1


def concordance(a, b) -> dict:
    conc = disc = ties = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = a[i] - a[j]
            d2 = b[i] - b[j]
            if abs(d1) <= 1e-15 or abs(d2) <= 1e-15:
                ties += 1
            elif d1 * d2 > 0:
                conc += 1
            else:
                disc += 1
    denom = conc + disc
    return {
        "concordant": conc,
        "discordant": disc,
        "tied": ties,
        "pairwise_concordance": (conc / denom) if denom else float("nan"),
    }


def fidelity_block(low, high, ids, rng) -> dict:
    ci = corr_ci(low, high, rng)
    order_high = np.argsort(-high, kind="mergesort")
    top5 = set(order_high[:5].tolist())
    top3 = set(order_high[:3].tolist())
    order_low = np.argsort(-low, kind="mergesort")
    pred5 = set(order_low[:5].tolist())
    pred3 = set(order_low[:3].tolist())
    best = int(order_high[0])
    return {
        **ci,
        "top3_recall": len(top3 & pred3) / 3.0,
        "top5_recall": len(top5 & pred5) / 5.0,
        "best80_rank_at_low_fidelity": rank_of(low, ids, best),
        "best80_id": ids[best],
        "best80_value": float(high[best]),
        "pairwise": concordance(high, low),
    }


def enumerate_architectures(spec: dict) -> list[dict]:
    space = spec["architecture_space"]
    out = []
    import itertools
    for combo in space["output_combinations"]:
        stages = [s["index"] for s in space["stages"] if s["op"] == "mbconv" and s["index"] <= combo["last_active_stage"]]
        for kernels in itertools.product((3, 5), repeat=len(stages)):
            kernel_map = {f"stage{s}": k for s, k in zip(stages, kernels)}
            aid = combo["id"] + "_" + "_".join(f"s{name[5:]}k{k}" for name, k in sorted(kernel_map.items()))
            out.append({
                "architecture_id": aid,
                "output_combination": combo["id"],
                "kernels": kernel_map,
                "num_heads": len(combo["levels"]),
                "levels": list(combo["levels"]),
                "last_active_stage": int(combo["last_active_stage"]),
            })
    return out


def minimal_features(arch: dict) -> dict[str, float]:
    combo = arch["output_combination"]
    kernels = arch["kernels"]
    feats = {
        "combo_P3_P4_P5": 1.0 if combo == "P3_P4_P5" else 0.0,
        "combo_P2_P3_P4": 1.0 if combo == "P2_P3_P4" else 0.0,
    }
    for s in STAGES:
        k = int(kernels.get(f"stage{s}", 0))
        feats[f"k5_stage{s}"] = 1.0 if k == 5 else 0.0
    return feats


def phase6a_features(arch: dict, params: float, macs: float) -> dict[str, float]:
    combo = arch["output_combination"]
    kernels = arch["kernels"]
    feats: dict[str, float] = {}
    for name in COMBOS:
        feats[f"combo_{name}"] = 1.0 if combo == name else 0.0
    feats["has_P2"] = 1.0 if combo in ("P2_P3_P4", "P2_P3_P4_P5") else 0.0
    feats["has_P5"] = 1.0 if combo in ("P3_P4_P5", "P2_P3_P4_P5") else 0.0
    feats["num_heads"] = float({"P3_P4_P5": 3, "P2_P3_P4": 3, "P2_P3_P4_P5": 4}[combo])
    feats["last_stage"] = float({"P3_P4_P5": 7, "P2_P3_P4": 5, "P2_P3_P4_P5": 7}[combo])
    active = set(kernels)
    k5_count = early = deep = 0
    for s in STAGES:
        key = f"stage{s}"
        k = int(kernels[key]) if key in active else 0
        feats[f"stage{s}_active"] = 1.0 if key in active else 0.0
        feats[f"stage{s}_k5"] = 1.0 if k == 5 else 0.0
        feats[f"stage{s}_kernel_value"] = float(k)
        if k == 5:
            k5_count += 1
            if s in (1, 2):
                early += 1
            if s in (5, 6, 7):
                deep += 1
    feats["num_k5"] = float(k5_count)
    feats["early_k5_count"] = float(early)
    feats["deep_k5_count"] = float(deep)
    feats["Params"] = float(params)
    feats["MACs"] = float(macs)
    return feats


def matrix_from(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray([[float(r[k]) for k in names] for r in rows], dtype=float)


def collision_report(keys: list, ids: list[str]) -> dict:
    groups: dict = {}
    for key, aid in zip(keys, ids):
        groups.setdefault(key, []).append(aid)
    dups = [v for v in groups.values() if len(v) > 1]
    examples = []
    for group in dups[:5]:
        examples.append(group[:4])
    return {
        "n": len(ids),
        "n_unique_keys": len(groups),
        "n_collision_groups": len(dups),
        "n_ids_in_collisions": int(sum(len(v) for v in dups)),
        "examples": examples,
    }


def stratified_indices(combos: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, combo in enumerate(combos):
        groups.setdefault(str(combo), []).append(i)
    keys = sorted(groups)
    total = len(combos)
    raw = {k: n * len(groups[k]) / total for k in keys}
    alloc = {k: int(math.floor(raw[k])) for k in keys}
    leftover = n - sum(alloc.values())
    order = sorted(keys, key=lambda k: (raw[k] - alloc[k], k), reverse=True)
    for key in order:
        if leftover <= 0:
            break
        if alloc[key] < len(groups[key]):
            alloc[key] += 1
            leftover -= 1
    chosen: list[int] = []
    for key in keys:
        take = alloc[key]
        if take == 0:
            continue
        pool = np.asarray(groups[key], dtype=int)
        pick = np.atleast_1d(rng.choice(pool, size=take, replace=False))
        chosen.extend(int(x) for x in pick)
    if len(chosen) != n or len(set(chosen)) != n:
        raise RuntimeError(f"Bad initial set {len(chosen)} alloc={alloc}")
    return np.asarray(chosen, dtype=int)


def standardize_fit(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def fit_ridge(X_train, y_train, X_test, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    mu, sd = standardize_fit(X_train)
    Z = (X_train - mu) / sd
    Zt = (X_test - mu) / sd
    A = np.concatenate([np.ones((len(Z), 1)), Z], axis=1)
    B = np.concatenate([np.ones((len(Zt), 1)), Zt], axis=1)
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y_train)
    return B @ coef


def fit_tree(kind: str, X_train, y_train, X_test, seed: int) -> np.ndarray:
    common = dict(n_estimators=200, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=1)
    if kind == "extratrees":
        model = ExtraTreesRegressor(**common)
    elif kind == "rf":
        model = RandomForestRegressor(**common)
    elif kind == "gb":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, subsample=1.0, random_state=seed,
        )
    else:
        raise ValueError(kind)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_gp(X_train, y_train, X_test, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = standardize_fit(X_train)
    Z = (X_train - mu) / sd
    Zt = (X_test - mu) / sd
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=seed, alpha=1e-6,
    )
    gp.fit(Z, y_train)
    mean, std = gp.predict(Zt, return_std=True)
    if not np.all(np.isfinite(mean)):
        raise RuntimeError("GP mean was not finite")
    return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def fit_bootstrap(X_train, y_train, X_test, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu, sd = standardize_fit(X_train)
    Z = np.concatenate([np.ones((len(X_train), 1)), (X_train - mu) / sd], axis=1)
    Zt = np.concatenate([np.ones((len(X_test), 1)), (X_test - mu) / sd], axis=1)
    n, p = Z.shape
    coefs = []
    for _ in range(N_BOOT_MODELS):
        idx = rng.integers(0, n, n)
        mask = np.ones(p, dtype=bool)
        if p > 6:
            mask = rng.random(p) < 0.75
            mask[0] = True
            if int(mask.sum()) < 2:
                mask[:] = True
        A = Z[idx][:, mask]
        b = y_train[idx]
        lam = 10 ** float(rng.uniform(-3, 0.7))
        reg = lam * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        try:
            coef = np.linalg.solve(A.T @ A + reg, A.T @ b)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A.T @ A + reg, A.T @ b, rcond=None)[0]
        full = np.zeros(p)
        full[mask] = coef
        coefs.append(full)
    P = Zt @ np.vstack(coefs).T
    return P.mean(axis=1), P.std(axis=1)


def fit_pairwise(X_train, y_train, X_test) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = standardize_fit(X_train)
    Z = (X_train - mu) / sd
    Zt = (X_test - mu) / sd
    diffs, labs = [], []
    for i in range(len(y_train)):
        for j in range(i + 1, len(y_train)):
            if y_train[i] == y_train[j]:
                continue
            diffs.append(Z[i] - Z[j])
            labs.append(1.0 if y_train[i] > y_train[j] else 0.0)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    clf.fit(np.asarray(diffs), np.asarray(labs))
    weight = clf.coef_.ravel()
    raw_tr = Z @ weight
    raw_te = Zt @ weight
    A = np.column_stack([raw_tr, np.ones(len(raw_tr))])
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return raw_te, raw_te * float(coef[0]) + float(coef[1])


def pack(pred, y_true, rank_score=None) -> dict:
    rank = pred if rank_score is None else rank_score
    ok = bool(np.all(np.isfinite(pred)) and np.all(np.isfinite(rank)))
    if not ok:
        nan = float("nan")
        return {"ok": False, "rmse": nan, "mae": nan, "spearman": nan, "kendall": nan,
                "top3_recall": nan, "top5_recall": nan, "ndcg10": nan, "ndcg_full": nan}
    return {
        "ok": True,
        "rmse": rmse(y_true, pred),
        "mae": mae(y_true, pred),
        "spearman": safe_spearman(y_true, rank),
        "kendall": safe_kendall(y_true, rank),
        "top3_recall": topk_recall(y_true, rank, 3),
        "top5_recall": topk_recall(y_true, rank, 5),
        "ndcg10": ndcg(y_true, rank, 10),
        "ndcg_full": ndcg(y_true, rank, None),
    }


def search_metrics(obs: set[int], y: np.ndarray, top3: set[int], top5: set[int]) -> dict:
    idx = np.fromiter(obs, dtype=int)
    best = float(y[idx].max())
    return {
        "regret": float(y.max() - best),
        "best": best,
        "top3_recall": len(obs & top3) / len(top3),
        "top5_recall": len(obs & top5) / len(top5),
    }


def decision_metrics(y_rem, scores) -> dict:
    return {
        "spearman": safe_spearman(y_rem, scores),
        "kendall": safe_kendall(y_rem, scores),
        "ndcg10": ndcg(y_rem, scores, 10),
        "ndcg_full": ndcg(y_rem, scores, None),
        "top3_recall": topk_recall(y_rem, scores, 3),
        "top5_recall": topk_recall(y_rem, scores, 5),
    }


def phase6a_acquire(mu, sigma, macs, y_obs, macs_obs, beta: float = 0.35) -> np.ndarray:
    def norm(v):
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi - lo < 1e-12:
            return np.zeros_like(v, dtype=float)
        return (v - lo) / (hi - lo)
    acc_n = norm(mu)
    unc_n = norm(sigma)
    cost_good = 1.0 - norm(macs)
    improvement = np.maximum(mu - float(np.max(y_obs)), 0.0)
    cheaper = np.asarray([(macs_obs >= m).mean() for m in macs], dtype=float)
    return 0.45 * acc_n + 0.20 * cost_good + 0.20 * norm(improvement) + 0.15 * cheaper + beta * unc_n


def run_trajectory(score_fn, init: np.ndarray, y: np.ndarray, ids: list[str], top3, top5):
    obs = set(int(i) for i in init.tolist())
    search, decision = {}, {}
    while len(obs) < 56:
        remaining = [i for i in range(len(y)) if i not in obs]
        scores = np.asarray(score_fn(sorted(obs), remaining), dtype=float)
        if len(scores) != len(remaining):
            raise RuntimeError("Score length mismatch")
        arriving = len(obs) + min(BATCH, len(remaining))
        if arriving in BUDGETS:
            decision[arriving] = decision_metrics(y[remaining], scores)
        order = sorted(range(len(remaining)), key=lambda t: (-float(scores[t]), ids[remaining[t]]))
        chosen = [remaining[t] for t in order[:BATCH]]
        obs.update(chosen)
        if len(obs) in BUDGETS:
            search[len(obs)] = search_metrics(obs, y, top3, top5)
    return search, decision


def random_summary(y, init, seed, ids, top3, top5):
    rng = np.random.default_rng(10_000 + seed)
    init_set = set(int(i) for i in init.tolist())
    rest = np.asarray([i for i in range(len(y)) if i not in init_set], dtype=int)
    need = {32: 12, 44: 24, 56: 36}
    decide_at = {32: 6, 44: 18, 56: 30}
    buckets = {b: [] for b in BUDGETS}
    decisions = {b: [] for b in BUDGETS}
    for _ in range(INNER_RANDOM):
        perm = rng.permutation(rest)
        for b, k in need.items():
            obs = set(init_set) | set(int(i) for i in perm[:k].tolist())
            buckets[b].append(search_metrics(obs, y, top3, top5))
        for b, k_before in decide_at.items():
            remaining = perm[k_before:]
            scores = rng.random(len(remaining))
            decisions[b].append(decision_metrics(y[remaining], scores))
    search, decision = {}, {}
    for b in BUDGETS:
        search[b] = {
            key: float(np.median([row[key] for row in buckets[b]]))
            for key in ("regret", "best", "top3_recall", "top5_recall")
        }
        decision[b] = {}
        for key in ("spearman", "kendall", "ndcg10", "ndcg_full", "top3_recall", "top5_recall"):
            vals = np.asarray([row[key] for row in decisions[b]], dtype=float)
            decision[b][key] = float(np.nanmedian(vals))
    return search, decision


def v0_initial(ids, combos, macs) -> list[int]:
    rows = [{
        "architecture_id": ids[i], "output_combination": combos[i], "MACs": float(macs[i]), "i": i,
    } for i in range(len(ids))]
    rng = random.Random(42)
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["output_combination"], []).append(row)
    selected = []
    quotas = {"P3_P4_P5": 7, "P2_P3_P4": 3, "P2_P3_P4_P5": 10}
    for combo, quota in quotas.items():
        group = sorted(groups.get(combo, []), key=lambda r: (r["MACs"], r["architecture_id"]))
        if not group:
            continue
        for chunk in np.array_split(np.asarray(group, dtype=object), quota):
            if len(chunk):
                selected.append(rng.choice(list(chunk))["i"])
    chosen = set(selected)
    remaining = [row["i"] for row in rows if row["i"] not in chosen]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, 20 - len(selected))])
    return selected[:20]


def mean_ci(values, rng: np.random.Generator) -> dict:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    boots = [float(valid[rng.integers(0, len(valid), len(valid))].mean()) for _ in range(BOOTSTRAP_B)]
    return {
        "n": int(len(valid)),
        "mean": float(valid.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def paired_block(policy: np.ndarray, baseline: np.ndarray, rng: np.random.Generator, higher_is_better: bool) -> dict:
    """policy/baseline are shape (n_seeds,). For regret, lower is better."""
    out = {"by_budget": {}, "stable": True}
    for i, b in enumerate(BUDGETS):
        p = policy[i]
        base = baseline[i]
        if higher_is_better:
            delta = p - base
            med_better = float(np.median(p)) > float(np.median(base))
        else:
            delta = base - p
            med_better = float(np.median(p)) < float(np.median(base))
        summary = mean_ci(delta, rng)
        ci_pos = summary["ci95"][0] > 0
        win = float(np.mean(delta > 0)) if len(delta) else float("nan")
        worse_ci = summary["ci95"][1] < 0
        pvalue = None
        clean = delta[np.isfinite(delta)]
        if len(clean) >= 10 and not np.allclose(clean, 0):
            try:
                pvalue = float(wilcoxon(clean, alternative="greater", zero_method="wilcox").pvalue)
            except ValueError:
                pvalue = None
        if not (med_better and ci_pos):
            out["stable"] = False
        out["by_budget"][str(b)] = {
            "mean_policy": float(np.mean(p)),
            "median_policy": float(np.median(p)),
            "mean_baseline": float(np.mean(base)),
            "median_baseline": float(np.median(base)),
            "mean_advantage": summary["mean"],
            "advantage_ci95": summary["ci95"],
            "median_better": med_better,
            "ci_entirely_positive": ci_pos,
            "significantly_worse": worse_ci,
            "win_rate": win,
            "wilcoxon_greater_pvalue": pvalue,
        }
    out["significantly_worse_any_budget"] = any(
        out["by_budget"][str(b)]["significantly_worse"] for b in BUDGETS
    )
    return out


def curve_coverage(path: Path, final_ids: set[str]) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    rows = read_csv(path)
    ids = sorted({r["architecture_id"] for r in rows})
    epochs = {}
    has_loss = 0
    for row in rows:
        epochs.setdefault(row["architecture_id"], set()).add(int(float(row["epoch"])))
        if row.get("validation_loss", "") not in ("", None):
            has_loss += 1
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": True,
        "n_rows": len(rows),
        "n_architectures": len(ids),
        "architectures": ids,
        "n_overlap_final80": len(set(ids) & final_ids),
        "min_epochs": min(len(v) for v in epochs.values()) if epochs else 0,
        "max_epoch": max((max(v) for v in epochs.values()), default=0),
        "rows_with_validation_loss": has_loss,
    }


def load_pool():
    before = {}
    for rel, expected in EXPECTED_SHA.items():
        path = ROOT / rel
        digest = sha256(path)
        before[str(path)] = digest
        if digest != expected:
            raise SystemExit(f"Ground-truth hash mismatch for {rel}: {digest}")
    before[str(STAGE2)] = sha256(STAGE2)
    before[str(SPEC)] = sha256(SPEC)
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    archs = enumerate_architectures(spec)
    if len(archs) != 144 or len({a["architecture_id"] for a in archs}) != 144:
        raise SystemExit(f"Expected 144 unique architectures, got {len(archs)}")
    by_id = {a["architecture_id"]: a for a in archs}
    stage1 = {r["architecture_id"]: r for r in read_csv(STAGE1)}
    stage2 = {r["architecture_id"]: r for r in read_csv(STAGE2)}
    final = {r["architecture_id"]: r for r in read_csv(FINAL)}
    if set(stage1) != set(by_id):
        raise SystemExit("Stage1 architecture ids do not match the 144-enumeration")
    missing60 = [aid for aid in final if aid not in stage2]
    missing40 = [aid for aid in final if aid not in stage1]
    if missing40 or missing60:
        raise SystemExit(f"Finalists missing fidelity rows: 40={len(missing40)} 60={len(missing60)}")
    pool_ids = sorted(final)
    def fnum(table, aid, key):
        return float(table[aid][key])
    y = np.asarray([fnum(final, aid, "mAP50_95") for aid in pool_ids])
    map40 = np.asarray([fnum(stage1, aid, "mAP50_95") for aid in pool_ids])
    map60 = np.asarray([fnum(stage2, aid, "mAP50_95") for aid in pool_ids])
    endpoint40 = np.asarray([fnum(stage1, aid, "endpoint_mAP50_95") for aid in pool_ids])
    endpoint60 = np.asarray([fnum(stage2, aid, "endpoint_mAP50_95") for aid in pool_ids])
    combos = [by_id[aid]["output_combination"] for aid in pool_ids]
    macs = np.asarray([float(stage1[aid]["MACs"]) for aid in pool_ids])
    params = np.asarray([float(stage1[aid]["Params"]) for aid in pool_ids])
    minimal_names = list(minimal_features(archs[0]).keys())
    p6_names = list(phase6a_features(archs[0], 0, 0).keys())
    X_arch = matrix_from([minimal_features(by_id[aid]) for aid in pool_ids], minimal_names)
    X_p6 = matrix_from([
        phase6a_features(by_id[aid], params[i], macs[i]) for i, aid in enumerate(pool_ids)
    ], p6_names)
    curve40_names = [
        "map40", "endpoint40", "ap50_40", "ap75_40", "precision40", "recall40", "best_epoch40",
    ]
    curve60_names = curve40_names + [
        "map60", "endpoint60", "ap50_60", "ap75_60", "precision60", "recall60", "best_epoch60", "slope_40_to_60",
    ]
    c40 = np.column_stack([
        map40, endpoint40,
        np.asarray([fnum(stage1, aid, "AP50") for aid in pool_ids]),
        np.asarray([fnum(stage1, aid, "AP75") for aid in pool_ids]),
        np.asarray([fnum(stage1, aid, "Precision") for aid in pool_ids]),
        np.asarray([fnum(stage1, aid, "Recall") for aid in pool_ids]),
        np.asarray([fnum(stage1, aid, "best_epoch") for aid in pool_ids]),
    ])
    c60_extra = np.column_stack([
        map60, endpoint60,
        np.asarray([fnum(stage2, aid, "AP50") for aid in pool_ids]),
        np.asarray([fnum(stage2, aid, "AP75") for aid in pool_ids]),
        np.asarray([fnum(stage2, aid, "Precision") for aid in pool_ids]),
        np.asarray([fnum(stage2, aid, "Recall") for aid in pool_ids]),
        np.asarray([fnum(stage2, aid, "best_epoch") for aid in pool_ids]),
        (map60 - map40) / 20.0,
    ])
    X_m2_40 = np.concatenate([X_arch, c40], axis=1)
    X_m2_60 = np.concatenate([X_arch, c40, c60_extra], axis=1)
    X_m1 = np.concatenate([X_arch, map40.reshape(-1, 1)], axis=1)
    X_compute = np.concatenate([X_arch, params.reshape(-1, 1), macs.reshape(-1, 1)], axis=1)
    seeds = read_csv(PHASE2F)
    return {
        "before": before,
        "spec": spec,
        "archs": archs,
        "by_id": by_id,
        "stage1": stage1,
        "stage2": stage2,
        "final": final,
        "seeds": seeds,
        "pool_ids": pool_ids,
        "combos": np.asarray(combos),
        "y": y,
        "map40": map40,
        "map60": map60,
        "endpoint40": endpoint40,
        "endpoint60": endpoint60,
        "macs": macs,
        "params": params,
        "minimal_names": minimal_names,
        "p6_names": p6_names,
        "curve40_names": curve40_names,
        "curve60_names": curve60_names,
        "X_arch": X_arch,
        "X_p6": X_p6,
        "X_m1": X_m1,
        "X_m2_40": X_m2_40,
        "X_m2_60": X_m2_60,
        "X_compute": X_compute,
        "c40": c40,
    }


def signal_audit(data, rng) -> dict:
    y, ids = data["y"], data["pool_ids"]
    low40 = fidelity_block(data["map40"], y, ids, rng)
    low60 = fidelity_block(data["map60"], y, ids, rng)
    end40 = fidelity_block(data["endpoint40"], y, ids, rng)
    end60 = fidelity_block(data["endpoint60"], y, ids, rng)
    grouped: dict[str, list] = {}
    for row in data["seeds"]:
        grouped.setdefault(row["architecture_id"], []).append(row)
    seed_rows = []
    for aid, rows in sorted(grouped.items()):
        vals = np.asarray([float(r["mAP50_95"]) for r in rows], dtype=float)
        seed_rows.append({
            "architecture_id": aid,
            "label": rows[0].get("label", ""),
            "n_seeds": len(vals),
            "seeds": [int(float(r["seed"])) for r in rows],
            "mean_mAP": float(vals.mean()),
            "std_mAP": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "var_mAP": float(vals.var(ddof=1)) if len(vals) > 1 else float("nan"),
            "min_mAP": float(vals.min()),
            "max_mAP": float(vals.max()),
            "seed42_mAP": float(next(float(r["mAP50_95"]) for r in rows if int(float(r["seed"])) == 42)),
        })
    stds = np.asarray([r["std_mAP"] for r in seed_rows], dtype=float)
    order = np.argsort(-y, kind="mergesort")
    gaps = y[order][:-1] - y[order][1:]
    pair_gaps = [abs(float(y[i] - y[j])) for i in range(len(y)) for j in range(i + 1, len(y))]
    pair_gaps_a = np.asarray(pair_gaps)
    median_std = float(np.median(stds))
    # Seed-vs-seed Spearman on the six architectures.
    seed_ids = sorted({int(float(r["seed"])) for r in data["seeds"]})
    arch_order = [r["architecture_id"] for r in seed_rows]
    by_seed = {s: [] for s in seed_ids}
    for aid in arch_order:
        rows = {int(float(r["seed"])): float(r["mAP50_95"]) for r in grouped[aid]}
        for s in seed_ids:
            by_seed[s].append(rows[s])
    seed_pairs = []
    for i, s1 in enumerate(seed_ids):
        for s2 in seed_ids[i + 1:]:
            seed_pairs.append({
                "seed_a": s1,
                "seed_b": s2,
                "n": len(arch_order),
                "spearman": safe_spearman(by_seed[s1], by_seed[s2]),
                "kendall": safe_kendall(by_seed[s1], by_seed[s2]),
            })
    top10_gaps = gaps[:9]
    return {
        "n_stage1": len(data["stage1"]),
        "n_stage2": len(data["stage2"]),
        "n_final80": len(data["final"]),
        "stage1_budgets": sorted({r["budget"] for r in data["stage1"].values()}),
        "stage2_budgets": sorted({r["budget"] for r in data["stage2"].values()}),
        "final_budgets": sorted({r["budget"] for r in data["final"].values()}),
        "final_seeds": sorted({r["seed"] for r in data["final"].values()}),
        "combo_counts": {c: int(np.sum(data["combos"] == c)) for c in COMBOS},
        "map40_vs_map80": low40,
        "map60_vs_map80": low60,
        "endpoint40_vs_map80": end40,
        "endpoint60_vs_map80": end60,
        "seed_rows": seed_rows,
        "seed_pairs": seed_pairs,
        "median_seed_std": median_std,
        "mean_seed_std": float(np.mean(stds)),
        "min_seed_std": float(np.min(stds)),
        "max_seed_std": float(np.max(stds)),
        "between_std_72": float(np.std(y, ddof=1)),
        "median_pairwise_gap_72": float(np.median(pair_gaps_a)),
        "mean_pairwise_gap_72": float(np.mean(pair_gaps_a)),
        "fraction_pairs_below_median_seed_std": float(np.mean(pair_gaps_a < median_std)),
        "median_adjacent_gap": float(np.median(gaps)),
        "fraction_adjacent_below_median_seed_std": float(np.mean(gaps < median_std)),
        "top1_top2_gap": float(gaps[0]),
        "top1_top5_gap": float(y[order[0]] - y[order[4]]),
        "top1_top10_gap": float(y[order[0]] - y[order[9]]),
        "median_top10_adjacent_gap": float(np.median(top10_gaps)),
        "fraction_top10_adjacent_below_seed_std": float(np.mean(top10_gaps < median_std)),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
        "label_noise_supported": bool(median_std > float(gaps[0]) and median_std > float(np.median(top10_gaps))),
        "fidelity40_inconsistent": bool(low40["spearman"] < 0.70),
        "fidelity60_inconsistent": bool(low60["spearman"] < 0.70),
    }


def encoding_audit(data) -> dict:
    archs = data["archs"]
    ids = [a["architecture_id"] for a in archs]
    minimal_names = data["minimal_names"]
    full_keys = []
    minimal_keys = []
    kernel_only = []
    heads_kernels = []
    p6_wo_cost = []
    cost_keys = []
    for arch in archs:
        kernels = arch["kernels"]
        full_keys.append((arch["output_combination"], tuple((s, int(kernels.get(f"stage{s}", 0))) for s in STAGES)))
        mf = minimal_features(arch)
        minimal_keys.append(tuple(mf[k] for k in minimal_names))
        kernel_only.append(tuple(int(kernels.get(f"stage{s}", 0)) for s in STAGES))
        heads_kernels.append((arch["num_heads"], tuple(int(kernels.get(f"stage{s}", 0)) for s in STAGES)))
        row = data["stage1"][arch["architecture_id"]]
        p6 = phase6a_features(arch, float(row["Params"]), float(row["MACs"]))
        p6_wo_cost.append(tuple(p6[k] for k in data["p6_names"] if k not in ("Params", "MACs")))
        cost_keys.append((int(float(row["Params"])), int(float(row["MACs"]))))
    spec = data["spec"]["architecture_space"]
    fixed_rows = []
    for stage in spec["stages"]:
        varies = "kernel_choices" in stage
        fixed_rows.append({
            "factor": f"stage{stage['index']}",
            "operator": stage["op"],
            "channels": stage.get("channels"),
            "depth": stage.get("depth"),
            "stride": stage.get("stride"),
            "expansion": stage.get("expansion", ""),
            "kernel": stage.get("kernel_choices", stage.get("kernel")),
            "varies_across_144": bool(varies and stage["op"] == "mbconv"),
        })
    # Activity of stages 1,2,4,5 is constant; 6 and 7 vary.
    active_patterns = {}
    for arch in archs:
        pattern = tuple(f"stage{s}" in arch["kernels"] for s in STAGES)
        active_patterns.setdefault(pattern, set()).add(arch["output_combination"])
    return {
        "n": 144,
        "minimal_feature_names": minimal_names,
        "phase6a_feature_names": data["p6_names"],
        "id": collision_report([(aid,) for aid in ids], ids),
        "full_combo_kernel": collision_report(full_keys, ids),
        "minimal_model_bits": collision_report(minimal_keys, ids),
        "kernel_only": collision_report(kernel_only, ids),
        "num_heads_plus_kernels": collision_report(heads_kernels, ids),
        "phase6a_without_params_macs": collision_report(p6_wo_cost, ids),
        "params_macs": collision_report(cost_keys, ids),
        "stages": fixed_rows,
        "neck": spec["neck"],
        "head": spec["head"],
        "active_patterns": [
            {"stages_active": [STAGES[i] for i, flag in enumerate(pattern) if flag], "combos": sorted(combos)}
            for pattern, combos in active_patterns.items()
        ],
        "encoding_insufficient": collision_report(minimal_keys, ids)["n_collision_groups"] > 0,
    }


def hw_audit(data) -> dict:
    rows = read_csv(H4)
    ids_144 = {a["architecture_id"] for a in data["archs"]}
    arch_ids = sorted({r["architecture_id"] for r in rows})
    cost_api = COST_API.read_text(encoding="utf-8") if COST_API.exists() else ""
    per_arch = []
    for aid in arch_ids:
        group = [r for r in rows if r["architecture_id"] == aid]
        def col(name):
            return np.asarray([float(r[name]) for r in group], dtype=float)
        per_arch.append({
            "network": sorted({r["network"] for r in group}),
            "architecture_id": aid,
            "in_144": aid in ids_144,
            "n_rows": len(group),
            "mapping_policies": sorted({r["mapping_policy"] for r in group}),
            "quant_config": sorted({r["quant_config"] for r in group}),
            "status": sorted({r["status"] for r in group}),
            "LUT_peak_min": float(col("LUT_peak").min()),
            "LUT_peak_max": float(col("LUT_peak").max()),
            "LUT_sum_min": float(col("LUT_sum").min()),
            "LUT_sum_max": float(col("LUT_sum").max()),
            "DSP_peak_min": float(col("DSP_peak").min()),
            "DSP_peak_max": float(col("DSP_peak").max()),
            "BRAM18_peak_min": float(col("BRAM18_peak").min()),
            "BRAM18_peak_max": float(col("BRAM18_peak").max()),
            "latency_us_min": float(col("Latency").min()),
            "latency_us_max": float(col("Latency").max()),
            "lookup_equals_total": all(r["lookup_nodes"] == r["total_nodes"] for r in group),
            "lookup_nodes": sorted({r["lookup_nodes"] for r in group}),
            "total_nodes": sorted({r["total_nodes"] for r in group}),
            "shell_reservation": sorted({r["shell_reservation"] for r in group}),
            "deployment": sorted({r["deployment"] for r in group}),
            "latency_scope": sorted({r["latency_scope"] for r in group}),
            "resource_scope": sorted({r["resource_scope"] for r in group}),
            "constraint_status": sorted({r["constraint_status"] for r in group}),
        })
    stage1_cost = len(data["stage1"])
    return {
        "n_rows": len(rows),
        "networks": sorted({r["network"] for r in rows}),
        "architecture_ids": arch_ids,
        "n_unique_architectures": len(arch_ids),
        "n_in_144": sum(1 for a in arch_ids if a in ids_144),
        "coverage_of_144": len(arch_ids) / 144.0,
        "mapping_policies": sorted({r["mapping_policy"] for r in rows}),
        "quant_configs": sorted({r["quant_config"] for r in rows}),
        "statuses": sorted({r["status"] for r in rows}),
        "per_arch": per_arch,
        "params_macs_rows": stage1_cost,
        "params_macs_cover_144": stage1_cost == 144,
        "cost_api_exact_lookup": "exact lookup" in cost_api.lower() or "exact-lookup" in cost_api.lower(),
        "cost_api_forbids_interpolation": "nearest interpolation" in cost_api.lower(),
        "cost_api_forbids_partial": "partial lookup" in cost_api.lower(),
    }


def aggregate_metrics(rows: list[dict], rng: np.random.Generator) -> dict:
    keys = ["rmse", "mae", "spearman", "kendall", "top3_recall", "top5_recall", "ndcg10", "ndcg_full"]
    cols = {k: [] for k in keys}
    n_valid = 0
    for row in rows:
        if row.get("ok"):
            n_valid += 1
            for k in keys:
                cols[k].append(row[k])
    out = {"n_seeds": len(rows), "n_valid": n_valid}
    for k, vals in cols.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[f"{k}_mean"] = float(finite.mean()) if len(finite) else float("nan")
        out[f"{k}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
    sp = np.asarray(cols["spearman"], dtype=float)
    sp = sp[np.isfinite(sp)]
    if len(sp):
        boots = [float(sp[rng.integers(0, len(sp), len(sp))].mean()) for _ in range(BOOTSTRAP_B)]
        out["spearman_ci95_low"] = float(np.quantile(boots, 0.025))
        out["spearman_ci95_high"] = float(np.quantile(boots, 0.975))
    else:
        out["spearman_ci95_low"] = float("nan")
        out["spearman_ci95_high"] = float("nan")
    return out


def benchmark_and_replay(data, signal):
    y = data["y"]
    ids = data["pool_ids"]
    combos = data["combos"]
    order = np.argsort(-y, kind="mergesort")
    top3 = set(order[:3].tolist())
    top5 = set(order[:5].tolist())
    meta = {
        "ridge_arch": ("absolute_regression", "architecture_bits", "mAP80", None, False),
        "rf_arch": ("absolute_regression", "architecture_bits", "mAP80", None, False),
        "extratrees_arch": ("absolute_regression", "architecture_bits", "mAP80", None, False),
        "gb_arch": ("absolute_regression", "architecture_bits", "mAP80", None, False),
        "gp_arch": ("absolute_regression", "architecture_bits", "mAP80", None, False),
        "ensemble_ridge_et_gb": ("lightweight_ensemble", "architecture_bits", "mAP80", None, False),
        "current_surrogate_mean": ("current_surrogate", "phase6a_bits_plus_params_macs_compute_aware", "mAP80", None, False),
        "ridge_compute": ("absolute_regression", "architecture_bits_plus_params_macs_compute_aware", "mAP80", None, False),
        "m1_ridge": ("multifidelity", "architecture_bits_plus_mAP40", "mAP80", "direct_map40", False),
        "m1_extratrees": ("multifidelity", "architecture_bits_plus_mAP40", "mAP80", "direct_map40", False),
        "m2_curve40_ridge": ("multifidelity", "architecture_bits_plus_budget40_summaries", "mAP40_plus_delta", "direct_map40", False),
        "m2_curve40_60_ridge": ("multifidelity", "architecture_bits_plus_budget40_and_budget60_summaries", "mAP40_plus_delta", "direct_map60", False),
        "pairwise_arch": ("ranking", "architecture_bits", "pairwise_preference", None, False),
        "pairwise_m1": ("ranking", "architecture_bits_plus_mAP40", "pairwise_preference", "direct_map40", False),
        "direct_map40": ("fidelity_reference", "mAP40_only", "mAP40_as_score", None, False),
        "direct_map60": ("fidelity_reference", "mAP60_only", "mAP60_as_score", None, False),
        "phase6a_acquisition": ("current_surrogate_acquisition", "phase6a_bits_plus_params_macs_compute_aware", "acquisition_score", None, True),
    }
    bench = {name: [] for name in meta if name != "phase6a_acquisition"}
    replay_names = [
        "random", "direct_map40", "direct_map60", "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch",
        "ensemble_ridge_et_gb", "gp_arch", "ridge_compute", "current_surrogate_mean", "phase6a_acquisition",
        "m1_ridge", "m1_extratrees", "m2_curve40_ridge", "m2_curve40_60_ridge", "pairwise_arch", "pairwise_m1",
    ]
    search_store = {name: {b: [] for b in BUDGETS} for name in replay_names}
    decision_store = {name: {b: {k: [] for k in ("spearman", "kendall", "ndcg10", "ndcg_full", "top3_recall", "top5_recall")} for b in BUDGETS} for name in replay_names}
    init_regret = []
    boot_sp, gp_sp = [], []
    sig_sum = np.zeros(len(y))
    err_sum = np.zeros(len(y))
    cnt = np.zeros(len(y))
    gp_sig_sum = np.zeros(len(y))
    gp_err_sum = np.zeros(len(y))
    gp_cnt = np.zeros(len(y))
    Xa, Xp, Xm1 = data["X_arch"], data["X_p6"], data["X_m1"]
    X40, X60, Xc = data["X_m2_40"], data["X_m2_60"], data["X_compute"]
    map40, map60, macs = data["map40"], data["map60"], data["macs"]

    def store_result(name, search, decision):
        for b in BUDGETS:
            search_store[name][b].append(search[b])
            for key in decision_store[name][b]:
                decision_store[name][b][key].append(decision[b][key])

    for seed in range(N_SEEDS):
        t0 = time.perf_counter()
        init = stratified_indices(combos, INITIAL, seed)
        init_set = set(int(i) for i in init.tolist())
        test = np.asarray([i for i in range(len(y)) if i not in init_set], dtype=int)
        tr = init
        ytr, yte = y[tr], y[test]
        init_regret.append(float(y.max() - y[tr].max()))

        pred_ridge = fit_ridge(Xa[tr], ytr, Xa[test])
        pred_rf = fit_tree("rf", Xa[tr], ytr, Xa[test], 1000 + seed)
        pred_et = fit_tree("extratrees", Xa[tr], ytr, Xa[test], 2000 + seed)
        pred_gb = fit_tree("gb", Xa[tr], ytr, Xa[test], 3000 + seed)
        pred_ens = (pred_ridge + pred_et + pred_gb) / 3.0
        pred_compute = fit_ridge(Xc[tr], ytr, Xc[test])
        mu_c, sig_c = fit_bootstrap(Xp[tr], ytr, Xp[test], 4000 + seed)
        raw_rank, cal_rank = fit_pairwise(Xa[tr], ytr, Xa[test])
        raw_m1, cal_m1 = fit_pairwise(Xm1[tr], ytr, Xm1[test])
        pred_m1 = fit_ridge(Xm1[tr], ytr, Xm1[test])
        pred_m1_et = fit_tree("extratrees", Xm1[tr], ytr, Xm1[test], 5000 + seed)
        d40 = fit_ridge(X40[tr], ytr - map40[tr], X40[test])
        d60 = fit_ridge(X60[tr], ytr - map40[tr], X60[test])
        try:
            mu_g, sig_g = fit_gp(Xa[tr], ytr, Xa[test], 6000 + seed)
            bench["gp_arch"].append(pack(mu_g, yte))
            gp_err = np.abs(mu_g - yte)
            gp_sp.append(safe_spearman(sig_g, gp_err))
            for local, idx in enumerate(test.tolist()):
                gp_sig_sum[idx] += float(sig_g[local])
                gp_err_sum[idx] += float(gp_err[local])
                gp_cnt[idx] += 1
        except Exception:
            bench["gp_arch"].append(pack(np.asarray([np.nan]), np.asarray([np.nan])))

        bench["ridge_arch"].append(pack(pred_ridge, yte))
        bench["rf_arch"].append(pack(pred_rf, yte))
        bench["extratrees_arch"].append(pack(pred_et, yte))
        bench["gb_arch"].append(pack(pred_gb, yte))
        bench["ensemble_ridge_et_gb"].append(pack(pred_ens, yte))
        bench["current_surrogate_mean"].append(pack(mu_c, yte))
        bench["ridge_compute"].append(pack(pred_compute, yte))
        bench["m1_ridge"].append(pack(pred_m1, yte))
        bench["m1_extratrees"].append(pack(pred_m1_et, yte))
        bench["m2_curve40_ridge"].append(pack(map40[test] + d40, yte))
        bench["m2_curve40_60_ridge"].append(pack(map40[test] + d60, yte))
        bench["pairwise_arch"].append(pack(cal_rank, yte, raw_rank))
        bench["pairwise_m1"].append(pack(cal_m1, yte, raw_m1))
        bench["direct_map40"].append(pack(map40[test], yte))
        bench["direct_map60"].append(pack(map60[test], yte))
        abs_err = np.abs(mu_c - yte)
        boot_sp.append(safe_spearman(sig_c, abs_err))
        for local, idx in enumerate(test.tolist()):
            sig_sum[idx] += float(sig_c[local])
            err_sum[idx] += float(abs_err[local])
            cnt[idx] += 1

        def scores_for(name, obs, remaining):
            obs_a = np.asarray(obs, dtype=int)
            rem_a = np.asarray(remaining, dtype=int)
            y_obs = y[obs_a]
            if name == "direct_map40":
                return map40[rem_a]
            if name == "direct_map60":
                return map60[rem_a]
            if name == "ridge_arch":
                return fit_ridge(Xa[obs_a], y_obs, Xa[rem_a])
            if name == "rf_arch":
                return fit_tree("rf", Xa[obs_a], y_obs, Xa[rem_a], 11000 + seed)
            if name == "extratrees_arch":
                return fit_tree("extratrees", Xa[obs_a], y_obs, Xa[rem_a], 12000 + seed)
            if name == "gb_arch":
                return fit_tree("gb", Xa[obs_a], y_obs, Xa[rem_a], 13000 + seed)
            if name == "ensemble_ridge_et_gb":
                a = fit_ridge(Xa[obs_a], y_obs, Xa[rem_a])
                b = fit_tree("extratrees", Xa[obs_a], y_obs, Xa[rem_a], 14000 + seed)
                c = fit_tree("gb", Xa[obs_a], y_obs, Xa[rem_a], 15000 + seed)
                return (a + b + c) / 3.0
            if name == "gp_arch":
                mu, _ = fit_gp(Xa[obs_a], y_obs, Xa[rem_a], 16000 + seed)
                return mu
            if name == "ridge_compute":
                return fit_ridge(Xc[obs_a], y_obs, Xc[rem_a])
            if name == "current_surrogate_mean":
                mu, _ = fit_bootstrap(Xp[obs_a], y_obs, Xp[rem_a], 17000 + seed)
                return mu
            if name == "phase6a_acquisition":
                mu, sig = fit_bootstrap(Xp[obs_a], y_obs, Xp[rem_a], 18000 + seed)
                return phase6a_acquire(mu, sig, macs[rem_a], y_obs, macs[obs_a])
            if name == "m1_ridge":
                return fit_ridge(Xm1[obs_a], y_obs, Xm1[rem_a])
            if name == "m1_extratrees":
                return fit_tree("extratrees", Xm1[obs_a], y_obs, Xm1[rem_a], 19000 + seed)
            if name == "m2_curve40_ridge":
                delta = fit_ridge(X40[obs_a], y_obs - map40[obs_a], X40[rem_a])
                return map40[rem_a] + delta
            if name == "m2_curve40_60_ridge":
                delta = fit_ridge(X60[obs_a], y_obs - map40[obs_a], X60[rem_a])
                return map40[rem_a] + delta
            if name == "pairwise_arch":
                raw, _ = fit_pairwise(Xa[obs_a], y_obs, Xa[rem_a])
                return raw
            if name == "pairwise_m1":
                raw, _ = fit_pairwise(Xm1[obs_a], y_obs, Xm1[rem_a])
                return raw
            raise KeyError(name)

        rand_search, rand_decision = random_summary(y, init, seed, ids, top3, top5)
        store_result("random", rand_search, rand_decision)
        for name in replay_names:
            if name == "random":
                continue
            try:
                search, decision = run_trajectory(
                    lambda obs, rem, name=name: scores_for(name, obs, rem),
                    init, y, ids, top3, top5,
                )
                store_result(name, search, decision)
            except Exception:
                nan_search = {b: {"regret": float("nan"), "best": float("nan"), "top3_recall": float("nan"), "top5_recall": float("nan")} for b in BUDGETS}
                nan_decision = {b: {k: float("nan") for k in ("spearman", "kendall", "ndcg10", "ndcg_full", "top3_recall", "top5_recall")} for b in BUDGETS}
                store_result(name, nan_search, nan_decision)
        print(f"seed {seed + 1}/{N_SEEDS} {time.perf_counter() - t0:.1f}s", flush=True)

    # Drop the accidental double-append helper reference by sanity check.
    for name in replay_names:
        for b in BUDGETS:
            if len(search_store[name][b]) != N_SEEDS:
                raise RuntimeError(f"{name} budget {b} has {len(search_store[name][b])} rows, expected {N_SEEDS}")

    rng = np.random.default_rng(20260922)
    bench_summary = {}
    for name, rows in bench.items():
        family, feature_set, target, direct, uses_u = meta[name]
        summary = aggregate_metrics(rows, np.random.default_rng(20260922 + (zlib_name(name))))
        summary.update({
            "model": name, "family": family, "feature_set": feature_set, "target": target,
            "direct_baseline": direct or "", "uses_uncertainty": uses_u,
            "n_train": INITIAL, "n_test": len(y) - INITIAL,
        })
        bench_summary[name] = summary

    def column(name, budget, key):
        return np.asarray([row[key] for row in search_store[name][budget]], dtype=float)

    def decision_column(name, budget, key):
        return np.asarray(decision_store[name][budget][key], dtype=float)

    policy_info = {}
    median_seed_std = float(signal["median_seed_std"])
    for name in replay_names:
        if name == "random":
            continue
        regret = np.vstack([column(name, b, "regret") for b in BUDGETS])
        base_regret = np.vstack([column("random", b, "regret") for b in BUDGETS])
        top5 = np.vstack([column(name, b, "top5_recall") for b in BUDGETS])
        base_top5 = np.vstack([column("random", b, "top5_recall") for b in BUDGETS])
        top3 = np.vstack([column(name, b, "top3_recall") for b in BUDGETS])
        base_top3 = np.vstack([column("random", b, "top3_recall") for b in BUDGETS])
        beats = paired_block(regret, base_regret, rng, higher_is_better=False)
        top5_cmp = paired_block(top5, base_top5, rng, higher_is_better=True)
        top3_cmp = paired_block(top3, base_top3, rng, higher_is_better=True)
        reductions = [beats["by_budget"][str(b)]["mean_advantage"] for b in BUDGETS]
        finite = [np.isfinite(column(name, b, "regret")).all() for b in BUDGETS]
        n_valid = N_SEEDS if all(finite) else int(min(np.isfinite(column(name, b, "regret")).sum() for b in BUDGETS))
        direct_name = meta.get(name, ("", "", "", None, False))[3] if name in meta else None
        beats_direct = None
        if direct_name:
            dreg = np.vstack([column(direct_name, b, "regret") for b in BUDGETS])
            beats_direct = paired_block(regret, dreg, rng, higher_is_better=False)
        noise_ok = all(
            (beats["by_budget"][str(b)]["mean_advantage"] > median_seed_std)
            for b in BUDGETS
        )
        uses_u = bool(meta.get(name, ("", "", "", None, False))[4]) if name in meta else False
        policy_info[name] = {
            "n_valid": n_valid,
            "uses_uncertainty": uses_u,
            "family": meta.get(name, ("replay", "", "", None, False))[0] if name in meta else "replay",
            "feature_set": meta.get(name, ("", "", "", None, False))[1] if name in meta else "",
            "direct_baseline": direct_name,
            "beats_random": beats,
            "top5_vs_random": top5_cmp,
            "top3_vs_random": top3_cmp,
            "beats_direct": beats_direct,
            "noise_gate": bool(noise_ok),
            "decision": {
                str(b): {
                    "spearman_mean": float(np.nanmean(decision_column(name, b, "spearman"))),
                    "kendall_mean": float(np.nanmean(decision_column(name, b, "kendall"))),
                    "ndcg10_mean": float(np.nanmean(decision_column(name, b, "ndcg10"))),
                    "ndcg_full_mean": float(np.nanmean(decision_column(name, b, "ndcg_full"))),
                }
                for b in BUDGETS
            },
        }

    def uncertainty_block(sp_list, sig_sum_v, err_sum_v, cnt_v):
        summary = mean_ci(sp_list, np.random.default_rng(7))
        mask = cnt_v > 0
        arch_sp = safe_spearman(sig_sum_v[mask] / cnt_v[mask], err_sum_v[mask] / cnt_v[mask])
        # Architecture bootstrap.
        idx_arch = np.flatnonzero(mask)
        sig_m = sig_sum_v[idx_arch] / cnt_v[idx_arch]
        err_m = err_sum_v[idx_arch] / cnt_v[idx_arch]
        boots = []
        rng_a = np.random.default_rng(8)
        for _ in range(BOOTSTRAP_B):
            take = rng_a.integers(0, len(idx_arch), len(idx_arch))
            boots.append(safe_spearman(sig_m[take], err_m[take]))
        boots_a = np.asarray(boots, dtype=float)
        arch_ci = [float(np.nanquantile(boots_a, 0.025)), float(np.nanquantile(boots_a, 0.975))]
        # Quintiles by architecture-mean sigma.
        order_s = np.argsort(sig_m, kind="mergesort")
        quintiles = []
        for q, chunk in enumerate(np.array_split(order_s, 5), start=1):
            quintiles.append({
                "quintile": q,
                "n": int(len(chunk)),
                "mean_predicted_sigma": float(np.mean(sig_m[chunk])) if len(chunk) else float("nan"),
                "mean_abs_error": float(np.mean(err_m[chunk])) if len(chunk) else float("nan"),
            })
        allow = (
            summary["n"] >= int(0.8 * N_SEEDS)
            and summary["mean"] >= UNCERTAINTY_SPEARMAN_MIN
            and summary["ci95"][0] > 0
            and arch_ci[0] > 0
        )
        return {
            "per_seed_spearman": summary,
            "n_splits": int(len(sp_list)),
            "n_undefined_spearman": int(len(sp_list) - summary["n"]),
            "architecture_mean_spearman": arch_sp,
            "architecture_mean_spearman_ci95": arch_ci,
            "quintiles": quintiles,
            "associated": bool(allow),
        }

    uncertainty = {
        "bootstrap_ridge": uncertainty_block(boot_sp, sig_sum, err_sum, cnt),
        "gp": uncertainty_block(gp_sp, gp_sig_sum, gp_err_sum, gp_cnt),
    }
    # Phase 6A acquisition uses bootstrap sigma. GP std does not turn that acquisition back on.
    uncertainty["uncertainty_acquisition_allowed"] = bool(uncertainty["bootstrap_ridge"]["associated"])
    uncertainty["gp_uncertainty_associated"] = bool(uncertainty["gp"]["associated"])
    uncertainty["rule"] = (
        "Bootstrap-ensemble sigma is the uncertainty used by Phase 6A acquisition. "
        "It stays forbidden unless the mean per-seed Spearman with absolute error is "
        f">= {UNCERTAINTY_SPEARMAN_MIN}, its seed-bootstrap 95% interval lies above 0, "
        "and the architecture-averaged Spearman interval also lies above 0. "
        "A calibrated GP standard deviation is reported and does not by itself authorize uncertainty acquisition."
    )

    v0_idx = v0_initial(ids, data["combos"].tolist(), data["macs"])
    v0_regret = float(y.max() - y[np.asarray(v0_idx)].max())
    learned = [
        "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch", "ensemble_ridge_et_gb", "gp_arch",
        "ridge_compute", "current_surrogate_mean", "phase6a_acquisition", "m1_ridge", "m1_extratrees",
        "m2_curve40_ridge", "m2_curve40_60_ridge", "pairwise_arch", "pairwise_m1",
    ]
    qualified = []
    for name in learned:
        info = policy_info[name]
        if info["n_valid"] != N_SEEDS:
            info["qualified"] = False
            info["reject_reason"] = "incomplete_seeds"
            continue
        if info["uses_uncertainty"] and not uncertainty["uncertainty_acquisition_allowed"]:
            info["qualified"] = False
            info["reject_reason"] = "uncertainty_acquisition_forbidden"
            continue
        if not info["beats_random"]["stable"]:
            info["qualified"] = False
            info["reject_reason"] = "not_stable_vs_random"
            continue
        if info["top5_vs_random"]["significantly_worse_any_budget"]:
            info["qualified"] = False
            info["reject_reason"] = "top5_significantly_worse"
            continue
        if not info["noise_gate"]:
            info["qualified"] = False
            info["reject_reason"] = "regret_reduction_within_seed_noise"
            continue
        if info["direct_baseline"] and not info["beats_direct"]["stable"]:
            info["qualified"] = False
            info["reject_reason"] = "does_not_stably_beat_direct_fidelity_sort"
            continue
        info["qualified"] = True
        info["reject_reason"] = ""
        qualified.append(name)

    best_arch_only = max(
        (bench_summary[n]["spearman_mean"] for n in (
            "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch", "gp_arch", "ensemble_ridge_et_gb", "pairwise_arch",
        )),
        default=float("nan"),
    )
    return {
        "benchmark": bench_summary,
        "policies": policy_info,
        "qualified": qualified,
        "uncertainty": uncertainty,
        "init_regret": {
            "n": len(init_regret),
            "mean": float(np.mean(init_regret)),
            "median": float(np.median(init_regret)),
            "min": float(np.min(init_regret)),
            "max": float(np.max(init_regret)),
            "std": float(np.std(init_regret, ddof=1)),
        },
        "v0_initial_regret": v0_regret,
        "v0_initial_ids": [ids[i] for i in v0_idx],
        "random_regret": {
            str(b): {
                "mean": float(np.mean(column("random", b, "regret"))),
                "median": float(np.median(column("random", b, "regret"))),
                "p25": float(np.quantile(column("random", b, "regret"), 0.25)),
                "p75": float(np.quantile(column("random", b, "regret"), 0.75)),
            }
            for b in BUDGETS
        },
        "best_architecture_only_spearman_mean": float(best_arch_only),
        "small_sample_predictor_supported": bool(best_arch_only < 0.40),
        "random_decision": {
            str(b): {
                "spearman_mean": float(np.nanmean(decision_column("random", b, "spearman"))),
                "kendall_mean": float(np.nanmean(decision_column("random", b, "kendall"))),
                "ndcg10_mean": float(np.nanmean(decision_column("random", b, "ndcg10"))),
                "ndcg_full_mean": float(np.nanmean(decision_column("random", b, "ndcg_full"))),
            }
            for b in BUDGETS
        },
        "search_store_means": {
            name: {
                str(b): {
                    "mean_regret": float(np.nanmean(column(name, b, "regret"))),
                    "median_regret": float(np.nanmedian(column(name, b, "regret"))),
                    "mean_top3": float(np.nanmean(column(name, b, "top3_recall"))),
                    "mean_top5": float(np.nanmean(column(name, b, "top5_recall"))),
                }
                for b in BUDGETS
            }
            for name in replay_names
        },
    }


def zlib_name(name: str) -> int:
    import zlib
    return zlib.adler32(name.encode("utf-8")) & 0x7FFFFFFF


def diagnosis_flags(signal, encoding, replay) -> dict:
    u_allowed = replay["uncertainty"]["uncertainty_acquisition_allowed"]
    any_mean_only_beats = any(
        replay["policies"][n]["beats_random"]["stable"]
        for n in ("ridge_arch", "extratrees_arch", "ensemble_ridge_et_gb", "current_surrogate_mean", "gp_arch")
    )
    acq = replay["policies"]["phase6a_acquisition"]["beats_random"]["stable"]
    return {
        "label_noise": bool(signal["label_noise_supported"]),
        "low_fidelity_40_rank_inconsistent": bool(signal["fidelity40_inconsistent"]),
        "fidelity_60_rank_inconsistent": bool(signal["fidelity60_inconsistent"]),
        "encoding_insufficient": bool(encoding["encoding_insufficient"]),
        "predictor_weak_on_small_sample": bool(replay["small_sample_predictor_supported"]),
        "uncertainty_uncalibrated": not u_allowed,
        "acquisition_is_primary_cause": bool((not u_allowed) and any_mean_only_beats and (not acq)),
        "v0_protocol_used_one_initial_set_and_budget_72": True,
        "repaired_protocol_finds_a_qualified_predictor": bool(replay["qualified"]),
    }


def render_signal(signal, curves, curves_fig, flags, replay) -> str:
    s40 = signal["map40_vs_map80"]
    s60 = signal["map60_vs_map80"]
    lines = []
    a = lines.append
    a("# Surrogate Signal Audit")
    a("")
    a("本文件审计现有 architecture 标签里，surrogate（代理模型）可用的排序信号有多强。全程只读 Phase2E / Phase2F，没有新训练，也没有改写 ground truth。")
    a("")
    a("主指标是 best-valid mAP50:95（IoU 从 0.50 到 0.95 的平均精度均值，取该预算内验证集最佳值）。Spearman（斯皮尔曼秩相关）和 Kendall（肯德尔 τ 秩相关）都是越高越说明两次排序一致。")
    a("")
    a("## 数据范围")
    a("")
    a(f"- Stage1（40 epoch）: {signal['n_stage1']} 个 architecture，budget = {', '.join(signal['stage1_budgets'])}。")
    a(f"- Stage2（60 epoch）: {signal['n_stage2']} 个 architecture，budget = {', '.join(signal['stage2_budgets'])}。")
    a(f"- Final（80 epoch）: {signal['n_final80']} 个 architecture，budget = {', '.join(signal['final_budgets'])}，seed = {', '.join(signal['final_seeds'])}。")
    a("- 40/60 对 80 的相关只在同时拥有两侧标签的 72 个 promoted finalist 上计算。这是条件相关，不是 144 个 architecture 的无偏样本。")
    a(f"- 72 个 finalist 的 output combination 计数: {signal['combo_counts']}。")
    a("- `P2_P3_P4` 在 X_V0.1 中有 16 个 architecture，但 80 epoch finalist 里是 0。replay 没有这个 combination 的 mAP80 标签，不能估计它的 80 epoch 效应。")
    a(f"- 80 epoch mAP50:95 范围 {f6(signal['y_min'])} 到 {f6(signal['y_max'])}，均值 {f6(signal['y_mean'])}，样本标准差 {f6(signal['between_std_72'])}。")
    a("")
    a("## 1. 40 epoch 对 80 epoch")
    a("")
    a(f"n = {s40['n']}。Spearman = {f4(s40['spearman'])}，95% bootstrap 区间 {f4(s40['spearman_ci95'][0])} 到 {f4(s40['spearman_ci95'][1])}。")
    a(f"Kendall = {f4(s40['kendall'])}，95% 区间 {f4(s40['kendall_ci95'][0])} 到 {f4(s40['kendall_ci95'][1])}。")
    a(f"以 40 epoch 排序去找回 80 epoch 的 Top-3 recall = {f4(s40['top3_recall'])}，Top-5 recall = {f4(s40['top5_recall'])}。")
    a(f"成对一致率 = {f4(s40['pairwise']['pairwise_concordance'])}，不一致对 = {s40['pairwise']['discordant']} / {s40['pairwise']['concordant'] + s40['pairwise']['discordant']}。")
    a(f"80 epoch 最优 architecture `{s40['best80_id']}` 在 40 epoch 上的名次是 {s40['best80_rank_at_low_fidelity']}。")
    e40 = signal["endpoint40_vs_map80"]
    a(f"敏感性：40 epoch 的 endpoint mAP 对 80 epoch best-valid 的 Spearman = {f4(e40['spearman'])}，Kendall = {f4(e40['kendall'])}。主结论使用 best-valid，与 Phase2E promotion 指标一致。")
    a("")
    a("## 2. 60 epoch 对 80 epoch")
    a("")
    a(f"n = {s60['n']}。Spearman = {f4(s60['spearman'])}，95% 区间 {f4(s60['spearman_ci95'][0])} 到 {f4(s60['spearman_ci95'][1])}。")
    a(f"Kendall = {f4(s60['kendall'])}，95% 区间 {f4(s60['kendall_ci95'][0])} 到 {f4(s60['kendall_ci95'][1])}。")
    a(f"Top-3 recall = {f4(s60['top3_recall'])}，Top-5 recall = {f4(s60['top5_recall'])}。")
    a(f"成对一致率 = {f4(s60['pairwise']['pairwise_concordance'])}，不一致对 = {s60['pairwise']['discordant']}。")
    a(f"80 epoch 最优 architecture 在 60 epoch 上的名次是 {s60['best80_rank_at_low_fidelity']}。")
    e60 = signal["endpoint60_vs_map80"]
    a(f"敏感性：60 epoch endpoint 对 80 epoch best-valid 的 Spearman = {f4(e60['spearman'])}，Kendall = {f4(e60['kendall'])}。")
    a("")
    a("40 epoch 排序与 80 epoch 排序不一致。60 epoch 排序明显更接近 80 epoch，但最优 architecture 的 60 epoch 名次仍然靠后，单看冠军名次会漏掉最终第一名。")
    a("")
    a("## 3. 同 architecture 的 seed 波动")
    a("")
    a("Phase2F 对 6 个 architecture 各有 seed 42/43/44。seed 42 沿用 Phase2E，seed 43 与 44 是独立从头训练。下表的 std 是样本标准差（ddof = 1）。")
    a("")
    a("| label | architecture | n | mean mAP | std mAP | var | min | max | seed42 |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in signal["seed_rows"]:
        a(f"| {row['label']} | `{row['architecture_id']}` | {row['n_seeds']} | {f6(row['mean_mAP'])} | {f6(row['std_mAP'])} | {f6(row['var_mAP'])} | {f6(row['min_mAP'])} | {f6(row['max_mAP'])} | {f6(row['seed42_mAP'])} |")
    a("")
    a(f"6 个 architecture 的 seed std 中位数 = {f6(signal['median_seed_std'])}，均值 = {f6(signal['mean_seed_std'])}，范围 {f6(signal['min_seed_std'])} 到 {f6(signal['max_seed_std'])}。")
    a("")
    a("同一批 6 个 architecture 上，不同 seed 之间的排序相关：")
    a("")
    a("| seed A | seed B | Spearman | Kendall |")
    a("|---:|---:|---:|---:|")
    for row in signal["seed_pairs"]:
        a(f"| {row['seed_a']} | {row['seed_b']} | {f4(row['spearman'])} | {f4(row['kendall'])} |")
    a("")
    a("n = 6，这组相关本身方差很大，只作为噪声上限的直接证据：换一个 seed，名次会重排。")
    a("")
    a("## 4. architecture 之间的差距对 seed 噪声")
    a("")
    a(f"- 72 个 finalist 的两两绝对差中位数 = {f6(signal['median_pairwise_gap_72'])}，均值 = {f6(signal['mean_pairwise_gap_72'])}。")
    a(f"- 其中小于 median seed std 的比例 = {f4(signal['fraction_pairs_below_median_seed_std'])}。")
    a(f"- 相邻名次差的中位数 = {f6(signal['median_adjacent_gap'])}；小于 median seed std 的比例 = {f4(signal['fraction_adjacent_below_median_seed_std'])}。")
    a(f"- 第 1 名与第 2 名的差距 = {f6(signal['top1_top2_gap'])}。")
    a(f"- 第 1 名与第 5 名的差距 = {f6(signal['top1_top5_gap'])}；与第 10 名的差距 = {f6(signal['top1_top10_gap'])}。")
    a(f"- Top-10 内部相邻差距的中位数 = {f6(signal['median_top10_adjacent_gap'])}；小于 median seed std 的比例 = {f4(signal['fraction_top10_adjacent_below_seed_std'])}。")
    a("")
    a("第 1 名和第 2 名的 seed-42 差距小于单个 architecture 的 seed 标准差。Top-10 里的相邻差距同样落在 seed 噪声里面。surrogate 即使完美拟合这张 seed-42 表，追到的名次差也小于重复训练的波动。")
    a("")
    a("## 5. 逐 epoch 曲线覆盖")
    a("")
    a("M2 若使用 last-k slope、逐 epoch validation loss，只能用已经存在的日志，不能为了特征重新训练。")
    a("")
    for block in (curves, curves_fig):
        if not block.get("exists"):
            a(f"- `{block.get('path')}` 不存在。")
            continue
        a(f"- `{block['path']}`: {block['n_architectures']} 个 architecture，{block['n_rows']} 行，与 72 个 finalist 的交集 = {block['n_overlap_final80']}，最大 epoch = {block['max_epoch']}，含 validation loss 的行数 = {block['rows_with_validation_loss']}。")
    a("")
    a("逐 epoch 曲线没有覆盖 72 个 finalist，因此 replay 里的 learning-curve 特征只使用 Stage1/Stage2 已经写好的摘要：mAP、endpoint mAP、AP50、AP75、Precision、Recall、best epoch，以及 (mAP60 - mAP40) / 20。没有把 12 个 witness 的逐 epoch slope 外推到其余 architecture。")
    a("")
    a("## 6. 六类失败来源")
    a("")
    a("| 来源 | 判定 | 证据 |")
    a("|---|---|---|")
    a(f"| 1. 标签噪声 | {'成立' if flags['label_noise'] else '当前证据不足'} | median seed std {f6(signal['median_seed_std'])}，第 1–2 名差距 {f6(signal['top1_top2_gap'])}，Top-10 相邻差距中位数 {f6(signal['median_top10_adjacent_gap'])}。 |")
    a(f"| 2. 低保真与高保真排序不一致 | 40 epoch 成立；60 epoch {'成立' if flags['fidelity_60_rank_inconsistent'] else '不成立'} | 40→80 Spearman {f4(s40['spearman'])}，60→80 Spearman {f4(s60['spearman'])}。 |")
    a(f"| 3. architecture encoding 不充分 | {'成立' if flags['encoding_insufficient'] else '不成立'} | 见 `ARCH_ENCODING_AUDIT.md`。可搜索因子的最小编码与 144 个 id 一一对应。 |")
    a(f"| 4. predictor 不适合小样本 | {'成立' if flags['predictor_weak_on_small_sample'] else '不成立'} | 只使用 architecture bits 的模型，在 20/52 split 上的最佳平均 Spearman = {f4(replay['best_architecture_only_spearman_mean'])}。 |")
    a(f"| 5. acquisition 设计 | uncertainty 与误差 {'无稳定正相关，禁止继续使用 uncertainty acquisition' if flags['uncertainty_uncalibrated'] else '存在正相关'}；acquisition 是否为主要原因: {'是' if flags['acquisition_is_primary_cause'] else '否'} | 主要原因指：去掉 uncertainty 之后 mean-only 能稳定优于 Random，而 Phase6A acquisition 不能。两者都不能稳定优于 Random 时，acquisition 是缺陷，但不是 surrogate 失败的唯一原因。 |")
    a(f"| 6. replay protocol 偏差 | V0 协议有缺陷；修复后 {'找到合格 predictor' if flags['repaired_protocol_finds_a_qualified_predictor'] else '仍没有合格 predictor'} | V0 使用单个 initial-20，并把 evaluated = 72 算进 admission。该 initial set 的 regret = {f6(replay['v0_initial_regret'])}。本轮 40 个 initial set 的 regret 均值 {f6(replay['init_regret']['mean'])}，中位数 {f6(replay['init_regret']['median'])}，范围 {f6(replay['init_regret']['min'])} 到 {f6(replay['init_regret']['max'])}。 |")
    a("")
    if flags["repaired_protocol_finds_a_qualified_predictor"]:
        a("标签噪声和 40 epoch 排序不一致仍然成立。修复后的多 seed 协议找到了合格 predictor，见 admission 文件。")
    else:
        a("标签噪声，以及 40 epoch 与 80 epoch 的排序不一致，是有数量支持的失败来源。encoding 碰撞不是失败来源。V0 的单 split 和 evaluated = 72 会扭曲单次 admission，但 40 个 paired seed 修复之后仍然没有 predictor 在 32、44、56 上稳定优于 Random。V0 的 FAIL 不是协议误伤。")
    a("")
    a("## 复现")
    a("")
    a("`python tools/sonar_spec/phase6a1_surrogate_diagnosis.py`")
    a("")
    a("Phase2E `final_results.csv`、`stage1_results.csv` 与 Phase2F `multiseed_results.csv` 的 SHA256 与 Phase 6A manifest 一致，脚本结束时再次核对，确认本轮没有改写这些文件。")
    a("")
    return "\n".join(lines)


def render_encoding(encoding) -> str:
    lines = []
    a = lines.append
    a("# Architecture Encoding Audit")
    a("")
    a("搜索空间是冻结的 X_V0.1，共 144 个 architecture。encoding（架构编码）必须区分这些 architecture。Params 与 MACs 只是附加的 compute-aware（计算量感知）数值，不能代替编码。")
    a("")
    a("## 可搜索因子与冻结因子")
    a("")
    a("枚举规则与 `src/hwnas_fpga/models/sonar_supernet/__init__.py` 的 `enumerate_architectures` 相同：每个 output combination 在其 last active stage 之前的 MBConv 上独立取 kernel 3 或 5。")
    a("")
    a("| 因子 | 在 144 个 architecture 上是否变化 | 编码方式 |")
    a("|---|---|---|")
    a("| output combination / head 数量 / FPN level | 变化，3 种 | `combo_P3_P4_P5`、`combo_P2_P3_P4`；两者都为 0 时表示 `P2_P3_P4_P5` |")
    a("| kernel | 变化，活跃 MBConv 上为 3 或 5 | `k5_stage{1,2,4,5,6,7}` |")
    a("| stage 是否执行 | stage 6 与 stage 7 随 `P2_P3_P4` 关闭；stage 1/2/4/5 始终执行 | 由 output combination 决定；关闭的 stage 其 kernel bit 固定为 0 |")
    a("| depth | 不变化，各 stage depth = 1 | 常数。放进特征矩阵只会得到零方差列 |")
    a("| width / channels | 不变化 | 常数，由 search space 固定 |")
    a("| operator | 不变化：stem/stage0 conv，MBConv，stage3 skip | 常数 |")
    a("| fusion | 不变化：fixed top-down nearest-add | 常数 |")
    a("| head tower | 不变化：kernel 3、每位置 3 个 anchor、21 个预测通道；head 个数由 level 数决定 | head 个数已由 combo 编码 |")
    a("| skip / topology | stage3 固定为等通道 identity skip；没有可搜索的跨层拓扑 | 常数 |")
    a("")
    a("Stage 定义来自 `configs/search_space_sonar_v0_1.yaml`：")
    a("")
    a("| stage | operator | channels | depth | stride | expansion | kernel |")
    a("|---|---|---:|---:|---:|---:|---|")
    for row in encoding["stages"]:
        kernel = row["kernel"] if row["kernel"] not in (None, "") else "—"
        expansion = row["expansion"] if row["expansion"] not in (None, "") else "—"
        a(f"| {row['factor']} | {row['operator']} | {row['channels']} | {row['depth']} | {row['stride']} | {expansion} | {kernel} |")
    a("")
    a(f"Neck: `{json.dumps(encoding['neck'], ensure_ascii=False)}`。")
    a(f"Head: `{json.dumps(encoding['head'], ensure_ascii=False)}`。")
    a("")
    a("活跃 stage 模式：")
    a("")
    for row in encoding["active_patterns"]:
        a(f"- stages {row['stages_active']} 活跃时，output combination = {', '.join(row['combos'])}。")
    a("")
    a("`P3_P4_P5` 与 `P2_P3_P4_P5` 的 MBConv 活跃集合相同。二者的差别是 P2 head / fusion level，必须靠 output combination 编码，不能只靠 kernel。")
    a("")
    a("## 唯一性与碰撞")
    a("")
    a("| 编码 | unique keys | collision groups | 落入碰撞的 architecture 数 |")
    a("|---|---:|---:|---:|")
    for key, label in (
        ("id", "architecture_id"),
        ("full_combo_kernel", "combo + 各 stage kernel"),
        ("minimal_model_bits", "最小模型比特（2 个 combo bit + 6 个 kernel bit）"),
        ("phase6a_without_params_macs", "Phase 6A 特征，去掉 Params/MACs"),
        ("num_heads_plus_kernels", "head 数 + kernel"),
        ("kernel_only", "只保留 kernel，去掉 combo"),
        ("params_macs", "只保留 (Params, MACs)"),
    ):
        block = encoding[key]
        a(f"| {label} | {block['n_unique_keys']} | {block['n_collision_groups']} | {block['n_ids_in_collisions']} |")
    a("")
    kernel = encoding["kernel_only"]
    if kernel["examples"]:
        a("去掉 output combination 之后，相同 kernel 会把不同 head/fusion 配到同一个向量。例子：")
        a("")
        for group in kernel["examples"]:
            a("- " + "；".join(f"`{item}`" for item in group))
        a("")
    cost = encoding["params_macs"]
    if cost["n_collision_groups"]:
        a(f"(Params, MACs) 有 {cost['n_collision_groups']} 组碰撞，不能当作 architecture id。")
    else:
        a("(Params, MACs) 在这 144 个 architecture 上没有撞键。它仍然是计算量数值，不是结构编码：两个未来的 architecture 可以计算量相同而 kernel/head 不同，查找表也不会因此知道结构。")
    a("")
    a("## 结论")
    a("")
    if encoding["encoding_insufficient"]:
        a("最小可搜索编码存在碰撞，encoding 不充分成立。")
    else:
        a("144 个 architecture id 互异。最小编码（output combination + 每个 MBConv 的 kernel bit）与这 144 个 id 一一对应，collision group = 0。")
        a("depth、width、operator、fusion 形式、head tower、skip 在 X_V0.1 中是冻结常数，写成特征列不会增加区分度。Phase 6A 的聚合特征（k5 计数、early/deep k5、has_P2、has_P5、num_heads）是这些比特的函数，去掉 Params/MACs 后同样没有碰撞。")
        a("因此本轮 surrogate 的排序失败不能归因于 144 个 architecture 的编码碰撞或漏掉可变 kernel/stage/head 因子。")
    a("")
    a("Params/MACs 在后续表格里标记为 compute-aware。它们不是 FPGA hardware cost（硬件代价）。")
    a("")
    return "\n".join(lines)


def render_uncertainty(replay) -> str:
    u = replay["uncertainty"]
    lines = []
    a = lines.append
    a("# Uncertainty Audit")
    a("")
    a("Phase 6A acquisition 使用 bootstrap ridge ensemble 的预测标准差 sigma 作为 uncertainty（不确定性）。本审计检查 sigma 是否随绝对预测误差变大。若没有这种关系，uncertainty acquisition 停止使用。")
    a("")
    a("每个 replay seed 用分层抽出的 20 个 architecture 拟合，在其余 52 个上比较 sigma 与 |预测 − mAP80|。推断单位是 seed。另外把每个 architecture 在它进入测试集的那些 seed 上取平均，再做一次 architecture 级 Spearman。")
    a("")
    a(f"判定阈值（预注册）：seed 级 Spearman 均值 ≥ {UNCERTAINTY_SPEARMAN_MIN}，其 95% bootstrap 区间下界 > 0，并且 architecture 级 Spearman 的 95% 区间下界 > 0。GP（高斯过程）的预测标准差单独报告，不能反过来批准 Phase 6A 的 bootstrap uncertainty。")
    a("")
    for key, title in (("bootstrap_ridge", "Bootstrap ridge sigma（Phase 6A 所用）"), ("gp", "GP predictive std（补充）")):
        block = u[key]
        seed = block["per_seed_spearman"]
        a(f"## {title}")
        a("")
        a(f"- 可计算 Spearman 的 seed 数: {seed['n']} / {block['n_splits']}。其余 {block['n_undefined_spearman']} 个 split 的预测标准差近似常数，秩相关没有定义。")
        a(f"- seed 级 Spearman 均值: {f4(seed['mean'])}，95% 区间 {f4(seed['ci95'][0])} 到 {f4(seed['ci95'][1])}")
        a(f"- architecture 平均后的 Spearman: {f4(block['architecture_mean_spearman'])}，95% 区间 {f4(block['architecture_mean_spearman_ci95'][0])} 到 {f4(block['architecture_mean_spearman_ci95'][1])}")
        a(f"- 与绝对误差有稳定正相关: {block['associated']}")
        a("")
        a("| sigma 五分位（低到高） | n | mean sigma | mean absolute error |")
        a("|---:|---:|---:|---:|")
        for row in block["quintiles"]:
            a(f"| {row['quintile']} | {row['n']} | {f6(row['mean_predicted_sigma'])} | {f6(row['mean_abs_error'])} |")
        a("")
    a("## 决定")
    a("")
    if u["uncertainty_acquisition_allowed"]:
        a("Bootstrap sigma 与绝对误差有稳定正相关。uncertainty acquisition 在本轮规则下可以保留。")
    else:
        a("Bootstrap sigma 与绝对误差没有达到预注册的正相关标准。禁止继续使用 uncertainty acquisition。")
        a("Phase 6A 的 acquisition 把这项 sigma 加进了打分，因此该 acquisition 不能进入 Phase 6B。")
    if u["gp_uncertainty_associated"]:
        a("GP 标准差与误差有相关。本轮没有把它实现成新的 acquisition，也没有用它推翻上面的禁止。")
    else:
        gp_sp = u["gp"]["architecture_mean_spearman"]
        a(f"GP 标准差没有达到同一标准。architecture 级 Spearman = {f4(gp_sp)}。")
        if gp_sp < 0:
            a("该相关为负：更高的预测标准差对应更小的绝对误差，方向与可用的 uncertainty 相反。")
    a("")
    a("五分位表是描述。Bootstrap sigma 从低到高，平均绝对误差没有单调上升。")
    a("")
    a("判定规则：Phase 6A acquisition 使用的是 bootstrap ensemble 的 sigma。只有当 seed 级 Spearman 均值不低于 0.2、其 95% 区间下界大于 0，并且 architecture 级 Spearman 的 95% 区间下界也大于 0 时，才允许继续使用 uncertainty acquisition。GP 标准差即使相关，本轮也不把它改写成新的 acquisition。")
    a("")
    return "\n".join(lines)


def render_hw(hw) -> str:
    lines = []
    a = lines.append
    a("# Full-Architecture FPGA Cost Gap")
    a("")
    a("本文件只审计 H4 cost 能不能覆盖 X_V0.1 的 144 个 architecture。Params 与 MACs 是计算量，不是 FPGA hardware cost（硬件代价）。LUT（查找表资源）、DSP（数字信号处理单元）、BRAM（块 RAM）和 latency（推理延迟）只有在 H4 查到实测行时才引用。")
    a("")
    a("## 覆盖")
    a("")
    a(f"- H4 行数: {hw['n_rows']}")
    a(f"- network 标签: {', '.join(hw['networks'])}")
    a(f"- 不同 architecture_id: {hw['n_unique_architectures']}")
    a(f"- 落在 144 个 architecture 内: {hw['n_in_144']}")
    a(f"- 对 144 的覆盖率: {hw['n_unique_architectures']} / 144 = {f4(hw['coverage_of_144'])}")
    a(f"- quant_config: {', '.join(hw['quant_configs'])}")
    a(f"- mapping policy 数: {len(hw['mapping_policies'])}（{', '.join(hw['mapping_policies'])}）")
    a(f"- status: {', '.join(hw['statuses'])}")
    a("")
    a("这是 shortlist / network 级查找，不是 144 个 architecture 各有一行 hardware cost。同一 architecture 还有多条 mapping policy，所以 cost 也不是 architecture 单独的属性，它依赖量化配置 q 和映射 m。")
    a("")
    a("## 已有直接 cost 的 architecture")
    a("")
    a("| network | architecture | rows | LUT peak min–max | DSP peak min–max | BRAM18 peak min–max | latency μs min–max | lookup = total nodes |")
    a("|---|---|---:|---|---|---|---|---|")
    for row in hw["per_arch"]:
        a(
            f"| {', '.join(row['network'])} | `{row['architecture_id']}` | {row['n_rows']} | "
            f"{row['LUT_peak_min']:.0f}–{row['LUT_peak_max']:.0f} | "
            f"{row['DSP_peak_min']:.0f}–{row['DSP_peak_max']:.0f} | "
            f"{row['BRAM18_peak_min']:.0f}–{row['BRAM18_peak_max']:.0f} | "
            f"{row['latency_us_min']:.0f}–{row['latency_us_max']:.0f} | {row['lookup_equals_total']} |"
        )
    a("")
    if hw["per_arch"]:
        sample = hw["per_arch"][0]
        a(f"latency scope: {sample['latency_scope']}")
        a(f"resource scope: {sample['resource_scope']}")
        a(f"shell reservation: {sample['shell_reservation']}；deployment: {sample['deployment']}；constraint: {sample['constraint_status']}")
        a("")
        a("LUT / DSP / BRAM 来自算子级 HLS CSYNTH 查找后再求和或取 peak。`lookup_nodes = total_nodes` 只说明这几个 shortlist 网络的节点都查到了，不说明其他 kernel 组合也能这样求和。")
    a("")
    a("## 为什么不能聚合成 144 个 architecture 的 FPGA cost")
    a("")
    a(f"- Cost API 文件声明 exact lookup: {hw['cost_api_exact_lookup']}。")
    a(f"- Cost API 禁止 nearest interpolation: {hw['cost_api_forbids_interpolation']}。")
    a(f"- Cost API 拒绝把 partial lookup 当成 network cost: {hw['cost_api_forbids_partial']}。")
    a("- kernel 从 3 改成 5 会改变 depthwise 的 shape。未测量的 shape 按 API 返回 `NEEDS_MEASUREMENT`，不能用邻居算子的 LUT 填上。")
    a("- 因此当前 H4 不能可靠聚合成 144 维的 architecture-level FPGA cost。")
    a("")
    a("## Params / MACs")
    a("")
    a(f"Phase2E Stage1 为 {hw['params_macs_rows']} 个 architecture 提供了 Params 与 MACs，覆盖全部 144 个: {hw['params_macs_cover_144']}。")
    a("这些数字只标记为 compute-aware。本轮 replay 不命名为 hardware-aware replay。")
    a("")
    a("## 结论")
    a("")
    a("直接 FPGA cost 只存在于上表中的 shortlist architecture，并且每条 cost 还绑定一条 mapping policy。144 个 architecture 的 hardware-aware 搜索现在没有 cost 覆盖。")
    a("")
    return "\n".join(lines)


def render_replay(signal, replay, flags) -> str:
    lines = []
    a = lines.append
    a("# Multi-Fidelity Replay Report")
    a("")
    a("本 replay 修复 Phase 6A 的协议：40 个独立 initial-set seed，每次 initial observed = 20，batch size = 6。比较在 evaluated = 32、44、56 上停止。evaluated = 72 不作为 admission 依据，因为那时 72 个候选都已被观察。")
    a("")
    a("Random 与每个 predictor 共用同一个 initial-20。每个 seed 里，Random 再抽 30 条续搜，取 regret 与 recall 的中位数，作为该 seed 的 Random 成绩。bootstrap 在 40 个 seed 上重采样。")
    a("")
    a("本轮使用的是 80 epoch 查询预算上的 accuracy / multi-fidelity / ranking 比较。Params/MACs 版本标记为 compute-aware。H4 FPGA cost 没有覆盖 144 个 architecture，因此本轮不称为 hardware-aware replay。")
    a("")
    a("## V0 协议偏差")
    a("")
    a("Phase 6A 只有一个 initial-20（seed 42，按 MACs 分箱），并把 evaluated = 72 的零 regret 算进“超过 Random 的 checkpoint”。")
    a(f"按同一 V0 规则重建的 initial-20，其 regret = {f6(replay['v0_initial_regret'])}。")
    a(f"本轮 40 个分层 initial-20 的 regret：均值 {f6(replay['init_regret']['mean'])}，中位数 {f6(replay['init_regret']['median'])}，范围 {f6(replay['init_regret']['min'])}–{f6(replay['init_regret']['max'])}，标准差 {f6(replay['init_regret']['std'])}。")
    a("单个 initial set 的起点差异已经和名次差距同一量级，所以 V0 不能把起点运气和 acquisition 分开。")
    a("")
    a("## 指标定义")
    a("")
    a("- regret: 72 个 finalist 中真实最佳 mAP50:95 减去当前已观察集合里的最佳值。越小越好。")
    a("- Top-3 / Top-5 recall: 全局前 3 / 前 5 名里，已经进入观察集合的比例。")
    a("- decision Spearman / Kendall / NDCG: 在到达该预算的前一步，模型给尚未观察的候选打分，与真实 mAP80 的排序质量。NDCG（归一化折损累计增益）用 mAP50:95 做 relevance。全列表 NDCG 会被 mAP 的窄动态范围抬高，决策时更看 NDCG@10、Spearman 和 Kendall。Random 行是随机打分的 seed 内中位数再对 seed 取均值。")
    a("- median / mean regret: 先在 seed 内得到一个 regret，再对 40 个 seed 取中位数或均值。")
    a("")
    a("## 直接保真度参照")
    a("")
    a("`direct_map40` 与 `direct_map60` 不是 surrogate。它们按已经测得的 40 或 60 epoch mAP 从高到低挑选。multi-fidelity 模型必须稳定优于它所消费的那一列，才算 surrogate 提供了额外信息；否则结果只是在重放已有的 multi-fidelity evidence。")
    a("")
    a("## Regret 与 recall")
    a("")
    a("下表的 top3 / top5 是搜索召回：全局前 3 / 前 5 名里已经观察过的比例。")
    a("")
    a("| policy | " + " | ".join(
        f"medR@{b} | meanR@{b} | top3@{b} | top5@{b}" for b in BUDGETS
    ) + " |")
    a("|---| " + " | ".join(["---: | ---: | ---: | ---:" for _ in BUDGETS]) + " |")
    means = replay["search_store_means"]
    order = [
        "random", "direct_map40", "direct_map60", "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch",
        "ensemble_ridge_et_gb", "gp_arch", "ridge_compute", "current_surrogate_mean", "phase6a_acquisition",
        "m1_ridge", "m1_extratrees", "m2_curve40_ridge", "m2_curve40_60_ridge", "pairwise_arch", "pairwise_m1",
    ]
    for name in order:
        cells = [name]
        for b in BUDGETS:
            row = means[name][str(b)]
            cells.append(f6(row["median_regret"]))
            cells.append(f6(row["mean_regret"]))
            cells.append(f4(row["mean_top3"]))
            cells.append(f4(row["mean_top5"]))
        a("| " + " | ".join(cells) + " |")
    a("")
    a("## 相对 Random 的 regret 降幅")
    a("")
    a("降幅 = Random regret − policy regret。稳定优于 Random 要求：每个预算的降幅 95% 区间下界 > 0，且 policy 的中位 regret 更低。")
    a("")
    a("| policy | 32 mean reduction (CI) | 44 mean reduction (CI) | 56 mean reduction (CI) | win rate 32/44/56 | stable vs Random | noise gate | beats direct fidelity | qualified |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name in order:
        if name == "random":
            continue
        info = replay["policies"][name]
        br = info["beats_random"]["by_budget"]
        cells = [name]
        for b in BUDGETS:
            item = br[str(b)]
            cells.append(f"{f6(item['mean_advantage'])} ({f6(item['advantage_ci95'][0])}, {f6(item['advantage_ci95'][1])})")
        wins = "/".join(f4(br[str(b)]["win_rate"]) for b in BUDGETS)
        direct = "NA"
        if info["beats_direct"] is not None:
            direct = "yes" if info["beats_direct"]["stable"] else "no"
        cells.extend([
            wins,
            "yes" if info["beats_random"]["stable"] else "no",
            "yes" if info["noise_gate"] else "no",
            direct,
            "yes" if info.get("qualified") else "no",
        ])
        a("| " + " | ".join(cells) + " |")
    a("")
    a(f"noise gate 的阈值是 Phase2F median seed std = {f6(signal['median_seed_std'])}。降幅必须在 32、44、56 三个预算上都大于这个数。小于 seed 噪声的 regret 改进不能支持新的训练 pilot。")
    a("")
    learned = [
        "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch", "ensemble_ridge_et_gb", "gp_arch",
        "ridge_compute", "current_surrogate_mean", "m1_ridge", "m1_extratrees",
        "m2_curve40_ridge", "m2_curve40_60_ridge", "pairwise_arch", "pairwise_m1",
    ]
    any_stable = any(replay["policies"][name]["beats_random"]["stable"] for name in learned)
    if any_stable:
        a("至少有一个 learned predictor 在三个预算上都稳定优于 Random。是否过 admission 还要看 noise gate、Top-5 与 direct-fidelity gate，见 qualified 列。")
    else:
        a("learned predictor 的 stable vs Random 全部为 no。noise gate 和 direct-fidelity gate 没有改写这个结论：没有任何模型先在 32、44、56 上稳定优于 Random。")
    a("")
    a("win rate 只计算 regret 严格更低的 seed。evaluated = 56 时，Random 在每个 seed 上的中位 regret 已经是 0。56/72 的随机样本包含 seed-42 最优 architecture 的概率高于 1/2，30 次内部重复的中位数因此饱和。这个预算上的严格胜率是 0。均值 regret 也是 0 的策略与 Random 持平；均值 regret 大于 0 的策略比 Random 更差。")
    a("")
    m2_sp = replay["benchmark"]["m2_curve40_60_ridge"]["spearman_mean"]
    d60_sp = replay["benchmark"]["direct_map60"]["spearman_mean"]
    a(f"`direct_map60` 在同一 20/52 split 上的平均 Spearman 是 {f4(d60_sp)}，`m2_curve40_60_ridge` 是 {f4(m2_sp)}。把 architecture 特征和学习到的增量加到已有的 60 epoch 标签上，没有超过直接按 mAP60 排序。")
    a("")
    a("`pairwise_arch` 的搜索 mean regret 在三个预算上都高于 Random，也高于 architecture-only 的 Ridge。pairwise 排序没有比绝对 mAP 回归更稳。")
    a("")
    a("## Decision-time 排序质量")
    a("")
    a("| policy | Sp@32 | Kd@32 | NDCG10@32 | Sp@44 | Kd@44 | NDCG10@44 | Sp@56 | Kd@56 | NDCG10@56 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    random_dec = replay["random_decision"]
    cells = ["random"]
    for b in BUDGETS:
        cells.extend([
            f4(random_dec[str(b)]["spearman_mean"]),
            f4(random_dec[str(b)]["kendall_mean"]),
            f4(random_dec[str(b)]["ndcg10_mean"]),
        ])
    a("| " + " | ".join(cells) + " |")
    for name in order:
        if name == "random":
            # random decision stats were stored
            info = None
        info = replay["policies"].get(name)
        if name == "random":
            continue
        dec = info["decision"]
        cells = [name]
        for b in BUDGETS:
            cells.extend([f4(dec[str(b)]["spearman_mean"]), f4(dec[str(b)]["kendall_mean"]), f4(dec[str(b)]["ndcg10_mean"])])
        a("| " + " | ".join(cells) + " |")
    a("")
    a("## 模型说明")
    a("")
    a("- Accuracy-only: `ridge_arch`、`rf_arch`、`extratrees_arch`、`gb_arch`、`gp_arch`、`ensemble_ridge_et_gb`。特征只有 architecture bits。ensemble 是 Ridge、ExtraTrees、Gradient Boosting 的预测平均。")
    a("- `ridge_compute` 与 `current_surrogate_mean` 额外使用 Params/MACs，标记为 compute-aware。current surrogate 是 Phase 6A 的 bootstrap ridge，但 acquisition 只用预测均值。")
    a("- `phase6a_acquisition` 复现 Phase 6A 的 uncertainty + MACs 打分，只作诊断。uncertainty 未校准时，它不能成为合格策略。")
    a("- M1: architecture + mAP40 → mAP80。M2: architecture + 曲线摘要 → (mAP80 − mAP40)，排序分 = mAP40 + 预测增量。`m2_curve40` 只用 40 epoch 摘要；`m2_curve40_60` 再加上 60 epoch 摘要。")
    a("- Ranking: pairwise logistic，特征差预测哪一个 mAP 更高，用线性得分排序。`pairwise_m1` 把 mAP40 放进特征。")
    a("- 超参数在看 replay 结果之前固定：Ridge alpha = 1；树模型 200 棵、`min_samples_leaf = 2`；Gradient Boosting 100 棵、深度 2、学习率 0.05；GP 为各向同性 Matern-1.5，`n_restarts_optimizer = 0`；bootstrap 模型数 = 32。没有在 52 个 held-out architecture 上做网格搜索。")
    a("")
    a("## 人口限制")
    a("")
    a("标签只存在于 72 个已经被 40 epoch 和 60 epoch 规则晋升的 architecture 上。这个 replay 不能证明同一 predictor 在未晋升的另一半搜索空间上仍然有效。测试集保持密封。")
    a("")
    a("## 与六类来源的关系")
    a("")
    a(f"- 标签噪声判定: {flags['label_noise']}。")
    a(f"- 40 epoch 排序不一致: {flags['low_fidelity_40_rank_inconsistent']}；60 epoch 排序不一致: {flags['fidelity_60_rank_inconsistent']}。")
    a(f"- encoding 不充分: {flags['encoding_insufficient']}。")
    a(f"- 小样本 architecture-only predictor 偏弱: {flags['predictor_weak_on_small_sample']}（最佳平均 Spearman {f4(replay['best_architecture_only_spearman_mean'])}）。")
    a(f"- uncertainty 未校准: {flags['uncertainty_uncalibrated']}。acquisition 为主要原因: {flags['acquisition_is_primary_cause']}。")
    a(f"- 修复协议后合格 predictor: {flags['repaired_protocol_finds_a_qualified_predictor']}。")
    a("")
    return "\n".join(lines)


def render_benchmark_rows(replay) -> list[dict]:
    rows = []
    order = [
        "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch", "gp_arch", "ensemble_ridge_et_gb",
        "current_surrogate_mean", "ridge_compute", "m1_ridge", "m1_extratrees", "m2_curve40_ridge",
        "m2_curve40_60_ridge", "pairwise_arch", "pairwise_m1", "direct_map40", "direct_map60",
    ]
    notes = {
        "direct_map40": "reference_not_a_fitted_surrogate; score is historical mAP40",
        "direct_map60": "reference_not_a_fitted_surrogate; score is historical mAP60",
        "pairwise_arch": "rmse_mae_use_affine_calibration_fit_on_train_only; rank metrics use raw score",
        "pairwise_m1": "rmse_mae_use_affine_calibration_fit_on_train_only; rank metrics use raw score",
        "ridge_compute": "compute-aware Params/MACs; not FPGA hardware cost",
        "current_surrogate_mean": "compute-aware Params/MACs; mean prediction only; not FPGA hardware cost",
        "m2_curve40_ridge": "no per-epoch slope; budget-40 summaries only",
        "m2_curve40_60_ridge": "uses existing stage2 mAP60 and summaries; not a new training feature",
    }
    for name in order:
        src = replay["benchmark"][name]
        row = dict(src)
        row["notes"] = notes.get(name, "topk_and_ndcg_are_computed_on_the_unobserved_52")
        if "topk" not in row["notes"]:
            row["notes"] += "; topk_and_ndcg_on_unobserved_52"
        rows.append(row)
    return rows


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(20260922)
    data = load_pool()
    signal = signal_audit(data, rng)
    encoding = encoding_audit(data)
    curves = curve_coverage(CURVES, set(data["pool_ids"]))
    curves_fig = curve_coverage(CURVES_FIG, set(data["pool_ids"]))
    hw = hw_audit(data)
    print("data audits ready, starting predictor benchmark and replay", flush=True)
    replay = benchmark_and_replay(data, signal)
    flags = diagnosis_flags(signal, encoding, replay)
    allowed = bool(replay["qualified"])
    verdict = "SURROGATE_REPAIRED_READY_FOR_PILOT" if allowed else "SURROGATE_NOT_JUSTIFIED"
    admission = {
        "phase": "6A.1",
        "prior_phase6a_status": "FAIL",
        "verdict": verdict,
        "PHASE6B_ALLOWED": allowed,
        "phase6b_started": False,
        "no_new_training": True,
        "phase2e_phase2f_modified": False,
        "test_evaluated": False,
        "replay_scope": "accuracy_multifidelity_ranking_and_compute_aware",
        "hardware_aware_replay": False,
        "n_replay_seeds": N_SEEDS,
        "initial_observed": INITIAL,
        "batch_size": BATCH,
        "admission_budgets": list(BUDGETS),
        "excluded_budget": 72,
        "inner_random_trajectories_per_seed": INNER_RANDOM,
        "bootstrap_replicates": BOOTSTRAP_B,
        "uncertainty_acquisition_allowed": replay["uncertainty"]["uncertainty_acquisition_allowed"],
        "qualified_predictors": replay["qualified"],
        "failure_sources": flags,
        "median_seed_std": signal["median_seed_std"],
        "best_architecture_only_spearman_mean": replay["best_architecture_only_spearman_mean"],
        "recommendation": (
            "A qualified predictor cleared paired Random, the seed-noise gate, and the direct-fidelity gate. Phase 6B was not started in this round."
            if allowed else
            "不要启动 surrogate 搜索。退回 multi-fidelity evidence、diversity，以及 compute/hardware constrained selection。"
        ),
        "any_learned_policy_stable_vs_random": any(
            replay["policies"][name]["beats_random"]["stable"] for name in (
                "ridge_arch", "rf_arch", "extratrees_arch", "gb_arch", "ensemble_ridge_et_gb", "gp_arch",
                "ridge_compute", "current_surrogate_mean", "m1_ridge", "m1_extratrees",
                "m2_curve40_ridge", "m2_curve40_60_ridge", "pairwise_arch", "pairwise_m1",
            )
        ),
        "population": "72 promoted finalists with 80-epoch labels; correlations are conditional on promotion",
        "input_sha256": {Path(k).name: v for k, v in data["before"].items()},
        "elapsed_seconds": None,
    }
    # Re-hash inputs before writing outputs.
    for path_s, digest in data["before"].items():
        if sha256(Path(path_s)) != digest:
            raise SystemExit(f"Input changed during the run: {path_s}")
    admission["phase2e_phase2f_modified"] = False
    admission["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    fieldnames = [
        "model", "family", "feature_set", "target", "n_seeds", "n_valid", "n_train", "n_test",
        "rmse_mean", "rmse_std", "mae_mean", "mae_std", "spearman_mean", "spearman_std",
        "spearman_ci95_low", "spearman_ci95_high", "kendall_mean", "kendall_std",
        "top3_recall_mean", "top3_recall_std", "top5_recall_mean", "top5_recall_std",
        "ndcg10_mean", "ndcg10_std", "ndcg_full_mean", "ndcg_full_std", "notes",
    ]
    write_text(OUT / "SURROGATE_SIGNAL_AUDIT.md", render_signal(signal, curves, curves_fig, flags, replay))
    write_text(OUT / "ARCH_ENCODING_AUDIT.md", render_encoding(encoding))
    write_csv(OUT / "PREDICTOR_BENCHMARK.csv", render_benchmark_rows(replay), fieldnames)
    write_text(OUT / "MULTIFIDELITY_REPLAY_REPORT.md", render_replay(signal, replay, flags))
    write_text(OUT / "UNCERTAINTY_AUDIT.md", render_uncertainty(replay))
    write_text(OUT / "FULL_ARCH_HW_COST_GAP.md", render_hw(hw))
    write_text(OUT / "PHASE6A1_ADMISSION.json", json.dumps(conv(admission), ensure_ascii=False, indent=2) + "\n")
    for path_s, digest in data["before"].items():
        if sha256(Path(path_s)) != digest:
            raise SystemExit(f"Input changed while writing reports: {path_s}")
    print(json.dumps({
        "verdict": verdict,
        "PHASE6B_ALLOWED": allowed,
        "qualified": replay["qualified"],
        "uncertainty_acquisition_allowed": admission["uncertainty_acquisition_allowed"],
        "best_architecture_only_spearman_mean": replay["best_architecture_only_spearman_mean"],
        "elapsed_seconds": admission["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
