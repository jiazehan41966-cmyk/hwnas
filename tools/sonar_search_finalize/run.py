"""Receipt-bound postprocessing for sonar_joint_v1, outside its frozen source tree.

No training, data loading, checkpoint unpickling, or legacy B1/C/D admission.
The campaign's frozen accuracy gate defines Q(x) for its own confirmed shortlist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
BITS = {"W8A8": (8, 8), "HEAD_W8A16": (8, 16), "HEAD_W16A16": (16, 16)}
COST_PREFIXES = ("artifacts/fpga_h3/", "artifacts/sonar_joint_cost_v1/",
                 "reports/fpga_h3_cost_db/", "artifacts/fpga_h2/", "reports/fpga_h2_contract/")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bounded(root, rel):
    part = PurePosixPath(str(rel).replace("\\", "/"))
    if part.is_absolute() or ".." in part.parts or ":" in str(part):
        raise ValueError("Artifact path escapes campaign")
    path = (root / Path(*part.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact symlink escapes campaign")
    return path


def score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid mAP50:95")
    return value


class Evidence:
    def __init__(self, campaign):
        self.root = Path(campaign).resolve()
        self.hashes = {}

    def record(self, rel):
        path = bounded(self.root, rel)
        value = sha(path)
        old = self.hashes.get(str(rel))
        if old is not None and old != value:
            raise ValueError("Artifact changed during finalization: " + str(rel))
        self.hashes[str(rel)] = value
        return path, value

    def json(self, rel):
        path, _ = self.record(rel)
        return read(path)

    def receipt(self, mode, aid, seed, budget, freeze_hash):
        rel = f"tasks/{mode}_{aid}_{seed}_{budget}.json"
        receipt = self.json(rel)
        if receipt.get("freeze_sha256") != freeze_hash or not receipt.get("outputs"):
            raise ValueError("Receipt freeze/output binding mismatch")
        for output, expected in receipt["outputs"].items():
            if self.record(output)[1] != expected:
                raise ValueError("Receipt artifact hash mismatch: " + output)
        result = receipt["result"]
        if result.get("architecture_id") != aid or result.get("seed") != seed or result.get("test_evaluated") is not False:
            raise ValueError("Receipt candidate/seed/test binding mismatch")
        if mode == "train":
            expected_path = f"training/{aid}/seed_{seed}/best_{budget}.pt"
            if result.get("budget") != budget or result.get("checkpoint") != expected_path:
                raise ValueError("Training checkpoint budget/path mismatch")
            if receipt["outputs"].get(expected_path) != result.get("checkpoint_sha256"):
                raise ValueError("Training checkpoint hash mismatch")
            score(result["mAP50_95"])
        else:
            result_path = f"quantization/{aid}/seed_{seed}/result.json"
            if result_path not in receipt["outputs"] or self.json(result_path) != result:
                raise ValueError("Quantization result not bound to receipt")
        return result, receipt["outputs"]


def validate_freeze(ev):
    binding = ev.json("freeze.json")
    snapshot = ev.json("snapshot.json")
    if ev.record("source_snapshot.zip")[1] != snapshot["sha256"]:
        raise ValueError("Frozen source archive changed")
    if ev.record("cost_snapshot.json")[1] != binding["cost_snapshot_sha256"]:
        raise ValueError("Frozen cost snapshot changed")
    with zipfile.ZipFile(ev.root / "source_snapshot.zip") as archive:
        if set(archive.namelist()) != set(binding["files"]):
            raise ValueError("Frozen source archive membership mismatch")
        for rel, expected in binding["files"].items():
            if hashlib.sha256(archive.read(rel)).hexdigest() != expected:
                raise ValueError("Frozen archive file mismatch: " + rel)
        archived_cfg = yaml.safe_load(archive.read("configs/search/sonar_joint_v1.yaml"))
    cfg = binding["protocol"]
    if archived_cfg != cfg or binding.get("hashes", {}).get("config_hash") != digest(cfg):
        raise ValueError("Frozen protocol disagrees with archived configuration")
    if binding.get("hashes", {}).get("data_hash") != digest(binding["dataset"]):
        raise ValueError("Frozen dataset manifest identity changed")
    if (cfg.get("version") != "sonar_joint_v1" or cfg.get("test_policy") != "sealed"
            or cfg.get("evaluation_split") != "valid" or cfg.get("confirmation_seeds") != [42, 43, 44]
            or cfg.get("budgets") != [40, 60, 80] or cfg.get("populations") != [144, 108, 72]
            or cfg.get("shortlist") != 12 or cfg.get("quant_profiles") != list(BITS)):
        raise ValueError("Unsupported campaign protocol")
    frozen_cost = ev.json("cost_snapshot.json")
    aids = {r["architecture_id"] for r in frozen_cost["rows"]}
    if len(aids) != 144 or len(frozen_cost["rows"]) != 432:
        raise ValueError("Invalid frozen candidate coverage")
    return binding, cfg, aids


def ranking(rows):
    if len({r["architecture_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate architecture")
    return sorted(rows, key=lambda r: (-score(r["mAP50_95"]), r["architecture_id"]))


def validate_selection(ev, cfg, aids, freeze_hash):
    selected = set(aids)
    audit_ids = []
    for stage, (budget, count) in enumerate(zip(cfg["budgets"], cfg["populations"])):
        rows = ev.json(f"stage_{budget}.json")
        if len(rows) != count or {r["architecture_id"] for r in rows} != selected or rows != ranking(rows):
            raise ValueError("Stage population/ranking mismatch")
        for row in rows:
            actual, _ = ev.receipt("train", row["architecture_id"], 42, budget, freeze_hash)
            if row != actual:
                raise ValueError("Stage score disagrees with training receipt")
        if stage < 2:
            keep = cfg["populations"][stage + 1]
            for macro in ("P3_P4_P5", "P2_P3_P4", "P2_P3_P4_P5"):
                group = [r["architecture_id"] for r in rows[keep:] if r["architecture_id"].split("_s")[0] == macro]
                if group:
                    audit_ids.append(min(group, key=lambda aid: digest([42, stage, aid])))
            selected = {r["architecture_id"] for r in rows[:keep]}
    audit = ev.json("rejection_audit.json")
    if [r["architecture_id"] for r in audit] != audit_ids:
        raise ValueError("Rejected-architecture audit membership mismatch")
    for row in audit:
        actual, _ = ev.receipt("train", row["architecture_id"], 42, 80, freeze_hash)
        if row != actual or score(row["mAP50_95"]) > score(rows[4]["mAP50_95"]):
            raise ValueError("Rejected-architecture audit failed")
    shortlist = ev.json("shortlist.json")["architectures"]
    if shortlist != [r["architecture_id"] for r in rows[:cfg["shortlist"]]]:
        raise ValueError("Shortlist differs from frozen promotion rule")
    return shortlist


def validate_profile(profile, name, numeric_hash):
    if (profile.get("numeric_contract_hash") != numeric_hash or profile.get("zero_point") != 0
            or profile.get("rounding") != "nearest_even" or profile.get("scheme") != "symmetric"
            or profile.get("qat") is not False):
        raise ValueError("Quantization numeric contract mismatch")
    if (profile.get("bitwidth") != dict(activation=8, weight=8, accumulator=32, requant_product=64)
            or profile.get("weight_granularity") != "per_tensor"
            or profile.get("activation_granularity") != "per_tensor"):
        raise ValueError("Quantization granularity/accumulator contract mismatch")
    bypass = profile.get("fp32_bypass", {})
    if bypass.get("software_diagnostic_only") is not False or bypass.get("neck_adds_quantized") is not True:
        raise ValueError("Diagnostic FP32 bypass is not a deployable quantization profile")
    layers = profile.get("layers", [])
    if not layers or not any(x["name"].startswith("head.") for x in layers):
        raise ValueError("Quantization profile has no head layers")
    wb, ab = BITS[name]
    mixed = profile.get("mixed_precision", {})
    if (mixed.get("weight_bits_by_prefix") != {"head": wb}
            or mixed.get("activation_bits_by_prefix") != {"head": ab}):
        raise ValueError("Quantization head prefix contract mismatch")
    for layer in layers:
        expected = (ab, wb, ab) if layer["name"].startswith("head.") else (8, 8, 8)
        if tuple(layer.get(k) for k in ("input_bits", "weight_bits", "output_bits")) != expected:
            raise ValueError("Quantization layer bitwidth mismatch")
        if layer.get("fp32_weight") is not False or layer.get("fp32_activation") is not False:
            raise ValueError("FP32 layer bypass")


def load_evaluations(ev, binding, cfg, shortlist, freeze_hash, numeric_hash):
    cal = ev.json("calibration/calibration_manifest.json")
    if (cal.get("split") != "train" or cal.get("test_evaluated") is not False
            or cal.get("valid_used_for_selection") is not False or cal.get("augmentation") is not False
            or len(cal.get("entries", [])) != cfg["calibration"]["images"]):
        raise ValueError("Calibration is not the frozen train-only subset")
    if cal["entries"] != binding["dataset"]["train"][:cfg["calibration"]["images"]]:
        raise ValueError("Calibration identities differ from frozen first train images")
    cal_hash = ev.hashes["calibration/calibration_manifest.json"]
    evaluations = {}
    for aid in shortlist:
        evaluations[aid] = {}
        for seed in cfg["confirmation_seeds"]:
            train, _ = ev.receipt("train", aid, seed, 80, freeze_hash)
            result, outputs = ev.receipt("quant", aid, seed, 80, freeze_hash)
            if (result.get("freeze_sha256") != freeze_hash or result.get("split") != "valid"
                    or result.get("checkpoint_sha256") != train["checkpoint_sha256"]
                    or result.get("calibration_sha256") != cal_hash
                    or outputs.get(train["checkpoint"]) != train["checkpoint_sha256"]
                    or outputs.get("calibration/calibration_manifest.json") != cal_hash):
                raise ValueError("Quantized evaluation checkpoint/calibration/split mismatch")
            score(result["fp32"]["mAP50_95"])
            if set(result["profiles"]) != set(cfg["quant_profiles"]):
                raise ValueError("Quantized profile set mismatch")
            for q in cfg["quant_profiles"]:
                rel = f"quantization/{aid}/seed_{seed}/{q}_profile.json"
                if outputs.get(rel) != result["profiles"][q]["profile_sha256"]:
                    raise ValueError("Quantization profile is not receipt-bound")
                validate_profile(ev.json(rel), q, numeric_hash)
                score(result["profiles"][q]["mAP50_95"])
            evaluations[aid][str(seed)] = result
    return evaluations


def admitted(fp32, uniform, values, profile, gate):
    if not (len(fp32) == len(uniform) == len(values) == 3):
        raise ValueError("Three paired seeds required")
    for value in fp32 + uniform + values:
        score(value)
    fp, mean = sum(fp32) / 3, sum(values) / 3
    drop = fp - mean
    accepted = drop <= gate["absolute_drop"] and drop <= gate["relative_drop"] * fp
    if profile != "W8A8":
        accepted = accepted and mean > sum(uniform) / 3 and sum(a > b for a, b in zip(values, uniform)) >= gate["improved_seeds"]
    return bool(accepted)


def nondominated(points, minimize):
    """Pure Pareto extraction over already campaign-admitted rows; never mutates them."""
    def dominates(a, b):
        return (a["Accuracy"] >= b["Accuracy"] and all(a[k] <= b[k] for k in minimize)
                and (a["Accuracy"] > b["Accuracy"] or any(a[k] < b[k] for k in minimize)))
    return [p for p in points if not any(dominates(other, p) for other in points)]


def build_points(evals, db, cfg, freeze_hash):
    points, rejected = [], []
    for aid, runs in sorted(evals.items()):
        if sorted(runs) != [str(s) for s in cfg["confirmation_seeds"]]:
            raise ValueError("Campaign seed set mismatch")
        rr = [runs[str(seed)] for seed in cfg["confirmation_seeds"]]
        for seed, row in zip(cfg["confirmation_seeds"], rr):
            if (row.get("architecture_id") != aid or row.get("seed") != seed or row.get("split") != "valid"
                    or row.get("test_evaluated") is not False or row.get("freeze_sha256") != freeze_hash):
                raise ValueError("Campaign evaluation identity mismatch")
        fp = [r["fp32"]["mAP50_95"] for r in rr]
        uniform = [r["profiles"]["W8A8"]["mAP50_95"] for r in rr]
        for q in cfg["quant_profiles"]:
            values = [r["profiles"][q]["mAP50_95"] for r in rr]
            if not admitted(fp, uniform, values, q, cfg["admission"]):
                rejected.append(dict(architecture_id=aid, profile=q, reason="campaign_accuracy_gate"))
                continue
            for policy in cfg["mapping_policies"]:
                cost = db.estimate_cost(aid, q, policy)
                if (cost.get("status") != "LOOKED_UP" or cost.get("hardware_status") != "MEASURED"
                        or cost.get("measurement_valid") is not True
                        or cost.get("architecture_id") != aid or cost.get("profile") != q):
                    rejected.append(dict(architecture_id=aid, profile=q, mapping=policy, reason="hardware_not_verified"))
                    continue
                numeric = {"serial_op_estimate_us": cost.get("serial_op_estimate_us", cost.get("Latency")),
                           **{key: cost.get(key) for key in ("LUT", "DSP", "BRAM")}}
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in numeric.values()):
                    raise ValueError("Invalid hardware cost")
                # Fresh row: no old accuracy key, no old Q gate, no FP32 fallback.
                provenance = dict(freeze_sha256=freeze_hash, architecture_id=aid,
                                  seeds=cfg["confirmation_seeds"], split="valid", profile=q,
                                  checkpoint_sha256=[r["checkpoint_sha256"] for r in rr],
                                  calibration_sha256=[r["calibration_sha256"] for r in rr],
                                  quant_profile_sha256=[r["profiles"][q]["profile_sha256"] for r in rr])
                mean = sum(values) / 3
                points.append(dict(**numeric, **provenance, mapping_policy=policy,
                                   Accuracy=mean, std_mAP50_95=(sum((v-mean)**2 for v in values)/2)**.5,
                                   accuracy_status="MEASURED", hardware_status="MEASURED",
                                   q_status="CAMPAIGN_Q_ADMITTED", eligible=True,
                                   accuracy_key_sha256=digest(provenance),
                                   architecture_hash=cost.get("architecture_hash"),
                                   quant_contract_hash=cost.get("quant_contract_hash"),
                                   mapping_hash=cost.get("mapping_hash"), measurement_ids=cost.get("measurement_ids"),
                                   timing_eligible=cost.get("timing_eligible"),
                                   measurement_scope="OP_LEVEL_HLS", shell_reservation="UNKNOWN",
                                   board_measured_latency=False, deployment="NOT_CLAIMED", test_evaluated=False))
    return dict(points=points, rejected=rejected,
                accuracy_latency=nondominated(points, ["serial_op_estimate_us"]),
                accuracy_resources=nondominated(points, ["DSP", "LUT", "BRAM"]))


def finalize(campaign, root=ROOT, db_factory=None):
    ev = Evidence(campaign)
    binding, cfg, aids = validate_freeze(ev)
    freeze_hash = ev.hashes["freeze.json"]
    required = ["stage_40.json", "stage_60.json", "stage_80.json", "shortlist.json",
                "rejection_audit.json", "calibration/calibration_manifest.json"]
    absent = [rel for rel in required if not (ev.root / rel).is_file()]
    base = dict(created_at=datetime.now(timezone.utc).isoformat(), campaign=str(ev.root),
                freeze_sha256=freeze_hash, test_evaluated=False, deployment="NOT_CLAIMED",
                scope="campaign_shortlist_quantized_valid_accuracy_vs_operator_HLS_cost",
                aggregation_policy="sonar_joint_v1_receipt_reducer_v1", input_sha256=ev.hashes)
    if absent:
        return dict(base, status="PENDING_CAMPAIGN_EVIDENCE", missing=absent, points=[],
                    note="Missing local evidence is not a claim that the remote campaign is idle.")
    shortlist = validate_selection(ev, cfg, aids, freeze_hash)
    missing_receipts = [f"tasks/{mode}_{aid}_{seed}_80.json" for aid in shortlist
                        for seed in cfg["confirmation_seeds"] for mode in ("train", "quant")
                        if not (ev.root / f"tasks/{mode}_{aid}_{seed}_80.json").is_file()]
    if missing_receipts:
        return dict(base, status="PENDING_CAMPAIGN_EVIDENCE", missing=missing_receipts, points=[])
    # Current reducer source may differ; the consumed numeric/cost payload may not.
    for rel, expected in binding["files"].items():
        if rel.startswith(COST_PREFIXES) and sha(bounded(root, rel)) != expected:
            raise ValueError("Cost/contract payload drifted from campaign freeze: " + rel)
    if db_factory is None:
        sys.path.insert(0, str(root / "tools/fpga_h4"))
        from cost_api import CostDatabase
        db_factory = CostDatabase
    db = db_factory()
    evaluations = load_evaluations(ev, binding, cfg, shortlist, freeze_hash, db.hashes["numeric_contract_hash"])
    report = build_points(evaluations, db, cfg, freeze_hash)
    frozen_rows = {(r["architecture_id"], r["precision"]): r for r in ev.json("cost_snapshot.json")["rows"]}
    for point in report["points"]:
        if point["mapping_policy"] == "reference_min_latency":
            original = frozen_rows[(point["architecture_id"], point["profile"])]
            for key, old in (("serial_op_estimate_us", "Latency"), ("DSP", "DSP"), ("LUT", "LUT"), ("BRAM", "BRAM")):
                if not math.isclose(point[key], original[old], rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError("Frozen cost snapshot value changed")
    base["consumer_source_sha256"] = {p.relative_to(root).as_posix(): sha(p) for p in (root / "tools/fpga_h4").glob("*.py")}
    # Detect edits during evidence collection without rewriting any campaign artifact.
    for rel, expected in list(ev.hashes.items()):
        if sha(bounded(ev.root, rel)) != expected:
            raise ValueError("Evidence changed during finalization")
    return dict(base, **report, status="COMPLETE" if report["points"] else "COMPLETE_NO_ADMITTED_POINTS",
                shortlist=shortlist, q_scope="this_campaign_confirmed_shortlist_not_legacy_B1_C_D",
                global_optimum_claim=False, full_detector_deployment=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.campaign.resolve()):
        raise ValueError("Output must be a new directory outside the frozen campaign")
    report = finalize(args.campaign)
    report["reducer_sha256"] = sha(Path(__file__))
    report["invocation"] = sys.argv
    output.mkdir(parents=True, exist_ok=False)
    (output / "finalization.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "freeze_sha256", "test_evaluated", "deployment")}))


if __name__ == "__main__":
    main()
