"""Campaign reducer regressions; all invented scores exist only in test fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sonar_search_finalize", ROOT / "tools/sonar_search_finalize/run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config():
    return json.loads((ROOT / "results/sonar_joint_v1/freeze.json").read_text())["protocol"]


def evaluations(aid="new_architecture_outside_B1_C_D", freeze="frozen"):
    return {aid: {str(seed): dict(architecture_id=aid, seed=seed, freeze_sha256=freeze,
            split="valid", test_evaluated=False, fp32={"mAP50_95": .5},
            checkpoint_sha256=f"checkpoint_{seed}", calibration_sha256="calibration",
            profiles={q: dict(mAP50_95=.499 if q == "W8A8" else .5,
                            profile_sha256=f"{q}_{seed}") for q in m.BITS}) for seed in (42, 43, 44)}}


class FakeDB:
    hashes = {"numeric_contract_hash": "numeric"}

    def __init__(self):
        self.rows = []

    def estimate_cost(self, aid, profile, policy):
        row = dict(architecture_id=aid, profile=profile, mapping_policy=policy,
                   status="LOOKED_UP", hardware_status="MEASURED", measurement_valid=True,
                   Latency=10 if profile == "W8A8" else 20, DSP=1, LUT=2, BRAM=3,
                   Accuracy=.99, accuracy_key={"legacy": True}, q_status="Q_NOT_ADMITTED",
                   eligible=False, quant_contract_hash="contract", timing_eligible=True)
        self.rows.append(row)
        return row


def test_nonlegacy_architecture_uses_campaign_q_without_mutating_cost_rows():
    db = FakeDB()
    result = m.build_points(evaluations(), db, config(), "frozen")
    assert len(result["points"]) == 18
    assert result["accuracy_latency"] and result["accuracy_resources"]
    assert sorted({r["Accuracy"] for r in result["points"]}) == pytest.approx([.499, .5])
    assert all(r["eligible"] and r["q_status"] == "CAMPAIGN_Q_ADMITTED" for r in result["points"])
    assert all(not r["eligible"] and r["Accuracy"] == .99 for r in db.rows)
    assert all("accuracy_key" not in r for r in result["points"])


def test_pareto_retains_accuracy_latency_tradeoff_and_drops_dominated_point():
    points = [dict(Accuracy=.5, latency=10), dict(Accuracy=.4, latency=5), dict(Accuracy=.4, latency=11)]
    before = copy.deepcopy(points)
    assert m.nondominated(points, ["latency"]) == points[:2]
    assert points == before


@pytest.mark.parametrize("key,value", [("split", "test"), ("test_evaluated", True),
                                      ("seed", 99), ("freeze_sha256", "wrong")])
def test_evaluation_identity_drift_rejected(key, value):
    ev = evaluations()
    next(iter(ev.values()))["42"][key] = value
    with pytest.raises(ValueError, match="identity"):
        m.build_points(ev, FakeDB(), config(), "frozen")


def test_missing_seed_and_nonfinite_score_rejected():
    ev = evaluations()
    next(iter(ev.values())).pop("44")
    with pytest.raises(ValueError, match="seed"):
        m.build_points(ev, FakeDB(), config(), "frozen")
    ev = evaluations()
    next(iter(ev.values()))["42"]["fp32"]["mAP50_95"] = float("nan")
    with pytest.raises(ValueError, match="mAP"):
        m.build_points(ev, FakeDB(), config(), "frozen")


def test_mixed_precision_still_requires_two_paired_improvements():
    gate = config()["admission"]
    assert not m.admitted([.5]*3, [.499]*3, [.499, .498, .503], "HEAD_W8A16", gate)
    assert not m.admitted([.5]*3, [.499]*3, [.45]*3, "W8A8", gate)


def test_missing_hardware_excluded_without_score_substitution():
    class Missing(FakeDB):
        def estimate_cost(self, *args):
            return {"hardware_status": "NEEDS_MEASUREMENT"}
    result = m.build_points(evaluations(), Missing(), config(), "frozen")
    assert not result["points"] and len(result["rejected"]) == 18


def profile(q):
    w, a = m.BITS[q]
    return dict(numeric_contract_hash="numeric", zero_point=0, rounding="nearest_even", scheme="symmetric", qat=False,
                bitwidth=dict(activation=8, weight=8, accumulator=32, requant_product=64),
                weight_granularity="per_tensor", activation_granularity="per_tensor",
                mixed_precision=dict(weight_bits_by_prefix={"head": w}, activation_bits_by_prefix={"head": a}),
                fp32_bypass=dict(software_diagnostic_only=False, neck_adds_quantized=True),
                layers=[dict(name="backbone.conv", input_bits=8, weight_bits=8, output_bits=8,
                             fp32_weight=False, fp32_activation=False),
                        dict(name="head.conv", input_bits=a, weight_bits=w, output_bits=a,
                             fp32_weight=False, fp32_activation=False)])


def test_mixed_profile_cannot_be_int8_or_float_substitute():
    p = profile("W8A8")
    with pytest.raises(ValueError, match="prefix|bitwidth"):
        m.validate_profile(p, "HEAD_W8A16", "numeric")
    p["fp32_bypass"]["software_diagnostic_only"] = True
    with pytest.raises(ValueError, match="bypass"):
        m.validate_profile(p, "W8A8", "numeric")


@pytest.mark.parametrize("field,value", [("weight_granularity", "per_channel"), ("activation_granularity", "per_channel"),
    ("bitwidth", dict(activation=8, weight=8, accumulator=16, requant_product=64))])
def test_profile_granularity_and_accumulator_must_match_hardware(field, value):
    p = profile("W8A8")
    p[field] = value
    with pytest.raises(ValueError, match="granularity/accumulator"):
        m.validate_profile(p, "W8A8", "numeric")


def write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


@pytest.fixture
def campaign(tmp_path):
    run = tmp_path / "campaign"
    run.mkdir()
    cfg = config()
    # The real immutable architecture IDs, but synthetic scores/checkpoint bytes.
    rows = json.loads((ROOT / "results/sonar_joint_v1/cost_snapshot.json").read_text())["rows"]
    aids = sorted({r["architecture_id"] for r in rows})
    costs = [dict(architecture_id=aid, precision=q, Latency=10 if q == "W8A8" else 20, DSP=1, LUT=2, BRAM=3)
             for aid in aids for q in m.BITS]
    write(run, "cost_snapshot.json", {"rows": costs})
    cfg_bytes = json.dumps(cfg).encode()
    config_rel = "configs/search/sonar_joint_v1.yaml"
    with zipfile.ZipFile(run / "source_snapshot.zip", "w") as archive:
        archive.writestr(config_rel, cfg_bytes)
    write(run, "snapshot.json", {"sha256": m.sha(run / "source_snapshot.zip")})
    train = [dict(image=f"train/images/{i:03}.png", annotation=f"train/labels/{i:03}.txt",
                  image_sha256=str(i), annotation_sha256=str(i)) for i in range(64)]
    binding = dict(protocol=cfg, files={config_rel: __import__("hashlib").sha256(cfg_bytes).hexdigest()},
                   dataset={"train": train}, cost_snapshot_sha256=m.sha(run / "cost_snapshot.json"),
                   hashes=dict(config_hash=m.digest(cfg), data_hash=m.digest({"train": train})))
    write(run, "freeze.json", binding)
    freeze_hash = m.sha(run / "freeze.json")

    def train_receipt(aid, seed, budget):
        rel = f"training/{aid}/seed_{seed}/best_{budget}.pt"
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"synthetic-checkpoint-{aid}-{seed}-{budget}".encode())
        result = dict(architecture_id=aid, seed=seed, budget=budget, best_epoch=1,
                      mAP50_95=.5 - aids.index(aid)*.001, checkpoint=rel,
                      checkpoint_sha256=m.sha(p), test_evaluated=False)
        write(run, f"tasks/train_{aid}_{seed}_{budget}.json",
              dict(freeze_sha256=freeze_hash, result=result, outputs={rel: m.sha(p)}))
        return result

    selected, audit_ids = aids, []
    for stage, (budget, count) in enumerate(zip(cfg["budgets"], cfg["populations"])):
        stage_rows = [train_receipt(aid, 42, budget) for aid in selected]
        write(run, f"stage_{budget}.json", stage_rows)
        if stage < 2:
            keep = cfg["populations"][stage+1]
            for macro in ("P3_P4_P5", "P2_P3_P4", "P2_P3_P4_P5"):
                group = [aid for aid in selected[keep:] if aid.split("_s")[0] == macro]
                if group:
                    audit_ids.append(min(group, key=lambda aid: m.digest([42, stage, aid])))
            selected = selected[:keep]
    write(run, "rejection_audit.json", [train_receipt(aid, 42, 80) for aid in audit_ids])
    shortlist = selected[:12]
    write(run, "shortlist.json", {"architectures": shortlist})
    cal = dict(split="train", test_evaluated=False, valid_used_for_selection=False, augmentation=False, entries=train)
    cal_rel = "calibration/calibration_manifest.json"
    write(run, cal_rel, cal)
    for aid in shortlist:
        for seed in cfg["confirmation_seeds"]:
            tr = train_receipt(aid, seed, 80)
            result = evaluations(aid, freeze_hash)[aid][str(seed)]
            result.update(checkpoint_sha256=tr["checkpoint_sha256"], calibration_sha256=m.sha(run / cal_rel))
            outputs = {tr["checkpoint"]: tr["checkpoint_sha256"], cal_rel: m.sha(run / cal_rel)}
            for q in m.BITS:
                rel = f"quantization/{aid}/seed_{seed}/{q}_profile.json"
                write(run, rel, profile(q))
                outputs[rel] = m.sha(run / rel)
                result["profiles"][q]["profile_sha256"] = outputs[rel]
            rel = f"quantization/{aid}/seed_{seed}/result.json"
            write(run, rel, result)
            outputs[rel] = m.sha(run / rel)
            write(run, f"tasks/quant_{aid}_{seed}_80.json", dict(freeze_sha256=freeze_hash, result=result, outputs=outputs))
    return run


def test_complete_receipt_chain_finalizes_new_shortlist(campaign):
    before = {p.relative_to(campaign).as_posix(): m.sha(p) for p in campaign.rglob("*") if p.is_file()}
    report = m.finalize(campaign, db_factory=FakeDB)
    assert report["status"] == "COMPLETE" and len(report["points"]) == 216
    assert report["accuracy_latency"] and report["test_evaluated"] is False
    assert report["full_detector_deployment"] is False
    assert before == {p.relative_to(campaign).as_posix(): m.sha(p) for p in campaign.rglob("*") if p.is_file()}


def test_incomplete_campaign_reports_pending_without_cost_query(campaign):
    (campaign / "stage_80.json").unlink()
    def should_not_query():
        pytest.fail("Hardware lookup should not run on incomplete evidence")
    report = m.finalize(campaign, db_factory=should_not_query)
    assert report["status"] == "PENDING_CAMPAIGN_EVIDENCE" and not report["points"]


def test_checkpoint_byte_tampering_is_detected(campaign):
    next((campaign / "training").rglob("best_40.pt")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        m.finalize(campaign, db_factory=FakeDB)


def test_shortlist_cannot_be_replaced_by_better_unselected_architecture(campaign):
    p = campaign / "shortlist.json"
    obj = m.read(p)
    obj["architectures"].reverse()
    write(campaign, "shortlist.json", obj)
    with pytest.raises(ValueError, match="Shortlist"):
        m.finalize(campaign, db_factory=FakeDB)


def test_missing_confirmation_does_not_publish_partial_front(campaign):
    next((campaign / "tasks").glob("quant_*_44_80.json")).unlink()
    report = m.finalize(campaign, db_factory=FakeDB)
    assert report["status"] == "PENDING_CAMPAIGN_EVIDENCE" and not report["points"]


def test_freeze_cannot_silently_change_admission_policy(campaign):
    binding = m.read(campaign / "freeze.json")
    binding["protocol"]["admission"]["absolute_drop"] = 1
    binding["hashes"]["config_hash"] = m.digest(binding["protocol"])
    write(campaign, "freeze.json", binding)
    with pytest.raises(ValueError, match="archived configuration"):
        m.finalize(campaign, db_factory=FakeDB)


@pytest.mark.parametrize("rel", ["../outside", "C:/outside", "/outside"])
def test_receipt_paths_cannot_escape_campaign(tmp_path, rel):
    with pytest.raises(ValueError, match="escapes"):
        m.bounded(tmp_path, rel)
