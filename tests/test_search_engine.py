"""SYNTHETIC orchestration fixtures only: no accuracy or hardware claims."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import yaml
from experiment import make_record,sha_file
from space import Genotype,enumerate_genotypes,canonical_hash
from quant import Q
from hw import HardwareContext,HardwareResult
import search.engine as engine

ROOT=Path(__file__).resolve().parents[1]


def cfg():
    return yaml.safe_load((ROOT/"configs/search.yaml").read_text(encoding="utf-8"))


def run_info(config):
    return dict(config=config,config_hash=canonical_hash(config),git=dict(commit="a"*40),source_manifest_hash="b"*64)


def point(row,config):
    # Explicit synthetic stand-in for independently tested artifact/HW verifier.
    return dict(architecture_id=row["architecture_id"],arch_hash=row["arch_hash"],
        profile=row["quantization"]["profile"],mapping_policy=row["hardware_context"]["mapping_policy"],
        Accuracy=row["accuracy"]["mAP50_95"],Latency=10.,LUT=20,DSP=3,BRAM=4,cost_pressure=.1,
        record_hash=row["record_hash"],dataset_sha256=row["accuracy"]["dataset_sha256"])


def records(g,budget,seed,phase,config,*,score=.898,unknown=False,artifacts=None,accuracy_extra=None):
    result=[];run=run_info(config)
    for profile in config["quant_profiles"]:
        q=Q(g,profile)
        for mapping in config["mapping_policies"]:
            context=HardwareContext("SYNTHETIC_DEVICE",200.,mapping,"hls_report","SYNTHETIC_TEST","operator_hls")
            hardware=HardwareResult("NEEDS_MEASUREMENT" if unknown else "READY",g.arch_hash,q.q_hash,
                context.context_hash,{} if unknown else dict(latency_us=10.,lut=20,dsp=3,bram18=4),
                "operator_hls",None if unknown else "c"*64).to_dict()
            accuracy=dict(mAP50_95=score if profile=="W8A8" else score+.001,
                fp32_mAP50_95=.9,split="valid",test_evaluated=False,
                checkpoint_sha256="d"*64,quant_artifact_sha256="e"*64,
                calibration_sha256="f"*64,dataset_sha256="1"*64)
            accuracy.update(accuracy_extra or {})
            result.append(make_record(g,q,hardware,run=run,seed=seed,budget=budget,
                stage=phase,accuracy=accuracy,artifacts=artifacts,context=context.to_dict()))
    return result


def fake_select(rows,config,*,stage,previous):
    registry={g.arch_hash:g for g in enumerate_genotypes()}
    targets=((6,48),(4,32),(2,10))[stage]
    selected=[]
    for pan,count in zip((False,True),targets):
        selected.extend([h for h in previous if registry[h].pan==pan][:count])
    audits=[]
    if stage<2:
        for pan in (False,True):
            rejected=[registry[h] for h in previous if h not in selected and registry[h].pan==pan]
            if rejected:audits.append(min(rejected,key=lambda g:canonical_hash([config["search_seed"],stage,g.architecture_id])).arch_hash)
    return dict(status="READY",selected_hashes=selected,rejection_audits=audits,
                points=[point(row,config) for row in rows])


class SyntheticStore:
    writes=[]
    def __init__(self,path):self.path=path
    def write(self,row):self.writes.append(row)


@pytest.fixture
def synthetic(monkeypatch):
    config=cfg();SyntheticStore.writes=[]
    monkeypatch.setattr(engine,"verify_run",lambda *a,**k:run_info(config))
    monkeypatch.setattr(engine,"ResultStore",SyntheticStore)
    monkeypatch.setattr(engine,"select_stage",fake_select)
    monkeypatch.setattr(engine,"_point",point)
    monkeypatch.setattr(engine,"_confirmed_accuracy",lambda row:row["accuracy"])
    return config


def test_synthetic_full_staged_driver_counts_audits_and_paired_confirmation(synthetic,tmp_path):
    calls=[]
    def evaluator(g,budget,seed,phase):
        calls.append((g.arch_hash,budget,seed,phase,g.pan))
        return records(g,budget,seed,phase,synthetic,score=.88 if phase=="rejection_audit" else .898)
    result=engine.run_staged(synthetic,evaluator,tmp_path)
    assert result["status"]=="COMPLETE_VALIDATION_CAMPAIGN"
    assert [sum(c[1]==b and c[3]=="search" for c in calls) for b in (40,60,80)]==[72,54,36]
    audit=[c for c in calls if c[3]=="rejection_audit"]
    assert len(audit)==4 and sum(c[4] for c in audit)==2 and all(c[1]==80 for c in audit)
    confirmation=[c for c in calls if c[3]=="confirmation"]
    assert len(confirmation)==36 and {c[2] for c in confirmation}=={42,43,44}
    assert len(result["shortlist"])==12
    assert result["n_independent_searches_represented"]==1
    assert result["confirmation_seeds_are_not_independent_search_repetitions"] is True
    assert len(SyntheticStore.writes)==3636
    assert (tmp_path/"joint_pareto.json").is_file()


@pytest.mark.parametrize("missing",["decision","measurement"])
def test_missing_measurement_or_incomplete_record_stops_before_next_candidate(synthetic,tmp_path,missing):
    calls=[]
    def evaluator(g,budget,seed,phase):
        calls.append((budget,phase))
        rows=records(g,budget,seed,phase,synthetic,unknown=missing=="measurement")
        return rows[:1] if missing=="decision" else rows
    result=engine.run_staged(synthetic,evaluator,tmp_path)
    assert result["status"]=="NEEDS_MEASUREMENT"
    assert calls==[(40,"search")]
    assert not (tmp_path/"stage_60.json").exists()


def test_rejection_audit_failure_prevents_confirmation(synthetic,tmp_path):
    phases=[]
    def evaluator(g,budget,seed,phase):
        phases.append(phase)
        return records(g,budget,seed,phase,synthetic,score=.999 if phase=="rejection_audit" else .898)
    # Keep all synthetic profile scores in [0,1].
    result=engine.run_staged(synthetic,evaluator,tmp_path)
    assert result["status"]=="REJECTION_AUDIT_FAILED"
    assert result["violations"] and "confirmation" not in phases


def test_paired_admission_requires_retention_and_two_paired_improvements():
    gate=dict(absolute_drop=.003,relative_drop=.2,improved_seeds=2)
    assert engine.paired_admit([.9]*3,[.898]*3,[.898]*3,"W8A8",gate)
    assert not engine.paired_admit([.9]*3,[.898]*3,[.896]*3,"W8A8",gate)
    assert not engine.paired_admit([.9]*3,[.898]*3,[.901,.898,.898],"HEAD_W8A16",gate)
    assert engine.paired_admit([.9]*3,[.898]*3,[.899,.899,.898],"HEAD_W8A16",gate)
    with pytest.raises(ValueError):engine.paired_admit([.9],[.8],[.9],"W8A8",gate)


def test_confirmation_missing_seed_and_unpaired_checkpoint_are_rejected(monkeypatch):
    config=cfg();g=Genotype(False,(3,3,3))
    monkeypatch.setattr(engine,"_point",point)
    monkeypatch.setattr(engine,"_confirmed_accuracy",lambda row:row["accuracy"])
    rows=[row for seed in (42,43,44) for row in records(g,80,seed,"confirmation",config)]
    assert engine.confirmed_points(rows,config,[g.arch_hash])["points"]
    with pytest.raises(ValueError):engine.confirmed_points(rows[:-1],config,[g.arch_hash])
    # Rebuild a valid immutable row with a different FP32 checkpoint for one precision.
    changed=deepcopy(rows)
    changed[-1]["accuracy"]["checkpoint_sha256"]="2"*64
    changed[-1]["record_hash"]=canonical_hash({k:v for k,v in changed[-1].items() if k!="record_hash"})
    with pytest.raises(ValueError):engine.confirmed_points(changed,config,[g.arch_hash])


def write_json(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")
    return dict(path=str(path),sha256=sha_file(path))


def test_confirmation_scores_bound_to_real_json_receipts(tmp_path):
    config=cfg();g=Genotype(False,(3,3,3));q=Q(g)
    checkpoint=tmp_path/"best_80.pt";checkpoint.write_bytes(b"SYNTHETIC checkpoint, never loaded")
    datasets=dict(train=dict(entries=[]),valid=dict(entries=[]))
    binding=dict(genotype=g.to_dict(),arch_hash=g.arch_hash,seed=42,budget=80,
                 dataset_sha256=canonical_hash(datasets),test_evaluated=False)
    write_json(checkpoint.with_suffix(".binding.json"),dict(binding=binding,checkpoint_sha256=sha_file(checkpoint)))
    artifacts=dict(checkpoint=dict(path=str(checkpoint),sha256=sha_file(checkpoint)),
        dataset=write_json(tmp_path/"dataset.json",datasets))
    training=dict(binding=binding,mAP50_95=.9,checkpoint=checkpoint.name,
                  checkpoint_sha256=sha_file(checkpoint),test_evaluated=False)
    artifacts["training_result"]=write_json(tmp_path/"training.json",training)
    extra=dict(checkpoint_sha256=sha_file(checkpoint),dataset_sha256=artifacts["dataset"]["sha256"])
    row=records(g,80,42,"confirmation",config,accuracy_extra=extra)[0]
    quant=dict(genotype=g.to_dict(),arch_hash=g.arch_hash,q_hash=q.q_hash,seed=42,budget=80,profile="W8A8",
        split="valid",test_evaluated=False,deployment_claim=False,
        software_integer_parity="PASS_SOFTWARE_INTEGER_FAKE_PARITY",**{k:v for k,v in row["accuracy"].items() if k not in {"split","test_evaluated"}})
    artifacts["quantization_result"]=write_json(tmp_path/"quantization.json",quant)
    row["artifacts"]=artifacts
    assert engine._confirmed_accuracy(row)["fp32_mAP50_95"]==.9
    training["mAP50_95"]=.91
    artifacts["training_result"]=write_json(tmp_path/"training.json",training)
    with pytest.raises(ValueError):engine._confirmed_accuracy(row)


def test_confirmation_filters_infeasible_mapping_without_losing_paired_scores(monkeypatch):
    config=cfg();g=Genotype(False,(3,3,3))
    excluded=config["mapping_policies"][-1]
    monkeypatch.setattr(engine,"_point",lambda row,c:None if row["hardware_context"]["mapping_policy"]==excluded else point(row,c))
    monkeypatch.setattr(engine,"_confirmed_accuracy",lambda row:row["accuracy"])
    rows=[row for seed in (42,43,44) for row in records(g,80,seed,"confirmation",config)]
    result=engine.confirmed_points(rows,config,[g.arch_hash])
    assert len(result["points"])==15
    assert len(result["rejected"])==3
    assert all(p["mapping_policy"]!=excluded for p in result["points"])
