"""Runtime-independent 72 -> 54 -> 36 -> 12 campaign orchestration.

An explicitly supplied evaluator owns all training/PTQ execution. This module
only validates and stores returned records, promotes measured candidates, runs
the original rejection audit, and applies paired three-seed confirmation.

Evaluator contract: evaluator(genotype,budget,seed,phase) returns one candidate
record for every configured precision x mapping pair. Phase is ``search``,
``rejection_audit`` or ``confirmation``. Confirmation records additionally retain
``quantization_result`` and ``training_result`` JSON artifacts and expose
``accuracy.fp32_mAP50_95``. These scores must match the retained receipts.
No model, tensor runtime, dataset loader or hardware execution is imported.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from experiment import ResultStore,validate_record,validate_config,verify_run,save_json,sha_file
from space import Genotype,enumerate_genotypes,canonical_hash
from .screening import select_stage,_point
from . import policy


def paired_admit(fp32,uniform,values,profile,gate):
    """Exact legacy sonar_search.run.admit mean-drop and paired-seed rule."""
    if len(fp32)!=3 or len(uniform)!=3 or len(values)!=3:
        raise ValueError("Three paired seeds required")
    for value in fp32+uniform+values: policy.score(value)
    f=sum(fp32)/3;q=sum(values)/3;drop=f-q
    ok=drop<=gate["absolute_drop"] and drop<=gate["relative_drop"]*f
    if profile!="W8A8":
        ok=ok and q>sum(uniform)/3 and sum(a>b for a,b in zip(values,uniform))>=gate["improved_seeds"]
    return bool(ok)


def _read_artifact(record,name):
    item=record.get("artifacts",{}).get(name)
    if not item or sha_file(item["path"])!=item["sha256"]:
        raise ValueError("Missing/drifted retained artifact: "+name)
    return json.loads(Path(item["path"]).read_text(encoding="utf-8"))


def _confirmed_accuracy(record):
    """Tie both scores to actual quantization/training and dataset receipts."""
    accuracy=record["accuracy"]
    quant=_read_artifact(record,"quantization_result")
    training=_read_artifact(record,"training_result")
    datasets=_read_artifact(record,"dataset")
    checkpoint=Path(record["artifacts"]["checkpoint"]["path"])
    sidecar=json.loads(checkpoint.with_suffix(".binding.json").read_text(encoding="utf-8"))
    binding=training["binding"]
    if (binding!=sidecar["binding"] or binding["genotype"]!=record["genotype"]
        or binding["arch_hash"]!=record["arch_hash"] or binding["seed"]!=record["seed"]
        or binding["budget"]!=80 or binding["test_evaluated"] is not False
        or binding["dataset_sha256"]!=canonical_hash(datasets)
        or training["checkpoint_sha256"]!=accuracy["checkpoint_sha256"]
        or training["checkpoint"]!=checkpoint.name or training["test_evaluated"] is not False):
        raise ValueError("Confirmation FP32 training/checkpoint/dataset identity mismatch")
    for field in ("genotype","arch_hash","q_hash","seed","budget"):
        if quant[field]!=record[field]:raise ValueError("Confirmation PTQ receipt identity mismatch: "+field)
    if (quant["profile"]!=record["quantization"]["profile"] or quant["split"]!="valid"
        or quant["test_evaluated"] is not False or quant["deployment_claim"] is not False
        or quant["software_integer_parity"]!="PASS_SOFTWARE_INTEGER_FAKE_PARITY"):
        raise ValueError("Confirmation requires actual valid-only integer-parity PTQ")
    for field in ("mAP50_95","fp32_mAP50_95","checkpoint_sha256","quant_artifact_sha256","calibration_sha256","dataset_sha256"):
        if quant[field]!=accuracy[field]:raise ValueError("Confirmation quantized accuracy binding mismatch: "+field)
    if training["mAP50_95"]!=accuracy["fp32_mAP50_95"]:
        raise ValueError("FP32 score does not belong to the selected trained checkpoint")
    policy.score(accuracy["mAP50_95"]);policy.score(accuracy["fp32_mAP50_95"])
    return accuracy


def confirmed_points(records,config,shortlist):
    """Validate three paired seeds and summarize admitted quantized costs/AP."""
    seeds=config["confirmation_seeds"]
    if seeds!=[42,43,44]:raise ValueError("Registered paired seeds required")
    expected=set(shortlist);grouped={};dataset_hashes=set();provenance=set()
    for record in records:
        validate_record(record)
        if (record["arch_hash"] not in expected or record["budget"]!=80
            or record["seed"] not in seeds or record["stage"]!="confirmation"
            or record["config_hash"]!=canonical_hash(config)):
            raise ValueError("Confirmation record phase/config/candidate/seed/budget mismatch")
        point=_point(record,config)
        accuracy=_confirmed_accuracy(record)
        dataset_hashes.add(accuracy["dataset_sha256"])
        provenance.add((record["git_commit"],record["source_manifest_hash"]))
        key=(record["arch_hash"],record["seed"],record["quantization"]["profile"],record["hardware_context"]["mapping_policy"])
        if key in grouped:raise ValueError("Duplicate confirmation precision/mapping record")
        grouped[key]=(record,point)
    if len(dataset_hashes)!=1 or len(provenance)!=1:
        raise ValueError("Confirmation sources/dataset bindings differ")
    expected_keys={(arch,seed,profile,mapping) for arch in expected for seed in seeds
                   for profile in config["quant_profiles"] for mapping in config["mapping_policies"]}
    if set(grouped)!=expected_keys:raise ValueError("Every shortlisted candidate requires three paired complete confirmations")
    points=[];rejected=[]
    for arch in sorted(expected):
        by_seed={}
        for seed in seeds:
            rows=[grouped[(arch,seed,q,m)][0] for q in config["quant_profiles"] for m in config["mapping_policies"]]
            checkpoints={r["accuracy"]["checkpoint_sha256"] for r in rows}
            fp={r["accuracy"]["fp32_mAP50_95"] for r in rows}
            calibration={r["accuracy"]["calibration_sha256"] for r in rows}
            if len(checkpoints)!=1 or len(fp)!=1 or len(calibration)!=1:
                raise ValueError("Within-seed precision alternatives must share FP32 checkpoint/calibration")
            for q in config["quant_profiles"]:
                identities={canonical_hash(r["accuracy"]) for r in rows if r["quantization"]["profile"]==q}
                if len(identities)!=1:raise ValueError("Mapping alternatives cannot change measured precision accuracy")
            by_seed[seed]=rows
        fp32=[by_seed[s][0]["accuracy"]["fp32_mAP50_95"] for s in seeds]
        uniform=[grouped[(arch,s,"W8A8",config["mapping_policies"][0])][0]["accuracy"]["mAP50_95"] for s in seeds]
        for profile in config["quant_profiles"]:
            values=[grouped[(arch,s,profile,config["mapping_policies"][0])][0]["accuracy"]["mAP50_95"] for s in seeds]
            if not paired_admit(fp32,uniform,values,profile,config["admission"]):
                rejected.append(dict(arch_hash=arch,profile=profile,reason="PAIRED_ACCURACY_GATE",values=values));continue
            mean=sum(values)/3
            for mapping in config["mapping_policies"]:
                rows=[grouped[(arch,s,profile,mapping)] for s in seeds]
                if any(p is None for _,p in rows):
                    rejected.append(dict(arch_hash=arch,profile=profile,mapping_policy=mapping,
                                         reason="HARDWARE_BUDGET_OR_SCOPE"))
                    continue
                point=deepcopy(rows[0][1])
                # Conservatively retain the worst of the three verified costs.
                for metric in policy.METRICS:point[metric]=max(p[metric] for _,p in rows)
                limits=config["hardware_budget"]["limits"]
                metric_names=dict(Latency="latency_us",LUT="lut",DSP="dsp",BRAM="bram18")
                point["cost_pressure"]=max(point[k]/limits[v] for k,v in metric_names.items())
                point.update(Accuracy=mean,mean_mAP50_95=mean,
                    std_mAP50_95=(sum((v-mean)**2 for v in values)/2)**.5,
                    n_seeds=3,confirmation_seeds=seeds,accuracy_admission="PAIRED_VALIDATION_PASS",
                    checkpoint_sha256=[r["accuracy"]["checkpoint_sha256"] for r,_ in rows],
                    quant_artifact_sha256=[r["accuracy"]["quant_artifact_sha256"] for r,_ in rows],
                    calibration_sha256=[r["accuracy"]["calibration_sha256"] for r,_ in rows],
                    confirmation_record_hashes=[r["record_hash"] for r,_ in rows])
                points.append(point)
    return dict(points=points,rejected=rejected,pareto=policy.fronts(points)[0] if points else [],
        scope="paired_valid_PTQ_accuracy_with_verified_costs",test_evaluated=False,
        independent_search_seed=config["search_seed"],n_independent_searches_represented=1,
        confirmation_seeds_are_not_independent_search_repetitions=True,deployment="NOT_CLAIMED")


def _collect(genotypes,budget,seed,phase,evaluator,store,config,run):
    records=[];expected_pairs={(q,m) for q in config["quant_profiles"] for m in config["mapping_policies"]}
    for genotype in genotypes:
        rows=evaluator(genotype,budget,seed,phase)
        if not isinstance(rows,list):raise TypeError("Evaluator must return a list of candidate records")
        seen=set();unknown=False
        for row in rows:
            validate_record(row)
            if (row["genotype"]!=genotype.to_dict() or row["arch_hash"]!=genotype.arch_hash
                or row["seed"]!=seed or row["budget"]!=budget or row["stage"]!=phase
                or row["git_commit"]!=run["git"]["commit"]
                or row["config_hash"]!=run["config_hash"]
                or row["source_manifest_hash"]!=run["source_manifest_hash"]):
                raise ValueError("Evaluator returned an unrelated candidate/config/source/phase")
            context=row.get("hardware_context")
            mapping=context["mapping_policy"] if context else None
            key=(row["quantization"]["profile"],mapping)
            if key in seen or key not in expected_pairs:raise ValueError("Duplicate/unregistered precision or mapping")
            seen.add(key)
            store.write(row);records.append(row)
            unknown=unknown or row.get("accuracy") is None or row["hardware"]["status"]!="READY"
        if seen!=expected_pairs or unknown:
            return records,dict(arch_hash=genotype.arch_hash,reason="INCOMPLETE_PRECISION_MAPPING_OR_MEASUREMENT",
                                expected_decisions=len(expected_pairs),received_decisions=len(seen))
    return records,None


def _report(output,name,value):
    save_json(Path(output)/name,value)
    return value


def run_staged(config,evaluator,output):
    """Execute the supplied evaluator only after explicit caller invocation.

    The output must already contain a create_run manifest. No implicit resume is
    supported: immutable candidate/phase receipts preserve every completed call.
    Missing measurements stop immediately before any next-stage evaluation.
    """
    validate_config(config)
    if config.get("admission")!=dict(absolute_drop=.003,relative_drop=.20,improved_seeds=2):
        raise ValueError("Preserve registered paired quantization admission")
    run=verify_run(output,current_source=True)
    if run["config"]!=config:raise ValueError("Campaign config differs from the run manifest")
    registry={g.arch_hash:g for g in enumerate_genotypes(config["dataset"]["num_classes"])}
    if len(registry)!=72:raise ValueError("Canonical 72-candidate registry required")
    output=Path(output)
    if (output/"campaign_result.json").exists():raise FileExistsError("Campaign already has a result; use a separate run")
    store=ResultStore(output);selected=list(registry);audit_hashes=[];stages=[];last=None
    def stopped(status,**kwargs):
        return _report(output,"campaign_result.json",dict(status=status,stages=stages,
            test_evaluated=False,deployment_claim=False,independent_search_seed=config["search_seed"],
            n_independent_searches_represented=1,confirmation_seeds_are_not_independent_search_repetitions=True,**kwargs))
    for stage,budget in enumerate(config["budgets"]):
        rows,missing=_collect([registry[h] for h in selected],budget,config["search_seed"],"search",evaluator,store,config,run)
        if missing:return stopped("NEEDS_MEASUREMENT",phase="search",budget=budget,missing=missing)
        last=select_stage(rows,config,stage=stage,previous=selected)
        _report(output,f"stage_{budget}.json",last);stages.append(last)
        if last["status"]!="READY":return stopped("NEEDS_MEASUREMENT",phase="promotion",budget=budget)
        target=config["populations"][stage+1] if stage<2 else config["shortlist"]
        chosen=last["selected_hashes"]
        if len(chosen)!=target or len(set(chosen))!=target or not set(chosen)<=set(selected):
            raise ValueError("Stage selection violated candidate count/membership")
        if stage<2:
            expected_audits=[]
            for pan in (False,True):
                pool=[registry[h] for h in selected if h not in chosen and registry[h].pan==pan]
                if pool:
                    expected_audits.append(min(pool,key=lambda g:canonical_hash([config["search_seed"],stage,g.architecture_id])).arch_hash)
            if set(last["rejection_audits"])!=set(expected_audits):raise ValueError("Rejection audit sampling changed")
            audit_hashes.extend(expected_audits)
        selected=chosen
    audit_rows,missing=_collect([registry[h] for h in sorted(set(audit_hashes))],80,config["search_seed"],"rejection_audit",evaluator,store,config,run)
    if missing:return stopped("NEEDS_MEASUREMENT",phase="rejection_audit",missing=missing)
    audit_points=[point for row in audit_rows if (point:=_point(row,config)) is not None]
    # Preserve the original final promoted-population top-five loss guard.
    final_points=last["points"]
    best=sorted((max(p["Accuracy"] for p in final_points if p["architecture_id"]==aid)
                 for aid in {p["architecture_id"] for p in final_points}),reverse=True)
    if not best:return stopped("NEEDS_MEASUREMENT",phase="rejection_threshold")
    threshold=best[min(4,len(best)-1)]
    violations=sorted({p["arch_hash"] for p in audit_points if p["Accuracy"]>threshold})
    _report(output,"rejection_audit.json",dict(candidate_hashes=sorted(set(audit_hashes)),
        points=audit_points,quantized_top5_threshold=threshold,violations=violations,test_evaluated=False))
    if violations:return stopped("REJECTION_AUDIT_FAILED",violations=violations)
    _report(output,"shortlist.json",dict(arch_hashes=selected,architectures=[registry[h].architecture_id for h in selected],
        scope="screened_promoted_population_not_global_optimum",test_evaluated=False))
    confirmations=[]
    for seed in config["confirmation_seeds"]:
        rows,missing=_collect([registry[h] for h in selected],80,seed,"confirmation",evaluator,store,config,run)
        if missing:return stopped("NEEDS_MEASUREMENT",phase="confirmation",seed=seed,missing=missing)
        confirmations.extend(rows)
    result=confirmed_points(confirmations,config,selected)
    _report(output,"joint_pareto.json",result)
    return stopped("COMPLETE_VALIDATION_CAMPAIGN" if result["points"] else "NO_ADMITTED_QUANTIZATION",shortlist=selected,confirmation=result)
