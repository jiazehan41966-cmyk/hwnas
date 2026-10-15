"""Export a bound integer deployment bundle and import measured evidence."""
import json
from pathlib import Path
import shutil

from experiment import verify_run, save_json, sha_file, ResultStore, make_record
from space import Genotype
from quant import Q
from hw import HardwareContext,evaluate
from .common import parser,print_result


def main():
    p=parser(__doc__)
    p.add_argument("--quantized-run",type=Path,required=True)
    p.add_argument("--context",type=Path,help="Exact HardwareContext JSON; default emits unknown screening cost")
    p.add_argument("--evidence",type=Path,help="Verified normalized report or HLS XML plus its sidecar")
    p.add_argument("--cache",type=Path)
    p.add_argument("--stage",choices=("search","confirmation","rejection_audit"),default="search")
    args=p.parse_args()
    root=args.quantized_run.resolve()
    run=verify_run(root,current_source=True)
    from experiment import load_config
    if load_config(args.config)!=run["config"]:raise ValueError("Deployment export config differs from source run")
    result=json.loads((root/"quantization_result.json").read_text(encoding="utf-8"))
    g=Genotype.from_dict(result["genotype"]);q=Q(g,result["profile"])
    if result["arch_hash"]!=g.arch_hash or result["q_hash"]!=q.q_hash:
        raise ValueError("PTQ result identity mismatch")
    for name,item in result["artifacts"].items():
        if sha_file(item["path"])!=item["sha256"]:raise ValueError("PTQ artifact changed: "+name)
    manifest_path=Path(result["artifacts"]["quant_artifact"]["path"])
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    # Validate all immutable PTQ leaves, not just the outer result JSON.
    from quant.executor import verify_saved_artifacts
    verify_saved_artifacts(manifest_path.parent,expected_arch_hash=g.arch_hash,expected_q_hash=q.q_hash)
    h=run["config"]["hardware"]
    if args.context:
        context=HardwareContext.from_dict(json.loads(args.context.read_text(encoding="utf-8")))
    else:
        context=HardwareContext(device=h["device"],clock_mhz=h["clock_mhz"],mapping_policy="reference_low",
            backend=h["backend"],tool_version=h.get("tool_version") or "UNMEASURED",scope=h["scope"])
    for key,expected in (("checkpoint_sha256",result["checkpoint_sha256"]),
                         ("quant_artifact_sha256",result["quant_artifact_sha256"])):
        declared=getattr(context,key)
        if declared is not None and declared!=expected:raise ValueError("Hardware context uses different PTQ/checkpoint identity")
    hardware=evaluate(g,q,context,cache_path=args.cache,evidence_path=args.evidence)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError("Bundle requires a new output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/"run.json",dict(run,parent_quantized_run=str(root)))
    shutil.copytree(manifest_path.parent,args.output/"integer")
    artifacts=dict(result["artifacts"])
    artifacts["quantization_result"]=dict(path=str(root/"quantization_result.json"),sha256=sha_file(root/"quantization_result.json"))
    if args.evidence:
        artifacts["hardware_evidence"]=dict(path=str(args.evidence.resolve()),sha256=sha_file(args.evidence))
    accuracy={k:result[k] for k in ("mAP50_95","split","test_evaluated","checkpoint_sha256",
        "quant_artifact_sha256","calibration_sha256","dataset_sha256")}
    accuracy["fp32_mAP50_95"] = result["fp32_mAP50_95"]
    record=make_record(g,q,hardware.to_dict(),run=run,seed=result["seed"],budget=result["budget"],
        accuracy=accuracy,artifacts=artifacts,stage=args.stage,context=context.to_dict())
    ResultStore(args.output).write(record)
    receipt=dict(status="INTEGER_BUNDLE_EXPORTED",arch_hash=g.arch_hash,q_hash=q.q_hash,
        hardware=hardware.to_dict(),deployment_claim=False,
        deployment_status="FULL_NETWORK_BOARD_RUNTIME_REQUIRES_SEPARATE_IMPLEMENTATION_AND_VALIDATION")
    save_json(args.output/"deployment_readiness.json",receipt)
    print_result(receipt)


if __name__ == "__main__":main()
