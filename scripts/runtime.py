"""Composition root: connect the pure search to explicit stage executors."""
import json
from pathlib import Path

from experiment import save_json,sha_file,make_record
from space import enumerate_genotypes,Genotype
from quant import Q
from hw import HardwareContext,evaluate
from .common import baseline_config


class CandidateEvaluator:
    def __init__(self,config,run,output,*,config_path,device):
        self.config,self.run,self.output,self.device=config,run,Path(output).resolve(),device
        h=config["hardware"]
        if h["scope"] in {"board","power"}:
            raise ValueError("The staged campaign needs implementation-level pretraining costs; checkpoint-specific board/power evidence is imported after candidate selection")
        index=h.get("evidence_index")
        if not index or not h.get("tool_version") or not h.get("evidence_index_sha256"):
            raise ValueError("Measured hardware index, its SHA256 and exact tool version are required before training")
        path=Path(index)
        path=path if path.is_absolute() else Path(config_path).resolve().parent/path
        if sha_file(path)!=h["evidence_index_sha256"]:raise ValueError("Hardware index content differs from committed config")
        self.evidence={}
        for row in json.loads(path.read_text(encoding="utf-8")):
            context=HardwareContext.from_dict(row["context"])
            if context.checkpoint_sha256 is not None or context.quant_artifact_sha256 is not None:
                raise ValueError("Shared pretraining costs cannot bind one seed/checkpoint's PTQ artifacts")
            evidence=Path(row["evidence"])
            evidence=evidence if evidence.is_absolute() else path.parent/evidence
            key=(row["arch_hash"],row["q_hash"],context.mapping_policy)
            if key in self.evidence:raise ValueError("Duplicate hardware index triple")
            if any(getattr(context,k)!=h[k] for k in ("device","clock_mhz","backend","tool_version","scope")):
                raise ValueError("Hardware index context differs from the experiment")
            self.evidence[key]=(context,evidence.resolve())
        required=set()
        self.cost_snapshot=[]
        # Verify complete cost coverage before opening any dataset or allocating a model.
        for g in enumerate_genotypes(config["dataset"]["num_classes"]):
            for profile in config["quant_profiles"]:
                q=Q(g,profile)
                for mapping in config["mapping_policies"]:
                    key=(g.arch_hash,q.q_hash,mapping);required.add(key)
                    if key not in self.evidence:raise ValueError("Missing measured cost for a registered joint decision")
                    context,evidence=self.evidence[key]
                    measured=evaluate(g,q,context,evidence_path=evidence)
                    if measured.status!="READY":raise ValueError("Unqualified hardware evidence: "+measured.reason)
                    self.cost_snapshot.append(measured.to_dict())
        if set(self.evidence)!=required:raise ValueError("Hardware index must cover exactly the current 1296 joint decisions")
        self.baseline=baseline_config(config,config_path=config_path)
        save_json(self.output/"hardware_preflight.json",dict(index_sha256=sha_file(path),rows=self.cost_snapshot,
            scope=h["scope"],deployment_claim=False,dataset_opened=False))

    def __call__(self,g,budget,seed,phase):
        from train.training import train_candidate,load_bound_checkpoint
        from quant.executor import verify_saved_artifacts
        from .quantize import execute as quantize
        folder=self.output/"training"/g.arch_hash/f"seed_{seed}"
        receipt_path=self.output/f"retrain_{g.arch_hash}_{seed}_{budget}.json"
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt["binding"]["genotype"]!=g.to_dict() or receipt["binding"]["seed"]!=seed
                    or receipt["binding"]["budget"]!=budget):raise ValueError("Training receipt identity differs")
            checkpoint=folder/receipt["checkpoint"]
            load_bound_checkpoint(checkpoint,receipt["binding"])
            if sha_file(checkpoint)!=receipt["checkpoint_sha256"]:raise ValueError("Training receipt checkpoint drift")
        else:
            receipt=train_candidate(g,self.baseline,folder,seed=seed,epochs=budget,device=self.device,
                resume=(folder/"latest.pt").exists())
            save_json(receipt_path,receipt)
        rows=[]
        for profile in self.config["quant_profiles"]:
            q=Q(g,profile)
            output=self.output/"quantization"/g.arch_hash/f"seed_{seed}"/f"budget_{budget}"/profile
            result_path=output/"quantization_result.json"
            if result_path.exists():
                result=json.loads(result_path.read_text(encoding="utf-8"))
                if (result["arch_hash"]!=g.arch_hash or result["q_hash"]!=q.q_hash
                        or result["checkpoint_sha256"]!=receipt["checkpoint_sha256"]):
                    raise ValueError("Existing PTQ receipt differs from candidate/checkpoint")
                verify_saved_artifacts(output/"ptq",expected_arch_hash=g.arch_hash,expected_q_hash=q.q_hash)
            else:
                result=quantize(self.output,output,profile=profile,device=self.device,receipt_path=receipt_path)
            accuracy={k:result[k] for k in ("mAP50_95","fp32_mAP50_95","split","test_evaluated", "checkpoint_sha256",
                "quant_artifact_sha256","calibration_sha256","dataset_sha256")}
            for mapping in self.config["mapping_policies"]:
                context,evidence=self.evidence[(g.arch_hash,q.q_hash,mapping)]
                for key in ("checkpoint_sha256","quant_artifact_sha256"):
                    if getattr(context,key) is not None and getattr(context,key)!=result[key]:
                        raise ValueError("Hardware evidence belongs to a different trained/quantized checkpoint")
                hw=evaluate(g,q,context,evidence_path=evidence)
                artifacts=dict(result["artifacts"],
                    quantization_result=dict(path=str(result_path),sha256=sha_file(result_path)),
                    hardware_evidence=dict(path=str(evidence),sha256=sha_file(evidence)))
                rows.append(make_record(g,q,hw.to_dict(),run=self.run,seed=seed,budget=budget,
                    accuracy=accuracy,artifacts=artifacts,stage=phase,context=context.to_dict()))
        return rows
