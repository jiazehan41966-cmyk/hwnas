"""Execute source-bound train64 PTQ, integer parity and full-valid evaluation."""
import json
from pathlib import Path
from experiment import create_run, verify_run, save_json, sha_file
from space import Genotype
from quant import Q, PROFILES
from .common import parser,print_result,baseline_config


def execute(training_run, output, *, profile, device="cpu", receipt_path=None):
    from train.training import load_bound_checkpoint
    from models import build_model
    from train.data import make_loader,bind_dataset
    from train import DenseAnchorDetectionLoss,loss_config_for_model,evaluate
    from quant.executor import TrainCalibration,prepare_ptq,verify_fake_integer_parity,save_ptq_artifacts
    training_run=Path(training_run).resolve()
    run=verify_run(training_run,current_source=True)
    results=[Path(receipt_path)] if receipt_path is not None else sorted(training_run.glob("retrain_*.json"))
    if len(results)!=1:
        raise ValueError("Quantization requires one explicit retraining receipt per input run")
    receipt=json.loads(results[0].read_text(encoding="utf-8"))
    binding=receipt["binding"]
    g=Genotype.from_dict(binding["genotype"])
    checkpoint=training_run/"training"/g.arch_hash/f"seed_{binding['seed']}"/receipt["checkpoint"]
    if sha_file(checkpoint)!=receipt["checkpoint_sha256"]:raise ValueError("Retrained checkpoint changed")
    state=load_bound_checkpoint(checkpoint,binding)
    config=run["config"]
    # The dataset root is already an absolute committed config value, or relative to configs/.
    baseline=baseline_config(config,config_path=run["config_path"])
    data=Path(baseline["dataset"]["data_dir"])
    train=make_loader(baseline,data,"train",binding["seed"],0)
    valid=make_loader(baseline,data,"valid",binding["seed"],0)
    datasets=dict(train=bind_dataset(train),valid=bind_dataset(valid))
    from space import canonical_hash
    if canonical_hash(datasets)!=binding["dataset_sha256"]:
        raise ValueError("Dataset differs from the bound training inputs")
    if output.exists() and any(output.iterdir()):raise FileExistsError("PTQ requires a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/"run.json",dict(run,parent_training_run=str(training_run)))
    save_json(output/"dataset_binding.json",datasets)
    model=build_model(g,binding["seed"]).to(device).eval()
    model.load_state_dict(state["model_state_dict"],strict=True)
    calibration=TrainCalibration(data,datasets["train"]["entries"],num_images=64)
    plan=Q(g,profile)
    prepared=prepare_ptq(model,calibration,plan=plan,device=device)
    witness=verify_fake_integer_parity(prepared,next(calibration.batches(device)))
    if witness["status"]!="PASS_SOFTWARE_INTEGER_FAKE_PARITY" or witness["mismatches"]!=0:
        raise ValueError("Actual software integer parity must pass before evaluation")
    artifact_dir=Path(save_ptq_artifacts(output/"ptq",prepared,witness=witness))
    metrics,_=evaluate(prepared.fake_model,valid,DenseAnchorDetectionLoss(loss_config_for_model(model,baseline)).to(device),baseline,device)
    calibration_path=output/"calibration.json"
    save_json(calibration_path,calibration.manifest)
    result=dict(genotype=g.to_dict(),arch_hash=g.arch_hash,q_hash=plan.q_hash,profile=profile,
        seed=binding["seed"],budget=binding["budget"],split="valid",test_evaluated=False,
        mAP50_95=metrics["mAP50_95"],fp32_mAP50_95=receipt["mAP50_95"],metrics=metrics,
        checkpoint_sha256=sha_file(checkpoint),
        quant_artifact_sha256=sha_file(artifact_dir/"artifact_manifest.json"),
        calibration_sha256=sha_file(calibration_path),dataset_sha256=sha_file(output/"dataset_binding.json"),
        software_integer_parity=witness["status"],deployment_claim=False,
        artifacts=dict(training_result=dict(path=str(results[0].resolve()),sha256=sha_file(results[0])),
                       checkpoint=dict(path=str(checkpoint),sha256=sha_file(checkpoint)),
                       quant_artifact=dict(path=str(artifact_dir/"artifact_manifest.json"),sha256=sha_file(artifact_dir/"artifact_manifest.json")),
                       calibration=dict(path=str(calibration_path.resolve()),sha256=sha_file(calibration_path)),
                       dataset=dict(path=str((output/"dataset_binding.json").resolve()),sha256=sha_file(output/"dataset_binding.json"))))
    save_json(output/"quantization_result.json",result)
    return result


def main():
    p=parser(__doc__)
    p.add_argument("--training-run",type=Path,required=True)
    p.add_argument("--profile",choices=PROFILES,default="W8A8")
    p.add_argument("--receipt",type=Path,help="Explicit retrain receipt when a run has several budgets/candidates")
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    # Compare against the exact training configuration; no hidden PTQ overrides.
    from experiment import load_config
    if load_config(args.config)!=verify_run(args.training_run)["config"]:
        raise ValueError("PTQ config differs from retraining run")
    print_result(execute(args.training_run,args.output.resolve(),profile=args.profile,device=args.device,receipt_path=args.receipt))


if __name__ == "__main__":main()
