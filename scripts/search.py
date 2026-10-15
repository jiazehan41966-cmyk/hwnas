"""Plan the 1296 decisions or screen existing records without touching datasets."""
from pathlib import Path
from experiment import create_run, ResultStore, make_record, save_json, verify_run
from space import enumerate_genotypes
from quant import Q
from hw import HardwareContext, evaluate
from search import select_stage
from .common import parser, print_result


def plan(config_path, output):
    config, run = create_run(config_path, output, command="search --plan", require_clean=False)
    store = ResultStore(output)
    count = 0
    hardware = config["hardware"]
    for g in enumerate_genotypes(config["dataset"]["num_classes"]):
        for profile in config["quant_profiles"]:
            q = Q(g,profile)
            for mapping in config["mapping_policies"]:
                context = HardwareContext(device=hardware["device"],clock_mhz=hardware["clock_mhz"],
                    mapping_policy=mapping,backend=hardware["backend"],
                    tool_version=hardware.get("tool_version") or "UNMEASURED",scope=hardware["scope"])
                result = evaluate(g,q,context)
                store.write(make_record(g,q,result.to_dict(),run=run,seed=config["search_seed"],
                    budget=config["budgets"][0],stage="plan",context=context.to_dict()))
                count += 1
    report = dict(status="NEEDS_MEASUREMENT",architectures=72,decisions=count,
        eligible_decisions=0,training_started=False,dataset_opened=False,test_evaluated=False,
        deployment_claim=False,scope=hardware["scope"],git_commit=run["git"]["commit"],
        reproducible=run["reproducible"],
        formal_comparison_gate=config["migration"]["formal_comparison_gate"])
    save_json(Path(output)/"plan.json",report)
    return report


def main():
    p = parser(__doc__)
    p.add_argument("--records",type=Path,help="An existing run containing actual scored candidate records")
    p.add_argument("--execute",action="store_true",help="Explicitly execute the retained multi-fidelity campaign after cost preflight")
    p.add_argument("--device",default="cuda")
    p.add_argument("--stage",type=int,choices=(0,1,2),default=0)
    p.add_argument("--previous",type=Path,help="JSON list of previous full architecture hashes")
    args = p.parse_args()
    if args.execute:
        if args.records:raise ValueError("Choose execution or existing-record screening")
        config,run=create_run(args.config,args.output,command="search --execute")
        from .runtime import CandidateEvaluator
        from search.engine import run_staged
        evaluator=CandidateEvaluator(config,run,args.output,config_path=args.config,device=args.device)
        result=run_staged(config,evaluator,args.output)
    elif args.records:
        import json
        from experiment import load_config
        config = load_config(args.config)
        run = verify_run(args.records)
        if config != run["config"]:
            raise ValueError("Selection config must exactly match the candidate run")
        rows = [r for r in ResultStore(args.records).read() if r["stage"]=="search" and
                r["budget"]==config["budgets"][args.stage] and r["seed"]==config["search_seed"]]
        if args.stage and not args.previous:
            raise ValueError("Later stages require their actual previous selected hashes")
        previous = (json.loads(args.previous.read_text(encoding="utf-8")) if args.previous else
                    [g.arch_hash for g in enumerate_genotypes(config["dataset"]["num_classes"])])
        result = select_stage(rows,config,stage=args.stage,previous=previous)
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Selection requires a new output directory")
        save_json(output/"selection.json",dict(result,input_run=str(args.records.resolve())))
    else:
        result = plan(args.config,args.output)
    print_result(result)


if __name__ == "__main__":
    main()
