"""Independently train a hashed candidate with the retained train/valid protocol."""
from experiment import create_run, verify_run, load_config, save_json
from .common import parser,genotype,baseline_config,print_result


def main():
    p=parser(__doc__,candidate=True)
    p.add_argument("--budget",type=int,choices=(40,60,80),default=80)
    p.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    p.add_argument("--device",default="cuda")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.resume:
        run=verify_run(args.output,current_source=True)
        cfg=load_config(args.config)
        if cfg != run["config"]: raise ValueError("Resume config changed")
    else:
        cfg,run=create_run(args.config,args.output,command="retrain")
    g=genotype(args,cfg)
    baseline=baseline_config(cfg,config_path=args.config)
    from train.training import train_candidate
    result=train_candidate(g,baseline,args.output/"training"/g.arch_hash/f"seed_{args.seed}",
        seed=args.seed,epochs=args.budget,device=args.device,resume=args.resume)
    save_json(args.output/f"retrain_{g.arch_hash}_{args.seed}_{args.budget}.json",result)
    print_result(result)


if __name__ == "__main__":main()
