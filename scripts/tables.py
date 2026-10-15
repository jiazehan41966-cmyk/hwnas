"""Generate tables 1/2/3 from verified candidate records, retaining missing values."""
import csv
from pathlib import Path
from experiment import ResultStore,verify_run,save_json
from search import pareto_records
from .common import parser,print_result


def _csv(path,fields,rows):
    with path.open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def generate(run_path,output):
    run=verify_run(run_path)
    records=ResultStore(run_path).read()
    if output.exists() and any(output.iterdir()):raise FileExistsError("Tables require a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    accuracy,hardware,seen=[],[],set()
    for r in records:
        base={k:r[k] for k in ("arch_hash","architecture_id","seed","budget","q_hash","git_commit")}
        base.update(profile=r["quantization"]["profile"],stage=r["stage"])
        key=(r["arch_hash"],r["q_hash"],r["seed"],r["budget"],r["stage"])
        score=r["accuracy"]["mAP50_95"] if r["accuracy"] else None
        if key not in seen:
            seen.add(key)
            accuracy.append(dict(base,mAP50_95=score,split="valid" if score is not None else "",status="EVALUATED" if score is not None else "NOT_EVALUATED"))
        h=r["hardware"]
        hardware.append(dict(base,**h["metrics"],evidence_level=h["evidence_level"],status=h["status"],context_hash=h["context_hash"],
                             mapping_policy=(r["hardware_context"] or {}).get("mapping_policy","")))
    basefields=["arch_hash","architecture_id","profile","q_hash","seed","budget","stage","git_commit"]
    _csv(output/"table1_accuracy.csv",basefields+["split","mAP50_95","status"],accuracy)
    _csv(output/"table2_hardware.csv",basefields+["context_hash","mapping_policy","evidence_level","status","latency_us","lut","dsp","bram18","power_w"],hardware)
    groups={}
    for row in records:
        groups.setdefault((row["seed"],row["budget"],row["stage"]),[]).append(row)
    front=[]
    for (seed,budget,stage),rows in groups.items():
        front.extend(dict(p,seed=seed,budget=budget,stage=stage) for p in pareto_records(rows,run["config"]))
    _csv(output/"table3_pareto.csv",["arch_hash","architecture_id","profile","mapping_policy","seed","budget","stage","Accuracy","Latency","LUT","DSP","BRAM","evidence_level","record_hash"],front)
    report=dict(records=len(records),accuracy_rows=len(accuracy),hardware_rows=len(hardware),pareto_rows=len(front),
        input_run=str(Path(run_path).resolve()),input_git_commit=run["git"]["commit"],missing_values="empty cells, never zero-imputed",
        test_evaluated=False,deployment_claim=False)
    save_json(output/"tables_manifest.json",report)
    return report


def main():
    p=parser(__doc__);p.add_argument("--run",type=Path,required=True);args=p.parse_args()
    print_result(generate(args.run,args.output.resolve()))


if __name__ == "__main__":main()
