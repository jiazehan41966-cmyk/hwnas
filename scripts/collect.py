"""Collect compatible result bundles into one auditable run for search/tables."""
from pathlib import Path
from experiment import verify_run,ResultStore,save_json,sha_file
from .common import parser,print_result


def main():
    p=parser(__doc__);p.add_argument("--input",type=Path,action="append",required=True);args=p.parse_args()
    runs=[verify_run(path) for path in args.input]
    first=runs[0]
    for run in runs:
        if any(run[k]!=first[k] for k in ("config_hash","source_manifest_hash","git")):
            raise ValueError("Cannot pool different code/config/provenance in a search run")
    records=[]
    for path in args.input:records.extend(ResultStore(path).read())
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError("Collect requires a new output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    sources=[dict(path=str(p.resolve()),run_sha256=sha_file(p/"run.json")) for p in args.input]
    save_json(args.output/"run.json",dict(first,collected_from=sources))
    store=ResultStore(args.output)
    for row in records:store.write(row)
    print_result(dict(records=len(records),output=str(args.output.resolve())))


if __name__ == "__main__":main()
