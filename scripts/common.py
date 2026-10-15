import argparse
import json
from pathlib import Path
import yaml

from experiment import ROOT, load_config
from space import Genotype, decode


def parser(description, *, candidate=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=ROOT / "configs/search.yaml")
    p.add_argument("--output", type=Path, required=True)
    if candidate:
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--genotype", type=Path)
        group.add_argument("--index", type=int, help="Registered encoding 0..71; output always binds its full hash")
    return p


def genotype(args, config):
    return (Genotype.from_json(args.genotype.read_text(encoding="utf-8")) if args.genotype
            else decode(args.index,config["dataset"]["num_classes"]))


def baseline_config(config, *, config_path):
    source = Path(config["baseline"])
    source = source if source.is_absolute() else ROOT / source
    baseline = yaml.safe_load(source.read_text(encoding="utf-8"))
    data_root = config["dataset"].get("root")
    if not data_root:
        raise ValueError("Set dataset.root in the committed YAML to your immutable YOLO dataset root")
    data_root = Path(data_root).expanduser()
    if not data_root.is_absolute():
        data_root = Path(config_path).resolve().parent / data_root
    baseline["dataset"]["data_dir"] = str(data_root.resolve())
    baseline["dataset"]["num_classes"] = config["dataset"]["num_classes"]
    return baseline


def print_result(result):
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
