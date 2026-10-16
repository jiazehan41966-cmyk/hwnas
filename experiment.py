"""Configuration, reproducibility manifests and immutable per-candidate records."""
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import re
from pathlib import Path
import platform
import subprocess
import sys

from space import Genotype, canonical_hash, validate_sha256
from quant import QuantizationPlan

ROOT = Path(__file__).resolve().parent


def sha_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_config(path):
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True, throw_on_missing=True)
    validate_config(config)
    return config


def validate_config(config):
    if config.get("space_version") != "elastic_neck72_v1":
        raise ValueError("Only the registered 72-candidate detection mainline is supported")
    if config.get("evaluation_split") != "valid" or config.get("test_policy") != "sealed":
        raise ValueError("Architecture selection must use valid; test must remain sealed")
    if config.get("budgets") != [40, 60, 80] or config.get("populations") != [72, 54, 36]:
        raise ValueError("Migration preserves original independent-training budget/population protocol")
    if config.get("search_seed") not in (42,43,44) or type(config["search_seed"]) is not int:
        raise ValueError("Registered search seeds are 42, 43, 44")
    if config.get("confirmation_seeds") != [42,43,44]:
        raise ValueError("Paired confirmation seeds must remain registered")
    if config.get("training") != dict(optimizer="AdamW", lr=.001,weight_decay=.0001,
            scheduler="none",amp=False,workers=0,initialization="independent_from_scratch"):
        raise ValueError("This migration preserves the registered training settings")
    if config.get("checkpoint_selection") != "best_valid_mAP50_95_earliest_tie":
        raise ValueError("Preserve validation checkpoint selection and earliest tie")
    if config.get("quant_profiles") != ["W8A8", "HEAD_W8A16", "HEAD_W16A16"]:
        raise ValueError("All three real precision profiles are required")
    mappings = config.get("mapping_policies", [])
    if len(mappings) != 6 or set(mappings) != {"reference_low", "reference_high", "reference_min_latency",
            "reference_min_dsp", "reference_min_lut", "reference_min_bram"}:
        raise ValueError("All six distinct mapping policies are required")
    if config.get("calibration") != dict(split="train", images=64, augmentation=False):
        raise ValueError("Preserve train-only fixed calibration")
    if config.get("shortlist") != 12:
        raise ValueError("Preserve the 12-candidate confirmation shortlist")
    if config.get("deployment_claim") is not False:
        raise ValueError("A software migration cannot assert deployment")
    limits = config["hardware_budget"]["limits"]
    if set(limits) != {"latency_us", "lut", "dsp", "bram18"} or any(
        isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x <= 0 for x in limits.values()):
        raise ValueError("Finite positive hardware limits with explicit units required")
    if config["hardware"]["scope"] not in ("operator_hls", "full_hls", "route", "board"):
        raise ValueError("A comparable hardware evidence scope is required")
    return config


def git_info():
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,stderr=subprocess.PIPE).decode("utf-8").strip()
    return dict(commit=git("rev-parse","HEAD"), branch=git("branch","--show-current"),
                dirty=bool(git("status","--porcelain=v1","--untracked-files=normal")))


def create_run(config_path, output, *, command, require_clean=True):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load_config(config_path)
    source = git_info()
    if source["dirty"] and require_clean:
        raise ValueError("Commit source/config changes before executing a reproducible experiment")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Each run requires a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(python=sys.version, platform=platform.platform(),
                       packages=sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()}))
    files = {}
    raw = subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode("utf-8")
    for relative in raw.split("\0"):
        # Dataset bytes are bound by the portable publication index and the
        # actual loader manifests. Planning must not read multi-GB archives.
        if relative.startswith("datasets/") and relative not in {
                "datasets/dataset_sources.json", "datasets/publication_summary.json"}:
            continue
        path = ROOT / relative
        if relative and path.is_file():
            files[relative] = sha_file(path)
    run = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
               command=command, argv=list(sys.argv), git=source, config=config, config_path=str(config_path), config_hash=canonical_hash(config),
               config_source_sha256=sha_file(config_path), source_files=files,
               source_manifest_hash=canonical_hash(files), environment=environment,
               reproducible=not source["dirty"], test_evaluated=False)
    from omegaconf import OmegaConf
    (output / "config.resolved.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)),encoding="utf-8")
    save_json(output / "run.json", run)
    return config, run


def verify_run(path, *, current_source=False):
    path = Path(path)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if canonical_hash(run["config"]) != run["config_hash"]:
        raise ValueError("Run configuration identity changed")
    if canonical_hash(run["source_files"]) != run["source_manifest_hash"]:
        raise ValueError("Run source manifest identity changed")
    if current_source:
        live = git_info()
        if live["dirty"] or live["commit"] != run["git"]["commit"]:
            raise ValueError("Resume requires the exact clean source commit")
        for relative, expected in run["source_files"].items():
            if sha_file(ROOT / relative) != expected:
                raise ValueError("Frozen source drift: " + relative)
    return run


def make_record(genotype, plan, hardware, *, run, seed, budget, accuracy=None, artifacts=None,
                stage="plan", context=None):
    row = dict(schema_version=1, genotype=genotype.to_dict(), arch_hash=genotype.arch_hash,
               architecture_id=genotype.architecture_id, legacy_arch_hash=genotype.legacy_arch_hash,
               quantization=plan.to_dict(), q_hash=plan.q_hash,
               git_commit=run["git"]["commit"], config_hash=run["config_hash"],
               source_manifest_hash=run["source_manifest_hash"], seed=seed, budget=budget,
               stage=stage, accuracy=accuracy, hardware=hardware, hardware_context=context,
               artifacts=artifacts or {}, test_evaluated=False, deployment_claim=False)
    validate_record(row)
    row["record_hash"] = canonical_hash(row)
    return row


def validate_record(row):
    if not isinstance(row.get("git_commit"),str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}",row["git_commit"]) is None:
        raise ValueError("Candidate requires a concrete Git commit")
    g, q = Genotype.from_dict(row["genotype"]), QuantizationPlan.from_dict(row["quantization"])
    if row.get("architecture_id") != g.architecture_id or row.get("legacy_arch_hash") != g.legacy_arch_hash:
        raise ValueError("Candidate display/legacy identities differ from its Genotype")
    if row["arch_hash"] != g.arch_hash or q.arch_hash != g.arch_hash or row["q_hash"] != q.q_hash:
        raise ValueError("Candidate/precision identity mismatch")
    for key in ("config_hash", "source_manifest_hash"):
        validate_sha256(row[key])
    if row.get("test_evaluated") is not False or row.get("deployment_claim") is not False:
        raise ValueError("Records must preserve selection/deployment evidence boundaries")
    hw = row["hardware"]
    from hw import HardwareContext, HardwareResult
    parsed_hardware = HardwareResult(**hw)
    context = row.get("hardware_context")
    if context is not None:
        context = HardwareContext.from_dict(context)
        if hw["context_hash"] != context.context_hash or hw["evidence_level"] != context.scope:
            raise ValueError("Hardware context hash/scope mismatch")
    if hw["arch_hash"] != g.arch_hash or hw["q_hash"] != q.q_hash:
        raise ValueError("Hardware identity mismatch")
    if type(row["seed"]) is not int or row["seed"] not in (42,43,44) or row["budget"] not in (40,60,80):
        raise ValueError("Unregistered seed or stage budget")
    accuracy = row.get("accuracy")
    if accuracy is not None:
        score = accuracy.get("mAP50_95")
        if (accuracy.get("split") != "valid" or accuracy.get("test_evaluated") is not False
                or isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Only finite validation mAP50_95 is eligible")
        for key in ("checkpoint_sha256", "quant_artifact_sha256", "calibration_sha256", "dataset_sha256"):
            validate_sha256(accuracy[key])
        if context is not None:
            for field in ("checkpoint_sha256", "quant_artifact_sha256"):
                if getattr(context,field) is not None and getattr(context,field) != accuracy[field]:
                    raise ValueError("Accuracy and hardware artifact identities differ")
    if "record_hash" in row:
        content = {k:v for k,v in row.items() if k != "record_hash"}
        if canonical_hash(content) != row["record_hash"]:
            raise ValueError("Candidate record content drift")
    return row


class ResultStore:
    def __init__(self, output):
        self.output = Path(output)

    def write(self, row):
        validate_record(row)
        folder = self.output / "candidates" / row["arch_hash"]
        save_json(folder / (row["record_hash"] + ".json"), row)

    def read(self):
        run = verify_run(self.output)
        records = []
        for path in sorted((self.output / "candidates").glob("*/*.json")):
            row = validate_record(json.loads(path.read_text(encoding="utf-8")))
            if any(row[k] != run[r] for k,r in (("config_hash","config_hash"),("source_manifest_hash","source_manifest_hash"))):
                raise ValueError("Candidate belongs to a different run")
            if row["git_commit"] != run["git"]["commit"]:
                raise ValueError("Candidate commit differs from run")
            records.append(row)
        return records
