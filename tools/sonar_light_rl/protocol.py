"""Frozen software diagnostic protocol and evidence-preserving interfaces."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tools.sonar_joint_v2 import policy
from tools.sonar_light_rl import VERSION

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_protocol():
    """Executable diagnostic defaults, explicitly not a frozen research budget."""
    return dict(version=VERSION, space_version="elastic_neck72_v1", seed=42,
        base_config="configs/search/sonar_joint_v2_elastic_v1.yaml",
        budgets=[40, 60, 80], episodes_per_stage=[8, 4, 2], populations=[72, 54, 36],
        learning_rate=.005, baseline_momentum=.9, entropy_weight=.001,
        reward=dict(accuracy_weight=1., latency_weight=.2, dsp_weight=.1,
                    bram_weight=.1, lut_weight=.1, energy_weight=0., constraint_penalty=1.),
        hardware_budget=dict(scope="operator_peak_and_serial_compute",
            limits=dict(Latency=120000., LUT=161280., DSP=1209., BRAM=436.),
            latency_unit="us", bram_unit="BRAM18"),
        promotion=dict(fraction=.15, accuracy_epsilon=.001, resource_fraction=.90),
        gpu_model=None, gpu_h_cap=None, research_protocol_frozen=False,
        evaluation_split="valid", test_policy="sealed", deployment_claim=False)


def validate_protocol(cfg):
    cfg = deepcopy(cfg)
    required = set(default_protocol())
    if set(cfg) != required:
        raise ValueError("Protocol keys differ: " + str(set(cfg) ^ required))
    if cfg["version"] != VERSION or cfg["space_version"] != "elastic_neck72_v1":
        raise ValueError("Separate lightweight RL strategy version required")
    if cfg["budgets"] != [40, 60, 80] or cfg["populations"] != [72, 54, 36]:
        raise ValueError("Preserve staged fidelity and population ceilings")
    if len(cfg["episodes_per_stage"]) != 3 or any(type(n) is not int or not 1 <= n <= 1296
                                                  for n in cfg["episodes_per_stage"]):
        raise ValueError("Three bounded episode counts required")
    if type(cfg["seed"]) is not int or cfg["seed"] not in (42, 43, 44):
        raise ValueError("Registered independent-search seed required")
    for k in ("learning_rate", "baseline_momentum", "entropy_weight"):
        if not policy.finite(cfg[k], positive=k == "learning_rate"):
            raise ValueError("Invalid controller parameter: " + k)
    if cfg["baseline_momentum"] >= 1:
        raise ValueError("Baseline momentum must be below one")
    if cfg["gpu_model"] is not None and (not isinstance(cfg["gpu_model"], str) or not cfg["gpu_model"].strip()):
        raise ValueError("GPU model must be a nonempty string or null")
    if cfg["gpu_h_cap"] is not None and not policy.finite(cfg["gpu_h_cap"], positive=True):
        raise ValueError("GPU budget must be positive or null")
    expected_reward = set(default_protocol()["reward"])
    if set(cfg["reward"]) != expected_reward or any(not policy.finite(v) for v in cfg["reward"].values()):
        raise ValueError("Explicit finite fixed reward weights required")
    if cfg["reward"]["energy_weight"] != 0 or cfg["reward"]["accuracy_weight"] <= 0:
        raise ValueError("Energy is unmeasured; positive accuracy reward required")
    policy.validate_budget(cfg["hardware_budget"])
    if cfg["hardware_budget"].get("latency_unit") != "us" or cfg["hardware_budget"].get("bram_unit") != "BRAM18":
        raise ValueError("Cost normalization requires us and BRAM18")
    for key, maximum in (("fraction", .5), ("accuracy_epsilon", 1), ("resource_fraction", 1)):
        if not policy.finite(cfg["promotion"].get(key), positive=key == "resource_fraction") or cfg["promotion"][key] > maximum:
            raise ValueError("Invalid stage promotion policy")
    if (cfg["evaluation_split"] != "valid" or cfg["test_policy"] != "sealed"
            or cfg["deployment_claim"] is not False or type(cfg["research_protocol_frozen"]) is not bool):
        raise ValueError("Validation-only selection and explicit research freeze required")
    return cfg


def source_binding():
    paths = ["tools/sonar_light_rl/" + name for name in ("__init__.py", "protocol.py", "controller.py", "run.py")]
    paths += ["src/hwnas_fpga/search/rl_searcher.py", "tools/sonar_joint_v2/policy.py",
              "tools/modnas_comparison/joint_v2_elastic.py", "tools/modnas_elastic/model.py"]
    backend = "tools/modnas_comparison/joint_cost_backend.py"
    if (ROOT / backend).is_file():
        paths.append(backend)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def environment_binding(cfg):
    """Bind base protocol plus the referenced registry; no claims of admission."""
    import yaml
    path = (ROOT / cfg["base_config"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Workspace base configuration required")
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ("hardware_cost_evidence", "software_backend_evidence"):
        if base.get(name):
            evidence = (ROOT / base[name]).resolve()
            if not evidence.is_relative_to(ROOT) or not evidence.is_file():
                raise ValueError("Configured evidence missing/outside workspace: " + name)
            files[str(evidence.relative_to(ROOT))] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    return dict(files=files, hardware_cost_backend=base.get("hardware_cost_backend"),
                scope="identity_only_not_hardware_admission")


def joint_key(row):
    return (row["architecture_id"], row["profile"], row["mapping_policy"])


def cost_status(row, requirement, cfg):
    """Unknown evidence must never become a hardware infeasibility reward."""
    identity = ("architecture_id", "architecture_hash", "profile", "mapping_policy", "space_version", "num_classes")
    if any(row.get(k) != requirement[k] for k in identity):
        raise ValueError("Hardware response does not bind requested joint candidate")
    if (row.get("status") != "LOOKED_UP" or row.get("measurement_valid") is not True
            or row.get("timing_eligible") is not True
            or any(not policy.finite(row.get(k)) for k in policy.METRICS)):
        return "NEEDS_MEASUREMENT"
    return "FEASIBLE" if policy.filter_costs([row], cfg["hardware_budget"])["rows"] else "OVER_BUDGET"


def provisional_handoff(point):
    return dict(schema_version=1, space_version=point["space_version"],
        architecture=point["architecture"], architecture_id=point["architecture_id"],
        architecture_hash=point["architecture_hash"], num_classes=point["num_classes"],
        profile=point["profile"], mapping_policy=point["mapping_policy"], seed=point["seed"],
        budget=point["budget"], artifacts={}, status="PENDING_THREE_SEED_CONFIRMATION",
        evidence_kind=point["evidence_kind"], formal_eligible=False, launch_hardware=False)
