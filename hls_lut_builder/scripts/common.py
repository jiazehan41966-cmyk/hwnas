from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Optional

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILDER_ROOT.parent

SHAPE_META_KEYS = {"parameters", "op_spec", "notes"}


@dataclass(frozen=True)
class HlsCase:
    case_name: str
    op_id: str
    operator: str
    shape_name: str
    implementation_name: str
    clock_profile_name: str
    top_function: str
    template_path: Path
    project_dir: Path
    source_filename: str
    solution_name: str
    part: str
    clock_period_ns: float
    target_clock_mhz: float
    flow_target: str
    cpp_standard: str
    parameters: dict[str, Any]
    op_spec: dict[str, Any]

    @property
    def source_path(self) -> Path:
        return self.project_dir / "src" / self.source_filename

    @property
    def metadata_path(self) -> Path:
        return self.project_dir / "metadata.json"

    @property
    def tcl_path(self) -> Path:
        return self.project_dir / "synth.tcl"

    @property
    def vivado_downstream_dir(self) -> Path:
        return self.project_dir / "vivado_downstream"

    @property
    def vivado_downstream_reports_dir(self) -> Path:
        return self.vivado_downstream_dir / "reports"

    @property
    def vivado_downstream_status_path(self) -> Path:
        return self.project_dir / "vivado_downstream_status.json"

    @property
    def report_candidates(self) -> tuple[Path, ...]:
        base = self.project_dir / "project" / self.solution_name / "syn" / "report"
        top = self.top_function
        return (
            base / f"{top}_csynth.xml",
            base / f"{top}_csynth.rpt",
            base / "csynth.xml",
            base / "csynth.rpt",
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "op_id": self.op_id,
            "operator": self.operator,
            "shape_name": self.shape_name,
            "implementation_name": self.implementation_name,
            "clock_profile_name": self.clock_profile_name,
            "top_function": self.top_function,
            "template_path": str(self.template_path),
            "project_dir": str(self.project_dir),
            "source_path": str(self.source_path),
            "solution_name": self.solution_name,
            "part": self.part,
            "clock_period_ns": self.clock_period_ns,
            "target_clock_mhz": self.target_clock_mhz,
            "flow_target": self.flow_target,
            "cpp_standard": self.cpp_standard,
            "parameters": self.parameters,
            "op_spec": self.op_spec,
            "report_candidates": [str(path) for path in self.report_candidates],
        }


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Builder config must be a mapping: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def resolve_workspace_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    config_path = Path(config["_config_path"])
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def sanitize_name(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "_")
    lowered = re.sub(r"[^a-z0-9_]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")


def render_template(template_text: str, parameters: dict[str, Any]) -> str:
    rendered = template_text
    for key, value in parameters.items():
        rendered = rendered.replace(f"__{key.upper()}__", str(value))

    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", rendered)))
    if unresolved:
        raise ValueError(f"Template still contains unresolved placeholders: {', '.join(unresolved)}")
    return rendered


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_tcl(case: HlsCase) -> str:
    include_dir = (BUILDER_ROOT / "include").as_posix()
    lines = [
        "open_project project",
        f"set_top {case.top_function}",
        f"add_files src/{case.source_filename} -cflags {{-std={case.cpp_standard} -O3 -I{include_dir}}}",
        f"open_solution -reset {case.solution_name} -flow_target {case.flow_target}",
        f"set_part {{{case.part}}}",
        f"create_clock -period {case.clock_period_ns:.3f} -name default",
        "set_clock_uncertainty 0.200",
        'puts "=== Starting synthesis ==="',
        'if {[catch {csynth_design} err]} {',
        '  puts "SYNTHESIS FAILED: $err"',
        "  exit 1",
        "}",
        f'set rpt_file "project/{case.solution_name}/syn/report/{case.top_function}_csynth.rpt"',
        f'set xml_file "project/{case.solution_name}/syn/report/{case.top_function}_csynth.xml"',
        "if {![file exists $rpt_file] && ![file exists $xml_file]} {",
        '  puts "ERROR: Report not generated at $rpt_file or $xml_file"',
        "  exit 2",
        "}",
        'puts "=== Synthesis completed successfully ==="',
        "exit",
    ]
    return "\n".join(lines) + "\n"


def build_cases(
    config: dict[str, Any],
    *,
    operator_filter: Optional[Iterable[str]] = None,
    case_filter: Optional[Iterable[str]] = None,
    project_root_override: str | Path | None = None,
    sort_cases: bool = True,
) -> list[HlsCase]:
    toolchain = config.get("toolchain", {})
    workspace = config.get("workspace", {})
    defaults = dict(config.get("defaults", {}))
    shapes = config.get("representative_shapes", {})
    operators = config.get("operators", {})
    impl_profiles = _load_impl_profiles(config)
    clock_profiles = _load_clock_profiles(config, toolchain)

    selected_operators = {sanitize_name(item) for item in operator_filter or ()}
    selected_cases = {sanitize_name(item) for item in case_filter or ()}

    if project_root_override is None:
        project_root = resolve_workspace_path(config, workspace["project_root"])
    else:
        override_path = Path(project_root_override).expanduser()
        project_root = override_path.resolve() if override_path.is_absolute() else (Path.cwd() / override_path).resolve()

    cases: list[HlsCase] = []
    for operator_name, operator_cfg in operators.items():
        operator_key = sanitize_name(operator_name)
        include_disabled = bool(selected_operators and operator_key in selected_operators)
        if not operator_cfg.get("enabled", True) and not include_disabled:
            continue
        if selected_operators and operator_key not in selected_operators:
            continue

        template_path = resolve_workspace_path(config, operator_cfg["template"])
        top_function = str(operator_cfg.get("top_function", operator_name))
        shape_refs = operator_cfg.get("shape_refs", [])
        if not shape_refs:
            raise ValueError(f"Operator {operator_name} must define a non-empty shape_refs list")

        for shape_name in shape_refs:
            if shape_name not in shapes:
                raise ValueError(f"Operator {operator_name} references missing shape {shape_name}")

            shape_cfg = shapes[shape_name]
            impl_refs = operator_cfg.get("implementation_refs") or tuple(impl_profiles.keys())
            clock_refs = operator_cfg.get("clock_profile_refs") or tuple(clock_profiles.keys())

            for impl_name in impl_refs:
                if impl_name not in impl_profiles:
                    raise ValueError(f"Operator {operator_name} references missing implementation profile {impl_name}")
                impl_cfg = impl_profiles[impl_name]

                for clock_name in clock_refs:
                    if clock_name not in clock_profiles:
                        raise ValueError(f"Operator {operator_name} references missing clock profile {clock_name}")
                    clock_cfg = clock_profiles[clock_name]

                    case_name = sanitize_name(f"{operator_name}__{shape_name}__{impl_name}__{clock_name}")
                    if selected_cases and case_name not in selected_cases:
                        continue

                    parameters = _build_parameters(
                        defaults=defaults,
                        operator_name=operator_name,
                        operator_cfg=operator_cfg,
                        shape_name=str(shape_name),
                        shape_cfg=shape_cfg,
                        implementation_name=str(impl_name),
                        implementation_cfg=impl_cfg,
                        clock_profile_name=str(clock_name),
                        clock_cfg=clock_cfg,
                    )
                    op_spec = _build_op_spec(operator_name, operator_cfg, shape_cfg, parameters)
                    op_id = _build_op_id(op_spec)

                    cases.append(
                        HlsCase(
                            case_name=case_name,
                            op_id=op_id,
                            operator=operator_name,
                            shape_name=str(shape_name),
                            implementation_name=str(impl_name),
                            clock_profile_name=str(clock_name),
                            top_function=top_function,
                            template_path=template_path,
                            project_dir=project_root / case_name,
                            source_filename=f"{top_function}.cpp",
                            solution_name=str(toolchain.get("solution_name", "solution1")),
                            part=str(toolchain.get("part", "xc7k325t-ffg676-2")),
                            clock_period_ns=float(clock_cfg["period_ns"]),
                            target_clock_mhz=float(clock_cfg["target_clock_mhz"]),
                            flow_target=str(toolchain.get("flow_target", "vivado")),
                            cpp_standard=str(toolchain.get("cpp_standard", "c++14")),
                            parameters=parameters,
                            op_spec=op_spec,
                        )
                    )

    if sort_cases:
        return sorted(cases, key=lambda item: item.case_name)
    return cases


def select_pilot_cases(cases: Iterable[HlsCase]) -> list[HlsCase]:
    selected: list[HlsCase] = []
    seen: set[str] = set()
    for case in cases:
        if case.operator in seen:
            continue
        selected.append(case)
        seen.add(case.operator)
    return selected


def materialize_case_project(case: HlsCase, *, force: bool = False) -> None:
    if case.project_dir.exists() and force:
        _safe_remove_tree(case.project_dir)

    (case.project_dir / "src").mkdir(parents=True, exist_ok=True)
    template_text = load_template(case.template_path)
    rendered_source = render_template(
        template_text,
        {
            "top_function": case.top_function,
            **case.parameters,
        },
    )
    case.source_path.write_text(rendered_source, encoding="utf-8")
    case.tcl_path.write_text(build_tcl(case), encoding="utf-8")
    case.metadata_path.write_text(
        json.dumps(case.to_metadata(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_project_metadata(project_root: str | Path) -> list[dict[str, Any]]:
    root = Path(project_root).expanduser().resolve()
    payloads: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        payloads.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return payloads


def find_first_existing_report(report_candidates: Iterable[str | Path]) -> Optional[Path]:
    for candidate in report_candidates:
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _safe_remove_tree(path: Path) -> None:
    resolved = path.expanduser().resolve()
    expected_root = resolved.parent
    if resolved == expected_root or expected_root not in resolved.parents:
        raise ValueError(f"Refusing to remove unexpected path: {resolved}")
    shutil.rmtree(resolved)


def _load_impl_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = config.get("implementation_profiles")
    if isinstance(payload, dict) and payload:
        return {str(name): dict(value or {}) for name, value in payload.items()}
    return {"default": {}}


def _load_clock_profiles(config: dict[str, Any], toolchain: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = config.get("clock_profiles")
    if isinstance(payload, dict) and payload:
        normalized: dict[str, dict[str, Any]] = {}
        for name, value in payload.items():
            cfg = dict(value or {})
            period_ns = float(cfg.get("period_ns", toolchain.get("clock_period_ns", 5.0)))
            normalized[str(name)] = {
                "period_ns": period_ns,
                "target_clock_mhz": float(cfg.get("target_clock_mhz", 1_000.0 / period_ns)),
            }
        return normalized

    default_period = float(toolchain.get("clock_period_ns", 5.0))
    return {
        "default": {
            "period_ns": default_period,
            "target_clock_mhz": 1_000.0 / default_period,
        }
    }


def _build_parameters(
    *,
    defaults: dict[str, Any],
    operator_name: str,
    operator_cfg: dict[str, Any],
    shape_name: str,
    shape_cfg: dict[str, Any],
    implementation_name: str,
    implementation_cfg: dict[str, Any],
    clock_profile_name: str,
    clock_cfg: dict[str, Any],
) -> dict[str, Any]:
    parameters: dict[str, Any] = dict(defaults)
    parameters.update(operator_cfg.get("parameters", {}))
    parameters.update(shape_cfg.get("parameters", {}))
    parameters.update(implementation_cfg)

    for key, value in shape_cfg.items():
        if key not in SHAPE_META_KEYS:
            parameters[key] = value

    feature_h = int(parameters.get("feature_h", parameters.get("input_height", 1)))
    feature_w = int(parameters.get("feature_w", parameters.get("input_width", feature_h)))
    kernel_size = int(parameters.get("kernel_size", 1))
    stride = int(parameters.get("stride", 1))
    padding = int(parameters.get("padding", kernel_size // 2))
    in_channels = int(parameters.get("in_channels", 1))
    out_channels = int(parameters.get("out_channels", in_channels))
    expand_ratio = int(parameters.get("expand_ratio", 1))
    bitwidth = int(parameters.get("bitwidth", 8))
    input_parallelism = int(parameters.get("input_parallelism", parameters.get("pi", 1)))
    output_parallelism = int(parameters.get("output_parallelism", parameters.get("po", 1)))
    unroll_factor = int(parameters.get("unroll_factor", parameters.get("kernel_unroll", 1)))

    out_h = int(parameters.get("out_h", ((feature_h + (2 * padding) - kernel_size) // stride) + 1))
    out_w = int(parameters.get("out_w", ((feature_w + (2 * padding) - kernel_size) // stride) + 1))

    if operator_name.startswith("dw"):
        out_channels = in_channels
        groups = in_channels
    else:
        groups = int(parameters.get("groups", 1))

    parameters.update(
        {
            "operator_name": operator_name,
            "shape_name": shape_name,
            "implementation_name": implementation_name,
            "clock_profile_name": clock_profile_name,
            "feature_h": feature_h,
            "feature_w": feature_w,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "expand_ratio": expand_ratio,
            "expanded_channels": max(1, in_channels * expand_ratio),
            "groups": groups,
            "out_h": out_h,
            "out_w": out_w,
            "bitwidth": bitwidth,
            "input_parallelism": input_parallelism,
            "output_parallelism": output_parallelism,
            "unroll_factor": unroll_factor,
            "array_partition_factor": int(parameters.get("array_partition_factor", max(input_parallelism, output_parallelism, 1))),
            "pipeline_ii": int(parameters.get("pipeline_ii", 1)),
            "target_clock_ns": float(clock_cfg["period_ns"]),
            "target_clock_mhz": float(clock_cfg["target_clock_mhz"]),
            "case_label": sanitize_name(f"{operator_name}_{shape_name}_{implementation_name}_{clock_profile_name}"),
        }
    )
    return parameters


def _build_op_spec(
    operator_name: str,
    operator_cfg: dict[str, Any],
    shape_cfg: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    op_spec: dict[str, Any] = dict(operator_cfg.get("op_spec_defaults", {}))
    op_spec.update(shape_cfg.get("op_spec", {}))

    op_spec.setdefault("op", operator_cfg.get("op_type", operator_name))
    op_spec.setdefault("kernel_size", int(parameters["kernel_size"]))
    op_spec.setdefault("in_channels", int(parameters["in_channels"]))
    op_spec.setdefault("out_channels", int(parameters["out_channels"]))
    op_spec.setdefault("stride", int(parameters["stride"]))
    op_spec.setdefault("groups", int(parameters["groups"]))
    op_spec.setdefault("expand_ratio", int(parameters.get("expand_ratio", 1)))
    op_spec.setdefault("input_resolution", [int(parameters["feature_h"]), int(parameters["feature_w"])])
    op_spec.setdefault("bitwidth", int(parameters["bitwidth"]))
    op_spec.setdefault("input_parallelism", int(parameters["input_parallelism"]))
    op_spec.setdefault("output_parallelism", int(parameters["output_parallelism"]))
    op_spec.setdefault("unroll_factor", int(parameters["unroll_factor"]))
    op_spec.setdefault("target_clock_mhz", float(parameters["target_clock_mhz"]))
    return op_spec


def _build_op_id(op_spec: dict[str, Any]) -> str:
    height, width = op_spec.get("input_resolution", [0, 0])
    tokens = [
        str(op_spec.get("op", "op")),
        f"h{int(height)}",
        f"w{int(width)}",
        f"cin{int(op_spec.get('in_channels', 0))}",
        f"cout{int(op_spec.get('out_channels', 0))}",
        f"k{int(op_spec.get('kernel_size', 1))}",
        f"s{int(op_spec.get('stride', 1))}",
        f"g{int(op_spec.get('groups', 1))}",
        f"e{int(op_spec.get('expand_ratio', 1))}",
        f"bw{int(op_spec.get('bitwidth', 8))}",
        f"pi{int(op_spec.get('input_parallelism', 1))}",
        f"po{int(op_spec.get('output_parallelism', 1))}",
        f"u{int(op_spec.get('unroll_factor', 1))}",
    ]
    target_clock = op_spec.get("target_clock_mhz")
    if target_clock is not None:
        tokens.append(f"clk{sanitize_name(f'{float(target_clock):.3f}mhz')}")
    return sanitize_name("_".join(tokens))
