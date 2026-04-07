#!/usr/bin/env python3
"""Sample architectures from a search space and report FPGA feasibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.interfaces import SearchCandidate
from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
    pick,
)
from hwnas_fpga.search_space import list_family_profiles, probe_search_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search-space feasibility probe")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--family-profile",
        type=str,
        default=None,
        choices=tuple(sorted(list_family_profiles().keys())),
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of random samples")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Run artifact root directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run directory name")
    parser.add_argument(
        "--skip-pruning",
        action="store_true",
        help="Probe the raw search space instead of the hardware-pruned one",
    )
    parser.add_argument(
        "--prefer-lightweight",
        action="store_true",
        help="Bias sampling toward lightweight choices instead of uniform random sampling",
    )
    return parser.parse_args()


def _search_space_payload(search_space) -> dict[str, object]:
    return {
        "input_channels": search_space.config.input_channels,
        "image_size": search_space.config.image_size,
        "stem_channels": search_space.config.stem_channels,
        "stem_stride": search_space.config.stem_stride,
        "post_stem_downsample_stride": search_space.config.post_stem_downsample_stride,
        "stage_strides": list(search_space.config.stage_strides),
        "stage_base_channels": (
            None
            if search_space.config.stage_base_channels is None
            else list(search_space.config.stage_base_channels)
        ),
        "width_multipliers": (
            None
            if search_space.config.width_multipliers is None
            else list(search_space.config.width_multipliers)
        ),
        "stage_channel_choices": (
            None
            if search_space.config.stage_channel_choices is None
            else [list(group) for group in search_space.config.stage_channel_choices]
        ),
        "stage_depth_choices": (
            None
            if search_space.config.stage_depth_choices is None
            else [list(group) for group in search_space.config.stage_depth_choices]
        ),
        "channel_choices": list(search_space.config.channel_choices),
        "depth_choices": list(search_space.config.depth_choices),
        "kernel_choices": list(search_space.config.kernel_choices),
        "expand_choices": list(search_space.config.expand_choices),
        "op_choices": list(search_space.config.op_choices),
        "head_conv_channels": search_space.config.head_conv_channels,
        "family_profile": search_space.config.family_profile,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("search_space", {})
    if args.family_profile is not None:
        config["search_space"]["family_profile"] = args.family_profile

    dataset_cfg = config.get("dataset", {})
    project_cfg = config.get("project", {})

    seed = int(pick(args.seed, project_cfg.get("seed"), 42))
    image_size = int(pick(args.image_size, dataset_cfg.get("image_size"), 224))
    input_channels = int(pick(args.input_channels, dataset_cfg.get("input_channels"), 1))
    num_classes = pick(args.num_classes, dataset_cfg.get("num_classes"), None)
    num_samples = int(pick(args.num_samples, config.get("probe", {}).get("num_samples"), 200))
    output_dir = pick(args.output_dir, project_cfg.get("output_dir"), "results")
    run_name = pick(args.run_name, project_cfg.get("run_name"), None)

    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    tracker = ExperimentTracker(
        output_root=output_root,
        project_name=project_cfg.get("name", "hwnas"),
        script_name="run_search_space_probe",
        search_method="space_probe",
        dataset_name=config.get("dataset", {}).get("name", "dummy"),
        run_name=run_name,
    )
    tracker.save_config(config, vars(args))

    try:
        with tracker.capture_console():
            print(f"Run dir: {tracker.root_dir}")
            print(f"Samples: {num_samples}")
            print(f"Seed: {seed}")
            print(
                "Family profile: "
                f"{config.get('search_space', {}).get('family_profile', 'default')}"
            )
            print(f"Prefer lightweight sampling: {args.prefer_lightweight}")
            print(f"Apply pruning: {not args.skip_pruning}")

            constraints = build_constraints(config)
            hardware_spec = build_hardware_spec(config)
            search_space = build_search_space(
                config,
                image_size=image_size,
                input_channels=input_channels,
                num_classes=num_classes,
                constraints=constraints,
            )
            tracker.write_search_space_summary(_search_space_payload(search_space))

            estimator = build_cost_estimator(
                config,
                hardware_spec=hardware_spec,
                constraints=constraints,
            )
            print(
                "Hardware profile: "
                f"{hardware_spec.name}, clock={hardware_spec.clock_mhz}MHz, "
                f"LUT={hardware_spec.max_lut}, DSP={hardware_spec.max_dsp}, BRAM={hardware_spec.max_bram}"
            )

            effective_space = search_space if args.skip_pruning else search_space.pre_prune(estimator)
            tracker.write_pruned_search_space_summary(
                {
                    **_search_space_payload(effective_space),
                    "pruned": not args.skip_pruning,
                }
            )

            baseline = effective_space.baseline_architecture()
            baseline_cost = estimator.estimate(baseline, effective_space)
            tracker.write_baseline(
                architecture=baseline.to_dict(),
                cost_estimate=baseline_cost,
            )

            print("\n=== Probing Search Space ===")
            records, summary = probe_search_space(
                effective_space,
                estimator,
                num_samples=num_samples,
                seed=seed,
                prefer_lightweight=args.prefer_lightweight,
            )

            candidates = []
            feasible_candidates = []
            infeasible_candidates = []
            for index, record in enumerate(records, start=1):
                candidate = SearchCandidate(
                    arch_id=record.arch_id,
                    encoding=record.architecture.to_dict(),
                    metrics=record.cost_estimate.to_candidate_metrics(),
                )
                candidates.append(candidate)
                if record.feasible:
                    feasible_candidates.append(candidate)
                else:
                    infeasible_candidates.append(candidate)

                tracker.record_candidate(
                    candidate,
                    feasible=record.feasible,
                    cost_estimate=record.cost_estimate,
                    history=None,
                    extra={
                        "probe_index": index,
                        "violations": list(record.violations),
                        "search_method": "space_probe",
                    },
                )

            tracker.update_search_state(
                total_evaluated=len(candidates),
                feasible=len(feasible_candidates),
                infeasible=len(infeasible_candidates),
                best_candidate=None,
                extra={
                    "mode": "space_probe",
                    "feasible_ratio": summary["feasible_ratio"],
                    "feasible_ratio_pct": summary["feasible_ratio_pct"],
                },
            )

            probe_summary_path = tracker.results_dir / "probe_summary.json"
            probe_summary_path.write_text(
                json.dumps(
                    {
                        "recorded_at": tracker.root_dir.name,
                        "num_samples": num_samples,
                        "seed": seed,
                        "prefer_lightweight": bool(args.prefer_lightweight),
                        "applied_pruning": not args.skip_pruning,
                        "summary": summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "Feasible ratio: "
                f"{summary['feasible_samples']}/{summary['total_samples']} "
                f"({summary['feasible_ratio_pct']:.2f}%)"
            )
            if summary["violation_counts"]:
                print("Top violations:")
                for violation, count in list(summary["violation_counts"].items())[:5]:
                    print(f"  - {violation}: {count}")

            tracker.finalize(
                status="completed",
                candidates=candidates,
                feasible_candidates=feasible_candidates,
                infeasible_candidates=infeasible_candidates,
                best_candidate=None,
                pareto_candidates=None,
                extra={
                    "mode": "space_probe",
                    "num_samples": num_samples,
                    "seed": seed,
                    "applied_pruning": not args.skip_pruning,
                    "prefer_lightweight": bool(args.prefer_lightweight),
                    "probe_summary": summary,
                },
            )

            print("\n=== Done ===")
    except Exception as exc:
        tracker.mark_failed(str(exc))
        raise


if __name__ == "__main__":
    main()
