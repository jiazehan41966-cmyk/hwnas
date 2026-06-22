"""Run official ProxylessNAS (reference clone) for NKSID and export HW-NAS artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate, SearchConstraints
from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    create_data_pipeline,
    load_config,
)
from hwnas_fpga.search import (
    ParetoFrontSelector,
    build_pareto_objectives,
    build_pareto_selection_summary,
    candidate_selection_score,
    compute_hypervolume,
    compute_pareto_front,
    compute_pareto_ranks,
    write_pareto_selection_artifacts,
)
from hwnas_fpga.search.official_proxyless_bridge import proxyless_net_config_to_architecture
from hwnas_fpga.search.searcher import BaseSearcher
from hwnas_fpga.training.trainer import evaluate_classifier

REPO_ROOT = Path(__file__).resolve().parents[3]
PROXYLESS_SEARCH = REPO_ROOT / "reference" / "ProxylessNAS" / "search"
MOBILE_REF_LATENCY_MS = 80.0
PROXYLESS_PATCH = REPO_ROOT / "patches" / "proxylessnas_hwnas_strict_current84.patch"


class HwnasLoaderDataProvider:
    """Minimal ProxylessNAS DataProvider backed by HW-NAS PyTorch loaders."""

    def __init__(
        self,
        *,
        train_loader,
        valid_loader,
        test_loader,
        input_channels: int,
        n_classes: int,
        image_size: int,
        data_dir: str,
    ) -> None:
        self.train = train_loader
        self.valid = valid_loader
        self.test = test_loader or valid_loader
        self._input_channels = int(input_channels)
        self._n_classes = int(n_classes)
        self._image_size = int(image_size)
        self._data_dir = str(data_dir)

    @staticmethod
    def name() -> str:
        return "nksid"

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return self._input_channels, self._image_size, self._image_size

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def save_path(self) -> str:
        return self._data_dir

    @property
    def data_url(self) -> str:
        return ""


def _ensure_proxyless_import_path() -> Path:
    if not (PROXYLESS_SEARCH / "nas_manager.py").exists():
        raise FileNotFoundError(
            "Official ProxylessNAS clone is missing. Run:\n"
            "  git submodule update --init --recursive reference/ProxylessNAS"
        )
    mix_op = PROXYLESS_SEARCH / "modules" / "mix_op.py"
    super_proxyless = PROXYLESS_SEARCH / "models" / "super_nets" / "super_proxyless.py"
    run_manager = PROXYLESS_SEARCH / "run_manager.py"
    patch_ready = (
        "1x1_Conv" in mix_op.read_text(encoding="utf-8")
        and "first_block_candidates" in super_proxyless.read_text(encoding="utf-8")
        and "def net_module" in run_manager.read_text(encoding="utf-8")
    )
    if not patch_ready:
        raise RuntimeError(
            "Official ProxylessNAS clone is present but does not include the HW-NAS "
            "strict-current84 compatibility patch. Apply it first:\n"
            f"  git -C {REPO_ROOT / 'reference' / 'ProxylessNAS'} apply {PROXYLESS_PATCH}"
        )
    search_path = str(PROXYLESS_SEARCH)
    if search_path not in sys.path:
        sys.path.insert(0, search_path)
    return PROXYLESS_SEARCH


def _resolve_official_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    search_cfg = config.get("search") or {}
    proxyless_cfg = search_cfg.get("proxyless") or {}
    official_cfg = search_cfg.get("proxyless_official") or {}
    space_cfg = config.get("search_space") or {}
    dataset_cfg = config.get("dataset") or {}
    training_cfg = config.get("training") or {}

    width_stages = official_cfg.get("width_stages") or space_cfg.get("stage_base_channels") or [16, 24, 32, 32]
    n_cell_stages = official_cfg.get("n_cell_stages")
    if n_cell_stages is None:
        depth_choices = space_cfg.get("stage_depth_choices") or [[1], [1], [1], [1]]
        n_cell_stages = [int(choice[0]) for choice in depth_choices]
    stride_stages = official_cfg.get("stride_stages") or space_cfg.get("stage_strides") or [1, 2, 2, 1]
    first_block_as_stage0 = bool(official_cfg.get("first_block_as_stage0", False))
    super_width_stages = official_cfg.get("super_width_stages")
    super_n_cell_stages = official_cfg.get("super_n_cell_stages")
    super_stride_stages = official_cfg.get("super_stride_stages")
    if first_block_as_stage0:
        if super_width_stages is None:
            super_width_stages = list(width_stages)[1:]
        if super_n_cell_stages is None:
            super_n_cell_stages = list(n_cell_stages)[1:]
        if super_stride_stages is None:
            super_stride_stages = list(stride_stages)[1:]
    else:
        super_width_stages = super_width_stages or width_stages
        super_n_cell_stages = super_n_cell_stages or n_cell_stages
        super_stride_stages = super_stride_stages or stride_stages

    conv_candidates = official_cfg.get("conv_candidates")
    if conv_candidates is None:
        conv_candidates = ["3x3_MBConv3", "3x3_MBConv6", "5x5_MBConv3", "5x5_MBConv6"]
    first_block_candidates = official_cfg.get("first_block_candidates") or ["3x3_MBConv1"]

    target_hardware = official_cfg.get("target_hardware", proxyless_cfg.get("target_hardware", "mobile"))
    if target_hardware in {"latency_ms", "latency"}:
        target_hardware = "mobile"

    ref_value = official_cfg.get("ref_value", proxyless_cfg.get("ref_value"))
    if ref_value is None and target_hardware == "mobile":
        ref_value = MOBILE_REF_LATENCY_MS

    return {
        "width_stages": [int(v) for v in width_stages],
        "n_cell_stages": [int(v) for v in n_cell_stages],
        "stride_stages": [int(v) for v in stride_stages],
        "super_width_stages": [int(v) for v in super_width_stages],
        "super_n_cell_stages": [int(v) for v in super_n_cell_stages],
        "super_stride_stages": [int(v) for v in super_stride_stages],
        "conv_candidates": list(conv_candidates),
        "first_block_candidates": list(first_block_candidates),
        "bridge_skip_fixed_first_block": bool(
            official_cfg.get("bridge_skip_fixed_first_block", not first_block_as_stage0)
        ),
        "width_mult": float(official_cfg.get("width_mult", 1.0)),
        "warmup_epochs": int(official_cfg.get("warmup_epochs", proxyless_cfg.get("warmup_epochs", 10))),
        "n_epochs": int(
            official_cfg.get(
                "n_epochs",
                proxyless_cfg.get("search_epochs", search_cfg.get("episodes", 300)),
            )
        ),
        "init_lr": float(official_cfg.get("init_lr", proxyless_cfg.get("weight_lr", 0.025))),
        "weight_decay": float(official_cfg.get("weight_decay", proxyless_cfg.get("weight_decay", 3e-4))),
        "momentum": float(official_cfg.get("momentum", proxyless_cfg.get("weight_momentum", 0.9))),
        "arch_lr": float(official_cfg.get("arch_lr", proxyless_cfg.get("arch_lr", 1e-3))),
        "arch_weight_decay": float(
            official_cfg.get("arch_weight_decay", proxyless_cfg.get("arch_weight_decay", 0.0))
        ),
        "grad_reg_loss_type": str(
            official_cfg.get("grad_reg_loss_type", proxyless_cfg.get("grad_reg_loss_type", "mul#log"))
        ),
        "grad_reg_loss_alpha": float(
            official_cfg.get("grad_reg_loss_alpha", proxyless_cfg.get("grad_reg_loss_alpha", 0.2))
        ),
        "grad_reg_loss_beta": float(
            official_cfg.get("grad_reg_loss_beta", proxyless_cfg.get("grad_reg_loss_beta", 0.3))
        ),
        "grad_update_arch_param_every": int(official_cfg.get("grad_update_arch_param_every", 5)),
        "grad_update_steps": int(official_cfg.get("grad_update_steps", 1)),
        "grad_binary_mode": str(official_cfg.get("grad_binary_mode", "full_v2")),
        "target_hardware": target_hardware,
        "ref_value": ref_value,
        "train_batch_size": int(training_cfg.get("batch_size", 32)),
        "test_batch_size": int(official_cfg.get("test_batch_size", training_cfg.get("batch_size", 32))),
        "num_workers": int(dataset_cfg.get("num_workers", 8)),
        "validation_frequency": int(official_cfg.get("validation_frequency", 1)),
        "print_frequency": int(official_cfg.get("print_frequency", 10)),
        "label_smoothing": float(official_cfg.get("label_smoothing", 0.0)),
    }


def _patch_first_conv_for_grayscale(super_net, input_channels: int) -> None:
    from modules.layers import ConvLayer

    first_conv = super_net.first_conv
    super_net.first_conv = ConvLayer(
        int(input_channels),
        first_conv.out_channels,
        kernel_size=3,
        stride=2,
        use_bn=True,
        act_func="relu6",
        ops_order="weight_bn_act",
    )


def _build_arch_search_manager(
    *,
    official_path: Path,
    settings: Mapping[str, Any],
    num_classes: int,
    input_channels: int,
    image_size: int,
    data_provider: HwnasLoaderDataProvider,
    manual_seed: int,
):
    from models import ImagenetRunConfig
    from models.super_nets.super_proxyless import SuperProxylessNASNets
    from nas_manager import ArchSearchRunManager, GradientArchSearchConfig

    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)

    run_config = ImagenetRunConfig(
        n_epochs=int(settings["n_epochs"]),
        init_lr=float(settings["init_lr"]),
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="nksid",
        train_batch_size=int(settings["train_batch_size"]),
        test_batch_size=int(settings["test_batch_size"]),
        valid_size=None,
        opt_type="sgd",
        opt_param={"momentum": float(settings["momentum"]), "nesterov": True},
        weight_decay=float(settings["weight_decay"]),
        label_smoothing=float(settings["label_smoothing"]),
        no_decay_keys="bn",
        model_init="he_fout",
        init_div_groups=False,
        validation_frequency=int(settings["validation_frequency"]),
        print_frequency=int(settings["print_frequency"]),
        n_worker=int(settings["num_workers"]),
        resize_scale=0.08,
        distort_color=None,
    )
    run_config._data_provider = data_provider

    super_net = SuperProxylessNASNets(
        width_stages=list(settings["super_width_stages"]),
        n_cell_stages=list(settings["super_n_cell_stages"]),
        stride_stages=list(settings["super_stride_stages"]),
        conv_candidates=list(settings["conv_candidates"]),
        n_classes=int(num_classes),
        width_mult=float(settings["width_mult"]),
        bn_param=(0.1, 1e-3),
        dropout_rate=0.0,
        first_block_candidates=list(settings["first_block_candidates"]),
    )
    _patch_first_conv_for_grayscale(super_net, input_channels)

    target_hardware = settings["target_hardware"]
    ref_value = settings["ref_value"]
    if target_hardware is None:
        ref_value = None

    grad_reg_loss_params = None
    if settings["grad_reg_loss_type"] == "mul#log":
        grad_reg_loss_params = {
            "alpha": float(settings["grad_reg_loss_alpha"]),
            "beta": float(settings["grad_reg_loss_beta"]),
        }

    arch_search_config = GradientArchSearchConfig(
        arch_init_type="normal",
        arch_init_ratio=1e-3,
        arch_opt_type="adam",
        arch_lr=float(settings["arch_lr"]),
        arch_opt_param={"betas": (0.0, 0.999), "eps": 1e-8},
        arch_weight_decay=float(settings["arch_weight_decay"]),
        target_hardware=target_hardware,
        ref_value=ref_value,
        grad_update_arch_param_every=int(settings["grad_update_arch_param_every"]),
        grad_update_steps=int(settings["grad_update_steps"]),
        grad_binary_mode=str(settings["grad_binary_mode"]),
        grad_data_batch=None,
        grad_reg_loss_type=str(settings["grad_reg_loss_type"]),
        grad_reg_loss_params=grad_reg_loss_params,
    )

    class TrackingArchSearchRunManager(ArchSearchRunManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.epoch_history: list[dict[str, Any]] = []

        def validate(self):
            valid_res, flops, latency = super().validate()
            val_loss, val_top1, val_top5 = valid_res
            self.epoch_history.append(
                {
                    "loss": float(val_loss),
                    "top1": float(val_top1),
                    "top5": float(val_top5),
                    "flops": float(flops),
                    "latency_ms": float(latency),
                    "warmup": bool(self.warmup),
                }
            )
            return valid_res, flops, latency

    official_path.mkdir(parents=True, exist_ok=True)
    return TrackingArchSearchRunManager(
        str(official_path),
        super_net,
        run_config,
        arch_search_config,
    ), run_config, settings


def _architecture_id(encoding: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(dict(encoding), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"official_proxyless_{digest[:12]}"


def _evaluate_official_net(
    normal_net: nn.Module,
    val_loader,
    *,
    device: str,
    num_classes: int,
) -> dict[str, float]:
    model = normal_net
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to("cpu")
        device = "cpu"
    return evaluate_classifier(
        model,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=num_classes,
    )


def _check_feasibility(constraints: SearchConstraints, cost_estimator, cost_estimate) -> bool:
    checker = BaseSearcher(search_space=None, cost_estimator=cost_estimator, constraints=constraints)
    return checker.check_feasibility(cost_estimate)


def run_official_proxyless_search(
    *,
    config_path: str,
    device: str = "cuda",
    resume: bool = False,
) -> Path:
    _ensure_proxyless_import_path()
    config = load_config(config_path)
    project_cfg = config.get("project") or {}
    dataset_cfg = config.get("dataset") or {}
    search_cfg = config.get("search") or {}
    training_cfg = config.get("training") or {}

    output_root = Path(project_cfg.get("output_dir", "results"))
    run_name = project_cfg.get("run_name")
    selection_metric = str(search_cfg.get("selection_metric", "macro_f1"))
    objective_weights = search_cfg.get("objective_weights") or {}

    tracker = ExperimentTracker(
        output_root,
        project_name=str(project_cfg.get("name", "official-proxyless")),
        script_name="run_official_proxyless_nksid",
        search_method="official_proxyless",
        dataset_name=str(dataset_cfg.get("name", "nksid")),
        run_name=run_name,
        resume=resume,
    )
    summary_path = tracker.results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "completed":
            print(f"[official_proxyless] already completed: {tracker.root_dir}")
            return tracker.root_dir

    tracker.save_config(config, {"config": config_path, "device": device, "resume": resume})
    official_path = tracker.root_dir / "official_proxyless_internal"
    settings = _resolve_official_settings(config)

    batch_size = int(training_cfg.get("batch_size", 32))
    image_size = int(dataset_cfg.get("image_size", 224))
    input_channels = int(dataset_cfg.get("input_channels", 1))
    fold = int(dataset_cfg.get("fold", 0))
    num_workers = int(dataset_cfg.get("num_workers", 8))
    data_dir = str(dataset_cfg.get("data_dir", "data/NKSID"))

    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    train_loader, val_loader, class_weights, num_classes = create_data_pipeline(
        dataset_name=str(dataset_cfg.get("name", "nksid")),
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=dataset_cfg.get("num_classes"),
        input_channels=input_channels,
        fold=fold,
        num_workers=num_workers,
        device=device,
    )
    search_space = build_search_space(
        config,
        image_size=image_size,
        input_channels=input_channels,
        num_classes=num_classes,
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=hardware_spec,
        constraints=constraints,
    )
    estimator.reset_lut_stats()
    data_provider = HwnasLoaderDataProvider(
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=val_loader,
        input_channels=input_channels,
        n_classes=num_classes,
        image_size=image_size,
        data_dir=data_dir,
    )

    tracker.write_dataset_summary(
        {
            "dataset": dataset_cfg.get("name"),
            "data_dir": data_dir,
            "fold": fold,
            "num_classes": num_classes,
            "input_channels": input_channels,
            "image_size": image_size,
            "class_weights": class_weights.tolist() if class_weights is not None else None,
        }
    )
    tracker.write_search_space_summary(
        {
            "backend": "reference/ProxylessNAS",
            "official_settings": settings,
            "hwnas_search_space_family": config.get("search_space", {}).get("macro_type"),
        }
    )

    manual_seed = int(project_cfg.get("seed", 42))
    manager, _, _ = _build_arch_search_manager(
        official_path=official_path,
        settings=settings,
        num_classes=num_classes,
        input_channels=input_channels,
        image_size=image_size,
        data_provider=data_provider,
        manual_seed=manual_seed,
    )

    with tracker.capture_console():
        print(f"[official_proxyless] run_dir={tracker.root_dir}")
        print(f"[official_proxyless] internal_path={official_path}")
        print(f"[official_proxyless] warmup_epochs={settings['warmup_epochs']} search_epochs={settings['n_epochs']}")
        print(f"[official_proxyless] target_hardware={settings['target_hardware']} ref_value={settings['ref_value']}")

        if resume:
            manager.load_model()
        if manager.warmup:
            manager.warm_up(warmup_epochs=int(settings["warmup_epochs"]))
        manager.train(fix_net_weights=False)

        normal_net = manager.net.cpu().convert_to_normal_net()
        net_config = normal_net.config
        architecture = proxyless_net_config_to_architecture(
            net_config,
            input_channels=input_channels,
            num_classes=num_classes,
            width_stages=settings["width_stages"],
            n_cell_stages=settings["n_cell_stages"],
            stride_stages=settings["stride_stages"],
            stem_stride=int(config.get("search_space", {}).get("stem_stride", 2)),
            post_stem_downsample_stride=int(
                config.get("search_space", {}).get("post_stem_downsample_stride", 1)
            ),
            head_conv_channels=config.get("search_space", {}).get("head_conv_channels"),
            skip_fixed_first_block=bool(settings["bridge_skip_fixed_first_block"]),
        )
        encoding = architecture.to_dict()
        arch_id = _architecture_id(encoding)

        eval_summary = _evaluate_official_net(
            normal_net,
            val_loader,
            device=device,
            num_classes=num_classes,
        )
        cost_estimate = estimator.estimate(architecture, search_space)
        feasible = _check_feasibility(constraints, estimator, cost_estimate)

        metrics = CandidateMetrics(
            accuracy=eval_summary.get("top1"),
            macro_f1=eval_summary.get("macro_f1"),
            weighted_f1=eval_summary.get("weighted_f1"),
            top1=eval_summary.get("top1"),
            top5=eval_summary.get("top5"),
            latency_ms=cost_estimate.latency_ms,
            energy_mj=cost_estimate.energy_mj,
            lut=cost_estimate.resource_lut,
            bram=cost_estimate.resource_bram,
            dsp=cost_estimate.resource_dsp,
            power_w=cost_estimate.power_w,
            memory_bandwidth_gbps=cost_estimate.memory_bandwidth_gbps,
            offchip_mem_mb=cost_estimate.offchip_mem_mb,
        )
        candidate = SearchCandidate(arch_id=arch_id, encoding=encoding, metrics=metrics)

        tracker.record_candidate(
            candidate,
            feasible=feasible,
            cost_estimate=cost_estimate,
            history={"official_epoch_history": manager.epoch_history},
            extra={
                "backend": "reference/ProxylessNAS",
                "official_net_config": net_config,
                "selection_metric": selection_metric,
            },
        )
        tracker.save_best_candidate(
            candidate,
            history={"official_epoch_history": manager.epoch_history},
            extra={
                "backend": "reference/ProxylessNAS",
                "official_net_config_path": str(official_path / "learned_net" / "net.config"),
                "selection_metric": selection_metric,
            },
        )

        candidates = [candidate]
        feasible_candidates = [candidate] if feasible else []
        infeasible_candidates = [] if feasible else [candidate]

        pareto_cfg = search_cfg.get("pareto") or {}
        pareto_topk = int(pareto_cfg.get("topk", 8))
        pareto_method = pareto_cfg.get("selection_method", "rank")
        pareto_objectives, pareto_directions = build_pareto_objectives(
            objective_weights,
            constraints,
            selection_metric=selection_metric,
        )
        pareto_front = compute_pareto_front(
            feasible_candidates,
            objectives=pareto_objectives,
            directions=pareto_directions,
        )
        selector = ParetoFrontSelector(
            objectives=pareto_objectives,
            directions=pareto_directions,
            selection_method=pareto_method,
        )
        pareto = selector.select(feasible_candidates, k=pareto_topk)
        pareto_ranks = compute_pareto_ranks(
            feasible_candidates,
            objectives=pareto_objectives,
            directions=pareto_directions,
        )
        hypervolume = 0.0
        if pareto_front:
            reference_point = selector._default_reference_point(pareto_front)
            hypervolume = compute_hypervolume(
                pareto_front,
                reference_point=reference_point,
                objectives=pareto_objectives,
                directions=pareto_directions,
            )

        pareto_summary = build_pareto_selection_summary(
            candidates=feasible_candidates,
            pareto_front=pareto_front,
            selected_candidates=pareto,
            ranks=pareto_ranks,
            objectives=pareto_objectives,
            directions=pareto_directions,
            selection_method=pareto_method,
            topk=pareto_topk,
            hypervolume=hypervolume,
        )
        write_pareto_selection_artifacts(tracker.results_dir, pareto_summary)

        lut_stats = estimator.get_lut_stats()
        (tracker.results_dir / "lut_stats.json").write_text(
            json.dumps(lut_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        status = "completed" if feasible else "failed"
        if not feasible:
            raise RuntimeError(
                "Official ProxylessNAS result is infeasible under strict board LUT constraints: "
                f"{cost_estimate.violations}"
            )

        tracker.finalize(
            status=status,
            candidates=candidates,
            feasible_candidates=feasible_candidates,
            infeasible_candidates=infeasible_candidates,
            best_candidate=candidate,
            pareto_candidates=pareto_front,
            extra={
                "selection_metric": selection_metric,
                "backend": "reference/ProxylessNAS",
                "official_settings": settings,
                "best_selection_score": candidate_selection_score(candidate, selection_metric),
            },
        )

        print(f"[official_proxyless] best {selection_metric}={candidate_selection_score(candidate, selection_metric):.4f}")
        print(f"[official_proxyless] strict LUT hits={lut_stats.get('hits')} true_misses={lut_stats.get('true_misses')}")

    return tracker.root_dir


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Official ProxylessNAS NKSID search")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    try:
        run_official_proxyless_search(
            config_path=args.config,
            device=args.device,
            resume=args.resume,
        )
        return 0
    except Exception as exc:
        print(f"[official_proxyless] failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
