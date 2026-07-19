"""Evidence-bounded HW-PR-NAS paper-spec helpers.

The pinned author repository is incomplete, so this module deliberately does
not claim to execute the author implementation.  It implements only the
paper-described Pareto-rank target and listwise ranking loss, plus a local
tabular architecture encoding for contract smoke tests.  Formal benchmark use
requires an explicit protocol amendment and must retain this provenance label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate
from hwnas_fpga.search.pareto import compute_pareto_ranks


HWPR_PINNED_COMMIT = "296c6576fbae2b277e56c704ff3b6e648ec4c2be"
HWPR_README_EXPECTED_FILES = (
    "surrogate_models/base_surrogate.py",
    "surrogate_models/accuracy_predictor.py",
    "surrogate_models/latency_predictor.py",
    "surrogate_models/energy_predictor.py",
)


def audit_hwpr_author_runtime(checkout: str | Path) -> dict[str, Any]:
    """Return static blockers that prevent calling the checkout author runtime."""

    root = Path(checkout).resolve()
    missing = [name for name in HWPR_README_EXPECTED_FILES if not (root / name).is_file()]
    search_text = (
        (root / "search_algo.py").read_text(encoding="utf-8", errors="replace")
        if (root / "search_algo.py").is_file()
        else ""
    )
    test_text = (
        (root / "test.py").read_text(encoding="utf-8", errors="replace")
        if (root / "test.py").is_file()
        else ""
    )
    blockers = []
    if missing:
        blockers.append("readme_listed_runtime_files_missing")
    if "valid_loss = valid_loss()" in search_text:
        blockers.append("undefined_valid_loss_call")
    if "evolution_search(args)" in test_text:
        blockers.append("test_entrypoint_signature_mismatch")
    return {
        "checkout": str(root),
        "author_runtime_ready": not blockers,
        "missing_readme_expected_files": missing,
        "blockers": blockers,
        "allowed_local_mode": "paper_spec_reimplementation_nonclaimable_smoke",
    }


def listmle_pareto_loss(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """ListMLE loss for ascending Pareto rank, following paper equations 7--8.

    Rank 0 is best.  Stable sorting makes tied-rank ordering deterministic; the
    paper does not specify a separate tie likelihood, so no unreported tie term
    is added here.
    """

    if scores.ndim != 1 or ranks.ndim != 1 or scores.numel() != ranks.numel():
        raise ValueError("scores and ranks must be equal-length one-dimensional tensors")
    if scores.numel() < 2:
        raise ValueError("listwise Pareto ranking loss needs at least two candidates")
    order = torch.argsort(ranks, stable=True)
    ordered = scores[order]
    suffix_logsumexp = torch.flip(
        torch.logcumsumexp(torch.flip(ordered, dims=(0,)), dim=0),
        dims=(0,),
    )
    return (suffix_logsumexp - ordered).mean()


class PaperSpecParetoRankMLP(nn.Module):
    """Three-layer local predictor matching only the paper's FCNN depth."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def encode_local_architectures(
    encodings: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode project stage-based architectures for a non-claimable smoke.

    This is intentionally labelled a local tabular encoding.  It is not the
    paper's missing feature+GCN+LSTM encoder.
    """

    if not encodings:
        raise ValueError("at least one architecture encoding is required")
    op_vocab = sorted(
        {
            str(block.get("op", "unknown"))
            for encoding in encodings
            for stage in encoding.get("stages", [])
            for block in stage.get("blocks", [])
        }
    )
    if not op_vocab:
        op_vocab = ["unknown"]
    max_stages = max(len(encoding.get("stages", [])) for encoding in encodings)
    max_blocks = max(
        (
            len(stage.get("blocks", []))
            for encoding in encodings
            for stage in encoding.get("stages", [])
        ),
        default=0,
    )
    rows: list[list[float]] = []
    for encoding in encodings:
        row = [
            _number(encoding.get("input_channels")),
            _number(encoding.get("stem_channels")),
            _number(encoding.get("stem_stride")),
            _number(encoding.get("post_stem_downsample_stride")),
            _number(encoding.get("head_conv_channels")),
            _number(encoding.get("head_channels")),
        ]
        stages = list(encoding.get("stages", []))
        for stage_index in range(max_stages):
            stage = stages[stage_index] if stage_index < len(stages) else {}
            row.extend(
                [
                    _number(stage.get("channels")),
                    _number(stage.get("depth")),
                    _number(stage.get("stride")),
                ]
            )
            blocks = list(stage.get("blocks", []))
            for block_index in range(max_blocks):
                block = blocks[block_index] if block_index < len(blocks) else {}
                op = str(block.get("op", "unknown"))
                row.extend(1.0 if op == candidate else 0.0 for candidate in op_vocab)
                row.extend(
                    [
                        _number(block.get("kernel_size")),
                        _number(block.get("expand_ratio")),
                        _number(block.get("stride")),
                    ]
                )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), {
        "encoding": "local_stage_tabular_v1",
        "paper_encoder_equivalent": False,
        "op_vocab": op_vocab,
        "max_stages": max_stages,
        "max_blocks_per_stage": max_blocks,
        "feature_dim": len(rows[0]),
    }


@dataclass(frozen=True)
class PaperSpecFitResult:
    ranks: list[int]
    scores: list[float]
    ordered_arch_ids: list[str]
    final_loss: float
    feature_schema: Mapping[str, Any]


def fit_paper_spec_surrogate(
    candidates: Sequence[SearchCandidate],
    *,
    objectives: Sequence[str],
    directions: Sequence[str],
    epochs: int = 100,
    learning_rate: float = 1e-2,
    seed: int = 42,
) -> PaperSpecFitResult:
    """Fit the bounded paper-spec loss on already evaluated candidates."""

    if len(candidates) < 2:
        raise ValueError("surrogate smoke needs at least two evaluated candidates")
    if len(objectives) != len(directions):
        raise ValueError("objectives and directions must have the same length")
    for candidate in candidates:
        missing = [
            name for name in objectives if getattr(candidate.metrics, name, None) is None
        ]
        if missing:
            raise ValueError(f"{candidate.arch_id}: missing objectives {missing}")

    matrix, schema = encode_local_architectures([candidate.encoding for candidate in candidates])
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    matrix = (matrix - mean) / np.where(std > 1e-8, std, 1.0)
    features = torch.from_numpy(matrix)
    ranks_list = compute_pareto_ranks(
        list(candidates), list(objectives), list(directions)
    )
    ranks = torch.tensor(ranks_list, dtype=torch.long)

    torch.manual_seed(int(seed))
    model = PaperSpecParetoRankMLP(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    final_loss = float("nan")
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        scores = model(features)
        loss = listmle_pareto_loss(scores, ranks)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    with torch.no_grad():
        scores_list = [float(value) for value in model(features).tolist()]
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-scores_list[index], candidates[index].arch_id),
    )
    return PaperSpecFitResult(
        ranks=ranks_list,
        scores=scores_list,
        ordered_arch_ids=[candidates[index].arch_id for index in order],
        final_loss=final_loss,
        feature_schema=schema,
    )


def candidate_from_record(record: Mapping[str, Any]) -> SearchCandidate:
    payload = dict(record.get("candidate") or record)
    return SearchCandidate(
        arch_id=str(payload["arch_id"]),
        encoding=dict(payload.get("encoding") or {}),
        metrics=CandidateMetrics(**dict(payload.get("metrics") or {})),
    )
