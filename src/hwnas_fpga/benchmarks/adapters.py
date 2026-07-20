"""Unified interface for in-repository and external paper baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import subprocess
from typing import Any, Callable, Mapping, Sequence


CLAIMABILITY_STATES = {
    "CLAIMABLE",
    "EXPLORATORY",
    "PENDING",
    "FROZEN",
    "NOT_MEASURED",
    "OFFICIAL_CODE_NOT_AVAILABLE",
}


@dataclass(frozen=True)
class AdapterContext:
    campaign_id: str
    paper_id: str
    method: str
    task: str
    fold: int
    seed: int
    data_dir: str
    output_dir: str
    config_sha: str
    data_sha: str
    split_sha: str
    code_commit: str | None
    code_state_sha: str
    claimability_status: str = "PENDING"

    def validate(self) -> None:
        if self.task not in {
            "closed_set",
            "open_long_tail",
            "nas_search",
            "hls_proxy",
            "artifact_workflow_reference",
        }:
            raise ValueError(f"unsupported task: {self.task}")
        if self.claimability_status not in CLAIMABILITY_STATES:
            raise ValueError(f"invalid claimability_status: {self.claimability_status}")
        for name in ("config_sha", "data_sha", "split_sha", "code_state_sha"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA256 hex digest")


@dataclass(frozen=True)
class PredictionRecord:
    campaign_id: str
    paper_id: str
    method: str
    fold: int
    seed: int
    sample_id: str
    target: int
    prediction: int
    confidence: float
    checkpoint_sha: str
    config_sha: str
    data_sha: str
    split_sha: str
    code_commit: str | None
    code_state_sha: str
    claimability_status: str
    logits: Sequence[float] | None = None
    unknown_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAdapter(ABC):
    """Required lifecycle for every paper baseline."""

    adapter_id: str

    @abstractmethod
    def prepare(self, context: AdapterContext) -> Mapping[str, Any]:
        """Validate environment, source pin, license state, and inputs."""

    @abstractmethod
    def fit(self, context: AdapterContext, **kwargs: Any) -> Mapping[str, Any]:
        """Train under the frozen inner-selection protocol."""

    @abstractmethod
    def predict(self, context: AdapterContext, split: str, **kwargs: Any) -> list[PredictionRecord]:
        """Produce standardized per-sample predictions."""

    @abstractmethod
    def export_hardware(self, context: AdapterContext, **kwargs: Any) -> Mapping[str, Any]:
        """Export ONNX/INT8/HLS inputs with hashes, or an explicit blocker."""

    @abstractmethod
    def collect_provenance(self, context: AdapterContext) -> Mapping[str, Any]:
        """Return source, environment, command, and artifact provenance."""


class BuiltinModelAdapter(BenchmarkAdapter):
    """Adapter that delegates model building/training to the canonical runner."""

    adapter_id = "builtin"

    def __init__(self, model_factory: Callable[..., Any] | None = None):
        self.model_factory = model_factory

    def prepare(self, context: AdapterContext) -> Mapping[str, Any]:
        context.validate()
        return {"adapter_id": self.adapter_id, "ready": True, "task": context.task}

    def build_model(self, **kwargs: Any) -> Any:
        if self.model_factory is None:
            raise RuntimeError("BuiltinModelAdapter requires a model_factory")
        return self.model_factory(**kwargs)

    def fit(self, context: AdapterContext, **kwargs: Any) -> Mapping[str, Any]:
        fit_fn = kwargs.get("fit_fn")
        if not callable(fit_fn):
            raise RuntimeError("canonical runner must supply fit_fn")
        return fit_fn()

    def predict(self, context: AdapterContext, split: str, **kwargs: Any) -> list[PredictionRecord]:
        predict_fn = kwargs.get("predict_fn")
        if not callable(predict_fn):
            raise RuntimeError("canonical runner must supply predict_fn")
        rows = predict_fn(split)
        return [row if isinstance(row, PredictionRecord) else PredictionRecord(**row) for row in rows]

    def export_hardware(self, context: AdapterContext, **kwargs: Any) -> Mapping[str, Any]:
        export_fn = kwargs.get("export_fn")
        if not callable(export_fn):
            return {"status": "PENDING", "blocker": "hardware_export_not_requested"}
        return export_fn()

    def collect_provenance(self, context: AdapterContext) -> Mapping[str, Any]:
        return {"adapter_id": self.adapter_id, "context": asdict(context)}


class ExternalCommandAdapter(BenchmarkAdapter):
    """Run a pinned author's adapter command without using a shell.

    The command must write standardized prediction JSONL. This bridge keeps
    execution inside ``run_eval_protocol.py`` while allowing each paper to use
    an isolated Python environment. It never interprets author-reported paper
    numbers as local results.
    """

    adapter_id = "external_command"

    def __init__(
        self,
        *,
        checkout: str | Path,
        python_executable: str | Path,
        fit_command: Sequence[str],
        predict_command: Sequence[str],
    ):
        self.checkout = Path(checkout).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.fit_command = list(fit_command)
        self.predict_command = list(predict_command)
        self.executed_commands: list[dict[str, Any]] = []

    def _format(self, command: Sequence[str], context: AdapterContext, **extra: Any) -> list[str]:
        values = {**asdict(context), **extra}
        return [str(token).format(**values) for token in command]

    def _run(self, command: Sequence[str], context: AdapterContext, **extra: Any) -> dict[str, Any]:
        formatted = self._format(command, context, **extra)
        argv = [str(self.python_executable), *formatted]
        environment = os.environ.copy()
        environment["HWNAS_BENCHMARK_CONTEXT"] = json.dumps(asdict(context), sort_keys=True)
        result = subprocess.run(
            argv,
            cwd=self.checkout,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        record = {
            "argv": argv,
            "cwd": str(self.checkout),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        self.executed_commands.append(record)
        if result.returncode != 0:
            raise RuntimeError(
                f"external adapter failed with {result.returncode}: {result.stderr[-1000:]}"
            )
        return record

    def prepare(self, context: AdapterContext) -> Mapping[str, Any]:
        context.validate()
        blockers = []
        if not self.checkout.is_dir():
            blockers.append("checkout_missing")
        if not self.python_executable.is_file():
            blockers.append("python_environment_missing")
        return {"adapter_id": self.adapter_id, "ready": not blockers, "blockers": blockers}

    def fit(self, context: AdapterContext, **kwargs: Any) -> Mapping[str, Any]:
        return self._run(self.fit_command, context, **kwargs)

    def predict(self, context: AdapterContext, split: str, **kwargs: Any) -> list[PredictionRecord]:
        prediction_path = Path(kwargs["prediction_path"]).resolve()
        self._run(
            self.predict_command,
            context,
            split=split,
            prediction_path=str(prediction_path),
            **{key: value for key, value in kwargs.items() if key != "prediction_path"},
        )
        rows = []
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(PredictionRecord(**json.loads(line)))
        return rows

    def export_hardware(self, context: AdapterContext, **kwargs: Any) -> Mapping[str, Any]:
        return {"status": "PENDING", "blocker": "external_export_command_not_configured"}

    def collect_provenance(self, context: AdapterContext) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "checkout": str(self.checkout),
            "python_executable": str(self.python_executable),
            "commands": self.executed_commands,
            "context": asdict(context),
        }
