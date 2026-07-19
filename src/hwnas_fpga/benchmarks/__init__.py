"""Evidence-aligned benchmark infrastructure for external research baselines."""

from .adapters import (
    AdapterContext,
    BenchmarkAdapter,
    BuiltinModelAdapter,
    ExternalCommandAdapter,
    PredictionRecord,
)
from .archive import CampaignPaths, validate_prediction_record
from .metrics import (
    ObjectiveSpec,
    calibration_summary,
    exact_hypervolume,
    normalize_objective_rows,
    open_set_summary,
)
from .registry import BenchmarkPaper, audit_source_checkout, load_paper_registry

__all__ = [
    "AdapterContext",
    "BenchmarkAdapter",
    "BenchmarkPaper",
    "BuiltinModelAdapter",
    "CampaignPaths",
    "ExternalCommandAdapter",
    "ObjectiveSpec",
    "PredictionRecord",
    "audit_source_checkout",
    "calibration_summary",
    "exact_hypervolume",
    "load_paper_registry",
    "normalize_objective_rows",
    "open_set_summary",
    "validate_prediction_record",
]
