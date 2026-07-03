"""Architecture encoding and legality checks for the search space."""

from .space import (
    ArchitectureSpec,
    BlockSpec,
    FAMILY_PROFILES,
    ResolvedBlockSpec,
    SearchSpace,
    SearchSpaceConfig,
    SONAR_OPS,
    StageSpec,
    list_family_profiles,
)
from .probe import ProbeRecord, probe_search_space, summarize_probe_records

__all__ = [
    "ArchitectureSpec",
    "BlockSpec",
    "FAMILY_PROFILES",
    "ProbeRecord",
    "ResolvedBlockSpec",
    "SearchSpace",
    "SearchSpaceConfig",
    "SONAR_OPS",
    "StageSpec",
    "list_family_profiles",
    "probe_search_space",
    "summarize_probe_records",
]
