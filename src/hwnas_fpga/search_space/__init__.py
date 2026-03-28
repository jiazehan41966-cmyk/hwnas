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

__all__ = [
    "ArchitectureSpec",
    "BlockSpec",
    "FAMILY_PROFILES",
    "ResolvedBlockSpec",
    "SearchSpace",
    "SearchSpaceConfig",
    "SONAR_OPS",
    "StageSpec",
    "list_family_profiles",
]
