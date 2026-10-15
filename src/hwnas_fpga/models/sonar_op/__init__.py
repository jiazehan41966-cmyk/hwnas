"""Isolated SonarOp V0.2-P0 prototype; not part of frozen Search Space V0.1."""

from .sonar_dw_reparam import ReparamDualRFDepthwise
from .candidate import (
    A1_ARCHITECTURE,
    A2_ARCHITECTURE,
    A3_ID,
    SONAROP_VERSION,
    SonarOpCandidateDetector,
    build_sonarop_candidate,
    fold_standard_stage7_depthwise,
    switch_candidate_to_deploy,
)

__all__ = [
    "ReparamDualRFDepthwise",
    "A1_ARCHITECTURE",
    "A2_ARCHITECTURE",
    "A3_ID",
    "SONAROP_VERSION",
    "SonarOpCandidateDetector",
    "build_sonarop_candidate",
    "fold_standard_stage7_depthwise",
    "switch_candidate_to_deploy",
]
