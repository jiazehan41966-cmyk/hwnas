"""Backward-compatible alias for the HLS backend module."""

from .hls_backend import HLSProjectConfig, attach_hls_report, create_hls_project_stub

__all__ = [
    "HLSProjectConfig",
    "attach_hls_report",
    "create_hls_project_stub",
]
