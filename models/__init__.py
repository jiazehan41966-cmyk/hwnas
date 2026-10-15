"""Genotype -> static PyTorch model; no training/search/backend imports."""
from .elastic import build_model, StaticElasticNeckDetector
__all__ = ["build_model", "StaticElasticNeckDetector"]
