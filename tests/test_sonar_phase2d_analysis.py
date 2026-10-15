"""Independent numerical checks for diagnosis conventions, not synthetic research evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'tools/sonar_spec/phase2d_analysis.py'
spec=importlib.util.spec_from_file_location('phase2d_analysis',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_known_rank_inversion_and_ties():
    assert m.association([1,2,3],[3,2,1])['kendall']==-1
    assert m.association([1,1,2],[1,1,3])['spearman']==pytest.approx(1)
    assert m.association([1,1,1],[1,2,3])['spearman'] is None


def test_paired_bootstrap_preserves_perfect_negative_association():
    result=m.bootstrap(np.arange(12),-np.arange(12),repeats=200)
    for stat in result.values():
        assert stat['estimate']==pytest.approx(-1)
        assert stat['ci95']==pytest.approx([-1,-1])


def test_near_ties_are_counted_as_pairs_and_adjacent_gaps_separately():
    d=m.distribution([0,.0003,.0006,1])
    assert d['near_tie_pairs_le_0.001']==3
    assert d['adjacent_gaps_le_0.001']==2
    assert d['exact_tie_pairs']==0
