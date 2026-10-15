"""Check real author solver parity, external aborts, and trace restoration."""
import sys

import numpy as np
import pytest
import torch

from tools.modnas_elastic.author import combine_gradients
from tools.modnas_comparison import safety


def _orthogonal(device="cpu"):
    return [[row.clone()] for row in torch.eye(3, device=device)]


def _assert_same(left, right):
    assert np.array_equal(left[1], right[1])
    assert left[2] == right[2]
    assert len(left[0]) == len(right[0])
    assert all(torch.equal(a, b) for a, b in zip(left[0], right[0]))


@pytest.mark.parametrize("normalize", [False, True])
def test_converging_real_solver_has_exact_output_parity(normalize):
    groups = _orthogonal()
    expected = combine_gradients(groups, normalize=normalize)
    previous = sys.gettrace()
    actual = safety.bounded_combine_gradients(groups, normalize=normalize)
    _assert_same(actual, expected)
    assert sys.gettrace() is previous


def test_actual_solver_iteration_overrun_aborts_and_restores_previous_trace():
    # Three orthogonal tasks need another loop after their first FW update.
    # A one-iteration allowance thus exercises the real unmodified solver's
    # overrun, rather than testing a fake replacement algorithm.
    events = []

    def existing_trace(frame, event, arg):
        if frame.f_code.co_name == "find_min_norm_element_FW":
            events.append(event)
        return existing_trace

    previous = sys.gettrace()
    sys.settrace(existing_trace)
    try:
        with pytest.raises(safety.MGDConvergenceError) as captured:
            safety.bounded_combine_gradients(_orthogonal(), max_iterations=1)
        assert captured.value.reason == "ITERATION_LIMIT_EXCEEDED"
        assert captured.value.iterations == 2
        assert sys.gettrace() is existing_trace
        assert "call" in events and "line" in events
    finally:
        sys.settrace(previous)


def test_deadline_abort_restores_trace(monkeypatch):
    times = iter([0.0, 10.0])
    monkeypatch.setattr(safety, "perf_counter", lambda: next(times))
    previous = sys.gettrace()
    with pytest.raises(safety.MGDConvergenceError) as captured:
        safety.bounded_combine_gradients(_orthogonal(), deadline_seconds=5.0)
    assert captured.value.reason == "DEADLINE_EXCEEDED"
    assert sys.gettrace() is previous


def test_singleton_and_zero_gradient_match_existing_bridge():
    groups = [[torch.tensor([3., 4.])]]
    _assert_same(safety.bounded_combine_gradients(groups, max_iterations=1), combine_gradients(groups))
    zeros = [[torch.zeros(2)], [torch.tensor([3., 4.])]]
    _assert_same(safety.bounded_combine_gradients(zeros, max_iterations=1), combine_gradients(zeros))


def test_bridge_failure_restores_previous_trace():
    previous = sys.gettrace()
    with pytest.raises(ValueError):
        safety.bounded_combine_gradients([[torch.tensor([float("nan")])]])
    assert sys.gettrace() is previous


@pytest.mark.parametrize("kwargs", [
    {"max_iterations": 0}, {"max_iterations": 1.5}, {"max_iterations": True},
    {"deadline_seconds": 0}, {"deadline_seconds": float("inf")},
    {"deadline_seconds": float("nan")},
])
def test_invalid_limits_rejected_without_replacing_trace(kwargs):
    previous = sys.gettrace()
    with pytest.raises(ValueError):
        safety.bounded_combine_gradients(_orthogonal(), **kwargs)
    assert sys.gettrace() is previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_gradients_have_exact_author_parity():
    groups = _orthogonal("cuda")
    _assert_same(safety.bounded_combine_gradients(groups), combine_gradients(groups))
