"""Bound unchanged author MGD execution without editing its source snapshot.

Engineering guard: the pinned MODNAS Frank-Wolfe implementation never increments
its own iter_count.  Count visits to its while-condition line externally and
check a monotonic deadline at Python line boundaries in that author file.
Converging computations and their outputs remain the author's implementation.

This current-thread Python tracing guard cannot interrupt a blocked native Torch
or CUDA call.  A campaign supervisor must also enforce a process wall-time limit.
The active debugger/profiler trace is chained where present and restored on exit.
"""
from __future__ import annotations

import ast
import inspect
import math
import numbers
import sys
import textwrap
from time import perf_counter

from tools.modnas_elastic.author import MinNormSolver, combine_gradients, verify_author_sources


class MGDConvergenceError(TimeoutError):
    """Original author solver exceeded an external iteration or time bound."""

    def __init__(self, reason, iterations, max_iterations, deadline_seconds):
        self.reason = reason
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.deadline_seconds = deadline_seconds
        super().__init__(
            f"Author MGD {reason}: while visits={iterations}, "
            f"max_iterations={max_iterations}, deadline_seconds={deadline_seconds}"
        )


def _loop_line(solver):
    source, first_line = inspect.getsourcelines(solver)
    tree = ast.parse(textwrap.dedent("".join(source)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
    if len(loops) != 1:
        raise RuntimeError("Unexpected author FW loop structure; guard must be re-reviewed")
    return first_line + loops[0].lineno - 1


class _AuthorTraceGuard:
    def __init__(self, solver, max_iterations, deadline_seconds):
        self.solver_code = solver.__code__
        self.author_filename = self.solver_code.co_filename
        self.loop_line = _loop_line(solver)
        self.max_iterations = max_iterations
        self.deadline_seconds = deadline_seconds
        self.iterations = 0
        self.previous_trace = None
        self.previous_locals = {}

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        self.started = perf_counter()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *_):
        sys.settrace(self.previous_trace)
        self.previous_locals.clear()

    def _abort(self, reason):
        raise MGDConvergenceError(reason, self.iterations,
                                  self.max_iterations, self.deadline_seconds)

    def _trace(self, frame, event, arg):
        if event == "call":
            previous_local = self.previous_trace(frame, event, arg) if self.previous_trace else None
            if frame.f_code.co_filename != self.author_filename:
                # Other frames retain their previous tracing behavior.  Only
                # the author file needs line events for this engineering guard.
                return previous_local
            self.previous_locals[frame] = previous_local
        else:
            previous_local = self.previous_locals.get(frame)
            if previous_local is not None:
                replacement = previous_local(frame, event, arg)
                if replacement is not None:
                    self.previous_locals[frame] = replacement
        if event == "line":
            if perf_counter() - self.started > self.deadline_seconds:
                self._abort("DEADLINE_EXCEEDED")
            if frame.f_code is self.solver_code and frame.f_lineno == self.loop_line:
                self.iterations += 1
                if self.iterations > self.max_iterations:
                    self._abort("ITERATION_LIMIT_EXCEEDED")
        elif event == "return":
            self.previous_locals.pop(frame, None)
        return self._trace


def bounded_combine_gradients(gradient_groups, normalize=True, *,
                              max_iterations=300, deadline_seconds=5.0):
    """Run the unchanged author aggregation, aborting rather than fabricating a result.

    Returns (combined_parameter_gradients, numpy_convex_weights, squared_norm),
    exactly like the existing bridge.  Singleton/zero-gradient handling remains
    in that bridge and invokes no solver loop.  An abort must fail the experiment
    step; callers must not silently substitute mean or last-iterate gradients.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, numbers.Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, numbers.Real) or not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be finite and positive")
    verify_author_sources()
    solver = MinNormSolver.find_min_norm_element_FW
    with _AuthorTraceGuard(solver, int(max_iterations), float(deadline_seconds)):
        return combine_gradients(gradient_groups, normalize=normalize)


__all__ = ["MGDConvergenceError", "bounded_combine_gradients"]
