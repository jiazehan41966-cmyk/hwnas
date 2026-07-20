"""Pareto-based multi-objective aging evolution for HW-NAS.

The search does not collapse task quality and hardware cost into one fixed
weighted reward.  Feasible candidates are ordered by non-dominated rank and
crowding distance; the oldest population member is evicted after each feasible
child is admitted.  An external Pareto archive retains non-dominated candidates
even after they age out of the parent population.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import SearchCandidate, SearchConstraints
from hwnas_fpga.search.pareto import (
    compute_crowding_distances,
    compute_pareto_front,
    compute_pareto_ranks,
    resolve_pareto_objectives,
)
from hwnas_fpga.search.searcher import RandomSearcher, candidate_selection_score
from hwnas_fpga.search_space import ArchitectureSpec, SearchSpace

if TYPE_CHECKING:
    from hwnas_fpga.experiment import ExperimentTracker


def architecture_signature(architecture: ArchitectureSpec | Mapping[str, Any]) -> str:
    payload = architecture.to_dict() if isinstance(architecture, ArchitectureSpec) else dict(architecture)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def crowding_distances(
    candidates: Sequence[SearchCandidate],
    objectives: Sequence[str],
) -> dict[str, float]:
    """Backward-compatible alias for the shared NSGA-II implementation."""

    return compute_crowding_distances(list(candidates), list(objectives))


class AgingEvolutionSearcher(RandomSearcher):
    """Multi-objective regularized evolution with an external Pareto archive."""

    def __init__(
        self,
        search_space: SearchSpace,
        cost_estimator: FPGACostEstimator,
        constraints: Optional[SearchConstraints] = None,
        *,
        seed: int = 42,
        eval_early_stopping_patience: Optional[int] = 2,
        selection_metric: str = "macro_f1",
        objective_weights: Optional[dict[str, float]] = None,
        pareto_config: Optional[dict[str, Any]] = None,
        robustness_config: Optional[Mapping[str, Any]] = None,
        population_size: int = 32,
        sample_size: int = 8,
        random_injection_probability: float = 0.10,
        crossover_probability: float = 0.50,
        survivor_selection: str = "pareto_crowding",
        max_mutation_attempts: int = 32,
        prefer_lightweight_initialization: bool = False,
    ) -> None:
        super().__init__(
            search_space=search_space,
            cost_estimator=cost_estimator,
            constraints=constraints,
            seed=seed,
            eval_early_stopping_patience=eval_early_stopping_patience,
            selection_metric=selection_metric,
            robustness_config=robustness_config,
        )
        if population_size <= 1:
            raise ValueError("population_size must be greater than one")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not 0.0 <= random_injection_probability <= 1.0:
            raise ValueError("random_injection_probability must be in [0, 1]")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0, 1]")
        if survivor_selection not in {"pareto_crowding", "oldest"}:
            raise ValueError("survivor_selection must be 'pareto_crowding' or 'oldest'")
        self.population_size = int(population_size)
        self.sample_size = min(int(sample_size), self.population_size)
        self.random_injection_probability = float(random_injection_probability)
        self.crossover_probability = float(crossover_probability)
        self.survivor_selection = str(survivor_selection)
        self.max_mutation_attempts = max(1, int(max_mutation_attempts))
        self.prefer_lightweight_initialization = bool(prefer_lightweight_initialization)
        self.objectives, self.directions = resolve_pareto_objectives(
            pareto_config,
            objective_weights,
            constraints,
            selection_metric=selection_metric,
        )
        self.population: deque[SearchCandidate] = deque()
        self.birth_indices: dict[str, int] = {}
        self.pareto_archive: list[SearchCandidate] = []
        self.seen_signatures: set[str] = set()
        self.best_candidate: Optional[SearchCandidate] = None
        self.space_exhausted = False

    def sample_candidate(self) -> ArchitectureSpec:
        return self.search_space.sample(
            seed=None,
            rng=self.rng,
            cost_estimator=self.estimator,
            apply_pruning=True,
            require_feasible=True,
            max_feasible_attempts=32,
            prefer_lightweight=self.prefer_lightweight_initialization,
        )

    def _unseen_random_architecture(self) -> Optional[ArchitectureSpec]:
        for _ in range(self.max_mutation_attempts):
            architecture = self.sample_candidate()
            if architecture_signature(architecture) not in self.seen_signatures:
                return architecture
        return None

    def mutate_architecture(
        self,
        parent: ArchitectureSpec,
    ) -> tuple[Optional[ArchitectureSpec], dict[str, Any]]:
        """Mutate one stage or block, falling back to an unseen random injection."""

        parent_signature = architecture_signature(parent)
        for attempt in range(1, self.max_mutation_attempts + 1):
            donor = self.sample_candidate()
            parent_payload = parent.to_dict()
            donor_payload = donor.to_dict()
            block_sites = [
                (stage_index, block_index)
                for stage_index, (left, right) in enumerate(
                    zip(parent_payload["stages"], donor_payload["stages"])
                )
                if len(left["blocks"]) == len(right["blocks"])
                for block_index in range(len(left["blocks"]))
            ]
            mutation_kinds = ["stage"] + (["block"] if block_sites else [])
            mutation_kind = self.rng.choice(mutation_kinds)
            if mutation_kind == "stage":
                stage_index = self.rng.randrange(len(parent_payload["stages"]))
                parent_payload["stages"][stage_index] = donor_payload["stages"][stage_index]
                location: dict[str, Any] = {"stage_index": stage_index}
            else:
                stage_index, block_index = self.rng.choice(block_sites)
                parent_payload["stages"][stage_index]["blocks"][block_index] = (
                    donor_payload["stages"][stage_index]["blocks"][block_index]
                )
                location = {
                    "stage_index": stage_index,
                    "block_index": block_index,
                }
            candidate = ArchitectureSpec.from_dict(parent_payload)
            signature = architecture_signature(candidate)
            if signature == parent_signature or signature in self.seen_signatures:
                continue
            validation_errors = self.search_space.validate(candidate)
            if validation_errors:
                continue
            return candidate, {
                "kind": mutation_kind,
                "attempt": attempt,
                **location,
            }

        fallback = self._unseen_random_architecture()
        if fallback is None:
            return None, {
                "kind": "space_exhausted",
                "attempt": self.max_mutation_attempts,
            }
        return fallback, {
            "kind": "random_fallback",
            "attempt": self.max_mutation_attempts,
        }

    def crossover_architectures(
        self,
        parent_a: ArchitectureSpec,
        parent_b: ArchitectureSpec,
    ) -> tuple[Optional[ArchitectureSpec], dict[str, Any]]:
        """Uniformly inherit stages from two parents and keep only valid unseen children."""

        signature_a = architecture_signature(parent_a)
        signature_b = architecture_signature(parent_b)
        if signature_a == signature_b:
            return None, {"kind": "crossover_failed", "reason": "identical_parents"}

        payload_a = parent_a.to_dict()
        payload_b = parent_b.to_dict()
        if len(payload_a.get("stages", [])) != len(payload_b.get("stages", [])):
            return None, {"kind": "crossover_failed", "reason": "stage_count_mismatch"}

        differing_stages = [
            index
            for index, (stage_a, stage_b) in enumerate(
                zip(payload_a["stages"], payload_b["stages"])
            )
            if stage_a != stage_b
        ]
        for attempt in range(1, self.max_mutation_attempts + 1):
            child_payload = parent_a.to_dict()
            inherited_stages = [
                index for index in differing_stages if self.rng.random() < 0.5
            ]
            if not inherited_stages and differing_stages:
                inherited_stages = [self.rng.choice(differing_stages)]
            for stage_index in inherited_stages:
                child_payload["stages"][stage_index] = payload_b["stages"][stage_index]

            child = ArchitectureSpec.from_dict(child_payload)
            signature = architecture_signature(child)
            if signature in {signature_a, signature_b} or signature in self.seen_signatures:
                continue
            if self.search_space.validate(child):
                continue
            return child, {
                "kind": "crossover",
                "operator": "two_parent_uniform_stage",
                "attempt": attempt,
                "inherited_from_parent_b_stage_indices": inherited_stages,
            }

        return None, {
            "kind": "crossover_failed",
            "reason": "no_valid_unseen_child",
            "attempt": self.max_mutation_attempts,
        }

    def _update_archive(self) -> None:
        self.pareto_archive = compute_pareto_front(
            list(self.feasible_candidates),
            objectives=list(self.objectives),
            directions=list(self.directions),
        )

    def _trim_population(self) -> list[str]:
        """Apply the configured survivor rule and return removed architecture ids."""

        if len(self.population) <= self.population_size:
            return []
        if self.survivor_selection == "oldest":
            removed = self.population.popleft()
            self.birth_indices.pop(removed.arch_id, None)
            return [removed.arch_id]

        population = list(self.population)
        ranks = compute_pareto_ranks(
            population,
            objectives=list(self.objectives),
            directions=list(self.directions),
        )
        crowding_by_id: dict[str, float] = {}
        for rank in sorted(set(ranks)):
            front = [
                candidate
                for candidate, candidate_rank in zip(population, ranks)
                if candidate_rank == rank
            ]
            crowding_by_id.update(crowding_distances(front, self.objectives))
        ranked = sorted(
            zip(population, ranks),
            key=lambda item: (
                item[1],
                -crowding_by_id.get(item[0].arch_id, 0.0),
                -self.birth_indices.get(item[0].arch_id, -1),
                item[0].arch_id,
            ),
        )
        survivor_ids = {
            candidate.arch_id for candidate, _ in ranked[: self.population_size]
        }
        removed_ids = [
            candidate.arch_id for candidate in population if candidate.arch_id not in survivor_ids
        ]
        self.population = deque(
            candidate for candidate in population if candidate.arch_id in survivor_ids
        )
        for arch_id in removed_ids:
            self.birth_indices.pop(arch_id, None)
        return removed_ids

    def _primary_representative(self) -> Optional[SearchCandidate]:
        pool = self.pareto_archive or list(self.feasible_candidates)
        if not pool:
            return None
        return max(
            pool,
            key=lambda candidate: candidate_selection_score(candidate, self.selection_metric),
        )

    def _tournament_parent(
        self,
        *,
        exclude_arch_ids: Optional[set[str]] = None,
    ) -> SearchCandidate:
        if not self.population:
            raise RuntimeError("cannot select a parent from an empty population")
        excluded = exclude_arch_ids or set()
        population = [
            candidate for candidate in self.population if candidate.arch_id not in excluded
        ]
        if not population:
            raise RuntimeError("no eligible parent remains after exclusions")
        tournament = self.rng.sample(
            population,
            k=min(self.sample_size, len(population)),
        )
        ranks = compute_pareto_ranks(
            tournament,
            objectives=list(self.objectives),
            directions=list(self.directions),
        )
        best_rank = min(ranks)
        front = [candidate for candidate, rank in zip(tournament, ranks) if rank == best_rank]
        distances = crowding_distances(front, self.objectives)
        best_distance = max(distances[candidate.arch_id] for candidate in front)
        diverse = [
            candidate
            for candidate in front
            if distances[candidate.arch_id] == best_distance
        ]
        return self.rng.choice(diverse)

    def restore_state(
        self,
        records: Sequence[Mapping[str, Any]],
        checkpoint: Mapping[str, Any],
    ) -> int:
        """Restore a resumable evolution state without retraining old candidates."""

        from hwnas_fpga.interfaces import CandidateMetrics

        self.evaluated_candidates.clear()
        self.feasible_candidates.clear()
        self.infeasible_candidates.clear()
        by_id: dict[str, SearchCandidate] = {}
        for record in records:
            payload = record.get("candidate") or {}
            candidate = SearchCandidate(
                arch_id=str(payload["arch_id"]),
                encoding=dict(payload["encoding"]),
                metrics=CandidateMetrics(**dict(payload.get("metrics") or {})),
            )
            by_id[candidate.arch_id] = candidate
            self.evaluated_candidates.append(candidate)
            if record.get("feasible", True):
                self.feasible_candidates.append(candidate)
            else:
                self.infeasible_candidates.append(candidate)
            self.seen_signatures.add(architecture_signature(candidate.encoding))

        self.population.clear()
        for row in checkpoint.get("population", []):
            arch_id = str(row.get("arch_id", ""))
            candidate = by_id.get(arch_id)
            if candidate is not None:
                self.population.append(candidate)
                self.birth_indices[arch_id] = int(row.get("birth_index", 0))
        self._update_archive()
        self.best_candidate = self._primary_representative()
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)
        self.space_exhausted = bool(checkpoint.get("space_exhausted", False))
        return int(checkpoint.get("next_iteration", len(self.evaluated_candidates)))

    def _checkpoint_payload(self, *, next_iteration: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "next_iteration": int(next_iteration),
            "rng_state": self.rng.getstate(),
            "population": [
                {
                    "arch_id": candidate.arch_id,
                    "birth_index": self.birth_indices.get(candidate.arch_id),
                }
                for candidate in self.population
            ],
            "pareto_archive_ids": [candidate.arch_id for candidate in self.pareto_archive],
            "seen_signature_count": len(self.seen_signatures),
            "space_exhausted": self.space_exhausted,
            "crossover_probability": self.crossover_probability,
            "survivor_selection": self.survivor_selection,
            "objectives": list(self.objectives),
            "directions": list(self.directions),
        }

    def search(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        num_candidates: int = 100,
        train_epochs: int = 3,
        device: str = "cpu",
        verbose: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        artifact_tracker: Optional["ExperimentTracker"] = None,
        start_iteration: int = 0,
        timeout_minutes: Optional[float] = None,
    ) -> Optional[SearchCandidate]:
        """Run count- or time-bounded multi-objective aging evolution."""

        started = perf_counter()
        timeout_seconds = None if timeout_minutes is None else float(timeout_minutes) * 60.0
        for iteration in range(int(start_iteration), int(num_candidates)):
            elapsed = perf_counter() - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                break

            parents: list[SearchCandidate] = []
            if len(self.population) < self.population_size:
                architecture = self._unseen_random_architecture()
                variation = {"kind": "initialization"}
            elif self.rng.random() < self.random_injection_probability:
                architecture = self._unseen_random_architecture()
                variation = {"kind": "random_injection"}
            else:
                parent_a = self._tournament_parent()
                parents = [parent_a]
                architecture = None
                variation: dict[str, Any]
                if len(self.population) >= 2 and self.rng.random() < self.crossover_probability:
                    parent_b = self._tournament_parent(exclude_arch_ids={parent_a.arch_id})
                    parents.append(parent_b)
                    architecture, variation = self.crossover_architectures(
                        ArchitectureSpec.from_dict(parent_a.encoding),
                        ArchitectureSpec.from_dict(parent_b.encoding),
                    )
                    if architecture is None:
                        crossover_failure = variation
                        architecture, variation = self.mutate_architecture(
                            ArchitectureSpec.from_dict(parent_a.encoding)
                        )
                        variation = {
                            **variation,
                            "fallback_after": crossover_failure,
                        }
                else:
                    architecture, variation = self.mutate_architecture(
                        ArchitectureSpec.from_dict(parent_a.encoding)
                    )
            if architecture is None:
                self.space_exhausted = True
                if verbose:
                    print("Aging evolution stopped: no unseen architecture found.")
                break

            signature = architecture_signature(architecture)
            candidate, feasible = self.evaluate_candidate(
                architecture,
                train_loader,
                val_loader,
                num_classes,
                train_epochs=train_epochs,
                device=device,
                class_weights=class_weights,
            )
            candidate.arch_id = f"aging_arch_{iteration}"
            self.seen_signatures.add(signature)
            removed_arch_ids: list[str] = []
            if feasible:
                self.population.append(candidate)
                self.birth_indices[candidate.arch_id] = iteration
                removed_arch_ids = self._trim_population()
                self._update_archive()
                previous_best = self.best_candidate
                self.best_candidate = self._primary_representative()
                if (
                    artifact_tracker
                    and self.best_candidate is candidate
                    and self.best_candidate is not previous_best
                    and self.last_trained_model is not None
                ):
                    artifact_tracker.save_representative_candidate(
                        "accuracy_first_compatibility",
                        candidate,
                        model_state_dict=self.last_trained_model.state_dict(),
                        history=self.last_training_history,
                        extra={
                            "selection_metric": self.selection_metric,
                            "role": "accuracy_first_compatibility",
                            "not_an_overall_best": True,
                            "iteration": iteration,
                        },
                    )

            extra = {
                "iteration": iteration,
                "search_method": "aging_evolution",
                "architecture_signature": signature,
                "parent_arch_id": parents[0].arch_id if parents else None,
                "parent_arch_ids": [parent.arch_id for parent in parents],
                "variation": variation,
                "mutation": variation,
                "population_size": len(self.population),
                "population_capacity": self.population_size,
                "removed_arch_ids": removed_arch_ids,
                "removed_oldest_arch_id": (
                    removed_arch_ids[0]
                    if self.survivor_selection == "oldest" and removed_arch_ids
                    else None
                ),
                "survivor_selection": self.survivor_selection,
                "pareto_archive_size": len(self.pareto_archive),
                "elapsed_search_seconds": perf_counter() - started,
            }
            if artifact_tracker:
                artifact_tracker.record_candidate(
                    candidate,
                    feasible=feasible,
                    cost_estimate=self.last_cost_estimate,
                    history=self.last_training_history,
                    extra=extra,
                )
                artifact_tracker.save_named_checkpoint(
                    "aging_latest.pt",
                    self._checkpoint_payload(next_iteration=iteration + 1),
                )
                artifact_tracker.update_search_state(
                    total_evaluated=len(self.evaluated_candidates),
                    feasible=len(self.feasible_candidates),
                    infeasible=len(self.infeasible_candidates),
                    best_candidate=None,
                    extra={
                        **extra,
                        "mode": "timeout" if timeout_seconds is not None else "count",
                        "population": [candidate.arch_id for candidate in self.population],
                        "pareto_archive": [candidate.arch_id for candidate in self.pareto_archive],
                        "accuracy_first_compatibility_arch_id": (
                            None if self.best_candidate is None else self.best_candidate.arch_id
                        ),
                    },
                )
            if verbose:
                score = candidate_selection_score(candidate, self.selection_metric)
                print(
                    f"[{iteration + 1}/{num_candidates}] {candidate.arch_id}: "
                    f"feasible={feasible}, {self.selection_metric}={score:.4f}, "
                    f"population={len(self.population)}, pareto={len(self.pareto_archive)}"
                )

        if verbose:
            print("\nMulti-objective aging evolution completed!")
            print(f"Total evaluated: {len(self.evaluated_candidates)}")
            print(f"Feasible: {len(self.feasible_candidates)}")
            print(f"Pareto archive: {len(self.pareto_archive)}")
        return self.best_candidate

    def get_search_summary(self) -> dict[str, Any]:
        summary = super().get_search_summary()
        summary.update(
            {
                "population_size": len(self.population),
                "population_capacity": self.population_size,
                "pareto_archive_size": len(self.pareto_archive),
                "seen_signature_count": len(self.seen_signatures),
                "space_exhausted": self.space_exhausted,
                "objectives": list(self.objectives),
                "directions": list(self.directions),
                "crossover_probability": self.crossover_probability,
                "survivor_selection": self.survivor_selection,
            }
        )
        return summary
