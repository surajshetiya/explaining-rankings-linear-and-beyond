"""Duplicate-aware importance sampling plus Kendall local search.

This experimental max-agreement heuristic is intended for data sets containing
many repeated feature rows.  It first collapses exact duplicate rows into point
classes and counts the full *directed class-comparison* population.  A sampled
edge is then drawn from the mixture

    q(e) = alpha * W_e / P + beta / |E| + gamma * H_e / sum(H),

where ``W_e`` is the number of original item comparisons represented by edge
``e``, ``P`` is the number of optimizable item comparisons, and ``H_e`` favors
small-margin edges under a linearized-MRC pilot.  The objective component keeps
faithfulness to the original Kendall measure, the class-balanced component
prevents large duplicate classes from hiding rare comparisons, and the hard
component spends draws near the pilot boundary.  Inverse-probability weights
``(W_e / P) / q(e)`` correct the biased proposal before local search.

The local optimizer uses weighted great-circle event sweeps in the span of the
sampled class differences.  Finalists and their antipodal reverses are scored
on the complete item-level comparison population.  Consequently the method is
a heuristic rather than the manuscript PTAS, but it retains the deterministic
1/2 maximum-agreement floor.  Exact duplicate rows remain tied in the output;
target ties are excluded; strict target comparisons inside one duplicate class
remain unavoidable disagreements.

For small k, an exact class-edge table gives the mixture distribution directly.
It needs O(k^2) int64 storage, independent of the number of original rows.
``max_class_matrix_entries`` is an automatic-switch threshold: above it, the
algorithm uses exact objective draws plus on-demand class-pair multiplicities
and never constructs the table.  The sparse proposal includes rejected class
pairs as null observations, which keeps its importance probabilities exact.

Dependencies: NumPy and the shared ranking modules in this directory.  Gurobi
is not used by this implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support package import and direct execution from Code/.
    from . import closest_linear_ranking_gurobi as _core
    from . import hybrid_linear_ranking_gurobi as _hybrid
    from . import linear_ranking_baselines as _baseline
    from . import max_agreement_linear_ranking_gurobi as _max
    from . import sampled_kendall_local_search_gurobi as _local
except ImportError:
    import closest_linear_ranking_gurobi as _core
    import hybrid_linear_ranking_gurobi as _hybrid
    import linear_ranking_baselines as _baseline
    import max_agreement_linear_ranking_gurobi as _max
    import sampled_kendall_local_search_gurobi as _local


_DEFAULT_SAMPLE_SIZE = 200
_DEFAULT_RESTARTS = 8
_DEFAULT_MAX_ITERATIONS = 250
_DEFAULT_PATIENCE = 40
_DEFAULT_GAUSSIAN_INTERVAL = 10
_DEFAULT_MAX_SAMPLE_SIZE = 1_000_000
_DEFAULT_MAX_CLASS_MATRIX_ENTRIES = 4_000_000
_DEFAULT_LMRC_CG_TOLERANCE = 1e-6
_DEFAULT_LMRC_CG_MAX_ITERATIONS = 50
_DEFAULT_OBJECTIVE_MIXTURE_WEIGHT = 0.50
_DEFAULT_CLASS_MIXTURE_WEIGHT = 0.25
_DEFAULT_HARD_MIXTURE_WEIGHT = 0.25
_HARDNESS_FLOOR = 0.05
_QUALITY_TOLERANCE_MULTIPLIER = 256.0
_TOP_SWEEP_INTERVALS_TO_VERIFY = 8
_SEED_SALT = 0x3C6EF372

__all__ = [
    "DuplicateAwareSampledKendallResult",
    "max_agreement_duplicate_aware_sampled_local_search",
]


@dataclass(frozen=True)
class DuplicateAwareSampledKendallResult:
    """Result of the duplicate-aware sampled Kendall local search."""

    ranking: np.ndarray
    direction: np.ndarray
    conditioned_scores: np.ndarray
    target_order: np.ndarray
    target_tie_ids: np.ndarray
    ranking_tie_ids: np.ndarray
    kendall_agreement: int
    kendall_tau: int
    total_comparisons: int
    forced_tie_disagreements: int
    optimizable_comparisons: int
    point_class_count: int
    positive_class_comparisons: int | None
    sampling_mode: str
    sample_size: int
    accepted_sample_size: int
    null_sample_size: int
    unique_sample_size: int
    unique_class_sample_size: int
    sampled_comparisons: np.ndarray
    sampled_class_comparisons: np.ndarray
    sample_multiplicities: np.ndarray
    sample_population_multiplicities: np.ndarray
    sample_proposal_probabilities: np.ndarray
    sample_importance_weights: np.ndarray
    best_empirical_agreement: float
    best_empirical_disagreement: float
    selected_empirical_agreement: float
    selected_empirical_disagreement: float
    initial_lmrc_empirical_agreement: float | None
    initial_lmrc_empirical_disagreement: float | None
    objective_mixture_weight: float
    class_mixture_weight: float
    hard_mixture_weight: float
    effective_sample_size: float
    sampled_class_coverage: float
    sample_collision_rate: float
    edge_multiplicity_cache_size: int
    pilot_margin_scale: float
    epsilon: float
    delta: float
    eta: float
    dimension_bound: int
    sample_dimension: int
    sample_condition_number: float
    sample_rank_tolerance: float
    sample_projection_used: bool
    theoretical_sample_size: int
    sampling_bound_met: bool
    restarts: int
    max_iterations_per_restart: int
    performed_iterations: int
    great_circle_sweeps: int
    strict_improvements: int
    plateau_moves: int
    full_candidate_evaluations: int
    candidate_count: int
    certificate_source: str
    certificate_restart: int
    certificate_orientation: int
    chosen_source: str
    chosen_restart: int
    chosen_orientation: int
    numerical_margin: float
    projected_instance: bool
    class_graph_runtime_seconds: float
    sampling_runtime_seconds: float
    optimization_runtime_seconds: float
    solver_runtime_seconds: float
    total_runtime_seconds: float
    restart_diagnostics: tuple[dict[str, Any], ...]

    @property
    def guaranteed_fraction_of_optimum(self) -> float:
        """Return the deterministic guarantee from antipodal full scoring."""

        return 1.0 if self.optimizable_comparisons == 0 else 0.5

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": "duplicate_aware_sampled_kendall_local_search",
            "sampling_strategy": (
                "directed point-class mixture with inverse-probability weights"
            ),
            "sampling_mode": self.sampling_mode,
            "ranking": self.ranking.tolist(),
            "direction": self.direction.tolist(),
            "conditioned_scores": self.conditioned_scores.tolist(),
            "target_order": self.target_order.tolist(),
            "target_tie_ids": self.target_tie_ids.tolist(),
            "ranking_tie_ids": self.ranking_tie_ids.tolist(),
            "kendall_agreement": self.kendall_agreement,
            "kendall_tau": self.kendall_tau,
            "total_comparisons": self.total_comparisons,
            "forced_tie_disagreements": self.forced_tie_disagreements,
            "optimizable_comparisons": self.optimizable_comparisons,
            "point_class_count": self.point_class_count,
            "positive_class_comparisons": self.positive_class_comparisons,
            "sample_size": self.sample_size,
            "accepted_sample_size": self.accepted_sample_size,
            "null_sample_size": self.null_sample_size,
            "unique_sample_size": self.unique_sample_size,
            "unique_class_sample_size": self.unique_class_sample_size,
            "sampled_comparisons": self.sampled_comparisons.tolist(),
            "sampled_class_comparisons": self.sampled_class_comparisons.tolist(),
            "sample_multiplicities": self.sample_multiplicities.tolist(),
            "sample_population_multiplicities": (
                self.sample_population_multiplicities.tolist()
            ),
            "sample_proposal_probabilities": (
                self.sample_proposal_probabilities.tolist()
            ),
            "sample_importance_weights": self.sample_importance_weights.tolist(),
            "best_empirical_agreement": self.best_empirical_agreement,
            "best_empirical_disagreement": self.best_empirical_disagreement,
            "selected_empirical_agreement": self.selected_empirical_agreement,
            "selected_empirical_disagreement": self.selected_empirical_disagreement,
            "initial_lmrc_empirical_agreement": (
                self.initial_lmrc_empirical_agreement
            ),
            "initial_lmrc_empirical_disagreement": (
                self.initial_lmrc_empirical_disagreement
            ),
            "objective_mixture_weight": self.objective_mixture_weight,
            "class_mixture_weight": self.class_mixture_weight,
            "hard_mixture_weight": self.hard_mixture_weight,
            "effective_sample_size": self.effective_sample_size,
            "sampled_class_coverage": self.sampled_class_coverage,
            "sample_collision_rate": self.sample_collision_rate,
            "edge_multiplicity_cache_size": self.edge_multiplicity_cache_size,
            "pilot_margin_scale": self.pilot_margin_scale,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "eta": self.eta,
            "dimension_bound": self.dimension_bound,
            "sample_dimension": self.sample_dimension,
            "sample_condition_number": self.sample_condition_number,
            "sample_rank_tolerance": self.sample_rank_tolerance,
            "sample_projection_used": self.sample_projection_used,
            "theoretical_sample_size": self.theoretical_sample_size,
            "sampling_bound_met": self.sampling_bound_met,
            "guaranteed_fraction_of_optimum": self.guaranteed_fraction_of_optimum,
            "restarts": self.restarts,
            "max_iterations_per_restart": self.max_iterations_per_restart,
            "performed_iterations": self.performed_iterations,
            "great_circle_sweeps": self.great_circle_sweeps,
            "strict_improvements": self.strict_improvements,
            "plateau_moves": self.plateau_moves,
            "full_candidate_evaluations": self.full_candidate_evaluations,
            "candidate_count": self.candidate_count,
            "certificate_source": self.certificate_source,
            "certificate_restart": self.certificate_restart,
            "certificate_orientation": self.certificate_orientation,
            "chosen_source": self.chosen_source,
            "chosen_restart": self.chosen_restart,
            "chosen_orientation": self.chosen_orientation,
            "numerical_margin": self.numerical_margin,
            "projected_instance": self.projected_instance,
            "class_graph_runtime_seconds": self.class_graph_runtime_seconds,
            "sampling_runtime_seconds": self.sampling_runtime_seconds,
            "optimization_runtime_seconds": self.optimization_runtime_seconds,
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "restart_diagnostics": [dict(item) for item in self.restart_diagnostics],
        }


@dataclass(frozen=True)
class _WeightedSearchRun:
    reduced_direction: np.ndarray
    margins: np.ndarray
    agreement: float
    robust_margin: float
    performed_iterations: int
    great_circle_sweeps: int
    strict_improvements: int
    plateau_moves: int
    stagnant_sweeps: int
    neighborhood_counts: dict[str, int]
    seed_source: str
    runtime_seconds: float


@dataclass(frozen=True)
class _Candidate:
    class_order: np.ndarray
    model_direction: np.ndarray
    class_scores: np.ndarray
    numerical_margin: float
    full_agreement: int
    full_disagreement: int
    empirical_agreement: float
    source: str
    restart: int
    orientation: int


@dataclass(frozen=True)
class _ImportanceSample:
    class_pairs: np.ndarray
    draw_counts: np.ndarray
    population_weights: np.ndarray
    proposal_probabilities: np.ndarray
    importance_weights: np.ndarray
    effective_sample_size: float
    accepted_sample_size: int
    null_sample_size: int
    edge_multiplicity_cache_size: int
    mixture: tuple[float, float, float]
    pilot_margin_scale: float


def _validate_mixture(
    objective_weight: float,
    class_weight: float,
    hard_weight: float,
) -> tuple[float, float, float]:
    values = np.asarray(
        (objective_weight, class_weight, hard_weight), dtype=np.float64
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("sampling mixture weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("at least one sampling mixture weight must be positive")
    normalized = values / total
    return float(normalized[0]), float(normalized[1]), float(normalized[2])


def _build_class_edge_population(
    ranking_input: _core._RankingInput,
    max_class_matrix_entries: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Count original item comparisons represented by every directed class edge."""

    started = time.perf_counter()
    class_count = ranking_input.point_class_count
    entries = class_count * class_count
    if max_class_matrix_entries is not None and entries > max_class_matrix_entries:
        raise ValueError(
            f"the duplicate-aware class table needs {entries} entries for "
            f"{class_count} distinct point classes, exceeding "
            f"max_class_matrix_entries={max_class_matrix_entries}; increase the "
            "guard only if the k-by-k int64 table fits memory, or use the ordinary "
            "sampled local-search implementation for a mostly unique data set"
        )

    weights = np.zeros((class_count, class_count), dtype=np.int64)
    seen = np.zeros(class_count, dtype=np.int64)
    class_ids = ranking_input.point_class_ids_order
    tie_ids = ranking_input.target_tie_ids_order
    n = int(class_ids.size)
    start = 0
    while start < n:
        end = start + 1
        while end < n and tie_ids[end] == tie_ids[start]:
            end += 1
        current_classes, current_counts = np.unique(
            class_ids[start:end], return_counts=True
        )
        current_counts = current_counts.astype(np.int64, copy=False)
        # Each earlier item is strictly preferred to each current-group item.
        # Updating only occupied current columns costs O(k) per (group, class)
        # incidence instead of forming an O(k^2) outer product per target group.
        weights[:, current_classes] += seen[:, None] * current_counts[None, :]
        seen[current_classes] += current_counts
        start = end

    # Cross-target comparisons within an exact point class are structurally
    # forced ties, not optimizable class edges.
    np.fill_diagonal(weights, 0)
    higher, lower = np.nonzero(weights)
    edge_weights = weights[higher, lower].astype(np.int64, copy=True)
    class_pairs = np.column_stack((higher, lower)).astype(np.int64, copy=False)
    if int(np.sum(edge_weights, dtype=np.int64)) != int(
        ranking_input.pair_index.modeled_comparisons
    ):
        raise RuntimeError("directed class-edge population accounting failed")
    return class_pairs, edge_weights, time.perf_counter() - started


def _hardness_distribution(
    class_basis: np.ndarray,
    class_pairs: np.ndarray,
    pilot_direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return a normalized, bounded inverse-margin edge distribution."""

    class_scores = class_basis @ pilot_direction
    margins = np.abs(
        class_scores[class_pairs[:, 0]] - class_scores[class_pairs[:, 1]]
    )
    positive = margins[margins > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    hardness = 1.0 / (_HARDNESS_FLOOR + margins / scale)
    hardness_sum = float(np.sum(hardness, dtype=np.float64))
    if not math.isfinite(hardness_sum) or hardness_sum <= 0.0:
        return np.full(margins.size, 1.0 / margins.size), scale
    return hardness / hardness_sum, scale


def _draw_importance_sample(
    class_pairs: np.ndarray,
    population_weights: np.ndarray,
    optimizable_comparisons: int,
    pilot_hardness: np.ndarray,
    sample_size: int,
    mixture: tuple[float, float, float],
    pilot_margin_scale: float,
    rng: np.random.Generator,
) -> _ImportanceSample:
    """Draw and aggregate class edges, returning self-normalized IPW weights."""

    objective_weight, class_weight, hard_weight = mixture
    class_probability = 1.0 / class_pairs.shape[0]
    # Reuse the private hardness buffer for q to avoid two extra O(|E|)
    # float64 arrays.  The caller discards this buffer after sampling.
    proposal = pilot_hardness
    proposal *= hard_weight
    proposal += class_weight * class_probability
    proposal += population_weights * (
        objective_weight / float(optimizable_comparisons)
    )
    proposal_sum = float(np.sum(proposal, dtype=np.float64))
    proposal /= proposal_sum
    cumulative = np.cumsum(proposal, dtype=np.float64)
    cumulative[-1] = 1.0
    draws = np.searchsorted(cumulative, rng.random(sample_size), side="right")
    selected, draw_counts = np.unique(draws, return_counts=True)
    draw_counts = draw_counts.astype(np.int64, copy=False)

    selected_objective_probabilities = (
        population_weights[selected].astype(np.float64)
        / float(optimizable_comparisons)
    )
    ratios = selected_objective_probabilities / proposal[selected]
    raw_aggregated_weights = draw_counts.astype(np.float64) * ratios
    raw_total = float(np.sum(raw_aggregated_weights, dtype=np.float64))
    if not math.isfinite(raw_total) or raw_total <= 0.0:
        raise RuntimeError("importance sample has invalid aggregate weight")
    normalized_weights = raw_aggregated_weights * (sample_size / raw_total)

    # Treat repeated draws as separate observations for the standard
    # importance-weight effective sample-size diagnostic.
    squared_total = float(
        np.sum(draw_counts.astype(np.float64) * ratios * ratios, dtype=np.float64)
    )
    effective_size = raw_total * raw_total / squared_total
    return _ImportanceSample(
        class_pairs=class_pairs[selected],
        draw_counts=draw_counts,
        population_weights=population_weights[selected],
        proposal_probabilities=proposal[selected],
        importance_weights=normalized_weights,
        effective_sample_size=effective_size,
        accepted_sample_size=sample_size,
        null_sample_size=0,
        edge_multiplicity_cache_size=0,
        mixture=mixture,
        pilot_margin_scale=pilot_margin_scale,
    )


def _stratified_component_counts(
    sample_size: int,
    mixture: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Allocate deterministic component counts with largest remainders."""

    expected = sample_size * np.asarray(mixture, dtype=np.float64)
    counts = np.floor(expected).astype(np.int64)
    remaining = sample_size - int(np.sum(counts, dtype=np.int64))
    if remaining:
        fractional = expected - counts
        order = np.argsort(-fractional, kind="stable")
        counts[order[:remaining]] += 1

    # When requested, retain at least one exact objective draw.  Besides being
    # useful statistically, this guarantees a non-null sparse sample for very
    # small sample sizes under the usual mixture.
    if mixture[0] > 0.0 and counts[0] == 0:
        donors = np.flatnonzero(counts[1:] > 0) + 1
        if donors.size:
            donor = int(donors[np.argmax(counts[donors])])
            counts[donor] -= 1
            counts[0] += 1
    actual = counts.astype(np.float64) / sample_size
    return counts, (float(actual[0]), float(actual[1]), float(actual[2]))


def _random_ordered_class_pairs(
    class_count: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw uniformly from the k(k-1) ordered pairs without rejection."""

    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    first = rng.integers(0, class_count, size=count, dtype=np.int64)
    second = rng.integers(0, class_count - 1, size=count, dtype=np.int64)
    second += second >= first
    return np.column_stack((first, second)).astype(np.int64, copy=False)


def _class_pair_multiplicities(
    ranking_input: _core._RankingInput,
    first: int,
    second: int,
    cache: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Return both directed item multiplicities for one unordered class pair."""

    class_count = ranking_input.point_class_count
    lower_class = min(first, second)
    upper_class = max(first, second)
    key = lower_class * class_count + upper_class
    cached = cache.get(key)
    if cached is None:
        pair_index = ranking_input.pair_index
        tie_ids = ranking_input.target_tie_ids_order
        lower_positions = pair_index.class_positions[
            pair_index.class_offsets[lower_class] : pair_index.class_offsets[
                lower_class + 1
            ]
        ]
        upper_positions = pair_index.class_positions[
            pair_index.class_offsets[upper_class] : pair_index.class_offsets[
                upper_class + 1
            ]
        ]
        lower_ties = tie_ids[lower_positions]
        upper_ties = tie_ids[upper_positions]
        if lower_ties.size <= upper_ties.size:
            lower_before_upper = int(
                np.sum(
                    upper_ties.size
                    - np.searchsorted(upper_ties, lower_ties, side="right"),
                    dtype=np.int64,
                )
            )
            upper_before_lower = int(
                np.sum(
                    np.searchsorted(upper_ties, lower_ties, side="left"),
                    dtype=np.int64,
                )
            )
        else:
            lower_before_upper = int(
                np.sum(
                    np.searchsorted(lower_ties, upper_ties, side="left"),
                    dtype=np.int64,
                )
            )
            upper_before_lower = int(
                np.sum(
                    lower_ties.size
                    - np.searchsorted(lower_ties, upper_ties, side="right"),
                    dtype=np.int64,
                )
            )
        cached = (lower_before_upper, upper_before_lower)
        cache[key] = cached
    if first == lower_class:
        return cached
    return cached[1], cached[0]


def _sparse_pilot_margin_scale(
    class_scores: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    """Estimate the typical class margin without enumerating class pairs."""

    pilot_count = min(16_384, max(512, 4 * sample_size))
    pilot_pairs = _random_ordered_class_pairs(
        class_scores.size, pilot_count, rng
    )
    margins = np.abs(
        class_scores[pilot_pairs[:, 0]] - class_scores[pilot_pairs[:, 1]]
    )
    positive = margins[margins > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return scale if math.isfinite(scale) and scale > 0.0 else 1.0


def _hard_acceptance(
    class_scores: np.ndarray,
    class_pairs: np.ndarray,
    scale: float,
) -> np.ndarray:
    margins = np.abs(
        class_scores[class_pairs[:, 0]] - class_scores[class_pairs[:, 1]]
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        probabilities = scale / (scale + margins)
    return np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)


def _draw_sparse_importance_sample(
    ranking_input: _core._RankingInput,
    class_basis: np.ndarray,
    pilot_direction: np.ndarray,
    sample_size: int,
    mixture: tuple[float, float, float],
    rng: np.random.Generator,
) -> _ImportanceSample:
    """Sample class edges with exact on-demand probabilities and no k^2 table.

    Class-balanced attempts draw uniformly from all k(k-1) ordered class
    pairs.  A pair with zero directed target multiplicity is a null draw.  Hard
    attempts use the same proposal and accept with a known inverse-margin
    probability.  Keeping null attempts in the sampling experiment makes
    ``q(e)`` known exactly, so the balance-heuristic weight ``p(e)/q(e)`` still
    targets the original item-level Kendall distribution.
    """

    component_counts, actual_mixture = _stratified_component_counts(
        sample_size, mixture
    )
    objective_count, class_count_draws, hard_count = map(int, component_counts)
    class_count = ranking_input.point_class_count
    ordered_pair_count = class_count * (class_count - 1)
    class_scores = class_basis @ pilot_direction
    margin_scale = _sparse_pilot_margin_scale(class_scores, sample_size, rng)
    multiplicity_cache: dict[int, tuple[int, int]] = {}
    accepted_parts: list[np.ndarray] = []

    if objective_count:
        pair_ids = rng.integers(
            0,
            ranking_input.pair_index.modeled_comparisons,
            size=objective_count,
            dtype=np.int64,
        )
        target_pairs = ranking_input.pair_index.decode(pair_ids)
        accepted_parts.append(
            ranking_input.point_class_ids_order[target_pairs].astype(
                np.int64, copy=False
            )
        )

    null_count = 0
    if class_count_draws:
        proposed = _random_ordered_class_pairs(class_count, class_count_draws, rng)
        accepted_mask = np.zeros(class_count_draws, dtype=bool)
        for index, (first_raw, second_raw) in enumerate(proposed):
            forward, _ = _class_pair_multiplicities(
                ranking_input,
                int(first_raw),
                int(second_raw),
                multiplicity_cache,
            )
            accepted_mask[index] = forward > 0
        accepted_parts.append(proposed[accepted_mask])
        null_count += int(np.count_nonzero(~accepted_mask))

    if hard_count:
        proposed = _random_ordered_class_pairs(class_count, hard_count, rng)
        acceptance = _hard_acceptance(class_scores, proposed, margin_scale)
        accepted_mask = rng.random(hard_count) < acceptance
        candidate_indices = np.flatnonzero(accepted_mask)
        for index_raw in candidate_indices:
            index = int(index_raw)
            forward, _ = _class_pair_multiplicities(
                ranking_input,
                int(proposed[index, 0]),
                int(proposed[index, 1]),
                multiplicity_cache,
            )
            if forward == 0:
                accepted_mask[index] = False
        accepted_parts.append(proposed[accepted_mask])
        null_count += int(np.count_nonzero(~accepted_mask))

    accepted = (
        np.concatenate(accepted_parts, axis=0)
        if accepted_parts
        else np.empty((0, 2), dtype=np.int64)
    )
    if accepted.shape[0] == 0:
        raise ValueError(
            "the sparse class proposal produced no optimizable comparison; "
            "use a positive objective_mixture_weight or increase sample_size"
        )
    edge_keys = accepted[:, 0] * class_count + accepted[:, 1]
    unique_keys, draw_counts = np.unique(edge_keys, return_counts=True)
    class_pairs = np.column_stack(
        (unique_keys // class_count, unique_keys % class_count)
    ).astype(np.int64, copy=False)
    draw_counts = draw_counts.astype(np.int64, copy=False)

    population_weights = np.empty(unique_keys.size, dtype=np.int64)
    for index, (first_raw, second_raw) in enumerate(class_pairs):
        forward, _ = _class_pair_multiplicities(
            ranking_input,
            int(first_raw),
            int(second_raw),
            multiplicity_cache,
        )
        if forward <= 0:
            raise RuntimeError("a sparse sampled class edge has zero multiplicity")
        population_weights[index] = forward

    objective_probability = population_weights.astype(np.float64)
    objective_probability /= float(ranking_input.pair_index.modeled_comparisons)
    hard_acceptance = _hard_acceptance(class_scores, class_pairs, margin_scale)
    objective_fraction, class_fraction, hard_fraction = actual_mixture
    proposal = (
        objective_fraction * objective_probability
        + class_fraction / ordered_pair_count
        + hard_fraction * hard_acceptance / ordered_pair_count
    )
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise RuntimeError("sparse importance proposal probability is invalid")
    ratios = objective_probability / proposal
    raw_weights = draw_counts.astype(np.float64) * ratios
    raw_total = float(np.sum(raw_weights, dtype=np.float64))
    if not math.isfinite(raw_total) or raw_total <= 0.0:
        raise RuntimeError("sparse importance sample has invalid aggregate weight")
    importance_weights = raw_weights * (sample_size / raw_total)
    squared_total = float(
        np.sum(draw_counts.astype(np.float64) * ratios * ratios, dtype=np.float64)
    )
    effective_size = raw_total * raw_total / squared_total
    accepted_size = int(accepted.shape[0])
    if accepted_size + null_count != sample_size:
        raise RuntimeError("sparse proposal draw accounting failed")
    return _ImportanceSample(
        class_pairs=class_pairs,
        draw_counts=draw_counts,
        population_weights=population_weights,
        proposal_probabilities=proposal,
        importance_weights=importance_weights,
        effective_sample_size=effective_size,
        accepted_sample_size=accepted_size,
        null_sample_size=null_count,
        edge_multiplicity_cache_size=len(multiplicity_cache),
        mixture=actual_mixture,
        pilot_margin_scale=margin_scale,
    )


def _representative_item_comparisons(
    ranking_input: _core._RankingInput,
    class_pairs: np.ndarray,
) -> np.ndarray:
    """Choose one correctly oriented original-row witness for each class edge."""

    pair_index = ranking_input.pair_index
    tie_ids = ranking_input.target_tie_ids_order
    target_pairs = np.empty_like(class_pairs)
    for edge_index, (higher_raw, lower_raw) in enumerate(class_pairs):
        higher = int(higher_raw)
        lower = int(lower_raw)
        higher_positions = pair_index.class_positions[
            pair_index.class_offsets[higher] : pair_index.class_offsets[higher + 1]
        ]
        lower_positions = pair_index.class_positions[
            pair_index.class_offsets[lower] : pair_index.class_offsets[lower + 1]
        ]
        lower_ties = tie_ids[lower_positions]
        witness_found = False
        for higher_position_raw in higher_positions:
            higher_position = int(higher_position_raw)
            lower_index = int(
                np.searchsorted(lower_ties, tie_ids[higher_position], side="right")
            )
            if lower_index < lower_positions.size:
                target_pairs[edge_index] = (
                    higher_position,
                    int(lower_positions[lower_index]),
                )
                witness_found = True
                break
        if not witness_found:
            raise RuntimeError("positive directed class edge has no item witness")
    return ranking_input.target_order[target_pairs]


def _weighted_agreement(
    margins: np.ndarray,
    weights: np.ndarray,
    tie_satisfied: np.ndarray,
) -> float:
    satisfied = _local._satisfied_mask(margins, tie_satisfied)
    return float(np.sum(weights[satisfied], dtype=np.float64))


def _agreement_tolerance(left: float, right: float) -> float:
    return (
        _QUALITY_TOLERANCE_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, abs(left), abs(right))
    )


def _quality_better(
    agreement: float,
    robust_margin: float,
    incumbent_agreement: float,
    incumbent_robust_margin: float,
) -> bool:
    tolerance = _agreement_tolerance(agreement, incumbent_agreement)
    if agreement > incumbent_agreement + tolerance:
        return True
    if abs(agreement - incumbent_agreement) > tolerance:
        return False
    margin_tolerance = (
        _QUALITY_TOLERANCE_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, abs(robust_margin), abs(incumbent_robust_margin))
    )
    return robust_margin > incumbent_robust_margin + margin_tolerance


def _group_circular_events(
    rays: np.ndarray,
    deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort circular boundary rays and stably sum coincident float events."""

    x_coordinates = rays[:, 0]
    y_coordinates = rays[:, 1]
    halves = np.where(
        (y_coordinates > 0.0) | ((y_coordinates == 0.0) & (x_coordinates >= 0.0)),
        0,
        1,
    ).astype(np.int8, copy=False)
    primary = np.where(halves == 0, -x_coordinates, x_coordinates)
    secondary = np.where(
        halves == 0,
        np.where(x_coordinates >= 0.0, y_coordinates, -y_coordinates),
        np.where(x_coordinates <= 0.0, -y_coordinates, y_coordinates),
    )
    order = np.lexsort((secondary, primary, halves))
    sorted_rays = rays[order]
    sorted_deltas = deltas[order]

    grouped_rays: list[np.ndarray] = []
    grouped_deltas: list[float] = []
    start = 0
    while start < sorted_rays.shape[0]:
        end = start + 1
        while (
            end < sorted_rays.shape[0]
            and _local._compare_boundary_rays(sorted_rays[start], sorted_rays[end])
            == 0
        ):
            end += 1
        grouped_rays.append(sorted_rays[start].copy())
        grouped_deltas.append(
            math.fsum(float(value) for value in sorted_deltas[start:end])
        )
        start = end
    return np.asarray(grouped_rays), np.asarray(grouped_deltas, dtype=np.float64)


def _best_great_circle_move(
    direction: np.ndarray,
    margins: np.ndarray,
    tangent: np.ndarray,
    comparison_coordinates: np.ndarray,
    weights: np.ndarray,
    tie_satisfied: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Return the best directly verified weighted cell on a great circle."""

    tangent_margins = comparison_coordinates @ tangent
    amplitudes = np.hypot(margins, tangent_margins)
    active = amplitudes > 0.0
    if not np.any(active):
        agreement = _weighted_agreement(margins, weights, tie_satisfied)
        return direction, margins, agreement, _local._robust_margin(margins), 0

    first_rays = (
        np.column_stack((tangent_margins[active], -margins[active]))
        / amplitudes[active, None]
    )
    active_weights = weights[active]
    event_rays = np.vstack((first_rays, -first_rays))
    event_deltas = np.concatenate((active_weights, -active_weights))
    group_rays, group_deltas = _group_circular_events(event_rays, event_deltas)

    reference_ray = _local._cell_midpoint_ray(group_rays[-1], group_rays[0])
    reference_margins = margins * reference_ray[0] + tangent_margins * reference_ray[1]
    current_agreement = _weighted_agreement(
        reference_margins, weights, tie_satisfied
    )
    interval_records: list[tuple[float, float, np.ndarray]] = [
        (
            current_agreement,
            _local._cell_width_score(group_rays[-1], group_rays[0]),
            reference_ray,
        )
    ]
    for index, lower_ray in enumerate(group_rays[:-1]):
        current_agreement += float(group_deltas[index])
        upper_ray = group_rays[index + 1]
        interval_records.append(
            (
                current_agreement,
                _local._cell_width_score(lower_ray, upper_ray),
                _local._cell_midpoint_ray(lower_ray, upper_ray),
            )
        )
    interval_records.sort(key=lambda item: (item[0], item[1]), reverse=True)

    best_direction = direction
    best_margins = margins
    best_agreement = _weighted_agreement(margins, weights, tie_satisfied)
    best_robustness = _local._robust_margin(margins)
    for _, _, midpoint_ray in interval_records[:_TOP_SWEEP_INTERVALS_TO_VERIFY]:
        candidate = _local._normalized_reduced(
            direction * midpoint_ray[0] + tangent * midpoint_ray[1]
        )
        if candidate is None:
            continue
        candidate_margins = comparison_coordinates @ candidate
        candidate_agreement = _weighted_agreement(
            candidate_margins, weights, tie_satisfied
        )
        candidate_robustness = _local._robust_margin(candidate_margins)
        if _quality_better(
            candidate_agreement,
            candidate_robustness,
            best_agreement,
            best_robustness,
        ):
            best_direction = candidate
            best_margins = candidate_margins
            best_agreement = candidate_agreement
            best_robustness = candidate_robustness
    return (
        best_direction,
        best_margins,
        best_agreement,
        best_robustness,
        int(event_rays.shape[0]),
    )


def _run_weighted_local_search(
    comparison_coordinates: np.ndarray,
    weights: np.ndarray,
    tie_satisfied: np.ndarray,
    initial_direction: np.ndarray,
    *,
    seed_source: str,
    max_iterations: int,
    patience: int,
    gaussian_interval: int,
    rng: np.random.Generator,
) -> _WeightedSearchRun:
    started = time.perf_counter()
    direction = _local._normalized_reduced(initial_direction)
    if direction is None:
        raise RuntimeError("duplicate-aware local search received a zero seed")
    margins = comparison_coordinates @ direction
    agreement = _weighted_agreement(margins, weights, tie_satisfied)
    robustness = _local._robust_margin(margins)
    strict_improvements = 0
    plateau_moves = 0
    performed = 0
    sweeps = 0
    stagnant = 0
    neighborhood_counts = {
        "coordinate": 0,
        "violated_pair": 0,
        "violated_aggregate": 0,
        "gaussian": 0,
    }
    dimension = direction.size

    for iteration in range(max_iterations):
        if stagnant >= patience:
            break
        performed += 1
        satisfied = _local._satisfied_mask(margins, tie_satisfied)
        violated = np.flatnonzero(~satisfied)
        if (iteration + 1) % gaussian_interval == 0:
            raw_tangent = rng.standard_normal(dimension)
            neighborhood = "gaussian"
        elif (iteration + 1) % 5 == 0 and violated.size:
            raw_tangent = comparison_coordinates[violated].T @ weights[violated]
            neighborhood = "violated_aggregate"
        elif iteration % 2 == 1 and violated.size:
            probabilities = weights[violated].copy()
            probabilities /= float(np.sum(probabilities, dtype=np.float64))
            raw_tangent = comparison_coordinates[int(rng.choice(violated, p=probabilities))]
            neighborhood = "violated_pair"
        else:
            raw_tangent = np.zeros(dimension, dtype=np.float64)
            raw_tangent[int(rng.integers(0, dimension))] = 1.0
            neighborhood = "coordinate"

        tangent = _local._tangent_direction(raw_tangent, direction, rng)
        if tangent is None:
            stagnant += 1
            continue
        neighborhood_counts[neighborhood] += 1
        sweeps += 1
        (
            candidate_direction,
            candidate_margins,
            candidate_agreement,
            candidate_robustness,
            _,
        ) = _best_great_circle_move(
            direction,
            margins,
            tangent,
            comparison_coordinates,
            weights,
            tie_satisfied,
        )
        tolerance = _agreement_tolerance(candidate_agreement, agreement)
        if candidate_agreement > agreement + tolerance:
            direction = candidate_direction
            margins = candidate_margins
            agreement = candidate_agreement
            robustness = candidate_robustness
            strict_improvements += 1
            stagnant = 0
        elif _quality_better(
            candidate_agreement, candidate_robustness, agreement, robustness
        ):
            direction = candidate_direction
            margins = candidate_margins
            agreement = candidate_agreement
            robustness = candidate_robustness
            plateau_moves += 1
            stagnant += 1
        else:
            stagnant += 1

    checked_margins = comparison_coordinates @ direction
    checked_agreement = _weighted_agreement(
        checked_margins, weights, tie_satisfied
    )
    if not math.isclose(
        checked_agreement,
        agreement,
        rel_tol=0.0,
        abs_tol=_agreement_tolerance(checked_agreement, agreement),
    ):
        raise RuntimeError("duplicate-aware local-search accounting changed")
    return _WeightedSearchRun(
        reduced_direction=direction,
        margins=checked_margins,
        agreement=checked_agreement,
        robust_margin=_local._robust_margin(checked_margins),
        performed_iterations=performed,
        great_circle_sweeps=sweeps,
        strict_improvements=strict_improvements,
        plateau_moves=plateau_moves,
        stagnant_sweeps=stagnant,
        neighborhood_counts=neighborhood_counts,
        seed_source=seed_source,
        runtime_seconds=time.perf_counter() - started,
    )


def _empirical_agreement(
    class_order: np.ndarray,
    class_pairs: np.ndarray,
    weights: np.ndarray,
) -> float:
    positions = np.empty(class_order.size, dtype=np.int64)
    positions[class_order] = np.arange(class_order.size, dtype=np.int64)
    satisfied = positions[class_pairs[:, 0]] < positions[class_pairs[:, 1]]
    return float(np.sum(weights[satisfied], dtype=np.float64))


def _make_candidate(
    ranking_input: _core._RankingInput,
    class_basis: np.ndarray,
    class_pairs: np.ndarray,
    weights: np.ndarray,
    sample_differences: np.ndarray,
    weak_direction: np.ndarray,
    *,
    source: str,
    restart: int,
    seed: int,
    evaluation_cache: dict[bytes, tuple[int, int, int, float]],
) -> _Candidate:
    raw_margins = sample_differences @ weak_direction
    margin_scale = float(np.max(np.abs(raw_margins), initial=0.0))
    robust_threshold = (
        _QUALITY_TOLERANCE_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, margin_scale)
    )
    active = raw_margins > robust_threshold
    try:
        direction, class_scores, class_order, margin = (
            _hybrid._strict_generic_candidate(
                class_basis,
                sample_differences,
                class_pairs,
                active,
                weak_direction,
                seed=seed,
            )
        )
    except RuntimeError:
        # Near-zero sampled margins can change sign while a direction is
        # normalized.  They are numerical ties, not reliable constraints.
        # The solver-free weak refinement protects robust score gaps and gives
        # a directly verified strict class order instead of failing the run.
        direction, class_scores, class_order, margin = (
            _max._generic_weak_refinement_witness(
                class_basis,
                weak_direction,
                seed=seed,
            )
        )
    cache_key = class_order.tobytes()
    cached = evaluation_cache.get(cache_key)
    if cached is None:
        _, _, _, agreement, disagreement, total = _max._evaluate_class_order(
            ranking_input, class_order
        )
        if total != ranking_input.pair_index.target_comparisons:
            raise RuntimeError("duplicate-aware comparison population changed")
        empirical = _empirical_agreement(class_order, class_pairs, weights)
        cached = (agreement, disagreement, total, empirical)
        evaluation_cache[cache_key] = cached
    agreement, disagreement, _, empirical = cached
    return _Candidate(
        class_order=class_order,
        model_direction=direction,
        class_scores=class_scores,
        numerical_margin=margin,
        full_agreement=agreement,
        full_disagreement=disagreement,
        empirical_agreement=empirical,
        source=source,
        restart=restart,
        orientation=1,
    )


def _reverse_candidate(
    ranking_input: _core._RankingInput,
    forward: _Candidate,
    sample_weight_total: float,
) -> _Candidate:
    agreement = ranking_input.pair_index.modeled_comparisons - forward.full_agreement
    return _Candidate(
        class_order=forward.class_order[::-1].copy(),
        model_direction=-forward.model_direction,
        class_scores=-forward.class_scores,
        numerical_margin=forward.numerical_margin,
        full_agreement=agreement,
        full_disagreement=ranking_input.pair_index.target_comparisons - agreement,
        empirical_agreement=sample_weight_total - forward.empirical_agreement,
        source=forward.source,
        restart=forward.restart,
        orientation=-1,
    )


def _candidate_key(candidate: _Candidate) -> tuple[float, ...]:
    return (
        float(candidate.full_agreement),
        candidate.empirical_agreement,
        candidate.numerical_margin,
        1.0 if candidate.source == "lmrc" else 0.0,
        float(-candidate.restart),
        float(candidate.orientation),
    )


def _empirical_candidate_key(candidate: _Candidate) -> tuple[float, ...]:
    return (
        candidate.empirical_agreement,
        float(candidate.full_agreement),
        candidate.numerical_margin,
        float(-candidate.restart),
        float(candidate.orientation),
    )


def _empty_result(
    ranking_input: _core._RankingInput,
    *,
    epsilon: float,
    delta: float,
    mixture: tuple[float, float, float],
    sample_rank_tolerance: float,
    restarts: int,
    max_iterations: int,
    total_started: float,
) -> DuplicateAwareSampledKendallResult:
    flattened, prediction_ids, prediction_order = _core._all_tied_prediction(
        ranking_input
    )
    agreement, disagreement, total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    target = ranking_input.target_order
    elapsed = time.perf_counter() - total_started
    return DuplicateAwareSampledKendallResult(
        ranking=target[flattened],
        direction=np.zeros(ranking_input.points.shape[1]),
        conditioned_scores=np.zeros(ranking_input.points.shape[0]),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=agreement,
        kendall_tau=disagreement,
        total_comparisons=total,
        forced_tie_disagreements=ranking_input.pair_index.forced_tie_disagreements,
        optimizable_comparisons=0,
        point_class_count=ranking_input.point_class_count,
        positive_class_comparisons=0,
        sampling_mode="empty",
        sample_size=0,
        accepted_sample_size=0,
        null_sample_size=0,
        unique_sample_size=0,
        unique_class_sample_size=0,
        sampled_comparisons=np.empty((0, 2), dtype=np.int64),
        sampled_class_comparisons=np.empty((0, 2), dtype=np.int64),
        sample_multiplicities=np.empty(0, dtype=np.int64),
        sample_population_multiplicities=np.empty(0, dtype=np.int64),
        sample_proposal_probabilities=np.empty(0),
        sample_importance_weights=np.empty(0),
        best_empirical_agreement=0.0,
        best_empirical_disagreement=0.0,
        selected_empirical_agreement=0.0,
        selected_empirical_disagreement=0.0,
        initial_lmrc_empirical_agreement=None,
        initial_lmrc_empirical_disagreement=None,
        objective_mixture_weight=mixture[0],
        class_mixture_weight=mixture[1],
        hard_mixture_weight=mixture[2],
        effective_sample_size=0.0,
        sampled_class_coverage=0.0,
        sample_collision_rate=0.0,
        edge_multiplicity_cache_size=0,
        pilot_margin_scale=0.0,
        epsilon=epsilon,
        delta=delta,
        eta=epsilon / 4.0,
        dimension_bound=0,
        sample_dimension=0,
        sample_condition_number=1.0,
        sample_rank_tolerance=sample_rank_tolerance,
        sample_projection_used=False,
        theoretical_sample_size=0,
        sampling_bound_met=True,
        restarts=restarts,
        max_iterations_per_restart=max_iterations,
        performed_iterations=0,
        great_circle_sweeps=0,
        strict_improvements=0,
        plateau_moves=0,
        full_candidate_evaluations=0,
        candidate_count=0,
        certificate_source="empty",
        certificate_restart=-1,
        certificate_orientation=1,
        chosen_source="empty",
        chosen_restart=-1,
        chosen_orientation=1,
        numerical_margin=0.0,
        projected_instance=False,
        class_graph_runtime_seconds=0.0,
        sampling_runtime_seconds=0.0,
        optimization_runtime_seconds=0.0,
        solver_runtime_seconds=0.0,
        total_runtime_seconds=elapsed,
        restart_diagnostics=(),
    )


def max_agreement_duplicate_aware_sampled_local_search(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    epsilon: float,
    delta: float,
    sample_size: int | None = _DEFAULT_SAMPLE_SIZE,
    restarts: int = _DEFAULT_RESTARTS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    patience: int = _DEFAULT_PATIENCE,
    gaussian_interval: int = _DEFAULT_GAUSSIAN_INTERVAL,
    ridge: float = 1e-6,
    use_lmrc_seed: bool = True,
    lmrc_cg_tolerance: float = _DEFAULT_LMRC_CG_TOLERANCE,
    lmrc_cg_max_iterations: int | None = _DEFAULT_LMRC_CG_MAX_ITERATIONS,
    objective_mixture_weight: float = _DEFAULT_OBJECTIVE_MIXTURE_WEIGHT,
    class_mixture_weight: float = _DEFAULT_CLASS_MIXTURE_WEIGHT,
    hard_mixture_weight: float = _DEFAULT_HARD_MIXTURE_WEIGHT,
    rank_tol: float | None = None,
    sample_rank_tol: float = 0.0,
    seed: int = 0,
    max_sample_size: int | None = _DEFAULT_MAX_SAMPLE_SIZE,
    max_class_matrix_entries: int | None = _DEFAULT_MAX_CLASS_MATRIX_ENTRIES,
) -> DuplicateAwareSampledKendallResult:
    """Run duplicate-aware importance sampling and weighted local search.

    ``epsilon`` and ``delta`` retain the same reference uniform-sample
    diagnostic used by the other sampled implementations.  The nonuniform,
    pilot-adaptive proposal does not claim that PTAS bound, so
    ``sampling_bound_met`` is false on every nonempty instance.  When the
    k-by-k class table would exceed ``max_class_matrix_entries`` (or that value
    is ``None``), sampling automatically switches to the sparse on-demand mode.
    """

    total_started = time.perf_counter()
    epsilon_value = _max._validate_open_probability(epsilon, "epsilon")
    delta_value = _max._validate_open_probability(delta, "delta")
    restart_count = _max._validate_positive_integer(restarts, "restarts")
    iteration_limit = _max._validate_positive_integer(max_iterations, "max_iterations")
    patience_value = _max._validate_positive_integer(patience, "patience")
    gaussian_frequency = _max._validate_positive_integer(
        gaussian_interval, "gaussian_interval"
    )
    ridge_value = _baseline._nonnegative_real(ridge, "ridge")
    use_lmrc = _local._validate_boolean(use_lmrc_seed, "use_lmrc_seed")
    cg_tolerance = _baseline._positive_real(lmrc_cg_tolerance, "lmrc_cg_tolerance")
    cg_maximum = _baseline._optional_positive_integer(
        lmrc_cg_max_iterations, "lmrc_cg_max_iterations"
    )
    rank_tolerance = _hybrid._validate_global_rank_tolerance(rank_tol)
    sample_tolerance = _hybrid._validate_sample_rank_tolerance(sample_rank_tol)
    requested_sample_size, sample_limit = _local._validate_sample_request(
        sample_size, max_sample_size
    )
    matrix_limit = _max._validate_optional_limit(
        max_class_matrix_entries, "max_class_matrix_entries"
    )
    mixture = _validate_mixture(
        objective_mixture_weight, class_mixture_weight, hard_mixture_weight
    )
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    optimizable = ranking_input.pair_index.modeled_comparisons
    if optimizable == 0:
        return _empty_result(
            ranking_input,
            epsilon=epsilon_value,
            delta=delta_value,
            mixture=mixture,
            sample_rank_tolerance=sample_tolerance,
            restarts=restart_count,
            max_iterations=iteration_limit,
            total_started=total_started,
        )

    x = ranking_input.points
    target = ranking_input.target_order
    parameterization = _core._score_parameterization(
        x,
        target,
        rank_tolerance,
        ranking_input.point_class_ids_order,
    )
    score_basis = parameterization.score_basis
    class_basis = score_basis[ranking_input.representative_target_positions]
    eta = epsilon_value / 4.0
    theoretical_sample_size = _max.vc_uniform_sample_size(
        parameterization.dimension_bound, eta, delta_value
    )
    actual_sample_size = (
        theoretical_sample_size if requested_sample_size is None else requested_sample_size
    )
    if actual_sample_size > _max._MAX_EXACT_INTEGER_IN_FLOAT64:
        raise ValueError(
            "sample_size exceeds the largest exactly weighted sample supported "
            "by this float64 implementation"
        )
    if sample_limit is not None and actual_sample_size > sample_limit:
        raise ValueError(
            f"sample_size={actual_sample_size} exceeds max_sample_size={sample_limit}; "
            "increase/disable the guard or request a smaller experimental sample"
        )

    normalized_seed = int(seed) % 2_000_000_000
    root_sequence = np.random.SeedSequence([normalized_seed, _SEED_SALT])
    sample_sequence, seed_sequence, run_sequence, refinement_sequence = (
        root_sequence.spawn(4)
    )

    # The pilot is used by the proposal even when the caller disables it as a
    # search seed/finalist.
    pilot_direction, _, _pilot_diagnostics = _baseline._linearized_mrc_direction(
        score_basis,
        ridge_value,
        cg_tolerance,
        cg_maximum,
        ranking_input.target_tie_ids_order,
    )
    sampling_started = time.perf_counter()
    matrix_entries = ranking_input.point_class_count**2
    use_exact_class_graph = (
        matrix_limit is not None and matrix_entries <= matrix_limit
    )
    if use_exact_class_graph:
        population_pairs, population_weights, class_graph_runtime = (
            _build_class_edge_population(ranking_input, matrix_limit)
        )
        if population_pairs.shape[0] == 0:
            raise RuntimeError("nonempty item comparison population has no class edge")
        positive_class_comparisons: int | None = int(population_pairs.shape[0])
        hardness, pilot_margin_scale = _hardness_distribution(
            class_basis, population_pairs, pilot_direction
        )
        importance_sample = _draw_importance_sample(
            population_pairs,
            population_weights,
            optimizable,
            hardness,
            actual_sample_size,
            mixture,
            pilot_margin_scale,
            np.random.default_rng(sample_sequence),
        )
        del population_pairs, population_weights, hardness
        sampling_mode = "exact_class_graph"
    else:
        class_graph_runtime = 0.0
        positive_class_comparisons = None
        importance_sample = _draw_sparse_importance_sample(
            ranking_input,
            class_basis,
            pilot_direction,
            actual_sample_size,
            mixture,
            np.random.default_rng(sample_sequence),
        )
        sampling_mode = "sparse_on_demand"
    sampling_runtime = time.perf_counter() - sampling_started

    class_pairs = importance_sample.class_pairs
    draw_counts = importance_sample.draw_counts
    selected_population_weights = importance_sample.population_weights
    selected_proposal = importance_sample.proposal_probabilities
    importance_weights = importance_sample.importance_weights
    effective_sample_size = importance_sample.effective_sample_size
    mixture = importance_sample.mixture
    pilot_margin_scale = importance_sample.pilot_margin_scale

    sample_parameterization = _hybrid._sample_parameterization(
        class_basis, class_pairs, sample_tolerance
    )
    comparison_coordinates = sample_parameterization.normalized_differences
    sample_differences = sample_parameterization.raw_differences
    tie_satisfied = _local._sample_tie_orientations(class_basis, class_pairs)
    sample_weight_total = float(np.sum(importance_weights, dtype=np.float64))
    if not math.isclose(sample_weight_total, actual_sample_size, rel_tol=1e-12):
        raise RuntimeError("normalized importance weights do not sum to sample size")

    optimization_started = time.perf_counter()
    projected_pilot = sample_parameterization.right_basis.T @ pilot_direction
    seed_rng = np.random.default_rng(seed_sequence)
    seeds = _local._seed_directions(
        comparison_coordinates,
        importance_weights,
        projected_pilot if use_lmrc else None,
        restart_count,
        seed_rng,
    )
    run_sequences = run_sequence.spawn(restart_count)
    runs: list[_WeightedSearchRun] = []
    restart_diagnostics: list[dict[str, Any]] = []
    for restart, ((seed_source, initial), child_sequence) in enumerate(
        zip(seeds, run_sequences, strict=True)
    ):
        run = _run_weighted_local_search(
            comparison_coordinates,
            importance_weights,
            tie_satisfied,
            initial,
            seed_source=seed_source,
            max_iterations=iteration_limit,
            patience=patience_value,
            gaussian_interval=gaussian_frequency,
            rng=np.random.default_rng(child_sequence),
        )
        runs.append(run)
        initial_agreement = _weighted_agreement(
            comparison_coordinates @ initial, importance_weights, tie_satisfied
        )
        restart_diagnostics.append(
            {
                "restart": restart,
                "seed_source": seed_source,
                "initial_empirical_agreement": initial_agreement,
                "final_search_empirical_agreement": run.agreement,
                "final_search_empirical_disagreement": (
                    sample_weight_total - run.agreement
                ),
                "robust_margin": run.robust_margin,
                "performed_iterations": run.performed_iterations,
                "great_circle_sweeps": run.great_circle_sweeps,
                "strict_improvements": run.strict_improvements,
                "plateau_moves": run.plateau_moves,
                "stagnant_sweeps": run.stagnant_sweeps,
                "neighborhood_counts": dict(run.neighborhood_counts),
                "runtime_seconds": run.runtime_seconds,
            }
        )

    candidates: list[_Candidate] = []
    evaluation_cache: dict[bytes, tuple[int, int, int, float]] = {}
    refinement_children = refinement_sequence.spawn(restart_count + int(use_lmrc))
    refinement_index = 0
    initial_lmrc_candidate: _Candidate | None = None
    if use_lmrc:
        refinement_seed = int(
            refinement_children[refinement_index].generate_state(1, dtype=np.uint32)[0]
            % 2_000_000_000
        )
        refinement_index += 1
        initial_lmrc_candidate = _make_candidate(
            ranking_input,
            class_basis,
            class_pairs,
            importance_weights,
            sample_differences,
            pilot_direction,
            source="lmrc",
            restart=-1,
            seed=refinement_seed,
            evaluation_cache=evaluation_cache,
        )
        candidates.extend(
            (
                initial_lmrc_candidate,
                _reverse_candidate(
                    ranking_input, initial_lmrc_candidate, sample_weight_total
                ),
            )
        )

    for restart, run in enumerate(runs):
        model_direction = sample_parameterization.to_full(run.reduced_direction)
        refinement_seed = int(
            refinement_children[refinement_index].generate_state(1, dtype=np.uint32)[0]
            % 2_000_000_000
        )
        refinement_index += 1
        forward = _make_candidate(
            ranking_input,
            class_basis,
            class_pairs,
            importance_weights,
            sample_differences,
            model_direction,
            source="duplicate_aware_sample_local_search",
            restart=restart,
            seed=refinement_seed,
            evaluation_cache=evaluation_cache,
        )
        reverse = _reverse_candidate(ranking_input, forward, sample_weight_total)
        if forward.full_agreement + reverse.full_agreement != optimizable or not math.isclose(
            forward.empirical_agreement + reverse.empirical_agreement,
            sample_weight_total,
            rel_tol=0.0,
            abs_tol=_agreement_tolerance(
                forward.empirical_agreement + reverse.empirical_agreement,
                sample_weight_total,
            ),
        ):
            raise RuntimeError("duplicate-aware antipodal accounting failed")
        candidates.extend((forward, reverse))
        restart_diagnostics[restart].update(
            {
                "refined_empirical_agreement": forward.empirical_agreement,
                "refined_empirical_disagreement": (
                    sample_weight_total - forward.empirical_agreement
                ),
                "full_agreement": forward.full_agreement,
                "full_disagreement": forward.full_disagreement,
            }
        )

    selected = max(candidates, key=_candidate_key)
    certificate = max(candidates, key=_empirical_candidate_key)
    if 2 * selected.full_agreement < optimizable:
        raise RuntimeError("duplicate-aware local search lost its antipodal half guarantee")

    flattened, prediction_ids, prediction_order = _core._prediction_from_class_order(
        ranking_input, selected.class_order
    )
    checked_agreement, checked_disagreement, checked_total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    if (
        checked_agreement != selected.full_agreement
        or checked_disagreement != selected.full_disagreement
        or checked_total != ranking_input.pair_index.target_comparisons
    ):
        raise RuntimeError("duplicate-aware full-data counts changed")
    ambient_direction = parameterization.to_ambient(selected.model_direction)
    representatives = ranking_input.representative_target_positions[selected.class_order]
    _core._verify_original_ranking(x, target[representatives], ambient_direction)
    target_scores = _max._class_scores_in_target_order(
        ranking_input, selected.class_scores
    )
    sampled_original = _representative_item_comparisons(ranking_input, class_pairs)
    covered_classes = np.unique(class_pairs).size
    optimization_runtime = time.perf_counter() - optimization_started

    return DuplicateAwareSampledKendallResult(
        ranking=target[flattened].astype(np.int64, copy=False),
        direction=ambient_direction.astype(np.float64, copy=False),
        conditioned_scores=_max._conditioned_scores_in_original_order(
            target, target_scores
        ),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=checked_agreement,
        kendall_tau=checked_disagreement,
        total_comparisons=checked_total,
        forced_tie_disagreements=ranking_input.pair_index.forced_tie_disagreements,
        optimizable_comparisons=optimizable,
        point_class_count=ranking_input.point_class_count,
        positive_class_comparisons=positive_class_comparisons,
        sampling_mode=sampling_mode,
        sample_size=actual_sample_size,
        accepted_sample_size=importance_sample.accepted_sample_size,
        null_sample_size=importance_sample.null_sample_size,
        unique_sample_size=int(class_pairs.shape[0]),
        unique_class_sample_size=int(class_pairs.shape[0]),
        sampled_comparisons=sampled_original.astype(np.int64, copy=False),
        sampled_class_comparisons=class_pairs.astype(np.int64, copy=False),
        sample_multiplicities=draw_counts,
        sample_population_multiplicities=selected_population_weights,
        sample_proposal_probabilities=selected_proposal,
        sample_importance_weights=importance_weights,
        best_empirical_agreement=certificate.empirical_agreement,
        best_empirical_disagreement=(
            sample_weight_total - certificate.empirical_agreement
        ),
        selected_empirical_agreement=selected.empirical_agreement,
        selected_empirical_disagreement=(
            sample_weight_total - selected.empirical_agreement
        ),
        initial_lmrc_empirical_agreement=(
            None
            if initial_lmrc_candidate is None
            else initial_lmrc_candidate.empirical_agreement
        ),
        initial_lmrc_empirical_disagreement=(
            None
            if initial_lmrc_candidate is None
            else sample_weight_total - initial_lmrc_candidate.empirical_agreement
        ),
        objective_mixture_weight=mixture[0],
        class_mixture_weight=mixture[1],
        hard_mixture_weight=mixture[2],
        effective_sample_size=effective_sample_size,
        sampled_class_coverage=covered_classes / ranking_input.point_class_count,
        sample_collision_rate=(
            1.0
            - class_pairs.shape[0] / importance_sample.accepted_sample_size
        ),
        edge_multiplicity_cache_size=(
            importance_sample.edge_multiplicity_cache_size
        ),
        pilot_margin_scale=pilot_margin_scale,
        epsilon=epsilon_value,
        delta=delta_value,
        eta=eta,
        dimension_bound=parameterization.dimension_bound,
        sample_dimension=sample_parameterization.dimension,
        sample_condition_number=sample_parameterization.condition_number,
        sample_rank_tolerance=sample_tolerance,
        sample_projection_used=sample_parameterization.projection_used,
        theoretical_sample_size=theoretical_sample_size,
        sampling_bound_met=False,
        restarts=restart_count,
        max_iterations_per_restart=iteration_limit,
        performed_iterations=sum(run.performed_iterations for run in runs),
        great_circle_sweeps=sum(run.great_circle_sweeps for run in runs),
        strict_improvements=sum(run.strict_improvements for run in runs),
        plateau_moves=sum(run.plateau_moves for run in runs),
        full_candidate_evaluations=len(evaluation_cache),
        candidate_count=len(candidates),
        certificate_source=certificate.source,
        certificate_restart=certificate.restart,
        certificate_orientation=certificate.orientation,
        chosen_source=selected.source,
        chosen_restart=selected.restart,
        chosen_orientation=selected.orientation,
        numerical_margin=selected.numerical_margin,
        projected_instance=parameterization.projected_instance,
        class_graph_runtime_seconds=class_graph_runtime,
        sampling_runtime_seconds=sampling_runtime,
        optimization_runtime_seconds=optimization_runtime,
        solver_runtime_seconds=0.0,
        total_runtime_seconds=time.perf_counter() - total_started,
        restart_diagnostics=tuple(restart_diagnostics),
    )


def _none_if_zero(value: str) -> int | None:
    try:
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("the value must be a nonnegative integer") from exc
    if integer < 0:
        raise argparse.ArgumentTypeError("the value must be a nonnegative integer")
    return None if integer == 0 else integer


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run duplicate-aware class-edge sampling and weighted Kendall local search."
        )
    )
    parser.add_argument(
        "points", type=Path, help="CSV, whitespace text, or .npy point matrix"
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-order",
        type=_core._parse_target_order,
        metavar="I,J,...",
        help="zero-based row permutation, highest first (default: row order)",
    )
    target_group.add_argument(
        "--target-ranks",
        type=_core._parse_target_ranks,
        metavar="R,R,...",
        help="numeric rank per original row; smaller is higher and equals are tied",
    )
    target_group.add_argument(
        "--target-ranks-file",
        type=Path,
        metavar="PATH",
        help="text or .npy rank vector aligned with the point rows",
    )
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument(
        "--sample-size", type=_none_if_zero, default=_DEFAULT_SAMPLE_SIZE, metavar="N"
    )
    parser.add_argument("--restarts", type=int, default=_DEFAULT_RESTARTS)
    parser.add_argument("--max-iterations", type=int, default=_DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--patience", type=int, default=_DEFAULT_PATIENCE)
    parser.add_argument(
        "--gaussian-interval", type=int, default=_DEFAULT_GAUSSIAN_INTERVAL
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--no-lmrc-seed", action="store_true")
    parser.add_argument(
        "--lmrc-cg-tolerance", type=float, default=_DEFAULT_LMRC_CG_TOLERANCE
    )
    parser.add_argument(
        "--lmrc-cg-max-iterations",
        type=_none_if_zero,
        default=_DEFAULT_LMRC_CG_MAX_ITERATIONS,
        metavar="N",
    )
    parser.add_argument(
        "--objective-mixture-weight",
        type=float,
        default=_DEFAULT_OBJECTIVE_MIXTURE_WEIGHT,
    )
    parser.add_argument(
        "--class-mixture-weight",
        type=float,
        default=_DEFAULT_CLASS_MIXTURE_WEIGHT,
    )
    parser.add_argument(
        "--hard-mixture-weight", type=float, default=_DEFAULT_HARD_MIXTURE_WEIGHT
    )
    parser.add_argument("--delimiter", default=",", help="text delimiter, or 'whitespace'")
    parser.add_argument("--rank-tol", type=float, default=None)
    parser.add_argument("--sample-rank-tol", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-sample-size",
        type=_none_if_zero,
        default=_DEFAULT_MAX_SAMPLE_SIZE,
        metavar="N",
        help="draw guard (default: 1000000); 0 disables",
    )
    parser.add_argument(
        "--max-class-matrix-entries",
        type=_none_if_zero,
        default=_DEFAULT_MAX_CLASS_MATRIX_ENTRIES,
        metavar="N",
        help=(
            "exact k-by-k class-table threshold (default: 4000000); "
            "larger instances, or 0, use sparse on-demand sampling"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        points = _core._load_points(args.points, args.delimiter)
        target_ranks = args.target_ranks
        if args.target_ranks_file is not None:
            target_ranks = _core._load_target_ranks(
                args.target_ranks_file, args.delimiter
            )
        result = max_agreement_duplicate_aware_sampled_local_search(
            points,
            args.target_order,
            target_ranks=target_ranks,
            epsilon=args.epsilon,
            delta=args.delta,
            sample_size=args.sample_size,
            restarts=args.restarts,
            max_iterations=args.max_iterations,
            patience=args.patience,
            gaussian_interval=args.gaussian_interval,
            ridge=args.ridge,
            use_lmrc_seed=not args.no_lmrc_seed,
            lmrc_cg_tolerance=args.lmrc_cg_tolerance,
            lmrc_cg_max_iterations=args.lmrc_cg_max_iterations,
            objective_mixture_weight=args.objective_mixture_weight,
            class_mixture_weight=args.class_mixture_weight,
            hard_mixture_weight=args.hard_mixture_weight,
            rank_tol=args.rank_tol,
            sample_rank_tol=args.sample_rank_tol,
            seed=args.seed,
            max_sample_size=args.max_sample_size,
            max_class_matrix_entries=args.max_class_matrix_entries,
        )
        rendered = json.dumps(result.to_dict(), indent=2)
        if args.output is None:
            print(rendered)
        else:
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
