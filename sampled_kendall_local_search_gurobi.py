"""PTAS-sampled local search for maximum-agreement linear ranking.

The exact empirical MIP from ``thm:logdim-similarity-ptas`` can be expensive
even when only 122 or 200 comparisons are sampled.  This module keeps the
theorem implementation's IID-with-replacement comparison sampler, but replaces
the binary empirical optimizer with a multistart local search over the weighted
sampled Kendall objective.

The inner loop never evaluates the full Kendall objective.  Aggregated sampled
comparison differences are compressed to their thin row span, whose dimension
is at most the number of distinct sampled class comparisons.  Each local move
uses a numerically verified weighted event sweep around a great circle in that
reduced space: the objective changes only where a sampled comparison crosses
zero.  For ``u`` sampled class comparisons and sample-span dimension ``r_s``, a
sweep costs ``O(u r_s + u log u)``.  The best predicted cells are evaluated
directly, and the incumbent is retained unless a verified nonworsening move is
found.  Full weak-Kendall scoring is performed only for the LMRC candidate and
one finalist per restart, with antipodal scores obtained algebraically.

This is a heuristic, not the similarity PTAS: local search does not certify the
empirical optimum.  Evaluating every finalist together with its antipodal
reverse nevertheless preserves the deterministic 1/2 maximum-agreement floor.
Target ties are excluded, exact duplicate point rows remain tied, and strict
target comparisons between duplicate rows remain unavoidable disagreements.

Dependencies: NumPy and the shared ranking modules in this directory.  The
search itself has no binary variables and performs no branch-and-bound.
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

try:  # Support both ``python -m Code...`` and direct execution from Code/.
    from . import closest_linear_ranking_gurobi as _core
    from . import hybrid_linear_ranking_gurobi as _hybrid
    from . import linear_ranking_baselines as _baseline
    from . import max_agreement_linear_ranking_gurobi as _max
except ImportError:
    import closest_linear_ranking_gurobi as _core
    import hybrid_linear_ranking_gurobi as _hybrid
    import linear_ranking_baselines as _baseline
    import max_agreement_linear_ranking_gurobi as _max


_DEFAULT_SAMPLE_SIZE = 200
_DEFAULT_RESTARTS = 8
_DEFAULT_MAX_ITERATIONS = 250
_DEFAULT_PATIENCE = 40
_DEFAULT_GAUSSIAN_INTERVAL = 10
_DEFAULT_MAX_SAMPLE_SIZE = 1_000_000
_DEFAULT_MAX_UNIQUE_COMPARISONS = 100_000
_DEFAULT_LMRC_CG_TOLERANCE = 1e-6
_DEFAULT_LMRC_CG_MAX_ITERATIONS = 50
_QUALITY_TOLERANCE_MULTIPLIER = 128.0
_TOP_SWEEP_INTERVALS_TO_VERIFY = 8
_SEED_SALT = 0xBB67AE85

__all__ = [
    "SampledKendallLocalSearchResult",
    "max_agreement_sampled_kendall_local_search",
]


@dataclass(frozen=True)
class SampledKendallLocalSearchResult:
    """Result of :func:`max_agreement_sampled_kendall_local_search`."""

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
    sample_size: int
    unique_sample_size: int
    unique_class_sample_size: int
    sampled_comparisons: np.ndarray
    sample_multiplicities: np.ndarray
    best_empirical_agreement: int
    best_empirical_disagreement: int
    selected_empirical_agreement: int
    selected_empirical_disagreement: int
    initial_lmrc_empirical_agreement: int | None
    initial_lmrc_empirical_disagreement: int | None
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
    optimization_runtime_seconds: float
    solver_runtime_seconds: float
    total_runtime_seconds: float
    restart_diagnostics: tuple[dict[str, Any], ...]

    @property
    def guaranteed_fraction_of_optimum(self) -> float:
        """Return the applicable maximum-agreement approximation fraction.

        The 1/2 floor is deterministic.  If the sufficient uniform-convergence
        sample size is met, the empirical agreement rate of one explicitly
        retained candidate gives the additional high-probability lower bound
        ``empirical_rate - eta``.  This is a candidate-quality bound, not an
        empirical-optimality or PTAS claim.
        """

        if self.optimizable_comparisons == 0:
            return 1.0
        if not self.sampling_bound_met or self.sample_size == 0:
            return 0.5
        empirical_rate = self.best_empirical_agreement / self.sample_size
        return max(0.5, min(1.0, empirical_rate - self.eta))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": "sampled_kendall_local_search",
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
            "sample_size": self.sample_size,
            "unique_sample_size": self.unique_sample_size,
            "unique_class_sample_size": self.unique_class_sample_size,
            "sampled_comparisons": self.sampled_comparisons.tolist(),
            "sample_multiplicities": self.sample_multiplicities.tolist(),
            "best_empirical_agreement": self.best_empirical_agreement,
            "best_empirical_disagreement": self.best_empirical_disagreement,
            "selected_empirical_agreement": self.selected_empirical_agreement,
            "selected_empirical_disagreement": (self.selected_empirical_disagreement),
            "initial_lmrc_empirical_agreement": (self.initial_lmrc_empirical_agreement),
            "initial_lmrc_empirical_disagreement": (
                self.initial_lmrc_empirical_disagreement
            ),
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
            "guaranteed_fraction_of_optimum": (self.guaranteed_fraction_of_optimum),
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
            "optimization_runtime_seconds": self.optimization_runtime_seconds,
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "restart_diagnostics": [dict(item) for item in self.restart_diagnostics],
        }


@dataclass(frozen=True)
class _SearchRun:
    reduced_direction: np.ndarray
    margins: np.ndarray
    agreement: int
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
    empirical_agreement: int
    source: str
    restart: int
    orientation: int


def _validate_boolean(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")  # noqa: TRY004
    return bool(value)


def _validate_sample_request(
    sample_size: int | None,
    max_sample_size: int | None,
) -> tuple[int | None, int | None]:
    sample_limit = _max._validate_optional_limit(max_sample_size, "max_sample_size")
    requested = (
        None
        if sample_size is None
        else _max._validate_positive_integer(sample_size, "sample_size")
    )
    if requested is not None and requested > _max._MAX_EXACT_INTEGER_IN_FLOAT64:
        raise ValueError(
            "sample_size exceeds the largest exactly weighted sample supported "
            "by this float64 implementation"
        )
    if requested is not None and sample_limit is not None and requested > sample_limit:
        raise ValueError(
            f"sample_size={requested} exceeds max_sample_size={sample_limit}; "
            "increase/disable the guard or use a smaller experimental sample"
        )
    return requested, sample_limit


def _satisfied_mask(margins: np.ndarray, tie_satisfied: np.ndarray) -> np.ndarray:
    return (margins > 0.0) | ((margins == 0.0) & tie_satisfied)


def _sample_agreement(
    margins: np.ndarray,
    multiplicities: np.ndarray,
    tie_satisfied: np.ndarray,
) -> int:
    satisfied = _satisfied_mask(margins, tie_satisfied)
    return int(np.sum(multiplicities[satisfied], dtype=np.int64))


def _robust_margin(margins: np.ndarray) -> float:
    """Return a scale-normalized lower-decile sampled clearance."""

    absolute = np.abs(margins)
    if absolute.size == 0:
        return 0.0
    index = math.floor(0.1 * (absolute.size - 1))
    return float(np.partition(absolute, index)[index])


def _quality_better(
    agreement: int,
    robust_margin: float,
    incumbent_agreement: int,
    incumbent_robust_margin: float,
) -> bool:
    if agreement != incumbent_agreement:
        return agreement > incumbent_agreement
    tolerance = (
        _QUALITY_TOLERANCE_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, abs(robust_margin), abs(incumbent_robust_margin))
    )
    return robust_margin > incumbent_robust_margin + tolerance


def _normalized_reduced(vector: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.isfinite(direction).all():
        return None
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return np.ascontiguousarray(direction / norm)


def _sample_tie_orientations(
    class_basis: np.ndarray,
    class_pairs: np.ndarray,
) -> np.ndarray:
    """Return the realizable lexicographic orientation for zero margins."""

    involved, inverse = np.unique(class_pairs, return_inverse=True)
    local_pairs = inverse.reshape(-1, 2)
    local_basis = np.ascontiguousarray(class_basis[involved])
    lexicographic_order = _max._descending_lexicographic_order(local_basis)
    positions = np.empty(involved.size, dtype=np.int64)
    positions[lexicographic_order] = np.arange(involved.size, dtype=np.int64)
    return positions[local_pairs[:, 0]] < positions[local_pairs[:, 1]]


def _ray_half(ray: np.ndarray) -> int:
    """Return the closed half-plane used to order a circular boundary ray."""

    x_coordinate = float(ray[0])
    y_coordinate = float(ray[1])
    return (
        0 if y_coordinate > 0.0 or (y_coordinate == 0.0 and x_coordinate >= 0.0) else 1
    )


def _compare_boundary_rays(left: np.ndarray, right: np.ndarray) -> int:
    """Compare two unit rays counterclockwise without converting to angles."""

    left_half = _ray_half(left)
    right_half = _ray_half(right)
    if left_half != right_half:
        return -1 if left_half < right_half else 1
    cross = math.fsum(
        (
            float(left[0]) * float(right[1]),
            -float(left[1]) * float(right[0]),
        )
    )
    if cross > 0.0:
        return -1
    if cross < 0.0:
        return 1
    return 0


def _group_circular_events(
    rays: np.ndarray,
    deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort circular boundary rays and group identical directions."""

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
    grouped_deltas: list[int] = []
    group_start = 0
    while group_start < sorted_rays.shape[0]:
        group_end = group_start + 1
        while (
            group_end < sorted_rays.shape[0]
            and _compare_boundary_rays(sorted_rays[group_start], sorted_rays[group_end])
            == 0
        ):
            group_end += 1
        grouped_rays.append(sorted_rays[group_start].copy())
        grouped_deltas.append(
            int(np.sum(sorted_deltas[group_start:group_end], dtype=np.int64))
        )
        group_start = group_end
    return (
        np.asarray(grouped_rays, dtype=np.float64),
        np.asarray(grouped_deltas, dtype=np.int64),
    )


def _cell_midpoint_ray(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return a stable interior ray for an adjacent circular event cell."""

    midpoint = lower + upper
    norm = float(np.linalg.norm(midpoint))
    if norm == 0.0:
        # Antipodal boundaries leave a semicircle; rotate the lower ray in the
        # counterclockwise direction to select its center.
        return np.asarray((-float(lower[1]), float(lower[0])), dtype=np.float64)
    return np.ascontiguousarray(midpoint / norm)


def _cell_width_score(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return a monotone clearance score for a cell of width at most pi."""

    dot = math.fsum(
        (
            float(lower[0]) * float(upper[0]),
            float(lower[1]) * float(upper[1]),
        )
    )
    return 1.0 - min(1.0, max(-1.0, dot))


def _best_great_circle_move(
    direction: np.ndarray,
    margins: np.ndarray,
    tangent: np.ndarray,
    comparison_coordinates: np.ndarray,
    multiplicities: np.ndarray,
    tie_satisfied: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, int]:
    """Return the best directly verified candidate cell on a great circle."""

    tangent_margins = comparison_coordinates @ tangent
    amplitudes = np.hypot(margins, tangent_margins)
    active = amplitudes > 0.0
    if not np.any(active):
        agreement = _sample_agreement(margins, multiplicities, tie_satisfied)
        return direction, margins, agreement, _robust_margin(margins), 0

    first_rays = (
        np.column_stack((tangent_margins[active], -margins[active]))
        / amplitudes[active, None]
    )
    active_weights = multiplicities[active]
    event_rays = np.vstack((first_rays, -first_rays))
    event_deltas = np.concatenate((active_weights, -active_weights)).astype(
        np.int64, copy=False
    )
    group_rays, group_deltas = _group_circular_events(event_rays, event_deltas)

    reference_ray = _cell_midpoint_ray(group_rays[-1], group_rays[0])
    reference_margins = margins * reference_ray[0] + tangent_margins * reference_ray[1]
    current_agreement = _sample_agreement(
        reference_margins, multiplicities, tie_satisfied
    )
    interval_records: list[tuple[int, float, np.ndarray]] = [
        (
            current_agreement,
            _cell_width_score(group_rays[-1], group_rays[0]),
            reference_ray,
        )
    ]

    for index, lower_ray in enumerate(group_rays[:-1]):
        current_agreement += int(group_deltas[index])
        upper_ray = group_rays[index + 1]
        midpoint_ray = _cell_midpoint_ray(lower_ray, upper_ray)
        interval_records.append(
            (
                current_agreement,
                _cell_width_score(lower_ray, upper_ray),
                midpoint_ray,
            )
        )

    interval_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
    incumbent_agreement = _sample_agreement(margins, multiplicities, tie_satisfied)
    incumbent_robustness = _robust_margin(margins)
    best_direction = direction
    best_margins = margins
    best_agreement = incumbent_agreement
    best_robustness = incumbent_robustness

    for _, _, midpoint_ray in interval_records[:_TOP_SWEEP_INTERVALS_TO_VERIFY]:
        candidate = direction * midpoint_ray[0] + tangent * midpoint_ray[1]
        candidate = _normalized_reduced(candidate)
        if candidate is None:
            continue
        candidate_margins = comparison_coordinates @ candidate
        candidate_agreement = _sample_agreement(
            candidate_margins, multiplicities, tie_satisfied
        )
        candidate_robustness = _robust_margin(candidate_margins)
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


def _tangent_direction(
    raw_direction: np.ndarray,
    incumbent: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    tangent = raw_direction - float(raw_direction @ incumbent) * incumbent
    normalized = _normalized_reduced(tangent)
    if normalized is not None:
        return normalized
    for _ in range(4):
        fallback = rng.standard_normal(incumbent.size)
        fallback -= float(fallback @ incumbent) * incumbent
        normalized = _normalized_reduced(fallback)
        if normalized is not None:
            return normalized
    return None


def _run_sample_local_search(
    comparison_coordinates: np.ndarray,
    multiplicities: np.ndarray,
    tie_satisfied: np.ndarray,
    initial_direction: np.ndarray,
    *,
    seed_source: str,
    max_iterations: int,
    patience: int,
    gaussian_interval: int,
    rng: np.random.Generator,
) -> _SearchRun:
    """Run monotone, numerically verified sampled line search from one seed."""

    started = time.perf_counter()
    direction = _normalized_reduced(initial_direction)
    if direction is None:
        raise RuntimeError("sampled local search received a zero seed")
    margins = comparison_coordinates @ direction
    agreement = _sample_agreement(margins, multiplicities, tie_satisfied)
    robustness = _robust_margin(margins)
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
        satisfied = _satisfied_mask(margins, tie_satisfied)
        violated = np.flatnonzero(~satisfied)

        if (iteration + 1) % gaussian_interval == 0:
            raw_tangent = rng.standard_normal(dimension)
            neighborhood = "gaussian"
        elif (iteration + 1) % 5 == 0 and violated.size:
            violated_weights = multiplicities[violated].astype(np.float64)
            raw_tangent = comparison_coordinates[violated].T @ violated_weights
            neighborhood = "violated_aggregate"
        elif iteration % 2 == 1 and violated.size:
            probabilities = multiplicities[violated].astype(np.float64)
            probabilities /= float(np.sum(probabilities))
            chosen = int(rng.choice(violated, p=probabilities))
            raw_tangent = comparison_coordinates[chosen]
            neighborhood = "violated_pair"
        else:
            coordinate = int(rng.integers(0, dimension))
            raw_tangent = np.zeros(dimension, dtype=np.float64)
            raw_tangent[coordinate] = 1.0
            neighborhood = "coordinate"

        tangent = _tangent_direction(raw_tangent, direction, rng)
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
            multiplicities,
            tie_satisfied,
        )
        if candidate_agreement > agreement:
            direction = candidate_direction
            margins = candidate_margins
            agreement = candidate_agreement
            robustness = candidate_robustness
            strict_improvements += 1
            stagnant = 0
        elif _quality_better(
            candidate_agreement,
            candidate_robustness,
            agreement,
            robustness,
        ):
            direction = candidate_direction
            margins = candidate_margins
            robustness = candidate_robustness
            plateau_moves += 1
            stagnant += 1
        else:
            stagnant += 1

    checked_margins = comparison_coordinates @ direction
    checked_agreement = _sample_agreement(
        checked_margins, multiplicities, tie_satisfied
    )
    if checked_agreement != agreement:
        raise RuntimeError("sampled local-search incumbent accounting changed")
    return _SearchRun(
        reduced_direction=direction,
        margins=checked_margins,
        agreement=agreement,
        robust_margin=_robust_margin(checked_margins),
        performed_iterations=performed,
        great_circle_sweeps=sweeps,
        strict_improvements=strict_improvements,
        plateau_moves=plateau_moves,
        stagnant_sweeps=stagnant,
        neighborhood_counts=neighborhood_counts,
        seed_source=seed_source,
        runtime_seconds=time.perf_counter() - started,
    )


def _candidate_key(candidate: _Candidate) -> tuple[float, ...]:
    source_priority = 1.0 if candidate.source == "lmrc" else 0.0
    return (
        float(candidate.full_agreement),
        float(candidate.empirical_agreement),
        float(candidate.numerical_margin),
        source_priority,
        float(-candidate.restart),
        float(candidate.orientation),
    )


def _empirical_candidate_key(candidate: _Candidate) -> tuple[float, ...]:
    return (
        float(candidate.empirical_agreement),
        float(candidate.full_agreement),
        float(candidate.numerical_margin),
        float(-candidate.restart),
        float(candidate.orientation),
    )


def _make_candidate(
    ranking_input: _core._RankingInput,
    class_basis: np.ndarray,
    class_pairs: np.ndarray,
    multiplicities: np.ndarray,
    sample_differences: np.ndarray,
    weak_direction: np.ndarray,
    *,
    source: str,
    restart: int,
    seed: int,
    evaluation_cache: dict[bytes, tuple[int, int, int, int]],
) -> _Candidate:
    """Genericize, fully score, and sample-score one forward direction."""

    raw_margins = sample_differences @ weak_direction
    active = raw_margins > 0.0
    direction, class_scores, class_order, margin = _hybrid._strict_generic_candidate(
        class_basis,
        sample_differences,
        class_pairs,
        active,
        weak_direction,
        seed=seed,
    )
    cache_key = class_order.tobytes()
    cached = evaluation_cache.get(cache_key)
    if cached is None:
        _, _, _, agreement, disagreement, total = _max._evaluate_class_order(
            ranking_input, class_order
        )
        if total != ranking_input.pair_index.target_comparisons:
            raise RuntimeError("sampled local-search comparison population changed")
        empirical = _max._empirical_agreement(class_order, class_pairs, multiplicities)
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
    sample_size: int,
) -> _Candidate:
    agreement = ranking_input.pair_index.modeled_comparisons - forward.full_agreement
    return _Candidate(
        class_order=forward.class_order[::-1].copy(),
        model_direction=-forward.model_direction,
        class_scores=-forward.class_scores,
        numerical_margin=forward.numerical_margin,
        full_agreement=agreement,
        full_disagreement=ranking_input.pair_index.target_comparisons - agreement,
        empirical_agreement=sample_size - forward.empirical_agreement,
        source=forward.source,
        restart=forward.restart,
        orientation=-1,
    )


def _seed_directions(
    comparison_coordinates: np.ndarray,
    multiplicities: np.ndarray,
    projected_lmrc: np.ndarray | None,
    restart_count: int,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()

    def add(source: str, raw: np.ndarray, *, allow_duplicate: bool = False) -> bool:
        if len(seeds) >= restart_count:
            return False
        normalized = _normalized_reduced(raw)
        if normalized is None:
            return False
        key = np.round(normalized, decimals=12).tobytes()
        if key in seen and not allow_duplicate:
            return False
        seen.add(key)
        seeds.append((source, normalized))
        return True

    if projected_lmrc is not None:
        add("projected_lmrc", projected_lmrc)
    weighted_centroid = comparison_coordinates.T @ multiplicities.astype(np.float64)
    add("weighted_sample_centroid", weighted_centroid)
    if projected_lmrc is not None:
        add("negative_projected_lmrc", -projected_lmrc)
    add("negative_weighted_sample_centroid", -weighted_centroid)
    duplicate_attempts = 0
    while len(seeds) < restart_count:
        raw = rng.standard_normal(comparison_coordinates.shape[1])
        if add("gaussian", raw):
            duplicate_attempts = 0
            continue
        duplicate_attempts += 1
        if duplicate_attempts >= 16:
            # In a one-dimensional sample span only two unit directions exist.
            # Repeated restarts are still valid because their neighborhood RNGs
            # differ, so stop insisting on a new seed after a bounded effort.
            add("gaussian_repeated", raw, allow_duplicate=True)
            duplicate_attempts = 0
    return seeds


def _empty_result(
    ranking_input: _core._RankingInput,
    *,
    epsilon: float,
    delta: float,
    sample_rank_tolerance: float,
    restarts: int,
    max_iterations: int,
    total_started: float,
) -> SampledKendallLocalSearchResult:
    """Return a verified result when no optimizable comparison exists."""

    x = ranking_input.points
    target = ranking_input.target_order
    if ranking_input.pair_index.modeled_comparisons != 0:
        raise RuntimeError("the empty local-search result has optimizable comparisons")
    flattened, prediction_ids, prediction_order = _core._all_tied_prediction(
        ranking_input
    )
    agreement, disagreement, total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    ambient_direction = np.zeros(x.shape[1], dtype=np.float64)
    conditioned = np.zeros(x.shape[0], dtype=np.float64)
    elapsed = time.perf_counter() - total_started
    return SampledKendallLocalSearchResult(
        ranking=target[flattened],
        direction=ambient_direction,
        conditioned_scores=conditioned,
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=agreement,
        kendall_tau=disagreement,
        total_comparisons=total,
        forced_tie_disagreements=(ranking_input.pair_index.forced_tie_disagreements),
        optimizable_comparisons=0,
        sample_size=0,
        unique_sample_size=0,
        unique_class_sample_size=0,
        sampled_comparisons=np.empty((0, 2), dtype=np.int64),
        sample_multiplicities=np.empty(0, dtype=np.int64),
        best_empirical_agreement=0,
        best_empirical_disagreement=0,
        selected_empirical_agreement=0,
        selected_empirical_disagreement=0,
        initial_lmrc_empirical_agreement=None,
        initial_lmrc_empirical_disagreement=None,
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
        optimization_runtime_seconds=0.0,
        solver_runtime_seconds=0.0,
        total_runtime_seconds=elapsed,
        restart_diagnostics=(),
    )


def max_agreement_sampled_kendall_local_search(
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
    rank_tol: float | None = None,
    sample_rank_tol: float = 0.0,
    seed: int = 0,
    max_sample_size: int | None = _DEFAULT_MAX_SAMPLE_SIZE,
    max_unique_comparisons: int | None = _DEFAULT_MAX_UNIQUE_COMPARISONS,
) -> SampledKendallLocalSearchResult:
    """Run scalable local search on an IID PTAS comparison sample.

    Search iterations use only the aggregated weighted sample.  Exactly one
    forward candidate per restart, the optional generically refined LMRC
    initializer, and their antipodal reverses enter full-data selection.
    If no target comparison is optimizable, the method returns the all-tied
    zero direction without sampling or running local search.
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
    use_lmrc = _validate_boolean(use_lmrc_seed, "use_lmrc_seed")
    cg_tolerance = _baseline._positive_real(lmrc_cg_tolerance, "lmrc_cg_tolerance")
    cg_maximum = _baseline._optional_positive_integer(
        lmrc_cg_max_iterations, "lmrc_cg_max_iterations"
    )
    rank_tolerance = _hybrid._validate_global_rank_tolerance(rank_tol)
    sample_tolerance = _hybrid._validate_sample_rank_tolerance(sample_rank_tol)
    requested_sample_size, sample_limit = _validate_sample_request(
        sample_size, max_sample_size
    )
    unique_limit = _max._validate_optional_limit(
        max_unique_comparisons, "max_unique_comparisons"
    )
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    normalized_seed = int(seed) % 2_000_000_000
    optimizable = ranking_input.pair_index.modeled_comparisons
    if optimizable == 0:
        return _empty_result(
            ranking_input,
            epsilon=epsilon_value,
            delta=delta_value,
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
        theoretical_sample_size
        if requested_sample_size is None
        else requested_sample_size
    )
    if actual_sample_size > _max._MAX_EXACT_INTEGER_IN_FLOAT64:
        raise ValueError(
            "sample_size exceeds the largest exactly weighted sample supported "
            "by this float64 implementation"
        )
    if sample_limit is not None and actual_sample_size > sample_limit:
        raise ValueError(
            f"sample_size={actual_sample_size} exceeds "
            f"max_sample_size={sample_limit}; increase/disable the guard or "
            "use a smaller experimental sample"
        )
    sampling_bound_met = actual_sample_size >= theoretical_sample_size

    root_sequence = np.random.SeedSequence([normalized_seed, _SEED_SALT])
    sample_sequence, seed_sequence, run_sequence, refinement_sequence = (
        root_sequence.spawn(4)
    )
    sample_rng = np.random.default_rng(sample_sequence)
    pair_ids, multiplicities = _max._sample_pair_multiset(
        optimizable,
        actual_sample_size,
        sample_rng,
        unique_limit,
    )
    target_pairs = ranking_input.pair_index.decode(pair_ids)
    class_pairs, class_multiplicities = _max._aggregate_class_comparisons(
        ranking_input, target_pairs, multiplicities
    )
    if int(np.sum(class_multiplicities, dtype=np.int64)) != actual_sample_size:
        raise RuntimeError("aggregated local-search sample lost multiplicity")
    sample_parameterization = _hybrid._sample_parameterization(
        class_basis, class_pairs, sample_tolerance
    )
    comparison_coordinates = sample_parameterization.normalized_differences
    sample_differences = sample_parameterization.raw_differences
    tie_satisfied = _sample_tie_orientations(class_basis, class_pairs)

    optimization_started = time.perf_counter()
    lmrc_direction: np.ndarray | None = None
    if use_lmrc:
        lmrc_direction, _, _lmrc_diagnostics = _baseline._linearized_mrc_direction(
            score_basis,
            ridge_value,
            cg_tolerance,
            cg_maximum,
            ranking_input.target_tie_ids_order,
        )
    projected_lmrc = (
        None
        if lmrc_direction is None
        else sample_parameterization.right_basis.T @ lmrc_direction
    )

    seed_rng = np.random.default_rng(seed_sequence)
    seeds = _seed_directions(
        comparison_coordinates,
        class_multiplicities,
        projected_lmrc,
        restart_count,
        seed_rng,
    )
    run_sequences = run_sequence.spawn(restart_count)
    runs: list[_SearchRun] = []
    restart_diagnostics: list[dict[str, Any]] = []
    for restart, ((seed_source, initial), child_sequence) in enumerate(
        zip(seeds, run_sequences, strict=True)
    ):
        run = _run_sample_local_search(
            comparison_coordinates,
            class_multiplicities,
            tie_satisfied,
            initial,
            seed_source=seed_source,
            max_iterations=iteration_limit,
            patience=patience_value,
            gaussian_interval=gaussian_frequency,
            rng=np.random.default_rng(child_sequence),
        )
        runs.append(run)
        restart_diagnostics.append(
            {
                "restart": restart,
                "seed_source": seed_source,
                "initial_empirical_agreement": _sample_agreement(
                    comparison_coordinates @ initial,
                    class_multiplicities,
                    tie_satisfied,
                ),
                "final_search_empirical_agreement": run.agreement,
                "final_search_empirical_disagreement": (
                    actual_sample_size - run.agreement
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
    forward_candidates: list[_Candidate] = []
    evaluation_cache: dict[bytes, tuple[int, int, int, int]] = {}
    refinement_children = refinement_sequence.spawn(restart_count + int(use_lmrc))
    refinement_index = 0
    initial_lmrc_candidate: _Candidate | None = None
    if lmrc_direction is not None:
        refinement_seed = int(
            refinement_children[refinement_index].generate_state(1, dtype=np.uint32)[0]
            % 2_000_000_000
        )
        refinement_index += 1
        initial_lmrc_candidate = _make_candidate(
            ranking_input,
            class_basis,
            class_pairs,
            class_multiplicities,
            sample_differences,
            lmrc_direction,
            source="lmrc",
            restart=-1,
            seed=refinement_seed,
            evaluation_cache=evaluation_cache,
        )
        reverse = _reverse_candidate(
            ranking_input, initial_lmrc_candidate, actual_sample_size
        )
        candidates.extend((initial_lmrc_candidate, reverse))
        forward_candidates.append(initial_lmrc_candidate)

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
            class_multiplicities,
            sample_differences,
            model_direction,
            source="sample_local_search",
            restart=restart,
            seed=refinement_seed,
            evaluation_cache=evaluation_cache,
        )
        reverse = _reverse_candidate(ranking_input, forward, actual_sample_size)
        if (
            forward.full_agreement + reverse.full_agreement != optimizable
            or forward.empirical_agreement + reverse.empirical_agreement
            != actual_sample_size
        ):
            raise RuntimeError("sampled local-search antipodal accounting failed")
        candidates.extend((forward, reverse))
        forward_candidates.append(forward)
        restart_diagnostics[restart].update(
            {
                "refined_empirical_agreement": forward.empirical_agreement,
                "refined_empirical_disagreement": (
                    actual_sample_size - forward.empirical_agreement
                ),
                "full_agreement": forward.full_agreement,
                "full_disagreement": forward.full_disagreement,
            }
        )

    if not candidates:
        raise RuntimeError("sampled local search generated no full-data candidate")
    selected = max(candidates, key=_candidate_key)
    certificate = max(candidates, key=_empirical_candidate_key)
    if (
        selected.full_agreement < certificate.full_agreement
        or certificate.empirical_agreement < selected.empirical_agreement
    ):
        raise RuntimeError("sampled local-search finalist selection is inconsistent")
    if 2 * selected.full_agreement < optimizable:
        raise RuntimeError("sampled local search lost its antipodal half guarantee")

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
        raise RuntimeError("sampled local-search full-data counts changed")

    ambient_direction = parameterization.to_ambient(selected.model_direction)
    representatives = ranking_input.representative_target_positions[
        selected.class_order
    ]
    _core._verify_original_ranking(x, target[representatives], ambient_direction)
    target_scores = _max._class_scores_in_target_order(
        ranking_input, selected.class_scores
    )
    selected_empirical_disagreement = actual_sample_size - selected.empirical_agreement
    best_empirical_disagreement = actual_sample_size - certificate.empirical_agreement
    optimization_runtime = time.perf_counter() - optimization_started
    total_runtime = time.perf_counter() - total_started
    sampled_original = target[target_pairs]

    return SampledKendallLocalSearchResult(
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
        forced_tie_disagreements=(ranking_input.pair_index.forced_tie_disagreements),
        optimizable_comparisons=optimizable,
        sample_size=actual_sample_size,
        unique_sample_size=int(pair_ids.size),
        unique_class_sample_size=int(class_pairs.shape[0]),
        sampled_comparisons=sampled_original.astype(np.int64, copy=False),
        sample_multiplicities=multiplicities.astype(np.int64, copy=False),
        best_empirical_agreement=certificate.empirical_agreement,
        best_empirical_disagreement=best_empirical_disagreement,
        selected_empirical_agreement=selected.empirical_agreement,
        selected_empirical_disagreement=selected_empirical_disagreement,
        initial_lmrc_empirical_agreement=(
            None
            if initial_lmrc_candidate is None
            else initial_lmrc_candidate.empirical_agreement
        ),
        initial_lmrc_empirical_disagreement=(
            None
            if initial_lmrc_candidate is None
            else actual_sample_size - initial_lmrc_candidate.empirical_agreement
        ),
        epsilon=epsilon_value,
        delta=delta_value,
        eta=eta,
        dimension_bound=parameterization.dimension_bound,
        sample_dimension=sample_parameterization.dimension,
        sample_condition_number=sample_parameterization.condition_number,
        sample_rank_tolerance=sample_tolerance,
        sample_projection_used=sample_parameterization.projection_used,
        theoretical_sample_size=theoretical_sample_size,
        sampling_bound_met=sampling_bound_met,
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
        optimization_runtime_seconds=optimization_runtime,
        solver_runtime_seconds=0.0,
        total_runtime_seconds=total_runtime,
        restart_diagnostics=tuple(restart_diagnostics),
    )


def _none_if_zero(value: str) -> int | None:
    try:
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "the value must be a nonnegative integer"
        ) from exc
    if integer < 0:
        raise argparse.ArgumentTypeError("the value must be a nonnegative integer")
    return None if integer == 0 else integer


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run great-circle local search on an IID PTAS comparison sample.")
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
        help=(
            "numeric rank per original row; smaller is higher and equal values are tied"
        ),
    )
    target_group.add_argument(
        "--target-ranks-file",
        type=Path,
        metavar="PATH",
        help="text or .npy rank vector aligned with the point rows",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="accuracy parameter used by the sampling diagnostic",
    )
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="failure-probability parameter used by the sampling diagnostic",
    )
    parser.add_argument(
        "--sample-size",
        type=_none_if_zero,
        default=_DEFAULT_SAMPLE_SIZE,
        metavar="N",
        help=(
            "IID comparison draws (default: 200); 0 uses the sufficient "
            "theoretical size determined by epsilon and delta"
        ),
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
        help="LMRC PCG limit (default: 50); 0 uses its adaptive limit",
    )
    parser.add_argument(
        "--delimiter", default=",", help="text delimiter, or 'whitespace'"
    )
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
        "--max-unique-comparisons",
        type=_none_if_zero,
        default=_DEFAULT_MAX_UNIQUE_COMPARISONS,
        metavar="N",
        help="unique sampled-item-pair guard (default: 100000); 0 disables",
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
        result = max_agreement_sampled_kendall_local_search(
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
            rank_tol=args.rank_tol,
            sample_rank_tol=args.sample_rank_tol,
            seed=args.seed,
            max_sample_size=args.max_sample_size,
            max_unique_comparisons=args.max_unique_comparisons,
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
