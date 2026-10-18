"""Optimized Gurobi implementations of the max-agreement theorems.

This module implements both algorithms from ``sections/logarithmic-dimension.tex``:

* :func:`max_agreement_half_approximation` implements
  ``thm:logdim-half-approximation``.
* :func:`max_agreement_similarity_ptas` implements
  ``thm:logdim-similarity-ptas``.

The deterministic algorithm constructs one directly verified generic linear
order of the distinct point classes, compares its agreement with that of its
reverse, and returns the better one.  If the target has no strict pair, it
instead returns the zero direction and one all-tied group.  Exact duplicate
rows remain tied.  The construction needs neither the proof's moment-curve
scan, an explicit O(n^2) pair list, nor a final solver call.

The randomized algorithm samples optimizable target comparisons independently
and uniformly with replacement, collapses duplicates, and solves the resulting
weighted empirical maximum-agreement problem to Gurobi optimality.  A seeded
generic perturbation inside the empirical optimum's feasible cone then produces
a strict order of the distinct point classes without an ill-conditioned
symbolic tie refinement; exact duplicate rows remain tied.  An explicit VC tail
inequality is used for the default sample size; a smaller caller-supplied sample
is allowed for experiments but is marked as not meeting that probability bound.

The numerical conditioning, ambient-coordinate back-map, and exact-fallback
ranking verifier are shared with ``closest_linear_ranking_gurobi.py``.  Keep the
two files in the same directory.

Dependencies: NumPy.  The similarity PTAS additionally requires gurobipy and a
working Gurobi license.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Support both ``python -m Code...`` and direct execution from Code/.
    from . import closest_linear_ranking_gurobi as _core
except ImportError:
    import closest_linear_ranking_gurobi as _core

gp = _core.gp
GRB = _core.GRB

_MAX_EXACT_INTEGER_IN_FLOAT64 = (1 << 53) - 1
_DEFAULT_MAX_SAMPLE_SIZE = 10_000_000
_DEFAULT_MAX_UNIQUE_COMPARISONS = 10_000
_SAMPLING_CHUNK_SIZE = 1_000_000
_DENSE_COUNT_LIMIT = 2_000_000
_GENERIC_REFINEMENT_TRIALS = 16
_GENERIC_REFINEMENT_MAX_TRIALS = 128
_GENERIC_REFINEMENT_SLACK_FRACTION = 0.25
_GENERIC_REFINEMENT_SEED_SALT = 0x6A09E667


@dataclass(frozen=True)
class HalfApproximationResult:
    """Result of :func:`max_agreement_half_approximation`.

    Array-valued attributes use original row indices except for
    ``conditioned_scores``, whose entries are merely stored in original row
    order.  Rankings and target orders are listed from highest to lowest.
    """

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
    base_agreement: int
    opposite_agreement: int
    chosen_orientation: int
    numerical_margin: float
    dimension_bound: int
    projected_instance: bool
    solver_runtime_seconds: float
    total_runtime_seconds: float

    @property
    def guaranteed_fraction_of_optimum(self) -> float:
        """The theorem's deterministic approximation factor."""

        return 0.5

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": "logdim_half_approximation",
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
            "base_agreement": self.base_agreement,
            "opposite_agreement": self.opposite_agreement,
            "chosen_orientation": self.chosen_orientation,
            "numerical_margin": self.numerical_margin,
            "dimension_bound": self.dimension_bound,
            "projected_instance": self.projected_instance,
            "guaranteed_fraction_of_optimum": (self.guaranteed_fraction_of_optimum),
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
        }


@dataclass(frozen=True)
class SimilarityPTASResult:
    """Result of :func:`max_agreement_similarity_ptas`.

    ``sampled_comparisons`` contains each distinct sampled comparison once,
    oriented according to ``target_order`` and expressed in original row
    indices.  ``sample_multiplicities`` stores its multiplicity in the IID
    with-replacement sample.
    """

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
    empirical_agreement: int
    empirical_optimum: int
    sample_size: int
    unique_sample_size: int
    sampled_comparisons: np.ndarray
    sample_multiplicities: np.ndarray
    epsilon: float
    delta: float
    eta: float
    dimension_bound: int
    theoretical_sample_size: int
    sampling_bound_met: bool
    selected_numerical_margin: float
    numerical_margin: float
    projected_instance: bool
    mip_node_count: float
    solver_runtime_seconds: float
    total_runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": "logdim_similarity_ptas",
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
            "empirical_agreement": self.empirical_agreement,
            "empirical_optimum": self.empirical_optimum,
            "sample_size": self.sample_size,
            "unique_sample_size": self.unique_sample_size,
            "sampled_comparisons": self.sampled_comparisons.tolist(),
            "sample_multiplicities": self.sample_multiplicities.tolist(),
            "epsilon": self.epsilon,
            "delta": self.delta,
            "eta": self.eta,
            "dimension_bound": self.dimension_bound,
            "theoretical_sample_size": self.theoretical_sample_size,
            "sampling_bound_met": self.sampling_bound_met,
            "selected_numerical_margin": self.selected_numerical_margin,
            "numerical_margin": self.numerical_margin,
            "projected_instance": self.projected_instance,
            "mip_node_count": self.mip_node_count,
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
        }


def _validate_open_probability(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number in (0, 1)") from exc
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return number


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    try:
        exact = bool(integer == value)
    except (TypeError, ValueError):
        exact = False
    if integer <= 0 or not exact:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _validate_optional_limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_integer(value, name)


def _log_vc_tail_bound(sample_size: int, vc_dimension: int, eta: float) -> float:
    """Log of ``8 (2 e s / v)^v exp(-s eta^2 / 32)``."""

    return (
        math.log(8.0)
        + vc_dimension * (math.log(2.0 * math.e / vc_dimension) + math.log(sample_size))
        - sample_size * eta * eta / 32.0
    )


def vc_uniform_sample_size(
    vc_dimension: int,
    eta: float,
    delta: float,
) -> int:
    """Return a sufficient sample size for uniform comparison convergence.

    The returned smallest integer on the decreasing branch satisfies

    ``8 * (2*e*s/v)**v * exp(-s*eta**2/32) <= delta``.

    This follows from a standard VC tail inequality together with Sauer's
    growth-function bound.  It is deliberately conservative and makes the
    universal constant hidden in the manuscript executable and auditable.

    Parameters
    ----------
    vc_dimension:
        A positive upper bound on the VC dimension.
    eta, delta:
        Numbers strictly between zero and one.
    """

    v = _validate_positive_integer(vc_dimension, "vc_dimension")
    eta_value = _validate_open_probability(eta, "eta")
    delta_value = _validate_open_probability(delta, "delta")
    eta_squared = eta_value * eta_value
    if eta_squared == 0.0:
        raise ValueError("eta is too small for a finite float64 sample bound")

    # The logarithmic bound is nonincreasing for s >= 32*v/eta^2.  It is
    # greater than one at s=v, so no valid sample size is skipped by beginning
    # the binary search on this monotone branch.
    if v > _MAX_EXACT_INTEGER_IN_FLOAT64:
        raise ValueError(
            "the VC sample bound exceeds the largest exactly weighted sample "
            "supported by this floating-point implementation"
        )
    monotone_start = 32.0 * float(v) / eta_squared
    if (
        not math.isfinite(monotone_start)
        or monotone_start > _MAX_EXACT_INTEGER_IN_FLOAT64
    ):
        raise ValueError(
            "the VC sample bound exceeds the largest exactly weighted sample "
            "supported by this floating-point implementation"
        )
    lower = max(v, int(math.ceil(monotone_start)))
    log_delta = math.log(delta_value)
    upper = lower
    while _log_vc_tail_bound(upper, v, eta_value) > log_delta:
        if upper > _MAX_EXACT_INTEGER_IN_FLOAT64 // 2:
            raise ValueError(
                "the VC sample bound exceeds the largest exactly weighted "
                "sample supported by this floating-point implementation"
            )
        upper *= 2

    left = lower
    right = upper
    while left < right:
        middle = (left + right) // 2
        if _log_vc_tail_bound(middle, v, eta_value) <= log_delta:
            right = middle
        else:
            left = middle + 1
    return left


def _descending_lexicographic_order(score_basis: np.ndarray) -> np.ndarray:
    """Return row positions in descending lexicographic order."""

    keys = tuple(
        score_basis[:, column] for column in range(score_basis.shape[1] - 1, -1, -1)
    )
    return np.lexsort(keys)[::-1].astype(np.int64, copy=False)


def _lexicographic_refinement_order(
    weak_scores: np.ndarray,
    score_basis: np.ndarray,
) -> np.ndarray:
    """Return a strict realizable refinement of a weak linear score order."""

    secondary_keys = tuple(
        score_basis[:, column] for column in range(score_basis.shape[1] - 1, -1, -1)
    )
    # lexsort's final key is primary.  Within every weak-score tie, the rows
    # are ordered by the realizable descending lexicographic direction.
    return np.lexsort(secondary_keys + (weak_scores,))[::-1].astype(
        np.int64, copy=False
    )


def _strict_order_witness(
    env: Any,
    score_basis: np.ndarray,
    order: np.ndarray,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find a normalized maximum-margin direction for a known feasible order."""

    n, dimension = score_basis.shape
    with gp.Model("strict_linear_ranking_witness", env=env) as model:
        _core._set_model_parameters(
            model,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            seed=seed,
            gurobi_params=gurobi_params,
        )
        q = model.addMVar(dimension, lb=-1.0, ub=1.0, name="q")
        score = model.addMVar(n, lb=-GRB.INFINITY, name="score")
        margin = model.addVar(lb=0.0, name="margin")
        model.addConstr(score == score_basis @ q)
        if n > 1:
            model.addConstr(score[order[:-1]] - score[order[1:]] >= margin)
        model.setObjective(margin, GRB.MAXIMIZE)
        status = _core._optimize_for_status(model)
        if status != GRB.OPTIMAL:
            raise _core._status_error(model, "strict ranking witness")
        direction = np.asarray(q.X, dtype=np.float64).reshape(-1)
        margin_value = float(margin.X)

    scores = score_basis @ direction
    gaps = scores[order[:-1]] - scores[order[1:]]
    if (
        not math.isfinite(margin_value)
        or margin_value <= 0.0
        or np.any(~np.isfinite(scores))
        or np.any(gaps <= 0.0)
        or not np.array_equal(np.argsort(-scores, kind="stable"), order)
    ):
        raise RuntimeError(
            "Gurobi did not return a numerically strict witness for the "
            "provably realizable order"
        )
    return direction, scores, min(margin_value, float(np.min(gaps)))


def _normalized_selected_witness(
    env: Any,
    score_basis: np.ndarray,
    target_pairs: np.ndarray,
    selected: np.ndarray,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a stable weak witness satisfying every selected comparison."""

    selected_pairs = target_pairs[selected]
    if selected_pairs.size == 0:
        raise RuntimeError("the empirical optimizer selected no comparisons")
    involved, inverse = np.unique(selected_pairs, return_inverse=True)
    local_pairs = inverse.reshape(-1, 2)
    dimension = score_basis.shape[1]

    with gp.Model("empirical_optimum_witness", env=env) as model:
        _core._set_model_parameters(
            model,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            seed=seed,
            gurobi_params=gurobi_params,
        )
        q = model.addMVar(dimension, lb=-1.0, ub=1.0, name="q")
        local_score = model.addMVar(involved.size, lb=-GRB.INFINITY, name="score")
        margin = model.addVar(lb=0.0, name="margin")
        model.addConstr(local_score == np.ascontiguousarray(score_basis[involved]) @ q)
        model.addConstr(
            local_score[local_pairs[:, 0]] - local_score[local_pairs[:, 1]] >= margin
        )
        model.setObjective(margin, GRB.MAXIMIZE)
        status = _core._optimize_for_status(model)
        if status != GRB.OPTIMAL:
            raise _core._status_error(model, "empirical optimum witness")
        direction = np.asarray(q.X, dtype=np.float64).reshape(-1)
        margin_value = float(margin.X)
        scores = score_basis @ direction
        gaps = scores[selected_pairs[:, 0]] - scores[selected_pairs[:, 1]]

        def valid_witness() -> bool:
            return bool(
                math.isfinite(margin_value)
                and margin_value > 0.0
                and np.isfinite(scores).all()
                and np.all(gaps > 0.0)
            )

        if not valid_witness():
            # Dual simplex can report an optimal tiny-margin witness whose
            # recomputed selected gaps lose their sign on nearly dependent
            # bases.  A clean primal-simplex resolve uses a different numeric
            # path while preserving the exact same LP and certificate.
            model.setParam("Method", 0)
            model.reset()
            status = _core._optimize_for_status(model)
            if status != GRB.OPTIMAL:
                raise _core._status_error(
                    model, "primal-simplex empirical optimum witness"
                )
            direction = np.asarray(q.X, dtype=np.float64).reshape(-1)
            margin_value = float(margin.X)
            scores = score_basis @ direction
            gaps = scores[selected_pairs[:, 0]] - scores[selected_pairs[:, 1]]

        if not valid_witness():
            raise RuntimeError(
                "Gurobi did not preserve the selected empirical comparisons in "
                "the normalized witness"
            )
    return direction, scores, min(margin_value, float(np.min(gaps)))


def _strict_candidate_margin(
    score_basis: np.ndarray,
    direction: np.ndarray,
    scores: np.ndarray,
    order: np.ndarray,
) -> float | None:
    """Return a checked adjacent margin, or ``None`` for a numerical tie."""

    if order.size <= 1:
        return 0.0
    score_gaps = scores[order[:-1]] - scores[order[1:]]
    if np.any(~np.isfinite(score_gaps)) or np.any(score_gaps <= 0.0):
        return None

    # Recheck signs by dotting adjacent point differences directly.  This is
    # intentionally independent of subtracting two already-rounded scores.
    # Chunking avoids an O(n*d) temporary on high-dimensional instances.
    dimension = score_basis.shape[1]
    bytes_per_row = max(1, 3 * dimension * np.dtype(np.float64).itemsize)
    chunk_rows = max(1, min(order.size - 1, (64 * 1024**2) // bytes_per_row))
    minimum = float(np.min(score_gaps))
    for begin in range(0, order.size - 1, chunk_rows):
        end = min(begin + chunk_rows, order.size - 1)
        differences = (
            score_basis[order[begin:end]] - score_basis[order[begin + 1 : end + 1]]
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            direct_gaps = differences @ direction
        if np.any(~np.isfinite(direct_gaps)) or np.any(direct_gaps <= 0.0):
            return None
        minimum = min(minimum, float(np.min(direct_gaps)))
    return minimum


def _generic_weak_refinement_witness(
    score_basis: np.ndarray,
    base_direction: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Numerically refine a weak linear direction without a certification LP.

    The baseline direction is infinity-normalized.  Score gaps larger than a
    conservative roundoff threshold are protected by bounding the complete
    score range of each perturbation; gaps at or below that threshold are
    treated as numerical ties.  Every returned order is induced directly by
    its returned direction and is independently checked through both computed
    scores and adjacent point differences.

    This avoids asking an LP to reproduce an infinitesimal lexicographic order,
    whose exact margin may be positive but far below solver feasibility scales.
    """

    basis = np.asarray(score_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] <= 1 or basis.shape[1] == 0:
        raise RuntimeError("generic weak refinement requires at least two classes")
    if not np.isfinite(basis).all():
        raise RuntimeError("the score basis contains nonfinite values")

    direction = np.asarray(base_direction, dtype=np.float64).reshape(-1)
    if direction.shape != (basis.shape[1],) or not np.isfinite(direction).all():
        raise RuntimeError("the weak refinement direction is invalid")
    direction_scale = float(np.max(np.abs(direction), initial=0.0))
    if direction_scale > 0.0:
        direction = np.ascontiguousarray(direction / direction_scale)
    else:
        direction = np.zeros(basis.shape[1], dtype=np.float64)

    base_scores = basis @ direction
    if not np.isfinite(base_scores).all():
        raise RuntimeError("the weak refinement scores are nonfinite")
    base_order = np.argsort(-base_scores, kind="stable")
    base_gaps = base_scores[base_order[:-1]] - base_scores[base_order[1:]]
    score_scale = float(np.max(np.abs(base_scores), initial=0.0))
    roundoff_threshold = (
        256.0 * np.finfo(np.float64).eps * max(1.0, score_scale)
    )
    robust_gaps = base_gaps[base_gaps > roundoff_threshold]
    minimum_robust_gap = (
        float(np.min(robust_gaps)) if robust_gaps.size else None
    )

    best_direction: np.ndarray | None = None
    best_scores: np.ndarray | None = None
    best_order: np.ndarray | None = None
    best_margin = -math.inf

    def consider(candidate: np.ndarray) -> None:
        nonlocal best_direction, best_scores, best_order, best_margin

        candidate_scale = float(np.max(np.abs(candidate), initial=0.0))
        if not math.isfinite(candidate_scale) or candidate_scale == 0.0:
            return
        normalized = np.ascontiguousarray(candidate / candidate_scale)
        scores = basis @ normalized
        if not np.isfinite(scores).all():
            return
        order = np.argsort(-scores, kind="stable").astype(np.int64, copy=False)
        margin = _strict_candidate_margin(basis, normalized, scores, order)
        if margin is None or margin <= best_margin:
            return
        best_direction = normalized
        best_scores = scores
        best_order = order
        best_margin = margin

    # Retain an already strict baseline direction without requiring any solver.
    consider(direction)

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), _GENERIC_REFINEMENT_SEED_SALT, 1])
    )
    for trial in range(_GENERIC_REFINEMENT_MAX_TRIALS):
        if trial >= _GENERIC_REFINEMENT_TRIALS and best_direction is not None:
            break
        perturbation = rng.standard_normal(basis.shape[1])
        perturbation_scale = float(
            np.max(np.abs(perturbation), initial=0.0)
        )
        if not math.isfinite(perturbation_scale) or perturbation_scale == 0.0:
            continue
        perturbation /= perturbation_scale
        perturbation_scores = basis @ perturbation
        score_range = float(np.ptp(perturbation_scores))
        if not math.isfinite(score_range) or score_range == 0.0:
            continue
        if minimum_robust_gap is None:
            amount = 1.0
        else:
            amount = min(
                1.0,
                _GENERIC_REFINEMENT_SLACK_FRACTION
                * minimum_robust_gap
                / score_range,
            )
        if not math.isfinite(amount) or amount <= 0.0:
            continue
        consider(direction + amount * perturbation)

    if (
        best_direction is None
        or best_scores is None
        or best_order is None
        or not math.isfinite(best_margin)
    ):
        raise RuntimeError(
            "failed to construct a numerically strict linear refinement; "
            "rescale nearly identical feature rows or try another seed"
        )
    return best_direction, best_scores, best_order, best_margin


def _generic_selected_refinement_witness(
    score_basis: np.ndarray,
    base_direction: np.ndarray,
    base_scores: np.ndarray,
    selected_pairs: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Strictly refine an empirical-optimum witness without losing its rows.

    The selected comparison cone has nonempty interior because
    :func:`_normalized_selected_witness` supplies a positive-margin point.
    A continuous generic perturbation remains inside that cone when its scale
    is bounded by the selected slack and, with probability one, avoids every
    score-tie hyperplane.  We use a separate seeded generator for reproducible
    experiments and retain the candidate with the largest checked margin.
    """

    n, dimension = score_basis.shape
    direction = np.asarray(base_direction, dtype=np.float64).reshape(-1)
    scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    pairs = np.asarray(selected_pairs, dtype=np.int64)
    if direction.size != dimension or scores.size != n:
        raise RuntimeError("the empirical witness has inconsistent dimensions")
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or pairs.size == 0:
        raise RuntimeError("the empirical optimizer selected no comparisons")
    if np.any(~np.isfinite(direction)) or np.any(~np.isfinite(scores)):
        raise RuntimeError("the empirical witness contains nonfinite values")

    selected_differences = np.ascontiguousarray(
        score_basis[pairs[:, 0]] - score_basis[pairs[:, 1]]
    )
    score_selected_gaps = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    direct_selected_gaps = selected_differences @ direction
    if (
        np.any(~np.isfinite(score_selected_gaps))
        or np.any(~np.isfinite(direct_selected_gaps))
        or np.any(score_selected_gaps <= 0.0)
        or np.any(direct_selected_gaps <= 0.0)
    ):
        raise RuntimeError(
            "the empirical witness does not numerically preserve every "
            "selected comparison"
        )
    selected_slacks = np.minimum(score_selected_gaps, direct_selected_gaps)

    best_direction: np.ndarray | None = None
    best_scores: np.ndarray | None = None
    best_order: np.ndarray | None = None
    best_margin = -math.inf

    def consider(candidate: np.ndarray) -> None:
        nonlocal best_direction, best_scores, best_order, best_margin

        scale = float(np.max(np.abs(candidate)))
        if not math.isfinite(scale) or scale == 0.0:
            return
        normalized = np.ascontiguousarray(candidate / scale)
        candidate_scores = score_basis @ normalized
        if np.any(~np.isfinite(candidate_scores)):
            return
        if np.any(candidate_scores[pairs[:, 0]] <= candidate_scores[pairs[:, 1]]):
            return
        candidate_order = np.argsort(-candidate_scores, kind="stable").astype(
            np.int64, copy=False
        )
        margin = _strict_candidate_margin(
            score_basis, normalized, candidate_scores, candidate_order
        )
        if margin is None or margin <= best_margin:
            return
        best_direction = normalized
        best_scores = candidate_scores
        best_order = candidate_order
        best_margin = margin

    # The maximum-margin selected witness is often already generic.
    consider(direction)

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), _GENERIC_REFINEMENT_SEED_SALT])
    )
    for trial in range(_GENERIC_REFINEMENT_MAX_TRIALS):
        if trial >= _GENERIC_REFINEMENT_TRIALS and best_direction is not None:
            break
        perturbation = rng.standard_normal(dimension)
        perturbation_scale = float(np.max(np.abs(perturbation)))
        if not math.isfinite(perturbation_scale) or perturbation_scale == 0.0:
            continue
        perturbation /= perturbation_scale
        selected_effects = selected_differences @ perturbation
        decreasing = selected_effects < 0.0
        if np.any(decreasing):
            scale_limit = float(
                np.min(
                    _GENERIC_REFINEMENT_SLACK_FRACTION
                    * selected_slacks[decreasing]
                    / -selected_effects[decreasing]
                )
            )
            perturbation_amount = min(1.0, scale_limit)
        else:
            perturbation_amount = 1.0
        if not math.isfinite(perturbation_amount) or perturbation_amount <= 0.0:
            continue
        consider(direction + perturbation_amount * perturbation)

    if best_direction is None or best_scores is None or best_order is None:
        raise RuntimeError(
            "failed to construct a numerically strict generic refinement "
            "while preserving every selected empirical comparison; rescale "
            "the input coordinates or try another seed"
        )
    return best_direction, best_scores, best_order, best_margin


def _sample_pair_multiset(
    total_comparisons: int,
    sample_size: int,
    rng: np.random.Generator,
    max_unique_comparisons: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample pair IDs IID with replacement and return sorted multiplicities."""

    if total_comparisons <= _DENSE_COUNT_LIMIT:
        counts = np.zeros(total_comparisons, dtype=np.int64)
        remaining = sample_size
        while remaining:
            draw_count = min(remaining, _SAMPLING_CHUNK_SIZE)
            if (
                max_unique_comparisons is not None
                and total_comparisons > max_unique_comparisons
            ):
                remaining_unique_budget = max_unique_comparisons - int(
                    np.count_nonzero(counts)
                )
                draw_count = min(
                    draw_count,
                    max(1_024, remaining_unique_budget + 1),
                )
            draws = rng.integers(
                0,
                total_comparisons,
                size=draw_count,
                dtype=np.int64,
            )
            counts += np.bincount(draws, minlength=total_comparisons).astype(
                np.int64, copy=False
            )
            if (
                max_unique_comparisons is not None
                and total_comparisons > max_unique_comparisons
                and np.count_nonzero(counts) > max_unique_comparisons
            ):
                raise ValueError(
                    "the sample already contains more than "
                    f"max_unique_comparisons={max_unique_comparisons}; increase "
                    "the guard or use a smaller experimental sample"
                )
            remaining -= draw_count
        pair_ids = np.flatnonzero(counts).astype(np.int64, copy=False)
        multiplicities = counts[pair_ids]
        if (
            max_unique_comparisons is not None
            and pair_ids.size > max_unique_comparisons
        ):
            raise ValueError(
                f"the sample contains {pair_ids.size} unique comparisons, "
                f"exceeding max_unique_comparisons={max_unique_comparisons}; "
                "increase the guard or use a smaller experimental sample"
            )
        return pair_ids, multiplicities

    aggregated: dict[int, int] = {}
    remaining = sample_size
    while remaining:
        draw_count = min(remaining, _SAMPLING_CHUNK_SIZE)
        if max_unique_comparisons is not None:
            remaining_unique_budget = max_unique_comparisons - len(aggregated)
            draw_count = min(
                draw_count,
                max(1_024, remaining_unique_budget + 1),
            )
        draws = rng.integers(
            0,
            total_comparisons,
            size=draw_count,
            dtype=np.int64,
        )
        values, counts = np.unique(draws, return_counts=True)
        for pair_id, count in zip(values.tolist(), counts.tolist()):
            aggregated[pair_id] = aggregated.get(pair_id, 0) + count
        if (
            max_unique_comparisons is not None
            and len(aggregated) > max_unique_comparisons
        ):
            raise ValueError(
                f"the sample already contains more than "
                f"max_unique_comparisons={max_unique_comparisons}; increase "
                "the guard or use a smaller experimental sample"
            )
        remaining -= draw_count

    pair_ids = np.fromiter(sorted(aggregated), dtype=np.int64)
    multiplicities = np.fromiter(
        (aggregated[int(pair_id)] for pair_id in pair_ids),
        dtype=np.int64,
        count=pair_ids.size,
    )
    return pair_ids, multiplicities


def _solve_empirical_optimum(
    env: Any,
    score_basis: np.ndarray,
    target_pairs: np.ndarray,
    multiplicities: np.ndarray,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> tuple[np.ndarray, int, float]:
    """Solve the weighted sampled maximum-agreement problem to MIP optimality."""

    involved, inverse = np.unique(target_pairs, return_inverse=True)
    local_pairs = inverse.reshape(-1, 2)
    dimension = score_basis.shape[1]
    unique_count = target_pairs.shape[0]

    with gp.Model("sampled_max_agreement", env=env) as model:
        _core._set_model_parameters(
            model,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            seed=seed,
            gurobi_params=gurobi_params,
        )
        # Exact empirical optimality is required by the theorem.  These two
        # settings therefore intentionally override any same-named caller
        # parameters.
        model.setParam("MIPGap", 0.0)
        model.setParam("MIPGapAbs", 0.0)
        q = model.addMVar(dimension, lb=-GRB.INFINITY, name="q")
        local_score = model.addMVar(involved.size, lb=-GRB.INFINITY, name="score")
        selected = model.addMVar(unique_count, vtype=GRB.BINARY, name="selected")
        model.addConstr(local_score == np.ascontiguousarray(score_basis[involved]) @ q)
        for index, (higher, lower) in enumerate(local_pairs):
            model.addGenConstrIndicator(
                selected[index],
                True,
                local_score[int(higher)] - local_score[int(lower)],
                GRB.GREATER_EQUAL,
                1.0,
            )
        objective = gp.quicksum(
            int(weight) * selected[index] for index, weight in enumerate(multiplicities)
        )
        # For every generic direction, its negation satisfies the complementary
        # sample occurrences.  Thus the empirical optimum is at least ceil(s/2).
        # Recording this theorem bound strengthens the root relaxation without
        # changing the integer optimum.
        sample_size = int(np.sum(multiplicities, dtype=np.int64))
        model.addConstr(objective >= (sample_size + 1) // 2)
        model.setObjective(objective, GRB.MAXIMIZE)
        status = _core._optimize_for_status(model)
        if status != GRB.OPTIMAL:
            raise _core._status_error(model, "exact empirical optimization")

        selected_values = np.asarray(selected.X, dtype=np.float64).reshape(-1)
        selected_mask = selected_values > 0.5
        empirical_optimum = int(np.sum(multiplicities[selected_mask], dtype=np.int64))
        objective_value = float(model.ObjVal)
        objective_bound = float(model.ObjBound)
        node_count = float(model.NodeCount)

    if (
        np.any(
            np.minimum(np.abs(selected_values), np.abs(1.0 - selected_values)) > 1e-5
        )
        or abs(objective_value - empirical_optimum) >= 0.25
        or abs(objective_bound - empirical_optimum) >= 0.25
    ):
        raise RuntimeError(
            "Gurobi's empirical optimum did not have an unambiguous integer "
            "objective certificate"
        )
    return selected_mask, empirical_optimum, node_count


def _agreement_from_target_positions(
    ranking_target_positions: np.ndarray,
) -> tuple[int, int, int]:
    n = int(ranking_target_positions.size)
    total = n * (n - 1) // 2
    distance = _core._count_inversions(ranking_target_positions)
    return total - distance, distance, total


def _empirical_agreement(
    ranking_target_positions: np.ndarray,
    target_pairs: np.ndarray,
    multiplicities: np.ndarray,
) -> int:
    positions = np.empty(ranking_target_positions.size, dtype=np.int64)
    positions[ranking_target_positions] = np.arange(
        ranking_target_positions.size, dtype=np.int64
    )
    satisfied = positions[target_pairs[:, 0]] < positions[target_pairs[:, 1]]
    return int(np.sum(multiplicities[satisfied], dtype=np.int64))


def _aggregate_class_comparisons(
    ranking_input: _core._RankingInput,
    target_pairs: np.ndarray,
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse sampled item comparisons to weighted point-class rows."""

    class_pairs = ranking_input.point_class_ids_order[target_pairs]
    if np.any(class_pairs[:, 0] == class_pairs[:, 1]):
        raise RuntimeError("the optimization sample contains a forced tie")
    unique_pairs, inverse = np.unique(class_pairs, axis=0, return_inverse=True)
    weights = np.bincount(
        inverse,
        weights=multiplicities.astype(np.float64),
        minlength=unique_pairs.shape[0],
    )
    integer_weights = weights.astype(np.int64)
    if not np.array_equal(weights, integer_weights.astype(np.float64)):
        raise RuntimeError("class-comparison aggregation lost integer weights")
    return np.ascontiguousarray(unique_pairs), integer_weights


def _evaluate_class_order(
    ranking_input: _core._RankingInput,
    class_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Expand and evaluate a strict point-class order as a weak item ranking."""

    flattened, prediction_ids, prediction_order = _core._prediction_from_class_order(
        ranking_input, class_order
    )
    agreement, disagreement, total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    return (
        flattened,
        prediction_ids,
        prediction_order,
        agreement,
        disagreement,
        total,
    )


def _class_scores_in_target_order(
    ranking_input: _core._RankingInput,
    class_scores: np.ndarray,
) -> np.ndarray:
    scores_by_class = np.empty(ranking_input.point_class_count, dtype=np.float64)
    scores_by_class[:] = class_scores
    return scores_by_class[ranking_input.point_class_ids_order]


def _conditioned_scores_in_original_order(
    target_order: np.ndarray,
    scores_target_order: np.ndarray,
) -> np.ndarray:
    scores = np.empty(target_order.size, dtype=np.float64)
    scores[target_order] = scores_target_order
    return scores


def max_agreement_half_approximation(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
) -> HalfApproximationResult:
    """Implement ``thm:logdim-half-approximation``.

    Unless the target is entirely tied, a deterministic seeded generic direction
    in the conditioned score space gives a directly verified strict order of the
    distinct point classes.  Its reverse
    is induced by negating that direction, so their agreements sum to all
    optimizable target comparisons.  Target ties are excluded and exact
    duplicate points remain tied in the output.

    ``rank_tol=None`` preserves the supplied float64 comparison geometry.
    Supplying a nonnegative cutoff explicitly runs the algorithm on the SVD-
    projected instance, as in the companion minimum-disagreement module.
    """

    total_started = time.perf_counter()
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    x = ranking_input.points
    target = ranking_input.target_order
    n, ambient_dimension = x.shape
    normalized_seed = int(seed) % 2_000_000_000
    forced_ties = ranking_input.pair_index.forced_tie_disagreements
    optimizable = ranking_input.pair_index.modeled_comparisons

    if (
        ranking_input.point_class_count == 1
        or ranking_input.pair_index.target_comparisons == 0
    ):
        flattened, prediction_ids, prediction_order = _core._all_tied_prediction(
            ranking_input
        )
        agreement, distance, total_comparisons = _core._weak_order_counts(
            ranking_input.target_tie_ids_order, prediction_order
        )
        elapsed = time.perf_counter() - total_started
        return HalfApproximationResult(
            ranking=target[flattened],
            direction=np.zeros(ambient_dimension, dtype=np.float64),
            conditioned_scores=np.zeros(n, dtype=np.float64),
            target_order=target.copy(),
            target_tie_ids=ranking_input.target_tie_ids.copy(),
            ranking_tie_ids=prediction_ids,
            kendall_agreement=agreement,
            kendall_tau=distance,
            total_comparisons=total_comparisons,
            forced_tie_disagreements=forced_ties,
            optimizable_comparisons=optimizable,
            base_agreement=0,
            opposite_agreement=0,
            chosen_orientation=1,
            numerical_margin=0.0,
            dimension_bound=0,
            projected_instance=False,
            solver_runtime_seconds=0.0,
            total_runtime_seconds=elapsed,
        )

    parameterization = _core._score_parameterization(
        x,
        target,
        rank_tol,
        ranking_input.point_class_ids_order,
    )
    score_basis = parameterization.score_basis
    class_basis = score_basis[ranking_input.representative_target_positions]
    base_direction, base_scores, base_order, margin = (
        _generic_weak_refinement_witness(
            class_basis,
            np.zeros(class_basis.shape[1], dtype=np.float64),
            seed=normalized_seed,
        )
    )
    (
        _,
        _,
        _,
        base_agreement,
        _,
        total_comparisons,
    ) = _evaluate_class_order(ranking_input, base_order)
    opposite_order = base_order[::-1].copy()
    (
        _,
        _,
        _,
        opposite_agreement,
        _,
        opposite_total,
    ) = _evaluate_class_order(ranking_input, opposite_order)
    if total_comparisons != opposite_total:
        raise RuntimeError("base/reverse comparison populations differ")
    if base_agreement >= opposite_agreement:
        chosen_order = base_order
        reduced_direction = base_direction
        target_scores = base_scores
        orientation = 1
    else:
        chosen_order = base_order[::-1].copy()
        reduced_direction = -base_direction
        target_scores = -base_scores
        orientation = -1
    solver_elapsed = 0.0

    ambient_direction = parameterization.to_ambient(reduced_direction)
    class_representatives = ranking_input.representative_target_positions[chosen_order]
    _core._verify_original_ranking(x, target[class_representatives], ambient_direction)
    target_scores = _class_scores_in_target_order(ranking_input, target_scores)
    (
        flattened,
        prediction_ids,
        _,
        agreement,
        distance,
        checked_total,
    ) = _evaluate_class_order(ranking_input, chosen_order)
    ranking = target[flattened]
    if (
        checked_total != total_comparisons
        or agreement + distance != total_comparisons
        or base_agreement + opposite_agreement != optimizable
        or 2 * agreement < optimizable
        or agreement != max(base_agreement, opposite_agreement)
        or distance < forced_ties
    ):
        raise RuntimeError("the deterministic half-approximation certificate failed")

    return HalfApproximationResult(
        ranking=ranking.astype(np.int64, copy=False),
        direction=ambient_direction.astype(np.float64, copy=False),
        conditioned_scores=_conditioned_scores_in_original_order(target, target_scores),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=agreement,
        kendall_tau=distance,
        total_comparisons=total_comparisons,
        forced_tie_disagreements=forced_ties,
        optimizable_comparisons=optimizable,
        base_agreement=base_agreement,
        opposite_agreement=opposite_agreement,
        chosen_orientation=orientation,
        numerical_margin=margin,
        dimension_bound=parameterization.dimension_bound,
        projected_instance=parameterization.projected_instance,
        solver_runtime_seconds=solver_elapsed,
        total_runtime_seconds=time.perf_counter() - total_started,
    )


def max_agreement_similarity_ptas(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    epsilon: float,
    delta: float,
    sample_size: int | None = None,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
    max_sample_size: int | None = _DEFAULT_MAX_SAMPLE_SIZE,
    max_unique_comparisons: int | None = _DEFAULT_MAX_UNIQUE_COMPARISONS,
) -> SimilarityPTASResult:
    """Implement the randomized scheme in ``thm:logdim-similarity-ptas``.

    The default sample size satisfies the explicit bound computed by
    :func:`vc_uniform_sample_size` with ``eta = epsilon/4``.  Under the theorem's
    exact empirical-optimization assumptions, the returned ranking is a
    ``(1-epsilon)`` approximation with probability at least ``1-delta``.
    A separately seeded generic perturbation is scaled to preserve every
    selected empirical comparison while removing all remaining ties between
    distinct point classes; exact duplicate rows remain tied.  An entirely
    tied target takes the shared zero-direction fast path.

    A caller-supplied ``sample_size`` is useful for practical experiments.  The
    result reports ``sampling_bound_met=False`` when it is below the sufficient
    VC bound, in which case this implementation makes no epsilon/delta claim.
    The two resource guards fail before an impractically large solve; pass
    ``None`` to disable either guard.
    """

    total_started = time.perf_counter()
    epsilon_value = _validate_open_probability(epsilon, "epsilon")
    delta_value = _validate_open_probability(delta, "delta")
    eta = epsilon_value / 4.0
    if eta == 0.0:
        raise ValueError("epsilon is too small for float64 eta=epsilon/4")
    sample_limit = _validate_optional_limit(max_sample_size, "max_sample_size")
    unique_limit = _validate_optional_limit(
        max_unique_comparisons, "max_unique_comparisons"
    )
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    x = ranking_input.points
    target = ranking_input.target_order
    normalized_seed = int(seed) % 2_000_000_000
    forced_ties = ranking_input.pair_index.forced_tie_disagreements
    optimizable = ranking_input.pair_index.modeled_comparisons

    if optimizable == 0:
        if sample_size is not None:
            _validate_positive_integer(sample_size, "sample_size")
        half = max_agreement_half_approximation(
            points,
            target_order,
            target_ranks=target_ranks,
            rank_tol=rank_tol,
            seed=seed,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            gurobi_params=gurobi_params,
        )
        elapsed = time.perf_counter() - total_started
        return SimilarityPTASResult(
            ranking=half.ranking.copy(),
            direction=half.direction.copy(),
            conditioned_scores=half.conditioned_scores.copy(),
            target_order=target.copy(),
            target_tie_ids=ranking_input.target_tie_ids.copy(),
            ranking_tie_ids=half.ranking_tie_ids.copy(),
            kendall_agreement=half.kendall_agreement,
            kendall_tau=half.kendall_tau,
            total_comparisons=half.total_comparisons,
            forced_tie_disagreements=forced_ties,
            optimizable_comparisons=0,
            empirical_agreement=0,
            empirical_optimum=0,
            sample_size=0,
            unique_sample_size=0,
            sampled_comparisons=np.empty((0, 2), dtype=np.int64),
            sample_multiplicities=np.empty(0, dtype=np.int64),
            epsilon=epsilon_value,
            delta=delta_value,
            eta=eta,
            dimension_bound=half.dimension_bound,
            theoretical_sample_size=0,
            sampling_bound_met=True,
            selected_numerical_margin=0.0,
            numerical_margin=0.0,
            projected_instance=half.projected_instance,
            mip_node_count=0.0,
            solver_runtime_seconds=half.solver_runtime_seconds,
            total_runtime_seconds=elapsed,
        )

    _core._require_gurobi()
    parameterization = _core._score_parameterization(
        x,
        target,
        rank_tol,
        ranking_input.point_class_ids_order,
    )
    score_basis = parameterization.score_basis
    class_basis = score_basis[ranking_input.representative_target_positions]
    dimension_bound = parameterization.dimension_bound
    theoretical_sample_size = vc_uniform_sample_size(dimension_bound, eta, delta_value)
    if sample_size is None:
        actual_sample_size = theoretical_sample_size
    else:
        actual_sample_size = _validate_positive_integer(sample_size, "sample_size")
    if actual_sample_size > _MAX_EXACT_INTEGER_IN_FLOAT64:
        raise ValueError(
            "sample_size exceeds the largest integer that Gurobi can weight "
            "exactly in a float64 objective"
        )
    if sample_limit is not None and actual_sample_size > sample_limit:
        raise ValueError(
            f"sample_size={actual_sample_size} exceeds "
            f"max_sample_size={sample_limit}; increase/disable the guard or "
            "supply a smaller experimental sample"
        )
    sampling_bound_met = actual_sample_size >= theoretical_sample_size

    total_comparisons = ranking_input.pair_index.target_comparisons
    rng = np.random.default_rng(normalized_seed)
    pair_ids, multiplicities = _sample_pair_multiset(
        optimizable,
        actual_sample_size,
        rng,
        unique_limit,
    )
    if int(np.sum(multiplicities, dtype=np.int64)) != actual_sample_size:
        raise RuntimeError("sample multiplicities do not sum to sample_size")
    target_pairs = ranking_input.pair_index.decode(pair_ids)
    class_pairs, class_multiplicities = _aggregate_class_comparisons(
        ranking_input, target_pairs, multiplicities
    )

    solver_started = time.perf_counter()
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", int(output_flag))
            env.start()
            selected, empirical_optimum, node_count = _solve_empirical_optimum(
                env,
                class_basis,
                class_pairs,
                class_multiplicities,
                output_flag=output_flag,
                threads=threads,
                feasibility_tol=feasibility_tol,
                seed=normalized_seed,
                gurobi_params=gurobi_params,
            )
            weak_direction, weak_scores, selected_margin = _normalized_selected_witness(
                env,
                class_basis,
                class_pairs,
                selected,
                output_flag=output_flag,
                threads=threads,
                feasibility_tol=feasibility_tol,
                seed=normalized_seed,
                gurobi_params=gurobi_params,
            )
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi error {exc.errno}: {exc}") from exc
    solver_elapsed = time.perf_counter() - solver_started

    (
        reduced_direction,
        target_scores,
        class_order,
        strict_margin,
    ) = _generic_selected_refinement_witness(
        class_basis,
        weak_direction,
        weak_scores,
        class_pairs[selected],
        seed=normalized_seed,
    )

    empirical_agreement = _empirical_agreement(
        class_order, class_pairs, class_multiplicities
    )
    if empirical_agreement != empirical_optimum:
        raise RuntimeError(
            "final strict perturbation changed the certified empirical optimum"
        )
    (
        flattened,
        prediction_ids,
        _,
        agreement,
        distance,
        checked_total,
    ) = _evaluate_class_order(ranking_input, class_order)
    if checked_total != total_comparisons or agreement + distance != checked_total:
        raise RuntimeError("full Kendall agreement/distance verification failed")

    ambient_direction = parameterization.to_ambient(reduced_direction)
    representative_positions = ranking_input.representative_target_positions[
        class_order
    ]
    _core._verify_original_ranking(
        x, target[representative_positions], ambient_direction
    )
    target_scores = _class_scores_in_target_order(ranking_input, target_scores)
    ranking = target[flattened]
    sampled_comparisons = target[target_pairs]

    return SimilarityPTASResult(
        ranking=ranking.astype(np.int64, copy=False),
        direction=ambient_direction.astype(np.float64, copy=False),
        conditioned_scores=_conditioned_scores_in_original_order(target, target_scores),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=agreement,
        kendall_tau=distance,
        total_comparisons=total_comparisons,
        forced_tie_disagreements=forced_ties,
        optimizable_comparisons=optimizable,
        empirical_agreement=empirical_agreement,
        empirical_optimum=empirical_optimum,
        sample_size=actual_sample_size,
        unique_sample_size=int(pair_ids.size),
        sampled_comparisons=sampled_comparisons,
        sample_multiplicities=multiplicities.copy(),
        epsilon=epsilon_value,
        delta=delta_value,
        eta=eta,
        dimension_bound=dimension_bound,
        theoretical_sample_size=theoretical_sample_size,
        sampling_bound_met=sampling_bound_met,
        selected_numerical_margin=selected_margin,
        numerical_margin=strict_margin,
        projected_instance=parameterization.projected_instance,
        mip_node_count=node_count,
        solver_runtime_seconds=solver_elapsed,
        total_runtime_seconds=time.perf_counter() - total_started,
    )


def _add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
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
        help="one-column/row text or .npy target-rank vector",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="text delimiter, or 'whitespace' (default: ',')",
    )
    parser.add_argument(
        "--rank-tol",
        type=float,
        default=None,
        help=(
            "optional SVD cutoff (default: no projection; any supplied value "
            "models a projected instance)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--solver-log", action="store_true", help="display Gurobi's solver log"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="optional JSON result path"
    )


def _none_if_zero(value: int) -> int | None:
    if value < 0:
        raise argparse.ArgumentTypeError("resource limits must be nonnegative")
    return None if value == 0 else value


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the optimized Gurobi implementations of the max-agreement "
            "linear-ranking theorems."
        )
    )
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    half_parser = subparsers.add_parser(
        "half", help="deterministic theorem thm:logdim-half-approximation"
    )
    _add_common_cli_arguments(half_parser)

    ptas_parser = subparsers.add_parser(
        "ptas", help="randomized theorem thm:logdim-similarity-ptas"
    )
    _add_common_cli_arguments(ptas_parser)
    ptas_parser.add_argument("--epsilon", type=float, required=True)
    ptas_parser.add_argument("--delta", type=float, required=True)
    ptas_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="experimental override; default uses the sufficient VC bound",
    )
    ptas_parser.add_argument(
        "--max-sample-size",
        type=_none_if_zero,
        default=_DEFAULT_MAX_SAMPLE_SIZE,
        metavar="N",
        help=("resource guard (default: 10000000); use 0 to disable"),
    )
    ptas_parser.add_argument(
        "--max-unique-comparisons",
        type=_none_if_zero,
        default=_DEFAULT_MAX_UNIQUE_COMPARISONS,
        metavar="N",
        help="MILP-size guard (default: 10000); use 0 to disable",
    )

    args = parser.parse_args()
    try:
        point_matrix = _core._load_points(args.points, args.delimiter)
        target_ranks = args.target_ranks
        if args.target_ranks_file is not None:
            target_ranks = _core._load_target_ranks(
                args.target_ranks_file, args.delimiter
            )
        common = {
            "target_ranks": target_ranks,
            "rank_tol": args.rank_tol,
            "seed": args.seed,
            "output_flag": args.solver_log,
            "threads": args.threads,
        }
        if args.algorithm == "half":
            result = max_agreement_half_approximation(
                point_matrix, args.target_order, **common
            )
        else:
            result = max_agreement_similarity_ptas(
                point_matrix,
                args.target_order,
                epsilon=args.epsilon,
                delta=args.delta,
                sample_size=args.sample_size,
                max_sample_size=args.max_sample_size,
                max_unique_comparisons=args.max_unique_comparisons,
                **common,
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
