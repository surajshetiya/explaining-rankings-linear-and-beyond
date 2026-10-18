"""Scalable experimental baselines for closest linear ranking.

The five public functions in this module implement the baselines described in
``sections/experiment.tex``:

* :func:`linearized_mrc_baseline` -- linearized maximum rank correlation;
* :func:`rank_svm_baseline` -- regularized pairwise hinge loss;
* :func:`linear_ranknet_baseline` -- regularized pairwise logistic loss;
* :func:`kendall_local_search_baseline` -- direct Kendall local search; and
* :func:`elastic_maxfs_baseline` -- an elastic maximum-feasible-subsystem
  heuristic with greedy add-back.

Every method uses the same weak-ranking convention and returns the same core
ranking, agreement, distance, and timing fields as the manuscript algorithms.
Target-tied pairs are excluded; exact duplicate point rows remain tied; and a
duplicate pair separated by the target is an unavoidable disagreement.  NumPy
baselines avoid a quadratic pair matrix: full-data RankSVM gradients use a
tie-aware ``O(n log n + nd)`` endpoint-counting routine, sampled losses decode
at most a caller-controlled number of eligible comparisons, and full-data
evaluation uses one ``O(n log n)`` weak-order count.  MaxFS uses a persistent
score-factorized LP.

The raw optimizer may produce additional score ties.  When an optimizable
comparison exists, a bounded deterministic generic perturbation preserves
robust score separations, refines numerical ties, and directly verifies a
strict order of the distinct point classes.  This finalization needs no solver.
An entirely tied target returns the zero direction and one output tie.  The
flattened ``ranking`` remains a permutation for compatibility;
``ranking_tie_ids`` carries the actual weak-order semantics.

Dependencies: NumPy.  Elastic MaxFS additionally requires gurobipy and a
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
    from . import max_agreement_linear_ranking_gurobi as _max
except ImportError:
    import closest_linear_ranking_gurobi as _core
    import max_agreement_linear_ranking_gurobi as _max

gp = _core.gp
GRB = _core.GRB

_DEFAULT_MAX_TRAINING_PAIRS = 100_000
_DEFAULT_MAXFS_ADD_BACK = 1_000
_DENSE_LINEAR_ALGEBRA_LIMIT = 512
_GRADIENT_NORM_LIMIT = 1_000.0

__all__ = [
    "BaselineRankingResult",
    "elastic_maxfs_baseline",
    "kendall_local_search_baseline",
    "linear_ranknet_baseline",
    "linearized_mrc_baseline",
    "rank_svm_baseline",
]


@dataclass(frozen=True)
class BaselineRankingResult:
    """Common result returned by all five experimental baselines.

    Rankings and target orders contain original zero-based row indices from
    highest to lowest.  ``conditioned_scores`` are stored in original row
    order.  The method-specific ``diagnostics`` dictionary contains only
    JSON-serializable values.
    """

    algorithm: str
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
    numerical_margin: float
    dimension_bound: int
    projected_instance: bool
    optimization_iterations: int
    surrogate_objective: float | None
    training_sample_size: int
    unique_training_comparisons: int
    optimization_runtime_seconds: float
    solver_runtime_seconds: float
    total_runtime_seconds: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": self.algorithm,
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
            "numerical_margin": self.numerical_margin,
            "dimension_bound": self.dimension_bound,
            "projected_instance": self.projected_instance,
            "optimization_iterations": self.optimization_iterations,
            "surrogate_objective": self.surrogate_objective,
            "training_sample_size": self.training_sample_size,
            "unique_training_comparisons": self.unique_training_comparisons,
            "optimization_runtime_seconds": self.optimization_runtime_seconds,
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "diagnostics": dict(self.diagnostics),
        }


def _positive_integer(value: int, name: str) -> int:
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


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    try:
        exact = bool(integer == value)
    except (TypeError, ValueError):
        exact = False
    if integer < 0 or not exact:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _positive_real(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _nonnegative_real(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _optional_positive_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _optional_nonnegative_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _fraction(value: float, name: str, *, include_one: bool = True) -> float:
    number = _nonnegative_real(value, name)
    upper_ok = number <= 1.0 if include_one else number < 1.0
    if not upper_ok:
        endpoint = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {endpoint}")
    return number


def _normalized(vector: np.ndarray) -> np.ndarray:
    q = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.isfinite(q).all():
        raise RuntimeError("the baseline optimizer produced a nonfinite direction")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm):
        max_abs = float(np.max(np.abs(q), initial=0.0))
        if max_abs == 0.0 or not math.isfinite(max_abs):
            raise RuntimeError("the baseline direction cannot be normalized")
        q = q / max_abs
        norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return np.zeros_like(q)
    return np.ascontiguousarray(q / norm)


def _trivial_result(
    algorithm: str,
    ranking_input: _core._RankingInput,
    total_started: float,
    *,
    diagnostics: Mapping[str, Any],
    surrogate_objective: float | None = 0.0,
) -> BaselineRankingResult:
    x = ranking_input.points
    target = ranking_input.target_order
    if ranking_input.pair_index.modeled_comparisons != 0:
        raise RuntimeError("the empty baseline result has trainable comparisons")
    flattened, prediction_ids, prediction_order = _core._all_tied_prediction(
        ranking_input
    )
    agreement, distance, total_comparisons = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    return BaselineRankingResult(
        algorithm=algorithm,
        ranking=target[flattened],
        direction=np.zeros(x.shape[1], dtype=np.float64),
        conditioned_scores=np.zeros(x.shape[0], dtype=np.float64),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=agreement,
        kendall_tau=distance,
        total_comparisons=total_comparisons,
        forced_tie_disagreements=(ranking_input.pair_index.forced_tie_disagreements),
        optimizable_comparisons=ranking_input.pair_index.modeled_comparisons,
        numerical_margin=0.0,
        dimension_bound=0,
        projected_instance=False,
        optimization_iterations=0,
        surrogate_objective=surrogate_objective,
        training_sample_size=0,
        unique_training_comparisons=0,
        optimization_runtime_seconds=0.0,
        solver_runtime_seconds=0.0,
        total_runtime_seconds=time.perf_counter() - total_started,
        diagnostics=dict(diagnostics),
    )


def _finalize_result(
    *,
    algorithm: str,
    ranking_input: _core._RankingInput,
    parameterization: Any,
    reduced_direction: np.ndarray,
    total_started: float,
    optimization_runtime: float,
    solver_runtime_prefix: float,
    optimization_iterations: int,
    surrogate_objective: float | None,
    training_sample_size: int,
    unique_training_comparisons: int,
    diagnostics: Mapping[str, Any],
    seed: int,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    gurobi_params: Mapping[str, Any] | None,
) -> BaselineRankingResult:
    """Refine, directly certify, map, and evaluate a reduced direction."""

    q = np.asarray(reduced_direction, dtype=np.float64).reshape(-1)
    x = ranking_input.points
    target = ranking_input.target_order
    score_basis = parameterization.score_basis
    if q.size != score_basis.shape[1] or not np.isfinite(q).all():
        raise RuntimeError("the baseline returned an invalid reduced direction")
    class_basis = score_basis[ranking_input.representative_target_positions]
    weak_class_scores = class_basis @ q
    if not np.isfinite(weak_class_scores).all():
        raise RuntimeError("the baseline produced nonfinite conditioned scores")
    normalized_seed = int(seed) % 2_000_000_000
    final_q, class_scores, class_order, margin = (
        _max._generic_weak_refinement_witness(
            class_basis,
            q,
            seed=normalized_seed,
        )
    )
    # These options remain in the common API because Elastic MaxFS uses them
    # during its own optimization.  NumPy baselines need no final solver call.
    del output_flag, threads, feasibility_tol, gurobi_params

    ambient_direction = parameterization.to_ambient(final_q)
    representative_positions = ranking_input.representative_target_positions[
        class_order
    ]
    _core._verify_original_ranking(
        x, target[representative_positions], ambient_direction
    )
    flattened, prediction_ids, prediction_order = _core._prediction_from_class_order(
        ranking_input, class_order
    )
    agreement, distance, total_comparisons = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    forced_ties = ranking_input.pair_index.forced_tie_disagreements
    if agreement + distance != total_comparisons or distance < forced_ties:
        raise RuntimeError("the baseline's Kendall counts are inconsistent")
    target_scores = _max._class_scores_in_target_order(ranking_input, class_scores)

    return BaselineRankingResult(
        algorithm=algorithm,
        ranking=target[flattened].astype(np.int64, copy=False),
        direction=ambient_direction.astype(np.float64, copy=False),
        conditioned_scores=_max._conditioned_scores_in_original_order(
            target, target_scores
        ),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=agreement,
        kendall_tau=distance,
        total_comparisons=total_comparisons,
        forced_tie_disagreements=forced_ties,
        optimizable_comparisons=ranking_input.pair_index.modeled_comparisons,
        numerical_margin=margin,
        dimension_bound=parameterization.dimension_bound,
        projected_instance=parameterization.projected_instance,
        optimization_iterations=int(optimization_iterations),
        surrogate_objective=(
            None if surrogate_objective is None else float(surrogate_objective)
        ),
        training_sample_size=int(training_sample_size),
        unique_training_comparisons=int(unique_training_comparisons),
        optimization_runtime_seconds=float(optimization_runtime),
        solver_runtime_seconds=float(solver_runtime_prefix),
        total_runtime_seconds=time.perf_counter() - total_started,
        diagnostics=dict(diagnostics),
    )


def _linearized_mrc_direction(
    score_basis: np.ndarray,
    ridge: float,
    cg_tolerance: float,
    cg_max_iterations: int | None,
    target_tie_ids_order: np.ndarray | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Solve the ridge-linearized MRC normal equations without a large Gram."""

    n, dimension = score_basis.shape
    centered = np.ascontiguousarray(score_basis - np.mean(score_basis, axis=0))
    if target_tie_ids_order is None:
        target_scores = (n - 1 - 2 * np.arange(n, dtype=np.float64)) / max(1, n - 1)
    else:
        tie_ids = np.asarray(target_tie_ids_order, dtype=np.int64)
        if tie_ids.shape != (n,) or np.any(tie_ids[1:] < tie_ids[:-1]):
            raise RuntimeError("target tie IDs are inconsistent with target order")
        changes = np.flatnonzero(tie_ids[1:] != tie_ids[:-1]) + 1
        starts = np.concatenate((np.array([0], dtype=np.int64), changes))
        ends = np.concatenate((changes, np.array([n], dtype=np.int64)))
        group_starts = starts[tie_ids]
        group_ends = ends[tie_ids]
        # Midrank response: (# strictly below - # strictly above)/(n-1).
        target_scores = (n - group_ends - group_starts) / max(1, n - 1)
        target_scores = target_scores.astype(np.float64, copy=False)
    rhs = centered.T @ target_scores / n
    mean_variance = float(np.sum(centered * centered) / (n * dimension))
    alpha = ridge * max(mean_variance, np.finfo(np.float64).tiny)
    method: str

    if dimension <= _DENSE_LINEAR_ALGEBRA_LIMIT and dimension <= n:
        gram = centered.T @ centered / n
        if alpha:
            gram.flat[:: dimension + 1] += alpha
            try:
                q = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                q = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        else:
            q = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        iterations = 1
        method = "dense_primal"
    elif n <= _DENSE_LINEAR_ALGEBRA_LIMIT and n < dimension:
        if alpha:
            dual_system = centered @ centered.T
            dual_system.flat[:: n + 1] += n * alpha
            try:
                correction = np.linalg.solve(dual_system, centered @ rhs)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(dual_system, centered @ rhs, rcond=None)[0]
            q = (rhs - centered.T @ correction) / alpha
            method = "dense_dual"
        else:
            q = np.linalg.lstsq(centered, target_scores, rcond=None)[0]
            method = "dense_minimum_norm"
        iterations = 1
    else:
        maximum = (
            min(2_000, max(50, 4 * dimension))
            if cg_max_iterations is None
            else cg_max_iterations
        )
        diagonal = np.sum(centered * centered, axis=0) / n + alpha
        diagonal = np.maximum(diagonal, np.finfo(np.float64).tiny)

        def multiply(vector: np.ndarray) -> np.ndarray:
            return centered.T @ (centered @ vector) / n + alpha * vector

        q = np.zeros(dimension, dtype=np.float64)
        residual = rhs.copy()
        preconditioned = residual / diagonal
        search = preconditioned.copy()
        residual_product = float(residual @ preconditioned)
        rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
        iterations = 0
        for iteration in range(1, maximum + 1):
            product = multiply(search)
            denominator = float(search @ product)
            if not math.isfinite(denominator) or denominator <= 0.0:
                break
            step = residual_product / denominator
            q += step * search
            residual -= step * product
            iterations = iteration
            if float(np.linalg.norm(residual)) <= cg_tolerance * rhs_norm:
                break
            next_preconditioned = residual / diagonal
            next_product = float(residual @ next_preconditioned)
            if residual_product == 0.0 or not math.isfinite(next_product):
                break
            search = next_preconditioned + (next_product / residual_product) * search
            preconditioned = next_preconditioned
            residual_product = next_product
        method = "matrix_free_pcg"

    if not np.isfinite(q).all():
        raise RuntimeError("linearized MRC produced a nonfinite direction")
    residual = centered.T @ (centered @ q) / n + alpha * q - rhs
    relative_residual = float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny
    )
    return (
        _normalized(q),
        iterations,
        {
            "ridge": float(ridge),
            "absolute_ridge": float(alpha),
            "linear_solver": method,
            "relative_normal_equation_residual": relative_residual,
        },
    )


def linearized_mrc_baseline(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    ridge: float = 1e-6,
    cg_tolerance: float = 1e-10,
    cg_max_iterations: int | None = None,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
) -> BaselineRankingResult:
    """Return the ridge-regularized linearized-MRC baseline."""

    total_started = time.perf_counter()
    ridge_value = _nonnegative_real(ridge, "ridge")
    tolerance = _positive_real(cg_tolerance, "cg_tolerance")
    maximum = _optional_positive_integer(cg_max_iterations, "cg_max_iterations")
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    if ranking_input.pair_index.modeled_comparisons == 0:
        return _trivial_result(
            "linearized_mrc",
            ranking_input,
            total_started,
            diagnostics={"ridge": ridge_value},
            surrogate_objective=None,
        )

    parameterization = _core._score_parameterization(
        ranking_input.points,
        ranking_input.target_order,
        rank_tol,
        ranking_input.point_class_ids_order,
    )
    optimization_started = time.perf_counter()
    q, iterations, diagnostics = _linearized_mrc_direction(
        parameterization.score_basis,
        ridge_value,
        tolerance,
        maximum,
        ranking_input.target_tie_ids_order,
    )
    optimization_runtime = time.perf_counter() - optimization_started
    return _finalize_result(
        algorithm="linearized_mrc",
        ranking_input=ranking_input,
        parameterization=parameterization,
        reduced_direction=q,
        total_started=total_started,
        optimization_runtime=optimization_runtime,
        solver_runtime_prefix=0.0,
        optimization_iterations=iterations,
        surrogate_objective=None,
        training_sample_size=0,
        unique_training_comparisons=0,
        diagnostics=diagnostics,
        seed=seed,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        gurobi_params=gurobi_params,
    )


def _pair_training_sample(
    total_comparisons: int,
    max_training_pairs: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, int, int, bool]:
    """Return ``None`` for the full pair set or a bounded IID sample."""

    if max_training_pairs is None or max_training_pairs >= total_comparisons:
        return None, total_comparisons, total_comparisons, False
    pair_ids = rng.integers(
        0,
        total_comparisons,
        size=max_training_pairs,
        dtype=np.int64,
    )
    return pair_ids, pair_ids.size, int(np.unique(pair_ids).size), True


def _fenwick_add(tree: np.ndarray, index: int) -> None:
    while index < tree.size:
        tree[index] += 1
        index += index & -index


def _fenwick_prefix(tree: np.ndarray, count: int) -> int:
    total = 0
    while count > 0:
        total += int(tree[count])
        count -= count & -count
    return total


def _full_hinge_objective_gradient(
    score_basis: np.ndarray,
    q: np.ndarray,
    regularization: float,
    ranking_input: _core._RankingInput | None = None,
) -> tuple[float, np.ndarray, int]:
    """Exact eligible-pair hinge objective/subgradient in ``O(n log n + nd)``.

    Target groups are queried as batches, so target-tied pairs never enter the
    sweep.  Same-point-class counts are subtracted explicitly; their scores are
    necessarily equal, but they are structural ties rather than training rows.
    """

    n = score_basis.shape[0]
    if ranking_input is None:
        target_tie_ids = np.arange(n, dtype=np.int64)
        point_class_ids = np.arange(n, dtype=np.int64)
        total = n * (n - 1) // 2
    else:
        target_tie_ids = ranking_input.target_tie_ids_order
        point_class_ids = ranking_input.point_class_ids_order
        if target_tie_ids.shape != (n,) or point_class_ids.shape != (n,):
            raise RuntimeError("ranking metadata and score basis disagree")
        total = ranking_input.pair_index.modeled_comparisons
    if total <= 0:
        raise RuntimeError("the full hinge objective has no eligible comparisons")

    scores = score_basis @ q
    coordinates = np.unique(scores)
    score_indices = np.searchsorted(coordinates, scores, side="left") + 1
    outgoing = np.zeros(n, dtype=np.int64)
    incoming = np.zeros(n, dtype=np.int64)
    changes = np.flatnonzero(target_tie_ids[1:] != target_tie_ids[:-1]) + 1
    group_starts = np.concatenate((np.array([0], dtype=np.int64), changes))
    group_ends = np.concatenate((changes, np.array([n], dtype=np.int64)))
    class_count = int(np.max(point_class_ids)) + 1

    tree = np.zeros(coordinates.size + 1, dtype=np.int64)
    class_seen = np.zeros(class_count, dtype=np.int64)
    held = 0
    for start_raw, end_raw in zip(group_starts[::-1], group_ends[::-1], strict=True):
        start = int(start_raw)
        end = int(end_raw)
        for higher in range(start, end):
            cutoff = int(
                np.searchsorted(coordinates, scores[higher] - 1.0, side="right")
            )
            active = held - _fenwick_prefix(tree, cutoff)
            outgoing[higher] = active - class_seen[point_class_ids[higher]]
        for higher in range(start, end):
            _fenwick_add(tree, int(score_indices[higher]))
            class_seen[point_class_ids[higher]] += 1
        held += end - start

    tree.fill(0)
    class_seen.fill(0)
    for start_raw, end_raw in zip(group_starts, group_ends, strict=True):
        start = int(start_raw)
        end = int(end_raw)
        for lower in range(start, end):
            cutoff = int(np.searchsorted(coordinates, scores[lower] + 1.0, side="left"))
            active = _fenwick_prefix(tree, cutoff)
            incoming[lower] = active - class_seen[point_class_ids[lower]]
        for lower in range(start, end):
            _fenwick_add(tree, int(score_indices[lower]))
            class_seen[point_class_ids[lower]] += 1

    if np.any(outgoing < 0) or np.any(incoming < 0):
        raise RuntimeError("tie-aware hinge endpoint counts became negative")

    coefficients = outgoing - incoming
    active = int(np.sum(outgoing, dtype=np.int64))
    hinge_sum = float(active - coefficients @ scores)
    roundoff = 128.0 * np.finfo(np.float64).eps * max(1.0, float(active))
    if hinge_sum < -roundoff or not math.isfinite(hinge_sum):
        raise RuntimeError("the all-pairs hinge objective became numerically invalid")
    hinge_sum = max(0.0, hinge_sum)
    objective = hinge_sum / total + 0.5 * regularization * float(q @ q)
    gradient = regularization * q - score_basis.T @ coefficients / total
    return objective, np.asarray(gradient, dtype=np.float64), active


def _stable_logistic_weights(margins: np.ndarray) -> np.ndarray:
    weights = np.empty_like(margins)
    nonnegative = margins >= 0.0
    exponential = np.exp(-margins[nonnegative])
    weights[nonnegative] = exponential / (1.0 + exponential)
    exponential = np.exp(margins[~nonnegative])
    weights[~nonnegative] = 1.0 / (1.0 + exponential)
    return weights


def _sampled_objective(
    score_basis: np.ndarray,
    q: np.ndarray,
    pairs: np.ndarray,
    loss: str,
    regularization: float,
) -> float:
    scores = score_basis @ q
    margins = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    if loss == "hinge":
        empirical = float(np.mean(np.maximum(0.0, 1.0 - margins)))
    else:
        empirical = float(np.mean(np.logaddexp(0.0, -margins)))
    return empirical + 0.5 * regularization * float(q @ q)


def _weighted_sampled_objective_gradient(
    score_basis: np.ndarray,
    q: np.ndarray,
    pairs: np.ndarray,
    multiplicities: np.ndarray,
    sample_size: int,
    loss: str,
    regularization: float,
) -> tuple[float, np.ndarray]:
    """Vectorized sampled objective/gradient via endpoint accumulation."""

    scores = score_basis @ q
    margins = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    if loss == "hinge":
        derivative_weights = multiplicities * (margins < 1.0)
        empirical_sum = float(multiplicities @ np.maximum(0.0, 1.0 - margins))
    else:
        derivative_weights = multiplicities * _stable_logistic_weights(margins)
        empirical_sum = float(multiplicities @ np.logaddexp(0.0, -margins))
    coefficients = np.bincount(
        pairs[:, 0], weights=derivative_weights, minlength=score_basis.shape[0]
    )
    coefficients -= np.bincount(
        pairs[:, 1], weights=derivative_weights, minlength=score_basis.shape[0]
    )
    objective = empirical_sum / sample_size + 0.5 * regularization * float(q @ q)
    gradient = regularization * q - score_basis.T @ coefficients / sample_size
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise RuntimeError(f"{loss} objective or gradient became nonfinite")
    return objective, np.asarray(gradient, dtype=np.float64)


def _sampled_pairwise_adam(
    score_basis: np.ndarray,
    pair_ids: np.ndarray,
    pair_index: int | _core._ComparablePairIndex,
    *,
    loss: str,
    regularization: float,
    epochs: int,
    batch_size: int | None,
    learning_rate: float,
    rng: np.random.Generator,
    initial_direction: np.ndarray,
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    _, dimension = score_basis.shape
    pairs = _core._decode_pair_ids(pair_ids, pair_index)
    zero = np.zeros(dimension, dtype=np.float64)
    initial = _normalized(initial_direction)
    zero_objective = _sampled_objective(score_basis, zero, pairs, loss, regularization)
    initial_objective = _sampled_objective(
        score_basis, initial, pairs, loss, regularization
    )
    q = initial.copy() if initial_objective < zero_objective else zero
    first_moment = np.zeros_like(q)
    second_moment = np.zeros_like(q)
    best_q = q.copy()
    best_objective = _sampled_objective(score_basis, q, pairs, loss, regularization)
    best_epoch = 0
    step_number = 0

    if batch_size is None:
        unique_ids, multiplicities = np.unique(pair_ids, return_counts=True)
        unique_pairs = _core._decode_pair_ids(unique_ids, pair_index)
        multiplicities = multiplicities.astype(np.float64, copy=False)

    for epoch in range(1, epochs + 1):
        if batch_size is None:
            _, gradient = _weighted_sampled_objective_gradient(
                score_basis,
                q,
                unique_pairs,
                multiplicities,
                pair_ids.size,
                loss,
                regularization,
            )
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > _GRADIENT_NORM_LIMIT:
                gradient *= _GRADIENT_NORM_LIMIT / gradient_norm
            step_number += 1
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
            corrected_first = first_moment / (1.0 - 0.9**step_number)
            corrected_second = second_moment / (1.0 - 0.999**step_number)
            q -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        else:
            permutation = rng.permutation(pair_ids.size)
            for begin in range(0, pair_ids.size, batch_size):
                batch = pairs[permutation[begin : begin + batch_size]]
                higher = batch[:, 0]
                lower = batch[:, 1]
                margins = score_basis[higher] @ q - score_basis[lower] @ q
                if loss == "hinge":
                    weights = (margins < 1.0).astype(np.float64)
                else:
                    weights = _stable_logistic_weights(margins)
                gradient = regularization * q
                gradient -= (
                    weights @ score_basis[higher] - weights @ score_basis[lower]
                ) / batch.shape[0]
                gradient_norm = float(np.linalg.norm(gradient))
                if gradient_norm > _GRADIENT_NORM_LIMIT:
                    gradient *= _GRADIENT_NORM_LIMIT / gradient_norm

                step_number += 1
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
                corrected_first = first_moment / (1.0 - 0.9**step_number)
                corrected_second = second_moment / (1.0 - 0.999**step_number)
                q -= (
                    learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
                )
        if not np.isfinite(q).all():
            raise RuntimeError(f"{loss} optimization produced nonfinite values")

        objective = _sampled_objective(score_basis, q, pairs, loss, regularization)
        if objective < best_objective:
            best_objective = objective
            best_q = q.copy()
            best_epoch = epoch

    return (
        best_q,
        best_objective,
        epochs,
        {
            "best_epoch": best_epoch,
            "optimizer": (
                "endpoint_full_batch_adam" if batch_size is None else "minibatch_adam"
            ),
            "pair_updates": int(pair_ids.size * epochs),
            "initialization": (
                "linearized_mrc" if initial_objective < zero_objective else "zero"
            ),
        },
    )


def _full_rank_svm_adam(
    score_basis: np.ndarray,
    *,
    regularization: float,
    epochs: int,
    learning_rate: float,
    initial_direction: np.ndarray,
    ranking_input: _core._RankingInput | None = None,
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    dimension = score_basis.shape[1]
    zero = np.zeros(dimension, dtype=np.float64)
    initial = _normalized(initial_direction)
    zero_objective = _full_hinge_objective_gradient(
        score_basis, zero, regularization, ranking_input
    )[0]
    initial_objective = _full_hinge_objective_gradient(
        score_basis, initial, regularization, ranking_input
    )[0]
    q = initial.copy() if initial_objective < zero_objective else zero
    first_moment = np.zeros_like(q)
    second_moment = np.zeros_like(q)
    best_q = q.copy()
    best_objective = math.inf
    best_epoch = 0
    last_active = 0

    for epoch in range(epochs + 1):
        objective, gradient, active = _full_hinge_objective_gradient(
            score_basis, q, regularization, ranking_input
        )
        if objective < best_objective:
            best_objective = objective
            best_q = q.copy()
            best_epoch = epoch
        last_active = active
        if epoch == epochs:
            break
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > _GRADIENT_NORM_LIMIT:
            gradient *= _GRADIENT_NORM_LIMIT / gradient_norm
        step = epoch + 1
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        q -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if not np.isfinite(q).all():
            raise RuntimeError("RankSVM optimization produced nonfinite values")

    return (
        best_q,
        best_objective,
        epochs,
        {
            "best_epoch": best_epoch,
            "optimizer": "full_pair_endpoint_adam",
            "active_hinge_pairs_at_last_iterate": last_active,
            "initialization": (
                "linearized_mrc" if initial_objective < zero_objective else "zero"
            ),
        },
    )


def _pairwise_baseline(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None,
    algorithm: str,
    loss: str,
    regularization: float,
    max_training_pairs: int | None,
    epochs: int,
    batch_size: int | None,
    learning_rate: float,
    rank_tol: float | None,
    seed: int,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    gurobi_params: Mapping[str, Any] | None,
) -> BaselineRankingResult:
    total_started = time.perf_counter()
    regularization_value = _nonnegative_real(regularization, "regularization")
    pair_limit = _optional_positive_integer(max_training_pairs, "max_training_pairs")
    epoch_count = _positive_integer(epochs, "epochs")
    validated_batch = _optional_positive_integer(batch_size, "batch_size")
    rate = _positive_real(learning_rate, "learning_rate")
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    if ranking_input.pair_index.modeled_comparisons == 0:
        return _trivial_result(
            algorithm,
            ranking_input,
            total_started,
            diagnostics={
                "loss": loss,
                "regularization": regularization_value,
                "comparison_population": "empty",
            },
        )

    parameterization = _core._score_parameterization(
        ranking_input.points,
        ranking_input.target_order,
        rank_tol,
        ranking_input.point_class_ids_order,
    )
    total_comparisons = ranking_input.pair_index.modeled_comparisons
    rng = np.random.default_rng(int(seed) % 2_000_000_000)
    pair_ids, sample_size, unique_size, sampled = _pair_training_sample(
        total_comparisons, pair_limit, rng
    )

    optimization_started = time.perf_counter()
    initial_q = _linearized_mrc_direction(
        parameterization.score_basis,
        1e-6,
        1e-6,
        50,
        ranking_input.target_tie_ids_order,
    )[0]
    if total_comparisons == 0:
        q = initial_q
        objective = 0.0
        iterations = 0
        diagnostics = {
            "best_epoch": 0,
            "optimizer": "empty_comparison_population",
            "initialization": "linearized_mrc",
        }
    elif loss == "hinge" and pair_ids is None:
        q, objective, iterations, diagnostics = _full_rank_svm_adam(
            parameterization.score_basis,
            regularization=regularization_value,
            epochs=epoch_count,
            learning_rate=rate,
            initial_direction=initial_q,
            ranking_input=ranking_input,
        )
    else:
        if pair_ids is None:
            pair_ids = np.arange(total_comparisons, dtype=np.int64)
        q, objective, iterations, diagnostics = _sampled_pairwise_adam(
            parameterization.score_basis,
            pair_ids,
            ranking_input.pair_index,
            loss=loss,
            regularization=regularization_value,
            epochs=epoch_count,
            batch_size=validated_batch,
            learning_rate=rate,
            rng=rng,
            initial_direction=initial_q,
        )
    optimization_runtime = time.perf_counter() - optimization_started
    if not math.isfinite(objective) or objective < 0.0:
        raise RuntimeError("the retained surrogate objective is invalid")
    diagnostics.update(
        {
            "loss": loss,
            "regularization": regularization_value,
            "learning_rate": rate,
            "comparison_population": (
                "empty"
                if total_comparisons == 0
                else ("sampled" if sampled else "full")
            ),
            "batch_size": validated_batch,
        }
    )
    return _finalize_result(
        algorithm=algorithm,
        ranking_input=ranking_input,
        parameterization=parameterization,
        reduced_direction=q,
        total_started=total_started,
        optimization_runtime=optimization_runtime,
        solver_runtime_prefix=0.0,
        optimization_iterations=iterations,
        surrogate_objective=objective,
        training_sample_size=sample_size,
        unique_training_comparisons=unique_size,
        diagnostics=diagnostics,
        seed=seed,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        gurobi_params=gurobi_params,
    )


def rank_svm_baseline(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    regularization: float = 1e-3,
    max_training_pairs: int | None = _DEFAULT_MAX_TRAINING_PAIRS,
    epochs: int = 100,
    batch_size: int | None = None,
    learning_rate: float = 0.03,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
) -> BaselineRankingResult:
    """Train a linear RankSVM on all eligible pairs or a uniform sample.

    When all eligible comparisons fit under ``max_training_pairs``, exact hinge
    objectives and subgradients are computed without materializing the pair
    set.  Setting the guard to ``None`` forces this scalable full-eligible-pair
    mode.
    """

    return _pairwise_baseline(
        points,
        target_order,
        target_ranks=target_ranks,
        algorithm="rank_svm",
        loss="hinge",
        regularization=regularization,
        max_training_pairs=max_training_pairs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rank_tol=rank_tol,
        seed=seed,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        gurobi_params=gurobi_params,
    )


def linear_ranknet_baseline(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    regularization: float = 1e-3,
    max_training_pairs: int | None = _DEFAULT_MAX_TRAINING_PAIRS,
    epochs: int = 100,
    batch_size: int | None = None,
    learning_rate: float = 0.03,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
) -> BaselineRankingResult:
    """Train linear RankNet on all eligible pairs or a bounded uniform sample."""

    return _pairwise_baseline(
        points,
        target_order,
        target_ranks=target_ranks,
        algorithm="linear_ranknet",
        loss="logistic",
        regularization=regularization,
        max_training_pairs=max_training_pairs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rank_tol=rank_tol,
        seed=seed,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        gurobi_params=gurobi_params,
    )


def _ranking_quality(
    class_scores: np.ndarray,
    class_basis: np.ndarray,
    ranking_input: _core._RankingInput,
) -> tuple[np.ndarray, int, float]:
    order = _max._lexicographic_refinement_order(class_scores, class_basis)
    _, _, prediction_order = _core._prediction_from_class_order(ranking_input, order)
    _, distance, _ = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    gaps = class_scores[order[:-1]] - class_scores[order[1:]]
    minimum_gap = float(np.min(gaps)) if gaps.size else 0.0
    return order, distance, minimum_gap


def _quality_better(
    distance: int,
    margin: float,
    incumbent_distance: int,
    incumbent_margin: float,
) -> bool:
    if distance != incumbent_distance:
        return distance < incumbent_distance
    tolerance = (
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(margin), abs(incumbent_margin))
    )
    return margin > incumbent_margin + tolerance


def kendall_local_search_baseline(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    restarts: int = 8,
    max_iterations: int = 250,
    initial_step: float = 0.25,
    step_decay: float = 0.5,
    patience: int = 25,
    min_step: float = 1e-4,
    ridge: float = 1e-6,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
) -> BaselineRankingResult:
    """Directly minimize full-data Kendall distance by multistart search.

    Each iteration tests both signs of one coordinate update.  Every tenth
    iteration also tests a Gaussian direction.  Score vectors are updated in
    ``O(n)`` for coordinate proposals, while exact Kendall evaluation takes
    ``O(n log n)``.  Non-worsening plateau moves are permitted; the globally
    retained solution is never worse than the linearized-MRC initialization.
    """

    total_started = time.perf_counter()
    restart_count = _positive_integer(restarts, "restarts")
    iteration_limit = _positive_integer(max_iterations, "max_iterations")
    initial_step_value = _positive_real(initial_step, "initial_step")
    decay = _positive_real(step_decay, "step_decay")
    if decay >= 1.0:
        raise ValueError("step_decay must lie strictly between 0 and 1")
    patience_value = _positive_integer(patience, "patience")
    minimum_step = _positive_real(min_step, "min_step")
    if minimum_step >= initial_step_value:
        raise ValueError("min_step must be smaller than initial_step")
    ridge_value = _nonnegative_real(ridge, "ridge")
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    if ranking_input.pair_index.modeled_comparisons == 0:
        return _trivial_result(
            "kendall_local_search",
            ranking_input,
            total_started,
            diagnostics={"restarts": restart_count},
            surrogate_objective=float(
                ranking_input.pair_index.forced_tie_disagreements
            ),
        )

    parameterization = _core._score_parameterization(
        ranking_input.points,
        ranking_input.target_order,
        rank_tol,
        ranking_input.point_class_ids_order,
    )
    score_basis = parameterization.score_basis
    class_basis = score_basis[ranking_input.representative_target_positions]
    total_comparisons = ranking_input.pair_index.modeled_comparisons
    rng = np.random.default_rng(int(seed) % 2_000_000_000)
    optimization_started = time.perf_counter()
    lmrc_q, _, lmrc_diagnostics = _linearized_mrc_direction(
        score_basis,
        ridge_value,
        1e-8,
        None,
        ranking_input.target_tie_ids_order,
    )
    lmrc_scores = class_basis @ lmrc_q
    lmrc_order, lmrc_distance, lmrc_margin = _ranking_quality(
        lmrc_scores, class_basis, ranking_input
    )

    if total_comparisons == 0:
        optimization_runtime = time.perf_counter() - optimization_started
        return _finalize_result(
            algorithm="kendall_local_search",
            ranking_input=ranking_input,
            parameterization=parameterization,
            reduced_direction=lmrc_q,
            total_started=total_started,
            optimization_runtime=optimization_runtime,
            solver_runtime_prefix=0.0,
            optimization_iterations=0,
            surrogate_objective=0.0,
            training_sample_size=0,
            unique_training_comparisons=0,
            diagnostics={
                "restarts": restart_count,
                "comparison_population": "empty",
                "initial_lmrc_kendall_tau": lmrc_distance,
                "ridge": ridge_value,
                "lmrc_linear_solver": lmrc_diagnostics["linear_solver"],
            },
            seed=seed,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            gurobi_params=gurobi_params,
        )

    global_q = lmrc_q.copy()
    global_order = lmrc_order.copy()
    global_distance = lmrc_distance
    global_margin = lmrc_margin
    objective_evaluations = 1
    strict_improvements = 0
    neutral_moves = 0
    performed_iterations = 0
    final_steps: list[float] = []

    seeds: list[np.ndarray] = [lmrc_q]
    if restart_count > 1:
        if np.any(lmrc_q != 0.0):
            seeds.append(_normalized(-lmrc_q))
        else:
            seeds.append(_normalized(rng.standard_normal(class_basis.shape[1])))
    while len(seeds) < restart_count:
        seeds.append(_normalized(rng.standard_normal(class_basis.shape[1])))

    for initial_q in seeds:
        q = initial_q.copy()
        scores = class_basis @ q
        order, distance, margin = _ranking_quality(scores, class_basis, ranking_input)
        objective_evaluations += 1
        step = initial_step_value
        stagnant = 0

        for iteration in range(iteration_limit):
            if step < minimum_step:
                break
            performed_iterations += 1
            coordinate = int(rng.integers(0, class_basis.shape[1]))
            candidates: list[tuple[np.ndarray, np.ndarray]] = []
            for sign in (-1.0, 1.0):
                raw_q = q.copy()
                raw_q[coordinate] += sign * step
                norm = float(np.linalg.norm(raw_q))
                if norm > 0.0 and math.isfinite(norm):
                    candidate_q = raw_q / norm
                    candidate_scores = (
                        scores + sign * step * class_basis[:, coordinate]
                    ) / norm
                    candidates.append((candidate_q, candidate_scores))

            if (iteration + 1) % 10 == 0:
                gaussian = _normalized(rng.standard_normal(class_basis.shape[1]))
                raw_q = q + step * gaussian
                norm = float(np.linalg.norm(raw_q))
                if norm > 0.0 and math.isfinite(norm):
                    candidates.append(
                        (
                            raw_q / norm,
                            (scores + step * (class_basis @ gaussian)) / norm,
                        )
                    )

            candidate_records: list[
                tuple[np.ndarray, np.ndarray, np.ndarray, int, float]
            ] = []
            for candidate_q, candidate_scores in candidates:
                candidate_order, candidate_distance, candidate_margin = (
                    _ranking_quality(candidate_scores, class_basis, ranking_input)
                )
                objective_evaluations += 1
                candidate_records.append(
                    (
                        candidate_q,
                        candidate_scores,
                        candidate_order,
                        candidate_distance,
                        candidate_margin,
                    )
                )
            if not candidate_records:
                stagnant += 1
            else:
                chosen = candidate_records[0]
                for candidate in candidate_records[1:]:
                    if _quality_better(
                        candidate[3], candidate[4], chosen[3], chosen[4]
                    ):
                        chosen = candidate
                strict = chosen[3] < distance
                same_objective = chosen[3] == distance
                changed_order = not np.array_equal(chosen[2], order)
                nonworsening_plateau = (
                    same_objective
                    and changed_order
                    and chosen[4]
                    >= margin
                    - 64.0
                    * np.finfo(np.float64).eps
                    * max(1.0, abs(chosen[4]), abs(margin))
                )
                if strict or nonworsening_plateau:
                    q, scores, order, distance, margin = chosen
                    if strict:
                        strict_improvements += 1
                        stagnant = 0
                    else:
                        neutral_moves += 1
                        stagnant += 1
                else:
                    stagnant += 1

            if _quality_better(distance, margin, global_distance, global_margin):
                global_q = q.copy()
                global_order = order.copy()
                global_distance = distance
                global_margin = margin
            if stagnant >= patience_value:
                step *= decay
                stagnant = 0
        final_steps.append(step)

    optimization_runtime = time.perf_counter() - optimization_started
    checked_order, checked_distance, _ = _ranking_quality(
        class_basis @ global_q, class_basis, ranking_input
    )
    if checked_distance != global_distance or not np.array_equal(
        checked_order, global_order
    ):
        raise RuntimeError("the local-search incumbent record is inconsistent")
    diagnostics: dict[str, Any] = {
        "restarts": restart_count,
        "max_iterations_per_restart": iteration_limit,
        "objective_evaluations": objective_evaluations,
        "strict_improvements": strict_improvements,
        "neutral_plateau_moves": neutral_moves,
        "initial_lmrc_kendall_tau": lmrc_distance,
        "initial_step": initial_step_value,
        "smallest_final_step": float(min(final_steps)),
        "ridge": ridge_value,
        "lmrc_linear_solver": lmrc_diagnostics["linear_solver"],
    }
    return _finalize_result(
        algorithm="kendall_local_search",
        ranking_input=ranking_input,
        parameterization=parameterization,
        reduced_direction=global_q,
        total_started=total_started,
        optimization_runtime=optimization_runtime,
        solver_runtime_prefix=0.0,
        optimization_iterations=performed_iterations,
        surrogate_objective=float(global_distance),
        training_sample_size=total_comparisons,
        unique_training_comparisons=total_comparisons,
        diagnostics=diagnostics,
        seed=seed,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        gurobi_params=gurobi_params,
    )


def _elastic_zero_threshold(total_weight: int, tolerance: float) -> float:
    # The relative threshold is useful for large sampled objectives, while the
    # cap preserves a plainly positive margin for every retained comparison.
    return min(0.25, tolerance * max(1, total_weight))


def elastic_maxfs_baseline(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    max_training_pairs: int | None = _DEFAULT_MAX_TRAINING_PAIRS,
    deletion_fraction: float = 0.05,
    max_removal_iterations: int = 1_000,
    max_add_back: int | None = _DEFAULT_MAXFS_ADD_BACK,
    elastic_tolerance: float = 1e-8,
    rank_tol: float | None = None,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
) -> BaselineRankingResult:
    """Run a Chinneck-style elastic MaxFS deletion/add-back heuristic.

    The model minimizes weighted elastic violation over either every eligible
    target comparison or an IID sample capped by ``max_training_pairs``.  At each
    removal round it drops the largest raw elastic slacks (one row when
    ``deletion_fraction=0``), warm-starts the persistent LP, and stops once the
    retained rows are jointly feasible.  Deleted rows are then tried in reverse
    removal order.  ``max_add_back=0`` disables add-back and ``None`` tests all
    deleted rows.

    This is a heuristic: neither the deleted subsystem nor its cardinality is
    claimed optimal.  Full-data agreement and disagreement are always
    recomputed independently from the returned weak ranking.
    """

    total_started = time.perf_counter()
    pair_limit = _optional_positive_integer(max_training_pairs, "max_training_pairs")
    removal_fraction = _fraction(deletion_fraction, "deletion_fraction")
    removal_limit = _positive_integer(max_removal_iterations, "max_removal_iterations")
    add_back_limit = _optional_nonnegative_integer(max_add_back, "max_add_back")
    tolerance = _positive_real(elastic_tolerance, "elastic_tolerance")
    if tolerance > 1e-3:
        raise ValueError("elastic_tolerance must not exceed 1e-3")
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    if ranking_input.pair_index.modeled_comparisons == 0:
        return _trivial_result(
            "elastic_maxfs",
            ranking_input,
            total_started,
            diagnostics={
                "comparison_population": "empty",
                "deleted_count": 0,
                "unique_deleted_count": 0,
            },
        )

    _core._require_gurobi()
    x = ranking_input.points
    target = ranking_input.target_order
    parameterization = _core._score_parameterization(
        x,
        target,
        rank_tol,
        ranking_input.point_class_ids_order,
    )
    score_basis = parameterization.score_basis
    class_basis = score_basis[ranking_input.representative_target_positions]
    n = x.shape[0]
    total_comparisons = ranking_input.pair_index.modeled_comparisons
    if total_comparisons == 0:
        return _finalize_result(
            algorithm="elastic_maxfs",
            ranking_input=ranking_input,
            parameterization=parameterization,
            reduced_direction=np.zeros(score_basis.shape[1], dtype=np.float64),
            total_started=total_started,
            optimization_runtime=0.0,
            solver_runtime_prefix=0.0,
            optimization_iterations=0,
            surrogate_objective=0.0,
            training_sample_size=0,
            unique_training_comparisons=0,
            diagnostics={
                "comparison_population": "empty",
                "deleted_count": 0,
                "unique_deleted_count": 0,
                "removed_comparisons": [],
                "removed_multiplicities": [],
            },
            seed=seed,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            gurobi_params=gurobi_params,
        )
    rng = np.random.default_rng(int(seed) % 2_000_000_000)
    sampled = pair_limit is not None and pair_limit < total_comparisons
    if sampled:
        pair_ids, multiplicities = _max._sample_pair_multiset(
            total_comparisons,
            int(pair_limit),
            rng,
            None,
        )
        training_sample_size = int(pair_limit)
    else:
        pair_ids = np.arange(total_comparisons, dtype=np.int64)
        multiplicities = np.ones(total_comparisons, dtype=np.int64)
        training_sample_size = total_comparisons
    pairs = _core._decode_pair_ids(pair_ids, ranking_input.pair_index)
    unique_count = int(pair_ids.size)
    if unique_count == 0 or int(np.sum(multiplicities)) != training_sample_size:
        raise RuntimeError("the MaxFS training multiset is inconsistent")

    objective_coefficients = multiplicities.astype(np.float64)
    active = np.ones(unique_count, dtype=bool)
    deleted_batches: list[np.ndarray] = []
    removal_iterations = 0
    add_back_attempts = 0
    add_back_successes = 0
    solver_optimizations = 0
    removal_limit_triggered = False
    final_elastic_objective = math.inf
    optimization_started = time.perf_counter()
    solver_started = time.perf_counter()

    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", int(output_flag))
            env.start()
            with gp.Model("elastic_maxfs_baseline", env=env) as model:
                _core._set_model_parameters(
                    model,
                    output_flag=output_flag,
                    threads=threads,
                    feasibility_tol=feasibility_tol,
                    seed=int(seed) % 2_000_000_000,
                    gurobi_params=gurobi_params,
                )
                q = model.addMVar(score_basis.shape[1], lb=-GRB.INFINITY, name="q")
                score = model.addMVar(n, lb=-GRB.INFINITY, name="score")
                elastic = model.addMVar(unique_count, lb=0.0, name="elastic")
                model.addConstr(score == score_basis @ q)
                model.addConstr(
                    score[pairs[:, 0]] - score[pairs[:, 1]] + elastic >= 1.0
                )
                model.setObjective(objective_coefficients @ elastic, GRB.MINIMIZE)
                model.update()

                while True:
                    status = _core._optimize_for_status(model)
                    solver_optimizations += 1
                    if status != GRB.OPTIMAL:
                        raise _core._status_error(model, "elastic MaxFS optimization")
                    final_elastic_objective = float(model.ObjVal)
                    active_weight = int(np.sum(multiplicities[active], dtype=np.int64))
                    zero_threshold = _elastic_zero_threshold(active_weight, tolerance)
                    if final_elastic_objective <= zero_threshold:
                        break

                    elastic_values = np.asarray(elastic.X, dtype=np.float64).reshape(-1)
                    active_indices = np.flatnonzero(active)
                    if active_indices.size == 0:
                        raise RuntimeError(
                            "the empty MaxFS subsystem had positive slack"
                        )
                    positive = active_indices[
                        elastic_values[active_indices] > max(tolerance, feasibility_tol)
                    ]
                    if positive.size == 0:
                        positive = active_indices[
                            np.argmax(elastic_values[active_indices]) : np.argmax(
                                elastic_values[active_indices]
                            )
                            + 1
                        ]

                    if removal_iterations >= removal_limit:
                        # Preserve termination even under an unusually long
                        # deletion chain; the fallback remains a valid MaxFS
                        # heuristic but is explicitly reported.
                        selected = active_indices
                        removal_limit_triggered = True
                    else:
                        remove_count = (
                            1
                            if removal_fraction == 0.0
                            else max(
                                1,
                                int(math.ceil(removal_fraction * positive.size)),
                            )
                        )
                        ordering = np.argsort(-elastic_values[positive], kind="stable")
                        selected = positive[ordering[:remove_count]]
                    active[selected] = False
                    objective_coefficients[selected] = 0.0
                    deleted_batches.append(selected.copy())
                    removal_iterations += 1
                    elastic.Obj = objective_coefficients
                    model.update()

                deleted_order = (
                    np.concatenate([batch[::-1] for batch in reversed(deleted_batches)])
                    if deleted_batches
                    else np.empty(0, dtype=np.int64)
                )
                attempted_indices = deleted_order
                if add_back_limit is not None:
                    attempted_indices = attempted_indices[:add_back_limit]
                for index_raw in attempted_indices:
                    index = int(index_raw)
                    add_back_attempts += 1
                    active[index] = True
                    objective_coefficients[index] = float(multiplicities[index])
                    elastic.Obj = objective_coefficients
                    model.update()
                    status = _core._optimize_for_status(model)
                    solver_optimizations += 1
                    if status != GRB.OPTIMAL:
                        raise _core._status_error(model, "MaxFS greedy add-back")
                    active_weight = int(np.sum(multiplicities[active], dtype=np.int64))
                    threshold = _elastic_zero_threshold(active_weight, tolerance)
                    if float(model.ObjVal) <= threshold:
                        add_back_successes += 1
                        continue

                    active[index] = False
                    objective_coefficients[index] = 0.0
                    elastic.Obj = objective_coefficients
                    model.update()
                    # A rejected row leaves the incumbent at a positive
                    # objective.  Reoptimize immediately before another test.
                    status = _core._optimize_for_status(model)
                    solver_optimizations += 1
                    if status != GRB.OPTIMAL:
                        raise _core._status_error(
                            model, "MaxFS state restoration after add-back"
                        )

                # Ensure the final q belongs to the final active subsystem,
                # including when add-back was disabled or truncated.
                elastic.Obj = objective_coefficients
                model.update()
                status = _core._optimize_for_status(model)
                solver_optimizations += 1
                if status != GRB.OPTIMAL:
                    raise _core._status_error(model, "final elastic MaxFS model")
                final_elastic_objective = float(model.ObjVal)
                reduced_direction = np.asarray(q.X, dtype=np.float64).reshape(-1)
                model_scores = np.asarray(score.X, dtype=np.float64).reshape(-1)
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi error {exc.errno}: {exc}") from exc

    solver_runtime = time.perf_counter() - solver_started
    optimization_runtime = time.perf_counter() - optimization_started
    active_weight = int(np.sum(multiplicities[active], dtype=np.int64))
    if final_elastic_objective > _elastic_zero_threshold(active_weight, tolerance):
        raise RuntimeError("the final MaxFS elastic subsystem was not feasible")
    active_gaps = model_scores[pairs[active, 0]] - model_scores[pairs[active, 1]]
    if active_gaps.size and np.any(active_gaps <= 0.0):
        raise RuntimeError("the MaxFS witness reverses an active comparison")

    deleted = ~active
    deleted_count = int(np.sum(multiplicities[deleted], dtype=np.int64))
    deleted_pairs_original = target[pairs[deleted]]
    normalized_direction = _normalized(reduced_direction)
    candidate_scores = class_basis @ normalized_direction
    candidate_order = _max._lexicographic_refinement_order(
        candidate_scores, class_basis
    )
    _, _, prediction_order = _core._prediction_from_class_order(
        ranking_input, candidate_order
    )
    if np.any(prediction_order[pairs[active, 0]] >= prediction_order[pairs[active, 1]]):
        raise RuntimeError("the strict MaxFS refinement lost an active comparison")
    sampled_disagreement = int(
        np.sum(
            multiplicities[
                prediction_order[pairs[:, 0]] >= prediction_order[pairs[:, 1]]
            ],
            dtype=np.int64,
        )
    )
    _, candidate_distance, candidate_total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    if candidate_total != ranking_input.pair_index.target_comparisons:
        raise RuntimeError("the MaxFS comparison population changed")
    if sampled_disagreement > deleted_count:
        raise RuntimeError("MaxFS sampled disagreement exceeds its deleted weight")
    if (
        not sampled
        and candidate_distance - ranking_input.pair_index.forced_tie_disagreements
        > deleted_count
    ):
        raise RuntimeError("full MaxFS distance exceeds its deleted comparisons")
    diagnostics = {
        "comparison_population": "sampled" if sampled else "full",
        "deletion_fraction": removal_fraction,
        "removal_iterations": removal_iterations,
        "removal_limit_triggered": removal_limit_triggered,
        "add_back_attempts": add_back_attempts,
        "add_back_successes": add_back_successes,
        "untested_deleted_comparisons": int(
            max(0, deleted_order.size - add_back_attempts)
        ),
        "unique_deleted_count": int(np.count_nonzero(deleted)),
        "deleted_count": deleted_count,
        "removed_comparisons": deleted_pairs_original.tolist(),
        "removed_multiplicities": multiplicities[deleted].tolist(),
        "final_elastic_objective": final_elastic_objective,
        "elastic_tolerance": tolerance,
        "sampled_disagreement": sampled_disagreement,
    }
    result = _finalize_result(
        algorithm="elastic_maxfs",
        ranking_input=ranking_input,
        parameterization=parameterization,
        reduced_direction=normalized_direction,
        total_started=total_started,
        optimization_runtime=optimization_runtime,
        solver_runtime_prefix=solver_runtime,
        optimization_iterations=solver_optimizations,
        surrogate_objective=float(deleted_count),
        training_sample_size=training_sample_size,
        unique_training_comparisons=unique_count,
        diagnostics=diagnostics,
        seed=seed,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        gurobi_params=gurobi_params,
    )

    if result.kendall_tau != candidate_distance:
        raise RuntimeError("the final MaxFS certificate changed the refined order")
    return result


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
        metavar="PATH",
        help="text or .npy vector of numeric ranks aligned with the point rows",
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
        description="Run scalable baselines for linear-ranking experiments."
    )
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    lmrc_parser = subparsers.add_parser("lmrc", help="linearized MRC baseline")
    _add_common_cli_arguments(lmrc_parser)
    lmrc_parser.add_argument("--ridge", type=float, default=1e-6)
    lmrc_parser.add_argument("--cg-tolerance", type=float, default=1e-10)
    lmrc_parser.add_argument("--cg-max-iterations", type=int, default=None)

    for command, help_text in (
        ("ranksvm", "pairwise-hinge RankSVM baseline"),
        ("ranknet", "pairwise-logistic linear RankNet baseline"),
    ):
        pair_parser = subparsers.add_parser(command, help=help_text)
        _add_common_cli_arguments(pair_parser)
        pair_parser.add_argument("--regularization", type=float, default=1e-3)
        pair_parser.add_argument(
            "--max-training-pairs",
            type=_none_if_zero,
            default=_DEFAULT_MAX_TRAINING_PAIRS,
            metavar="N",
            help="eligible-pair cap (default: 100000); use 0 for all eligible pairs",
        )
        pair_parser.add_argument("--epochs", type=int, default=100)
        pair_parser.add_argument(
            "--batch-size",
            type=_none_if_zero,
            default=None,
            metavar="N",
            help="optional mini-batch size; 0/default uses endpoint full-batch Adam",
        )
        pair_parser.add_argument("--learning-rate", type=float, default=0.03)

    local_parser = subparsers.add_parser(
        "local-search", help="direct full-data Kendall local search"
    )
    _add_common_cli_arguments(local_parser)
    local_parser.add_argument("--restarts", type=int, default=8)
    local_parser.add_argument("--max-iterations", type=int, default=250)
    local_parser.add_argument("--initial-step", type=float, default=0.25)
    local_parser.add_argument("--step-decay", type=float, default=0.5)
    local_parser.add_argument("--patience", type=int, default=25)
    local_parser.add_argument("--min-step", type=float, default=1e-4)
    local_parser.add_argument("--ridge", type=float, default=1e-6)

    maxfs_parser = subparsers.add_parser(
        "maxfs", help="elastic maximum-feasible-subsystem heuristic"
    )
    _add_common_cli_arguments(maxfs_parser)
    maxfs_parser.add_argument(
        "--max-training-pairs",
        type=_none_if_zero,
        default=_DEFAULT_MAX_TRAINING_PAIRS,
        metavar="N",
        help=(
            "eligible LP-comparison cap (default: 100000); use 0 for all eligible pairs"
        ),
    )
    maxfs_parser.add_argument("--deletion-fraction", type=float, default=0.05)
    maxfs_parser.add_argument("--max-removal-iterations", type=int, default=1_000)
    maxfs_parser.add_argument(
        "--max-add-back",
        type=int,
        default=_DEFAULT_MAXFS_ADD_BACK,
        metavar="N",
        help="maximum greedy tests (default: 1000); use 0 to disable",
    )
    maxfs_parser.add_argument("--elastic-tolerance", type=float, default=1e-8)

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
        if args.algorithm == "lmrc":
            result = linearized_mrc_baseline(
                point_matrix,
                args.target_order,
                ridge=args.ridge,
                cg_tolerance=args.cg_tolerance,
                cg_max_iterations=args.cg_max_iterations,
                **common,
            )
        elif args.algorithm == "ranksvm":
            result = rank_svm_baseline(
                point_matrix,
                args.target_order,
                regularization=args.regularization,
                max_training_pairs=args.max_training_pairs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                **common,
            )
        elif args.algorithm == "ranknet":
            result = linear_ranknet_baseline(
                point_matrix,
                args.target_order,
                regularization=args.regularization,
                max_training_pairs=args.max_training_pairs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                **common,
            )
        elif args.algorithm == "local-search":
            result = kendall_local_search_baseline(
                point_matrix,
                args.target_order,
                restarts=args.restarts,
                max_iterations=args.max_iterations,
                initial_step=args.initial_step,
                step_decay=args.step_decay,
                patience=args.patience,
                min_step=args.min_step,
                ridge=args.ridge,
                **common,
            )
        else:
            result = elastic_maxfs_baseline(
                point_matrix,
                args.target_order,
                max_training_pairs=args.max_training_pairs,
                deletion_fraction=args.deletion_fraction,
                max_removal_iterations=args.max_removal_iterations,
                max_add_back=args.max_add_back,
                elastic_tolerance=args.elastic_tolerance,
                **common,
            )
        rendered = json.dumps(result.to_dict(), indent=2)
        print(f"{result.to_dict()["algorithm"]} produced a KT of {result.to_dict()["kendall_tau"]} and {result.to_dict()["kendall_agreement"]}")
        if args.output is None:
            print(rendered)
        else:
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
