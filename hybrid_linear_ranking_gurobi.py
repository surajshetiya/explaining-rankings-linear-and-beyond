"""Sampled greedy-IIS hybrid for maximum-agreement linear ranking.

This module combines the uniform comparison sampling used by
``thm:logdim-similarity-ptas`` with the minimal-infeasible-subsystem deletion
idea used by ``thm:main-kt``.  It is intended for the regime in which exact
sampled maximum-agreement MIPs become slow: the sampled optimizer below uses
continuous LP/IIS steps followed by one minimum-norm convex QP, with no binary
variables or branch-and-bound.

For a fixed IID sample, equal directed point-class comparisons are aggregated
with integer multiplicities.  The optimizer is the weighted primal--dual
version of greedy IIS deletion.  If ``C`` is a comparison-minimal IIS and
``rho_e`` is the residual multiplicity of comparison ``e``, it raises the dual
weight of ``C`` by ``min(rho_e for e in C)`` and removes every newly tight
comparison.  Thus its empirical deletion certificate is within the largest
encountered IIS size (at most the sampled comparison dimension plus one) of
the empirical minimum disagreement.  A greedy add-back pass can only improve
the returned sample order.

The same sample is reused across restarts with different constraint orders.
Every forward greedy candidate and its antipodal reverse are evaluated on the
full weak target, and the best full-data candidate is returned.  The antipodal
pair gives a deterministic 1/2 max-agreement guarantee.  If the requested VC
sample bound is met, the empirical primal--dual gap additionally yields a
data-dependent high-probability agreement bound.  This method is nevertheless
reported as a hybrid heuristic, not as the exact empirical PTAS.

Dependencies: NumPy, gurobipy, and a working Gurobi license.
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

_DEFAULT_SAMPLE_SIZE = 200
_DEFAULT_RESTARTS = 3
_DEFAULT_MAX_ADD_BACK = 64
_DEFAULT_MAX_SAMPLE_SIZE = 1_000_000
_DEFAULT_MAX_UNIQUE_COMPARISONS = 100_000
_GENERIC_REFINEMENT_ATTEMPTS = 16
_GENERIC_REFINEMENT_SLACK_FRACTION = 0.25
_SEED_SALT = 0x3C6EF372

__all__ = [
    "GreedyIISHybridResult",
    "max_agreement_greedy_iis_hybrid",
]


@dataclass(frozen=True)
class GreedyIISHybridResult:
    """Result of :func:`max_agreement_greedy_iis_hybrid`.

    ``empirical_disagreement`` belongs to the best certified *forward* greedy
    candidate on the shared sample.  ``selected_empirical_disagreement``
    belongs to the full-data-selected output and can be larger when the reverse
    orientation generalizes better.  This distinction keeps the primal--dual
    sample certificate honest.
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
    sample_size: int
    unique_sample_size: int
    unique_class_sample_size: int
    sampled_comparisons: np.ndarray
    sample_multiplicities: np.ndarray
    empirical_agreement: int
    empirical_disagreement: int
    selected_empirical_agreement: int
    selected_empirical_disagreement: int
    empirical_disagreement_lower_bound: int
    empirical_deletion_upper_bound: int
    empirical_optimality_gap_upper_bound: int
    normalized_empirical_gap: float
    empirical_certificate_scope: str
    greedy_approximation_factor: int
    iis_rounds: int
    dual_raise_total: int
    restarts: int
    lower_bound_restart: int
    certificate_restart: int
    chosen_restart: int
    chosen_orientation: int
    epsilon: float
    delta: float
    eta: float
    dimension_bound: int
    sample_dimension: int
    sample_condition_number: float
    sample_rank_tolerance: float
    sample_projection_used: bool
    raw_iis_certified: bool
    theoretical_sample_size: int
    sampling_bound_met: bool
    numerical_margin: float
    projected_instance: bool
    solver_optimizations: int
    optimization_runtime_seconds: float
    solver_runtime_seconds: float
    total_runtime_seconds: float
    restart_diagnostics: tuple[dict[str, Any], ...]

    @property
    def guaranteed_fraction_of_optimum(self) -> float:
        """Return the applicable max-agreement approximation fraction.

        The 1/2 floor is deterministic.  The stronger data-dependent value is
        conditional on the VC event and therefore reported only when the
        theoretical sample bound is met, neither projection mode is active,
        and every reduced IIS has passed its precompression check.
        """

        if self.optimizable_comparisons == 0:
            return 1.0
        if (
            not self.sampling_bound_met
            or self.sample_projection_used
            or self.projected_instance
            or not self.raw_iis_certified
        ):
            return 0.5
        data_dependent = 1.0 - self.epsilon - 2.0 * self.normalized_empirical_gap
        return max(0.5, min(1.0, data_dependent))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": "max_agreement_greedy_iis_hybrid",
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
            "empirical_agreement": self.empirical_agreement,
            "empirical_disagreement": self.empirical_disagreement,
            "selected_empirical_agreement": self.selected_empirical_agreement,
            "selected_empirical_disagreement": (self.selected_empirical_disagreement),
            "empirical_disagreement_lower_bound": (
                self.empirical_disagreement_lower_bound
            ),
            "empirical_deletion_upper_bound": (self.empirical_deletion_upper_bound),
            "empirical_optimality_gap_upper_bound": (
                self.empirical_optimality_gap_upper_bound
            ),
            "normalized_empirical_gap": self.normalized_empirical_gap,
            "empirical_certificate_scope": self.empirical_certificate_scope,
            "greedy_approximation_factor": self.greedy_approximation_factor,
            "iis_rounds": self.iis_rounds,
            "dual_raise_total": self.dual_raise_total,
            "restarts": self.restarts,
            "lower_bound_restart": self.lower_bound_restart,
            "certificate_restart": self.certificate_restart,
            "chosen_restart": self.chosen_restart,
            "chosen_orientation": self.chosen_orientation,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "eta": self.eta,
            "dimension_bound": self.dimension_bound,
            "sample_dimension": self.sample_dimension,
            "sample_condition_number": self.sample_condition_number,
            "sample_rank_tolerance": self.sample_rank_tolerance,
            "sample_projection_used": self.sample_projection_used,
            "raw_iis_certified": self.raw_iis_certified,
            "theoretical_sample_size": self.theoretical_sample_size,
            "sampling_bound_met": self.sampling_bound_met,
            "guaranteed_fraction_of_optimum": (self.guaranteed_fraction_of_optimum),
            "numerical_margin": self.numerical_margin,
            "projected_instance": self.projected_instance,
            "solver_optimizations": self.solver_optimizations,
            "optimization_runtime_seconds": self.optimization_runtime_seconds,
            "solver_runtime_seconds": self.solver_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "restart_diagnostics": [dict(item) for item in self.restart_diagnostics],
        }


@dataclass(frozen=True)
class _SampleParameterization:
    normalized_differences: np.ndarray
    raw_differences: np.ndarray
    right_basis: np.ndarray
    dimension: int
    tolerance: float
    projection_used: bool
    condition_number: float

    def to_full(self, sample_direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(sample_direction, dtype=np.float64).reshape(-1)
        if direction.shape != (self.dimension,):
            raise RuntimeError("sample direction has inconsistent dimensions")
        return np.ascontiguousarray(self.right_basis @ direction)


@dataclass(frozen=True)
class _GreedyRun:
    reduced_direction: np.ndarray
    active_mask: np.ndarray
    dual_lower_bound: int
    deleted_weight_before_add_back: int
    primal_dual_deletion_upper_bound: int
    deleted_weight: int
    iis_rounds: int
    dual_raise_total: int
    max_iis_size: int
    iis_block_sizes: tuple[int, ...]
    raw_iis_checks: int
    add_back_attempts: int
    add_back_successes: int
    solver_optimizations: int
    solver_runtime_seconds: float


@dataclass(frozen=True)
class _Candidate:
    class_order: np.ndarray
    model_direction: np.ndarray
    class_scores: np.ndarray
    numerical_margin: float
    full_agreement: int
    full_disagreement: int
    empirical_agreement: int
    restart: int
    orientation: int


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


def _optional_nonnegative_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _validate_sample_rank_tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample_rank_tol must be a finite nonnegative number") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("sample_rank_tol must be a finite nonnegative number")
    return tolerance


def _validate_global_rank_tolerance(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "rank_tol must be a finite nonnegative number or None"
        ) from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rank_tol must be a finite nonnegative number or None")
    return tolerance


def _sample_parameterization(
    class_basis: np.ndarray,
    class_pairs: np.ndarray,
    sample_rank_tol: float,
) -> _SampleParameterization:
    """Compress sampled differences to their thin numerical row span."""

    differences = np.ascontiguousarray(
        class_basis[class_pairs[:, 0]] - class_basis[class_pairs[:, 1]]
    )
    if differences.ndim != 2 or differences.shape[0] == 0:
        raise RuntimeError("the hybrid sample contains no class comparisons")
    try:
        left, singular_values, right_transpose = np.linalg.svd(
            differences, full_matrices=False
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("sample-difference SVD did not converge") from exc
    if singular_values.size == 0 or not np.isfinite(singular_values).all():
        raise RuntimeError("sample-difference SVD returned no finite directions")

    positive_dimension = int(np.count_nonzero(singular_values > 0.0))
    dimension = int(np.count_nonzero(singular_values > sample_rank_tol))
    if dimension == 0:
        raise ValueError(
            "sample_rank_tol removed every sampled comparison direction; "
            "lower the cutoff"
        )
    coordinates = np.ascontiguousarray(
        left[:, :dimension] * singular_values[:dimension]
    )
    row_scales = np.max(np.abs(coordinates), axis=1)
    if np.any(~np.isfinite(row_scales)) or np.any(row_scales == 0.0):
        raise RuntimeError("a sampled comparison vanished in the reduced space")
    normalized = np.ascontiguousarray(coordinates / row_scales[:, None])
    smallest = float(singular_values[dimension - 1])
    condition_number = float(singular_values[0]) / smallest
    if not math.isfinite(condition_number):
        raise ValueError(
            "the sampled comparison span is too ill-conditioned for float64; "
            "rescale the input or set an explicit sample_rank_tol"
        )
    return _SampleParameterization(
        normalized_differences=normalized,
        raw_differences=differences,
        right_basis=np.ascontiguousarray(right_transpose[:dimension].T),
        dimension=dimension,
        tolerance=sample_rank_tol,
        projection_used=dimension < positive_dimension,
        condition_number=condition_number,
    )


def _compute_direct_iis(
    model: Any,
    comparison_constraints: list[Any],
    active_mask: np.ndarray,
    sample_dimension: int,
) -> np.ndarray:
    """Return a direct-model comparison-minimal IIS in original row IDs."""

    try:
        model.computeIIS()
    except gp.GurobiError as exc:
        numeric_error = getattr(GRB, "ERROR_NUMERIC", 10014)
        if exc.errno != numeric_error:
            raise
        model.setParam("IISMethod", 1)
        model.computeIIS()
    if int(model.IISMinimal) != 1:
        raise RuntimeError("Gurobi did not finish a minimal sampled IIS")
    flags = np.asarray(
        model.getAttr(GRB.Attr.IISConstr, comparison_constraints), dtype=bool
    )
    block = np.flatnonzero(flags & active_mask).astype(np.int64, copy=False)
    if block.size == 0:
        raise RuntimeError("the sampled IIS contains no active comparison")
    if np.any(flags & ~active_mask):
        raise RuntimeError("a disabled sampled comparison entered the IIS")
    if block.size > sample_dimension + 1:
        raise RuntimeError(
            f"sample IIS size {block.size} exceeds the reduced dimension "
            f"bound + 1={sample_dimension + 1}"
        )
    return block


def _verify_raw_iis_infeasibility(
    env: Any,
    raw_differences: np.ndarray,
    block: np.ndarray,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> None:
    """Re-certify a reduced-space IIS from the precompression rows.

    If ``A`` contains the block rows, feasibility can be restricted without
    loss to their row span: write ``q = A.T z`` and solve ``A A.T z >= 1``.
    The Gram formulation therefore uses at most ``len(block)`` variables even
    when the input has many thousands of features.  This independent check
    prevents a floating SVD artifact from being counted in the dual bound.
    """

    rows = np.ascontiguousarray(raw_differences[block], dtype=np.float64)
    row_scales = np.max(np.abs(rows), axis=1)
    if np.any(~np.isfinite(row_scales)) or np.any(row_scales == 0.0):
        raise RuntimeError("a raw sampled IIS row is numerically invalid")
    normalized = np.ascontiguousarray(rows / row_scales[:, None])
    gram = np.ascontiguousarray(normalized @ normalized.T)
    if np.any(~np.isfinite(gram)):
        raise RuntimeError("raw sampled IIS Gram matrix is not finite")

    with gp.Model("raw_sampled_iis_check", env=env) as model:
        _core._set_model_parameters(
            model,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            seed=seed,
            gurobi_params=gurobi_params,
        )
        coefficients = model.addMVar(block.size, lb=-GRB.INFINITY, name="z")
        coefficient_variables = coefficients.tolist()
        for row_index in range(block.size):
            expression = gp.LinExpr(gram[row_index].tolist(), coefficient_variables)
            model.addConstr(expression >= 1.0)
        model.setObjective(0.0, GRB.MINIMIZE)
        status = _core._optimize_for_status(model)

    if status == GRB.OPTIMAL:
        raise RuntimeError(
            "the sampled SVD reported an IIS that is feasible in the "
            "precompression comparison space; rescale the input or set an "
            "explicit sample_rank_tol and treat the result as projected"
        )
    if status != GRB.INFEASIBLE:
        raise RuntimeError(
            "Gurobi could not certify a sampled IIS in the precompression space"
        )


def _run_weighted_greedy(
    env: Any,
    sample_parameterization: _SampleParameterization,
    multiplicities: np.ndarray,
    *,
    restart_seed: int,
    max_add_back: int | None,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    gurobi_params: Mapping[str, Any] | None,
    raw_iis_cache: set[tuple[int, ...]],
) -> _GreedyRun:
    """Run weighted primal--dual IIS deletion and greedy add-back."""

    rows = sample_parameterization.normalized_differences
    unique_count = rows.shape[0]
    if multiplicities.shape != (unique_count,) or np.any(multiplicities <= 0):
        raise RuntimeError("sample multiplicities are inconsistent")
    rng = np.random.default_rng(restart_seed)
    insertion_order = rng.permutation(unique_count)
    residual = multiplicities.astype(np.int64, copy=True)
    active = np.ones(unique_count, dtype=bool)
    iis_block_sizes: list[int] = []
    dual_lower_bound = 0
    primal_dual_deletion_upper_bound = 0
    solver_optimizations = 0
    raw_iis_checks = 0
    add_back_attempts = 0
    add_back_successes = 0
    solver_started = time.perf_counter()

    with gp.Model("sampled_weighted_greedy_iis", env=env) as model:
        _core._set_model_parameters(
            model,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            seed=restart_seed,
            gurobi_params=gurobi_params,
        )
        direction = model.addMVar(
            sample_parameterization.dimension,
            lb=-GRB.INFINITY,
            name="q",
        )
        direction_variables = direction.tolist()
        comparison_constraints: list[Any] = [None] * unique_count
        for index_raw in insertion_order:
            index = int(index_raw)
            expression = gp.LinExpr(rows[index].tolist(), direction_variables)
            comparison_constraints[index] = model.addConstr(
                expression >= 1.0,
                name=f"comparison_{index}",
            )
        model.setObjective(0.0, GRB.MINIMIZE)
        model.update()

        while True:
            status = _core._optimize_for_status(model)
            solver_optimizations += 1
            if status == GRB.OPTIMAL:
                break
            if status != GRB.INFEASIBLE:
                raise _core._status_error(model, "sampled greedy feasibility")

            block = _compute_direct_iis(
                model,
                comparison_constraints,
                active,
                sample_parameterization.dimension,
            )
            if not sample_parameterization.projection_used:
                block_key = tuple(sorted(block.tolist()))
                if block_key not in raw_iis_cache:
                    _verify_raw_iis_infeasibility(
                        env,
                        sample_parameterization.raw_differences,
                        block,
                        output_flag=output_flag,
                        threads=threads,
                        feasibility_tol=feasibility_tol,
                        seed=restart_seed,
                        gurobi_params=gurobi_params,
                    )
                    raw_iis_cache.add(block_key)
                    solver_optimizations += 1
                raw_iis_checks += 1
            theta = int(np.min(residual[block]))
            if theta <= 0:
                raise RuntimeError("sampled IIS has nonpositive residual weight")
            residual[block] -= theta
            dual_lower_bound += theta
            primal_dual_deletion_upper_bound += theta * int(block.size)
            iis_block_sizes.append(int(block.size))
            exhausted = block[residual[block] == 0]
            if exhausted.size == 0:
                raise RuntimeError("weighted IIS update exhausted no comparison")
            for index_raw in exhausted:
                index = int(index_raw)
                active[index] = False
                comparison_constraints[index].RHS = -GRB.INFINITY
            model.update()

        deleted_before_add_back = int(np.sum(multiplicities[~active], dtype=np.int64))
        inactive = np.flatnonzero(~active)
        if inactive.size:
            random_ties = rng.random(inactive.size)
            priority = np.lexsort(
                (random_ties, -multiplicities[inactive].astype(np.float64))
            )
            attempted = inactive[priority]
            if max_add_back is not None:
                attempted = attempted[:max_add_back]
            for index_raw in attempted:
                index = int(index_raw)
                add_back_attempts += 1
                comparison_constraints[index].RHS = 1.0
                model.update()
                status = _core._optimize_for_status(model)
                solver_optimizations += 1
                if status == GRB.OPTIMAL:
                    active[index] = True
                    add_back_successes += 1
                elif status == GRB.INFEASIBLE:
                    comparison_constraints[index].RHS = -GRB.INFINITY
                    model.update()
                else:
                    raise _core._status_error(model, "sampled greedy add-back")

        # Among directions satisfying the retained comparisons, finish with
        # the minimum-norm witness.  This single convex QP is the hard-margin
        # analogue of the RankSVM regularizer and avoids an arbitrary LP
        # feasibility vertex in the common high-dimensional regime where the
        # complete sample is realizable.
        model.setObjective(direction @ direction, GRB.MINIMIZE)
        status = _core._optimize_for_status(model)
        solver_optimizations += 1
        if status != GRB.OPTIMAL:
            raise _core._status_error(model, "sampled minimum-norm witness")
        sample_direction = np.asarray(direction.X, dtype=np.float64).reshape(-1)

    solver_runtime = time.perf_counter() - solver_started
    deleted_weight = int(np.sum(multiplicities[~active], dtype=np.int64))
    max_iis_size = max(iis_block_sizes, default=1)
    if deleted_weight > deleted_before_add_back:
        raise RuntimeError("sampled add-back increased the deletion certificate")
    if deleted_before_add_back > primal_dual_deletion_upper_bound:
        raise RuntimeError("sampled deletion exceeds its primal--dual charge")
    if primal_dual_deletion_upper_bound > max_iis_size * dual_lower_bound:
        raise RuntimeError("weighted IIS primal--dual accounting failed")
    return _GreedyRun(
        reduced_direction=sample_direction,
        active_mask=active,
        dual_lower_bound=dual_lower_bound,
        deleted_weight_before_add_back=deleted_before_add_back,
        primal_dual_deletion_upper_bound=primal_dual_deletion_upper_bound,
        deleted_weight=deleted_weight,
        iis_rounds=len(iis_block_sizes),
        dual_raise_total=dual_lower_bound,
        max_iis_size=max_iis_size,
        iis_block_sizes=tuple(iis_block_sizes),
        raw_iis_checks=raw_iis_checks,
        add_back_attempts=add_back_attempts,
        add_back_successes=add_back_successes,
        solver_optimizations=solver_optimizations,
        solver_runtime_seconds=solver_runtime,
    )


def _strict_generic_candidate(
    class_basis: np.ndarray,
    sample_differences: np.ndarray,
    class_pairs: np.ndarray,
    active_mask: np.ndarray,
    weak_direction: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generically refine one feasible sampled direction in near-linear time."""

    direction = np.asarray(weak_direction, dtype=np.float64).reshape(-1)
    if direction.shape != (class_basis.shape[1],) or not np.isfinite(direction).all():
        raise RuntimeError("the sampled greedy direction is invalid")
    scale = float(np.max(np.abs(direction), initial=0.0))
    if scale > 0.0:
        direction = np.ascontiguousarray(direction / scale)

    active_differences = sample_differences[active_mask]
    if active_differences.size:
        active_gaps = active_differences @ direction
        if np.any(~np.isfinite(active_gaps)) or np.any(active_gaps <= 0.0):
            raise RuntimeError(
                "mapping the sampled feasible direction lost an active comparison"
            )
    else:
        active_gaps = np.empty(0, dtype=np.float64)

    def inspect(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        candidate_scale = float(np.max(np.abs(candidate), initial=0.0))
        if not math.isfinite(candidate_scale) or candidate_scale == 0.0:
            return None
        normalized = np.ascontiguousarray(candidate / candidate_scale)
        scores = class_basis @ normalized
        if np.any(~np.isfinite(scores)):
            return None
        order = np.argsort(-scores, kind="stable").astype(np.int64, copy=False)
        margin = _max._strict_candidate_margin(class_basis, normalized, scores, order)
        if margin is None:
            return None
        if active_differences.size:
            class_positions = np.empty(order.size, dtype=np.int64)
            class_positions[order] = np.arange(order.size, dtype=np.int64)
            if np.any(
                class_positions[class_pairs[active_mask, 0]]
                >= class_positions[class_pairs[active_mask, 1]]
            ):
                return None
        return normalized, scores, margin

    inspected = inspect(direction)
    if inspected is not None:
        normalized, scores, margin = inspected
        order = np.argsort(-scores, kind="stable").astype(np.int64, copy=False)
        return normalized, scores, order, margin

    rng = np.random.default_rng(np.random.SeedSequence([seed, _SEED_SALT]))
    for _ in range(_GENERIC_REFINEMENT_ATTEMPTS):
        perturbation = rng.standard_normal(class_basis.shape[1])
        perturbation_scale = float(np.max(np.abs(perturbation), initial=0.0))
        if not math.isfinite(perturbation_scale) or perturbation_scale == 0.0:
            continue
        perturbation /= perturbation_scale
        if active_differences.size:
            effects = active_differences @ perturbation
            decreasing = effects < 0.0
            if np.any(decreasing):
                limit = float(
                    np.min(
                        _GENERIC_REFINEMENT_SLACK_FRACTION
                        * active_gaps[decreasing]
                        / -effects[decreasing]
                    )
                )
                amount = min(1.0, limit)
            else:
                amount = 1.0
        else:
            amount = 1.0
        if not math.isfinite(amount) or amount <= 0.0:
            continue
        inspected = inspect(direction + amount * perturbation)
        if inspected is None:
            continue
        normalized, scores, margin = inspected
        order = np.argsort(-scores, kind="stable").astype(np.int64, copy=False)
        return normalized, scores, order, margin

    raise RuntimeError(
        "failed to construct a strict generic hybrid direction; rescale the input"
    )


def _make_candidate(
    ranking_input: _core._RankingInput,
    class_pairs: np.ndarray,
    multiplicities: np.ndarray,
    class_order: np.ndarray,
    reduced_direction: np.ndarray,
    class_scores: np.ndarray,
    numerical_margin: float,
    *,
    restart: int,
    orientation: int,
    evaluation_cache: dict[bytes, tuple[int, int, int, int]],
) -> _Candidate:
    cache_key = class_order.tobytes()
    cached = evaluation_cache.get(cache_key)
    if cached is None:
        _, _, _, agreement, disagreement, total = _max._evaluate_class_order(
            ranking_input, class_order
        )
        if total != ranking_input.pair_index.target_comparisons:
            raise RuntimeError("hybrid full comparison population changed")
        empirical = _max._empirical_agreement(class_order, class_pairs, multiplicities)
        cached = (agreement, disagreement, total, empirical)
        evaluation_cache[cache_key] = cached
    agreement, disagreement, total, empirical = cached
    return _Candidate(
        class_order=class_order,
        model_direction=reduced_direction,
        class_scores=class_scores,
        numerical_margin=numerical_margin,
        full_agreement=agreement,
        full_disagreement=disagreement,
        empirical_agreement=empirical,
        restart=restart,
        orientation=orientation,
    )


def _reverse_candidate(
    ranking_input: _core._RankingInput,
    forward: _Candidate,
    sample_size: int,
) -> _Candidate:
    """Construct the exact antipodal complement without a second full scan."""

    agreement = ranking_input.pair_index.modeled_comparisons - forward.full_agreement
    return _Candidate(
        class_order=forward.class_order[::-1].copy(),
        model_direction=-forward.model_direction,
        class_scores=-forward.class_scores,
        numerical_margin=forward.numerical_margin,
        full_agreement=agreement,
        full_disagreement=ranking_input.pair_index.target_comparisons - agreement,
        empirical_agreement=sample_size - forward.empirical_agreement,
        restart=forward.restart,
        orientation=-1,
    )


def _candidate_key(candidate: _Candidate) -> tuple[float, ...]:
    return (
        float(candidate.full_agreement),
        float(candidate.empirical_agreement),
        float(candidate.numerical_margin),
        float(-candidate.restart),
        float(candidate.orientation),
    )


def _trivial_result(
    ranking_input: _core._RankingInput,
    *,
    epsilon: float,
    delta: float,
    restarts: int,
    sample_rank_tolerance: float,
    total_started: float,
) -> GreedyIISHybridResult:
    if ranking_input.pair_index.modeled_comparisons != 0:
        raise RuntimeError("the empty hybrid result has optimizable comparisons")
    n, ambient_dimension = ranking_input.points.shape
    flattened, prediction_ids, prediction_order = _core._all_tied_prediction(
        ranking_input
    )
    agreement, disagreement, total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    elapsed = time.perf_counter() - total_started
    return GreedyIISHybridResult(
        ranking=ranking_input.target_order[flattened],
        direction=np.zeros(ambient_dimension, dtype=np.float64),
        conditioned_scores=np.zeros(n, dtype=np.float64),
        target_order=ranking_input.target_order.copy(),
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
        empirical_agreement=0,
        empirical_disagreement=0,
        selected_empirical_agreement=0,
        selected_empirical_disagreement=0,
        empirical_disagreement_lower_bound=0,
        empirical_deletion_upper_bound=0,
        empirical_optimality_gap_upper_bound=0,
        normalized_empirical_gap=0.0,
        empirical_certificate_scope="original",
        greedy_approximation_factor=1,
        iis_rounds=0,
        dual_raise_total=0,
        restarts=restarts,
        lower_bound_restart=-1,
        certificate_restart=-1,
        chosen_restart=-1,
        chosen_orientation=1,
        epsilon=epsilon,
        delta=delta,
        eta=epsilon / 4.0,
        dimension_bound=0,
        sample_dimension=0,
        sample_condition_number=1.0,
        sample_rank_tolerance=sample_rank_tolerance,
        sample_projection_used=False,
        raw_iis_certified=True,
        theoretical_sample_size=0,
        sampling_bound_met=True,
        numerical_margin=0.0,
        projected_instance=False,
        solver_optimizations=0,
        optimization_runtime_seconds=0.0,
        solver_runtime_seconds=0.0,
        total_runtime_seconds=elapsed,
        restart_diagnostics=(),
    )


def max_agreement_greedy_iis_hybrid(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None = None,
    *,
    target_ranks: Sequence[float] | np.ndarray | None = None,
    epsilon: float,
    delta: float,
    sample_size: int | None = _DEFAULT_SAMPLE_SIZE,
    restarts: int = _DEFAULT_RESTARTS,
    max_add_back: int | None = _DEFAULT_MAX_ADD_BACK,
    rank_tol: float | None = None,
    sample_rank_tol: float = 0.0,
    seed: int = 0,
    output_flag: bool = False,
    threads: int | None = None,
    feasibility_tol: float = 1e-8,
    gurobi_params: Mapping[str, Any] | None = None,
    max_sample_size: int | None = _DEFAULT_MAX_SAMPLE_SIZE,
    max_unique_comparisons: int | None = _DEFAULT_MAX_UNIQUE_COMPARISONS,
) -> GreedyIISHybridResult:
    """Run the sampled weighted greedy-IIS max-agreement hybrid.

    The method samples comparisons exactly as the similarity PTAS does, but
    replaces exact empirical binary optimization with a continuous weighted
    primal--dual IIS heuristic.  The same sample is reused across ``restarts``.
    The best forward greedy certificate supplies the empirical gap bound, while
    the returned ranking is selected by full-data agreement from every forward
    candidate and its reverse.

    ``sample_size=None`` uses the sufficient VC sample size.  Manual sizes such
    as 122 or 200 are usually more practical; when they are below the bound the
    result makes only the deterministic 1/2 agreement claim.
    If no target comparison is optimizable, the method returns the all-tied
    zero direction without sampling, conditioning, or invoking Gurobi.
    """

    total_started = time.perf_counter()
    epsilon_value = _max._validate_open_probability(epsilon, "epsilon")
    delta_value = _max._validate_open_probability(delta, "delta")
    eta = epsilon_value / 4.0
    restart_count = _max._validate_positive_integer(restarts, "restarts")
    add_back_limit = _optional_nonnegative_integer(max_add_back, "max_add_back")
    rank_tolerance = _validate_global_rank_tolerance(rank_tol)
    sample_tolerance = _validate_sample_rank_tolerance(sample_rank_tol)
    sample_limit = _max._validate_optional_limit(max_sample_size, "max_sample_size")
    unique_limit = _max._validate_optional_limit(
        max_unique_comparisons, "max_unique_comparisons"
    )
    requested_sample_size = (
        None
        if sample_size is None
        else _max._validate_positive_integer(sample_size, "sample_size")
    )
    if (
        requested_sample_size is not None
        and requested_sample_size > _max._MAX_EXACT_INTEGER_IN_FLOAT64
    ):
        raise ValueError(
            "sample_size exceeds the largest exactly weighted sample supported "
            "by this float64 implementation"
        )
    if (
        requested_sample_size is not None
        and sample_limit is not None
        and requested_sample_size > sample_limit
    ):
        raise ValueError(
            f"sample_size={requested_sample_size} exceeds "
            f"max_sample_size={sample_limit}; increase/disable the guard or "
            "use a smaller experimental sample"
        )
    ranking_input = _core._validate_ranking_input(points, target_order, target_ranks)
    optimizable = ranking_input.pair_index.modeled_comparisons
    if optimizable == 0:
        return _trivial_result(
            ranking_input,
            epsilon=epsilon_value,
            delta=delta_value,
            restarts=restart_count,
            sample_rank_tolerance=sample_tolerance,
            total_started=total_started,
        )

    x = ranking_input.points
    target = ranking_input.target_order
    forced_ties = ranking_input.pair_index.forced_tie_disagreements
    normalized_seed = int(seed) % 2_000_000_000
    parameterization = _core._score_parameterization(
        x,
        target,
        rank_tolerance,
        ranking_input.point_class_ids_order,
    )
    class_basis = parameterization.score_basis[
        ranking_input.representative_target_positions
    ]

    _core._require_gurobi()
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

    optimization_started = time.perf_counter()
    sample_rng = np.random.default_rng(normalized_seed)
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
        raise RuntimeError("aggregated hybrid sample lost multiplicity")
    sample_parameterization = _sample_parameterization(
        class_basis, class_pairs, sample_tolerance
    )

    candidates: list[_Candidate] = []
    forward_candidates: list[tuple[_Candidate, _GreedyRun]] = []
    restart_diagnostics: list[dict[str, Any]] = []
    evaluation_cache: dict[bytes, tuple[int, int, int, int]] = {}
    raw_iis_cache: set[tuple[int, ...]] = set()
    total_solver_runtime = 0.0
    total_solver_optimizations = 0
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", int(output_flag))
            env.start()
            for restart in range(restart_count):
                restart_seed = int(
                    np.random.SeedSequence(
                        [normalized_seed, restart, _SEED_SALT]
                    ).generate_state(1, dtype=np.uint32)[0]
                    % 2_000_000_000
                )
                run = _run_weighted_greedy(
                    env,
                    sample_parameterization,
                    class_multiplicities,
                    restart_seed=restart_seed,
                    max_add_back=add_back_limit,
                    output_flag=output_flag,
                    threads=threads,
                    feasibility_tol=feasibility_tol,
                    gurobi_params=gurobi_params,
                    raw_iis_cache=raw_iis_cache,
                )
                total_solver_runtime += run.solver_runtime_seconds
                total_solver_optimizations += run.solver_optimizations
                full_weak_direction = sample_parameterization.to_full(
                    run.reduced_direction
                )
                (
                    strict_direction,
                    class_scores,
                    class_order,
                    margin,
                ) = _strict_generic_candidate(
                    class_basis,
                    sample_parameterization.raw_differences,
                    class_pairs,
                    run.active_mask,
                    full_weak_direction,
                    seed=restart_seed,
                )
                forward = _make_candidate(
                    ranking_input,
                    class_pairs,
                    class_multiplicities,
                    class_order,
                    strict_direction,
                    class_scores,
                    margin,
                    restart=restart,
                    orientation=1,
                    evaluation_cache=evaluation_cache,
                )
                sample_disagreement = actual_sample_size - forward.empirical_agreement
                if (
                    sample_disagreement > run.deleted_weight
                    or run.dual_lower_bound > sample_disagreement
                    or run.deleted_weight > run.max_iis_size * run.dual_lower_bound
                ):
                    raise RuntimeError("sampled greedy disagreement certificate failed")
                reverse = _reverse_candidate(ranking_input, forward, actual_sample_size)
                if (
                    forward.full_agreement + reverse.full_agreement != optimizable
                    or forward.empirical_agreement + reverse.empirical_agreement
                    != actual_sample_size
                ):
                    raise RuntimeError("hybrid antipodal accounting failed")
                candidates.extend((forward, reverse))
                forward_candidates.append((forward, run))
                restart_diagnostics.append(
                    {
                        "restart": restart,
                        "seed": restart_seed,
                        "iis_rounds": run.iis_rounds,
                        "iis_block_sizes": list(run.iis_block_sizes),
                        "dual_raise_total": run.dual_raise_total,
                        "dual_lower_bound": run.dual_lower_bound,
                        "deleted_weight_before_add_back": (
                            run.deleted_weight_before_add_back
                        ),
                        "primal_dual_deletion_upper_bound": (
                            run.primal_dual_deletion_upper_bound
                        ),
                        "deleted_weight": run.deleted_weight,
                        "raw_iis_checks": run.raw_iis_checks,
                        "add_back_attempts": run.add_back_attempts,
                        "add_back_successes": run.add_back_successes,
                        "active_class_comparisons": int(
                            np.count_nonzero(run.active_mask)
                        ),
                        "empirical_agreement": forward.empirical_agreement,
                        "empirical_disagreement": sample_disagreement,
                        "full_agreement": forward.full_agreement,
                        "full_disagreement": forward.full_disagreement,
                        "solver_optimizations": run.solver_optimizations,
                        "solver_runtime_seconds": run.solver_runtime_seconds,
                    }
                )
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi error {exc.errno}: {exc}") from exc

    if not candidates or not forward_candidates:
        raise RuntimeError("the hybrid generated no candidate ranking")
    selected = max(candidates, key=_candidate_key)
    certificate_candidate, certificate_run = min(
        forward_candidates,
        key=lambda item: (
            actual_sample_size - item[0].empirical_agreement,
            item[1].deleted_weight,
            -item[1].dual_lower_bound,
            item[0].restart,
        ),
    )
    empirical_disagreement = (
        actual_sample_size - certificate_candidate.empirical_agreement
    )
    lower_bound_candidate, lower_bound_run = max(
        forward_candidates,
        key=lambda item: (item[1].dual_lower_bound, -item[0].restart),
    )
    empirical_lower_bound = lower_bound_run.dual_lower_bound
    if empirical_lower_bound > empirical_disagreement:
        raise RuntimeError("hybrid empirical lower bound exceeds its candidate")
    empirical_gap = empirical_disagreement - empirical_lower_bound
    normalized_gap = empirical_gap / actual_sample_size

    flattened, prediction_ids, prediction_order = _core._prediction_from_class_order(
        ranking_input, selected.class_order
    )
    checked_agreement, checked_distance, checked_total = _core._weak_order_counts(
        ranking_input.target_tie_ids_order, prediction_order
    )
    if (
        checked_agreement != selected.full_agreement
        or checked_distance != selected.full_disagreement
        or checked_total != ranking_input.pair_index.target_comparisons
        or 2 * checked_agreement < optimizable
    ):
        raise RuntimeError("hybrid full-data certificate failed")

    ambient_direction = parameterization.to_ambient(selected.model_direction)
    representatives = ranking_input.representative_target_positions[
        selected.class_order
    ]
    _core._verify_original_ranking(x, target[representatives], ambient_direction)
    target_scores = _max._class_scores_in_target_order(
        ranking_input, selected.class_scores
    )
    selected_empirical_disagreement = actual_sample_size - selected.empirical_agreement
    optimization_runtime = time.perf_counter() - optimization_started
    total_runtime = time.perf_counter() - total_started
    sampled_original = target[target_pairs]

    return GreedyIISHybridResult(
        ranking=target[flattened].astype(np.int64, copy=False),
        direction=ambient_direction.astype(np.float64, copy=False),
        conditioned_scores=_max._conditioned_scores_in_original_order(
            target, target_scores
        ),
        target_order=target.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        kendall_agreement=checked_agreement,
        kendall_tau=checked_distance,
        total_comparisons=checked_total,
        forced_tie_disagreements=forced_ties,
        optimizable_comparisons=optimizable,
        sample_size=actual_sample_size,
        unique_sample_size=int(pair_ids.size),
        unique_class_sample_size=int(class_pairs.shape[0]),
        sampled_comparisons=sampled_original.astype(np.int64, copy=False),
        sample_multiplicities=multiplicities.astype(np.int64, copy=False),
        empirical_agreement=certificate_candidate.empirical_agreement,
        empirical_disagreement=empirical_disagreement,
        selected_empirical_agreement=selected.empirical_agreement,
        selected_empirical_disagreement=selected_empirical_disagreement,
        empirical_disagreement_lower_bound=empirical_lower_bound,
        empirical_deletion_upper_bound=certificate_run.deleted_weight,
        empirical_optimality_gap_upper_bound=empirical_gap,
        normalized_empirical_gap=normalized_gap,
        empirical_certificate_scope=(
            "sample_projected"
            if sample_parameterization.projection_used
            else (
                "projected_global"
                if parameterization.projected_instance
                else "original"
            )
        ),
        greedy_approximation_factor=certificate_run.max_iis_size,
        iis_rounds=certificate_run.iis_rounds,
        dual_raise_total=certificate_run.dual_raise_total,
        restarts=restart_count,
        lower_bound_restart=lower_bound_candidate.restart,
        certificate_restart=certificate_candidate.restart,
        chosen_restart=selected.restart,
        chosen_orientation=selected.orientation,
        epsilon=epsilon_value,
        delta=delta_value,
        eta=eta,
        dimension_bound=parameterization.dimension_bound,
        sample_dimension=sample_parameterization.dimension,
        sample_condition_number=sample_parameterization.condition_number,
        sample_rank_tolerance=sample_tolerance,
        sample_projection_used=sample_parameterization.projection_used,
        raw_iis_certified=(
            not sample_parameterization.projection_used
            and all(
                run.raw_iis_checks == run.iis_rounds for _, run in forward_candidates
            )
        ),
        theoretical_sample_size=theoretical_sample_size,
        sampling_bound_met=sampling_bound_met,
        numerical_margin=selected.numerical_margin,
        projected_instance=parameterization.projected_instance,
        solver_optimizations=total_solver_optimizations,
        optimization_runtime_seconds=optimization_runtime,
        solver_runtime_seconds=total_solver_runtime,
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


def _nonnegative_cli_integer(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "the value must be a nonnegative integer"
        ) from exc
    if integer < 0:
        raise argparse.ArgumentTypeError("the value must be a nonnegative integer")
    return integer


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sampled greedy-IIS max-agreement hybrid without an "
            "empirical binary MIP."
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
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help=(
            "IID comparison draws (default: 200); omit the Python argument or "
            "set it to None in the API to use the theoretical bound"
        ),
    )
    parser.add_argument("--restarts", type=int, default=_DEFAULT_RESTARTS)
    parser.add_argument(
        "--max-add-back",
        type=_nonnegative_cli_integer,
        default=_DEFAULT_MAX_ADD_BACK,
        metavar="N",
        help="greedy add-back attempts (default: 64); 0 skips add-back",
    )
    parser.add_argument(
        "--delimiter", default=",", help="text delimiter, or 'whitespace'"
    )
    parser.add_argument("--rank-tol", type=float, default=None)
    parser.add_argument(
        "--sample-rank-tol",
        type=float,
        default=0.0,
        help="sample-difference SVD cutoff (default: 0, no positive-value truncation)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--solver-log", action="store_true")
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
        result = max_agreement_greedy_iis_hybrid(
            points,
            args.target_order,
            target_ranks=target_ranks,
            epsilon=args.epsilon,
            delta=args.delta,
            sample_size=args.sample_size,
            restarts=args.restarts,
            max_add_back=args.max_add_back,
            rank_tol=args.rank_tol,
            sample_rank_tol=args.sample_rank_tol,
            seed=args.seed,
            output_flag=args.solver_log,
            threads=args.threads,
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
