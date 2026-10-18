"""Seven polynomial ranking baselines with the linear experiment JSON schema.

RankSVM, polynomial-score RankNet, LMRC, full Kendall local search, and Elastic
MaxFS reuse the linear engines on lifted items. Preference RankRLS and a
logistic-smoothed MRC adaptation are implemented here using NumPy only.
This is not a reproduction of self-induced Gaussian MRC inference, nor a
neural-network RankNet. See README.md for objectives and primary references.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np

from polynomial_common import (
    DEFAULT_MAX_FEATURE_ENTRIES, DEFAULT_MAX_FEATURES, PolynomialRankingResult,
    add_common_arguments, core, emit_cli_result, linear, load_cli_input,
    none_if_zero, polynomial_features, wrap_result,
)

__all__ = [
    "polynomial_baseline", "polynomial_rank_svm_baseline",
    "polynomial_ranknet_baseline", "polynomial_rank_rls_baseline",
    "polynomial_smoothed_mrc_baseline", "polynomial_lmrc_baseline",
    "polynomial_kendall_local_search_baseline", "polynomial_elastic_maxfs_baseline",
]


class _PreferenceGraph:
    """Complete target-strict graph minus cross-target exact-duplicate edges.

    Apply its Laplacian in O(n), without materializing O(n^2) comparisons.
    Rows are in target order. Within-target and duplicate exclusions use only
    group/class/class-group sums, all with O(n) index storage.
    """

    def __init__(self, ranking_input: Any):
        groups = ranking_input.target_tie_ids_order
        classes = ranking_input.point_class_ids_order
        n = groups.size
        self.groups = groups
        self.classes = classes
        group_sizes = np.bincount(groups)
        class_sizes = np.bincount(classes)
        combinations, combined, sizes = np.unique(
            np.column_stack((classes, groups)), axis=0,
            return_inverse=True, return_counts=True,
        )
        self.combined = combined
        self.degree = (n - group_sizes[groups] - class_sizes[classes] + sizes[combined]).astype(np.float64)
        group_start = np.cumsum(group_sizes) - group_sizes
        target_balance = n - group_sizes[groups] - 2 * group_start[groups]
        prefix = np.cumsum(sizes) - sizes
        class_start = np.r_[0, np.flatnonzero(combinations[1:, 0] != combinations[:-1, 0]) + 1]
        class_offsets = np.empty(class_sizes.size, dtype=np.int64)
        class_offsets[combinations[class_start, 0]] = prefix[class_start]
        duplicate_above = prefix - class_offsets[combinations[:, 0]]
        duplicate_balance = class_sizes[classes] - sizes[combined] - 2 * duplicate_above[combined]
        self.balance = (target_balance - duplicate_balance).astype(np.float64)
        self.pair_count = ranking_input.pair_index.modeled_comparisons
        if int(np.sum(self.degree)) != 2 * self.pair_count:
            raise RuntimeError("preference Laplacian comparison count is inconsistent")

    def multiply(self, values: np.ndarray) -> np.ndarray:
        group_sum = np.bincount(self.groups, weights=values)
        class_sum = np.bincount(self.classes, weights=values)
        combined_sum = np.bincount(self.combined, weights=values)
        return (self.degree * values - float(np.sum(values))
                + group_sum[self.groups] + class_sum[self.classes]
                - combined_sum[self.combined])


def _surrogate_parameterization(points: np.ndarray, ranking_input: Any,
                                rank_tol: float | None) -> Any:
    """Keep the chosen polynomial feature metric for regularized baselines.

    Additional per-column conditioning would change ||a||^2 and, in kernel
    mode, destroy the polynomial kernel's RKHS regularizer. Inputs were already
    reversibly scaled before lifting. Only constant columns are omitted here.
    An explicitly requested SVD cutoff uses the legacy projected metric.
    """
    if rank_tol is not None:
        return core._score_parameterization(
            points, ranking_input.target_order, rank_tol, ranking_input.point_class_ids_order,
        )
    active = np.flatnonzero(np.any(points != points[:1], axis=0))
    return core._ScoreParameterization(
        score_basis=np.ascontiguousarray(points[ranking_input.target_order][:, active]),
        active_features=active, feature_shifts=np.zeros(active.size, dtype=np.int64),
        right_basis=None, singular_values=None,
        dimension_bound=min(active.size, ranking_input.point_class_count - 1),
        projected_instance=False, ambient_dimension=points.shape[1],
    )


def _pairwise_lifted(
    points: np.ndarray, target_order: Any = None, *, target_ranks: Any = None,
    loss: str, regularization: float = 1e-3, max_training_pairs: int | None = 100_000,
    epochs: int = 100, batch_size: int | None = None, learning_rate: float = 0.03,
    max_materialized_pairs: int | None = 1_000_000,
    rank_tol: float | None = None, seed: int = 0,
) -> Any:
    """Existing scalable hinge/logistic optimizers, preserving lifted L2 metric."""
    started = time.perf_counter()
    regularization = linear._nonnegative_real(regularization, "regularization")
    limit = linear._optional_positive_integer(max_training_pairs, "max_training_pairs")
    guard = linear._optional_positive_integer(max_materialized_pairs, "max_materialized_pairs")
    epochs = linear._positive_integer(epochs, "epochs")
    batch_size = linear._optional_positive_integer(batch_size, "batch_size")
    rate = linear._positive_real(learning_rate, "learning_rate")
    ranking_input = core._validate_ranking_input(points, target_order, target_ranks)
    algorithm = "rank_svm" if loss == "hinge" else "polynomial_score_ranknet"
    population = ranking_input.pair_index.modeled_comparisons
    if population == 0:
        return linear._trivial_result(algorithm, ranking_input, started,
                                      diagnostics={"comparison_population": "empty", "loss": loss})
    parameterization = _surrogate_parameterization(points, ranking_input, rank_tol)
    rng = np.random.default_rng(int(seed) % 2_000_000_000)
    # Full hinge uses endpoint counting without materializing pairs. Logistic
    # uses a bounded sample, or explicit full pairs only after this guard.
    sample_size = population if limit is None else min(limit, population)
    if loss != "hinge" or (limit is not None and limit < population):
        if guard is not None and sample_size > guard:
            raise ValueError("pairwise loss materialization exceeds the guard; lower max_training_pairs")
    ids, sample_size, unique_size, sampled = linear._pair_training_sample(population, limit, rng)
    optimization_started = time.perf_counter()
    initial = linear._linearized_mrc_direction(
        parameterization.score_basis, 1e-6, 1e-6, 50, ranking_input.target_tie_ids_order,
    )[0]
    if loss == "hinge" and ids is None:
        q, objective, iterations, diagnostics = linear._full_rank_svm_adam(
            parameterization.score_basis, regularization=regularization,
            epochs=epochs, learning_rate=rate, initial_direction=initial,
            ranking_input=ranking_input,
        )
    else:
        if ids is None:
            ids = np.arange(population, dtype=np.int64)
        q, objective, iterations, diagnostics = linear._sampled_pairwise_adam(
            parameterization.score_basis, ids, ranking_input.pair_index,
            loss=loss, regularization=regularization, epochs=epochs,
            batch_size=batch_size, learning_rate=rate, rng=rng, initial_direction=initial,
        )
    diagnostics.update({
        "loss": loss, "regularization": regularization,
        "comparison_population": "sampled" if sampled else "full",
        "learning_rate": rate, "batch_size": batch_size,
        "regularization_space": "projected_conditioned" if rank_tol is not None else "chosen_polynomial_feature_basis",
    })
    return linear._finalize_result(
        algorithm=algorithm, ranking_input=ranking_input, parameterization=parameterization,
        reduced_direction=q, total_started=started,
        optimization_runtime=time.perf_counter() - optimization_started,
        solver_runtime_prefix=0.0, optimization_iterations=iterations,
        surrogate_objective=objective, training_sample_size=sample_size,
        unique_training_comparisons=unique_size, diagnostics=diagnostics,
        seed=seed, output_flag=False, threads=None, feasibility_tol=1e-8, gurobi_params=None,
    )


def _rank_svm_lifted(points: np.ndarray, target_order: Any = None, **kwargs: Any) -> Any:
    return _pairwise_lifted(points, target_order, loss="hinge", **kwargs)


def _ranknet_lifted(points: np.ndarray, target_order: Any = None, **kwargs: Any) -> Any:
    return _pairwise_lifted(points, target_order, loss="logistic", **kwargs)


def _pcg(multiply: Any, rhs: np.ndarray, diagonal: np.ndarray,
         tolerance: float, maximum: int) -> tuple[np.ndarray, int, float]:
    """Preconditioned conjugate gradients; expose residual rather than hide limits."""
    q = np.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return q, 0, 0.0
    diagonal = np.maximum(diagonal, np.finfo(np.float64).tiny)
    z = residual / diagonal
    search = z.copy()
    product = float(residual @ z)
    iterations = 0
    for iteration in range(1, maximum + 1):
        applied = multiply(search)
        denominator = float(search @ applied)
        if not math.isfinite(denominator) or denominator <= 0.0:
            break
        step = product / denominator
        q += step * search
        residual -= step * applied
        iterations = iteration
        if float(np.linalg.norm(residual)) <= tolerance * rhs_norm:
            break
        next_z = residual / diagonal
        next_product = float(residual @ next_z)
        if not math.isfinite(next_product) or product == 0.0:
            break
        search = next_z + (next_product / product) * search
        z, product = next_z, next_product
    if not np.isfinite(q).all():
        raise RuntimeError("RankRLS normal equations produced nonfinite coefficients")
    relative = float(np.linalg.norm(multiply(q) - rhs)) / rhs_norm
    return q, iterations, relative


def _rank_rls_lifted(
    points: np.ndarray, target_order: Any = None, *, target_ranks: Any = None,
    regularization: float = 1e-3, max_training_pairs: int | None = None,
    cg_tolerance: float = 1e-8, cg_max_iterations: int | None = None,
    rank_tol: float | None = None, seed: int = 0,
) -> Any:
    """Preference RankRLS: mean (1 - (s_h - s_l))^2 / 2 + lambda ||q||^2 / 2.

    Only order information is used, not arbitrary differences of numeric rank
    labels. Full mode uses the implicit complete preference-graph Laplacian.
    Sample mode draws eligible original item pairs uniformly with replacement;
    repeated class edges retain their item-level multiplicity.
    """
    started = time.perf_counter()
    regularization = linear._nonnegative_real(regularization, "regularization")
    limit = linear._optional_positive_integer(max_training_pairs, "max_training_pairs")
    tolerance = linear._positive_real(cg_tolerance, "cg_tolerance")
    maximum = linear._optional_positive_integer(cg_max_iterations, "cg_max_iterations")
    ranking_input = core._validate_ranking_input(points, target_order, target_ranks)
    pairs_total = ranking_input.pair_index.modeled_comparisons
    if pairs_total == 0:
        return linear._trivial_result("rank_rls", ranking_input, started,
                                      diagnostics={"comparison_population": "empty"})
    parameterization = _surrogate_parameterization(points, ranking_input, rank_tol)
    basis = parameterization.score_basis
    dimension = basis.shape[1]
    maximum = maximum or min(2_000, max(50, 4 * dimension))
    optimization_started = time.perf_counter()
    sampled = limit is not None and limit < pairs_total
    if sampled:
        rng = np.random.default_rng(int(seed) % 2_000_000_000)
        ids, counts = np.unique(rng.integers(0, pairs_total, size=limit), return_counts=True)
        pairs = ranking_input.pair_index.decode(ids)
        higher, lower = pairs.T
        weights = counts.astype(np.float64)
        # Work on sampled endpoints only, rather than n scores each CG iteration.
        endpoints, local_pairs = np.unique(pairs, return_inverse=True)
        local_pairs = local_pairs.reshape(-1, 2)
        sampled_basis = basis[endpoints]
        h, l = local_pairs.T
        balance = (np.bincount(h, weights=weights, minlength=endpoints.size)
                   - np.bincount(l, weights=weights, minlength=endpoints.size))
        rhs = sampled_basis.T @ balance / limit

        def multiply(q: np.ndarray) -> np.ndarray:
            scores = sampled_basis @ q
            weighted_margins = weights * (scores[h] - scores[l])
            coefficients = (np.bincount(h, weights=weighted_margins, minlength=endpoints.size)
                            - np.bincount(l, weights=weighted_margins, minlength=endpoints.size))
            return sampled_basis.T @ coefficients / limit + regularization * q

        # Exact diagonal in bounded chunks; no sample-by-D difference matrix.
        diagonal = np.full(dimension, regularization)
        for begin in range(0, pairs.shape[0], 2_048):
            end = begin + 2_048
            difference = basis[higher[begin:end]] - basis[lower[begin:end]]
            diagonal += weights[begin:end] @ (difference * difference) / limit
        sample_size, unique_size = limit, ids.size
    else:
        graph = _PreferenceGraph(ranking_input)
        rhs = basis.T @ graph.balance / pairs_total

        def multiply(q: np.ndarray) -> np.ndarray:
            return basis.T @ graph.multiply(basis @ q) / pairs_total + regularization * q

        diagonal = np.full(dimension, regularization)
        for begin in range(0, basis.shape[0], 4_096):
            block = basis[begin : begin + 4_096]
            diagonal += graph.degree[begin : begin + 4_096] @ (block * block) / pairs_total
        sample_size = unique_size = pairs_total
    q, iterations, relative_residual = _pcg(multiply, rhs, diagonal, tolerance, maximum)
    objective = max(0.0, 0.5 + 0.5 * float(q @ multiply(q)) - float(rhs @ q))
    diagnostics = {
        "loss": "unit_preference_squared", "regularization": regularization,
        "optimizer": "matrix_free_preconditioned_cg",
        "comparison_population": "sampled" if sampled else "full_implicit_preference_graph",
        "cg_tolerance": tolerance, "cg_max_iterations": maximum,
        "relative_normal_equation_residual": relative_residual,
        "normal_equations_converged": relative_residual <= tolerance,
        "regularization_space": "projected_conditioned" if rank_tol is not None else "chosen_polynomial_feature_basis",
    }
    return linear._finalize_result(
        algorithm="rank_rls", ranking_input=ranking_input, parameterization=parameterization,
        reduced_direction=q, total_started=started,
        optimization_runtime=time.perf_counter() - optimization_started,
        solver_runtime_prefix=0.0, optimization_iterations=iterations,
        surrogate_objective=objective, training_sample_size=sample_size,
        unique_training_comparisons=unique_size, diagnostics=diagnostics,
        seed=seed, output_flag=False, threads=None, feasibility_tol=1e-8, gurobi_params=None,
    )


def _smoothed_objective_gradient(
    basis: np.ndarray, q: np.ndarray, pairs: np.ndarray, weights: np.ndarray,
    bandwidth: float,
) -> tuple[float, np.ndarray]:
    scores = basis @ q
    margins = (scores[pairs[:, 0]] - scores[pairs[:, 1]]) / bandwidth
    # Stable sigmoid, not log-sigmoid: this differs from the RankNet loss.
    positive = linear._stable_logistic_weights(-margins)
    total = float(np.sum(weights))
    derivative = weights * positive * (1.0 - positive) / bandwidth
    coefficients = (np.bincount(pairs[:, 0], weights=derivative, minlength=basis.shape[0])
                    - np.bincount(pairs[:, 1], weights=derivative, minlength=basis.shape[0]))
    objective = float(weights @ positive) / total
    gradient = basis.T @ coefficients / total
    return objective, gradient


def _smoothed_mrc_lifted(
    points: np.ndarray, target_order: Any = None, *, target_ranks: Any = None,
    max_training_pairs: int | None = 100_000, restarts: int = 4,
    epochs: int = 100, learning_rate: float = 1.0,
    bandwidths: Any = (1.0, 0.5, 0.2, 0.1),
    max_materialized_pairs: int | None = 1_000_000,
    rank_tol: float | None = None, seed: int = 0,
) -> Any:
    """Multistart projected ascent on a bandwidth-continuation sigmoid MRC.

    Enforce ||q||_2=1, with backtracking and warm starts between bandwidths.
    The smoothed-MRC literature motivates the surrogate; this implementation
    is a logistic smoothing adaptation, without statistical inference claims.
    """
    started = time.perf_counter()
    limit = linear._optional_positive_integer(max_training_pairs, "max_training_pairs")
    guard = linear._optional_positive_integer(max_materialized_pairs, "max_materialized_pairs")
    restart_count = linear._positive_integer(restarts, "restarts")
    epoch_count = linear._positive_integer(epochs, "epochs")
    rate = linear._positive_real(learning_rate, "learning_rate")
    bandwidth_values = tuple(linear._positive_real(h, "bandwidth") for h in bandwidths)
    if not bandwidth_values or any(a <= b for a, b in zip(bandwidth_values, bandwidth_values[1:])):
        raise ValueError("bandwidths must be nonempty and strictly decreasing")
    ranking_input = core._validate_ranking_input(points, target_order, target_ranks)
    population = ranking_input.pair_index.modeled_comparisons
    if population == 0:
        return linear._trivial_result("smoothed_mrc", ranking_input, started,
                                      diagnostics={"comparison_population": "empty"})
    requested = population if limit is None else min(population, limit)
    if guard is not None and requested > guard:
        raise ValueError("MRC pair materialization exceeds the guard; use max_training_pairs")
    parameterization = _surrogate_parameterization(points, ranking_input, rank_tol)
    basis = parameterization.score_basis
    rng = np.random.default_rng(int(seed) % 2_000_000_000)
    sampled = limit is not None and limit < population
    if sampled:
        ids, counts = np.unique(rng.integers(0, population, size=limit), return_counts=True)
    else:
        ids, counts = np.arange(population), np.ones(population, dtype=np.int64)
    pairs = ranking_input.pair_index.decode(ids)
    weights = counts.astype(np.float64)
    endpoints, local_pairs = np.unique(pairs, return_inverse=True)
    local_pairs = local_pairs.reshape(-1, 2)
    sampled_basis = basis[endpoints]
    optimization_started = time.perf_counter()
    initial = linear._linearized_mrc_direction(
        basis, 1e-6, 1e-6, 50, ranking_input.target_tie_ids_order,
    )[0]
    seeds = [initial, -initial]
    while len(seeds) < restart_count:
        seeds.append(linear._normalized(rng.normal(size=basis.shape[1])))
    best_q = initial.copy()
    best_objective = -math.inf
    iterations = 0
    per_restart: list[dict[str, Any]] = []
    for restart, initial_q in enumerate(seeds[:restart_count]):
        q = linear._normalized(initial_q)
        accepted = 0
        for bandwidth in bandwidth_values:
            step = rate
            for _ in range(epoch_count):
                iterations += 1
                objective, gradient = _smoothed_objective_gradient(
                    sampled_basis, q, local_pairs, weights, bandwidth,
                )
                tangent = gradient - float(q @ gradient) * q
                norm_squared = float(tangent @ tangent)
                if norm_squared <= 1e-20:
                    break
                accepted_move = False
                for _ in range(25):
                    candidate = linear._normalized(q + step * tangent)
                    candidate_objective, _ = _smoothed_objective_gradient(
                        sampled_basis, candidate, local_pairs, weights, bandwidth,
                    )
                    if candidate_objective >= objective + 1e-4 * step * norm_squared:
                        q = candidate
                        accepted += 1
                        step = min(rate, step * 1.5)
                        accepted_move = True
                        break
                    step *= 0.5
                if not accepted_move:
                    break
        objective, _ = _smoothed_objective_gradient(
            sampled_basis, q, local_pairs, weights, bandwidth_values[-1],
        )
        per_restart.append({"restart": restart, "smoothed_agreement": objective,
                            "accepted_moves": accepted})
        if objective > best_objective:
            best_q, best_objective = q.copy(), objective
    return linear._finalize_result(
        algorithm="smoothed_mrc", ranking_input=ranking_input, parameterization=parameterization,
        reduced_direction=best_q, total_started=started,
        optimization_runtime=time.perf_counter() - optimization_started,
        solver_runtime_prefix=0.0, optimization_iterations=iterations,
        surrogate_objective=best_objective, training_sample_size=requested,
        unique_training_comparisons=ids.size,
        diagnostics={"optimizer": "sphere_projected_ascent_with_backtracking",
                     "smoothing": "logistic_sigmoid_not_log_sigmoid",
                     "bandwidths": list(bandwidth_values), "restarts": restart_count,
                     "comparison_population": "sampled" if sampled else "full",
                     "restart_diagnostics": per_restart},
        seed=seed, output_flag=False, threads=None, feasibility_tol=1e-8, gurobi_params=None,
    )


_ENGINES = {
    "lmrc": linear.linearized_mrc_baseline,
    "ranksvm": _rank_svm_lifted,
    "ranknet": _ranknet_lifted,
    "rankrls": _rank_rls_lifted,
    "smoothed-mrc": _smoothed_mrc_lifted,
    "local": linear.kendall_local_search_baseline,
    "maxfs": linear.elastic_maxfs_baseline,
}


def polynomial_baseline(
    points: Any, target_order: Any = None, *, method: str,
    target_ranks: Any = None, degree: int = 2, feature_basis: str = "monomial",
    input_scaling: str = "power2", max_features: int | None = DEFAULT_MAX_FEATURES,
    max_feature_entries: int | None = DEFAULT_MAX_FEATURE_ENTRIES, **kwargs: Any,
) -> PolynomialRankingResult:
    """Common API; remaining keywords are the selected engine's options."""
    if method not in _ENGINES:
        raise ValueError(f"unknown baseline {method!r}; choose from {tuple(_ENGINES)}")
    started = time.perf_counter()
    features, feature_map = polynomial_features(
        points, degree=degree, feature_basis=feature_basis, input_scaling=input_scaling,
        max_features=max_features, max_feature_entries=max_feature_entries,
    )
    feature_runtime = time.perf_counter() - started
    result = _ENGINES[method](features, target_order, target_ranks=target_ranks, **kwargs)
    return wrap_result(result, feature_map, "polynomial_" + method.replace("-", "_"),
                       started, feature_runtime)


def polynomial_rank_svm_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="ranksvm", **kwargs)


def polynomial_ranknet_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="ranknet", **kwargs)


def polynomial_rank_rls_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="rankrls", **kwargs)


def polynomial_smoothed_mrc_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="smoothed-mrc", **kwargs)


def polynomial_lmrc_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="lmrc", **kwargs)


def polynomial_kendall_local_search_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="local", **kwargs)


def polynomial_elastic_maxfs_baseline(points: Any, target_order: Any = None, **kwargs: Any) -> PolynomialRankingResult:
    return polynomial_baseline(points, target_order, method="maxfs", **kwargs)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="method", required=True)
    for name in _ENGINES:
        subparser = subparsers.add_parser(name)
        add_common_arguments(subparser)
        if name == "lmrc":
            subparser.add_argument("--ridge", type=float, default=1e-6)
            subparser.add_argument("--cg-tolerance", type=float, default=1e-10)
            subparser.add_argument("--cg-max-iterations", type=none_if_zero, default=None)
        if name in {"ranksvm", "ranknet", "rankrls"}:
            subparser.add_argument("--regularization", type=float, default=1e-3)
        if name in {"ranksvm", "ranknet", "rankrls", "smoothed-mrc", "maxfs"}:
            subparser.add_argument("--max-training-pairs", type=none_if_zero,
                                   default=None if name == "rankrls" else 100_000)
        if name in {"ranksvm", "ranknet"}:
            subparser.add_argument("--epochs", type=int, default=100)
            subparser.add_argument("--batch-size", type=none_if_zero, default=None)
            subparser.add_argument("--learning-rate", type=float, default=0.03)
            subparser.add_argument("--max-materialized-pairs", type=none_if_zero, default=1_000_000)
        if name == "rankrls":
            subparser.add_argument("--cg-tolerance", type=float, default=1e-8)
            subparser.add_argument("--cg-max-iterations", type=none_if_zero, default=None)
        if name == "smoothed-mrc":
            subparser.add_argument("--restarts", type=int, default=4)
            subparser.add_argument("--epochs", type=int, default=100)
            subparser.add_argument("--learning-rate", type=float, default=1.0)
            subparser.add_argument("--bandwidths", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.1])
            subparser.add_argument("--max-materialized-pairs", type=none_if_zero, default=1_000_000)
        if name == "local":
            subparser.add_argument("--restarts", type=int, default=8)
            subparser.add_argument("--max-iterations", type=int, default=250)
            subparser.add_argument("--patience", type=int, default=25)
            subparser.add_argument("--initial-step", type=float, default=0.25)
            subparser.add_argument("--min-step", type=float, default=1e-4)
            subparser.add_argument("--step-decay", type=float, default=0.5)
            subparser.add_argument("--ridge", type=float, default=1e-6)
        if name == "maxfs":
            subparser.add_argument("--deletion-fraction", type=float, default=0.05)
            subparser.add_argument("--max-removal-iterations", type=int, default=1_000)
            subparser.add_argument("--max-add-back", type=int, default=1_000)
            subparser.add_argument("--elastic-tolerance", type=float, default=1e-8)
            subparser.add_argument("--threads", type=int)
            subparser.add_argument("--solver-log", action="store_true", help="solver log may precede JSON")
    args = parser.parse_args()
    try:
        points, common = load_cli_input(args)
        options = vars(args).copy()
        for key in ("method", "points", "delimiter", "output", "target_ranks_file", *common):
            options.pop(key, None)
        if "solver_log" in options:
            options["output_flag"] = options.pop("solver_log")
        emit_cli_result(polynomial_baseline(points, method=args.method, **common, **options), args.output)
    except (OSError, RuntimeError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
