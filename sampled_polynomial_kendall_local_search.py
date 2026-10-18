"""Polynomial feature lifting + duplicate-aware sampled Kendall local search.

The inner event-sweep search runs in the sampled difference span, not the full
polynomial dimension. Default importance sampling mixes item-weighted,
class-balanced, and pilot-hard comparisons; large class populations use sparse
on-demand multiplicities automatically. Uniform sampling is an ablation.
This is a heuristic, NOT the exact empirical optimizer required by the PTAS.
Finalist/antipodal selection retains the linear engine's deterministic 1/2
maximum-agreement floor on the computed features, including duplicate rows.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from polynomial_common import (
    DEFAULT_MAX_FEATURE_ENTRIES, DEFAULT_MAX_FEATURES, PolynomialRankingResult,
    add_common_arguments, emit_cli_result, load_cli_input, none_if_zero,
    polynomial_features, wrap_result,
)

import duplicate_aware_sampled_kendall_local_search as duplicate_search


def max_agreement_sampled_polynomial_local_search(
    points: Any, target_order: Any = None, *, target_ranks: Any = None,
    degree: int = 2, feature_basis: str = "monomial", input_scaling: str = "power2",
    max_features: int | None = DEFAULT_MAX_FEATURES,
    max_feature_entries: int | None = DEFAULT_MAX_FEATURE_ENTRIES,
    sampling: str = "duplicate-aware", epsilon: float = 0.8, delta: float = 0.8,
    sample_size: int | None = 200, restarts: int = 8, max_iterations: int = 250,
    patience: int = 40, gaussian_interval: int = 10, ridge: float = 1e-6,
    use_lmrc_seed: bool = True, lmrc_cg_tolerance: float = 1e-6,
    lmrc_cg_max_iterations: int | None = 50,
    objective_mixture_weight: float = 0.5, class_mixture_weight: float = 0.25,
    hard_mixture_weight: float = 0.25,
    max_class_matrix_entries: int | None = 4_000_000,
    max_sample_size: int | None = 1_000_000,
    rank_tol: float | None = None, sample_rank_tol: float = 0.0, seed: int = 0,
) -> PolynomialRankingResult:
    """Sample eligible item comparisons after lifting, preserving weak ranks.

    epsilon/delta supply reference sample-size diagnostics, not a PTAS claim.
    Leave rank_tol=None to retain the full lifted model. An explicit cutoff
    requests a projected-instance heuristic instead.
    """
    if sampling not in {"duplicate-aware", "uniform"}:
        raise ValueError("sampling must be duplicate-aware or uniform")
    started = time.perf_counter()
    features, feature_map = polynomial_features(
        points, degree=degree, feature_basis=feature_basis, input_scaling=input_scaling,
        max_features=max_features, max_feature_entries=max_feature_entries,
    )
    feature_runtime = time.perf_counter() - started
    options = {
        "target_ranks": target_ranks, "epsilon": epsilon, "delta": delta,
        "sample_size": sample_size, "restarts": restarts,
        "max_iterations": max_iterations, "patience": patience,
        "gaussian_interval": gaussian_interval, "ridge": ridge,
        "use_lmrc_seed": use_lmrc_seed, "lmrc_cg_tolerance": lmrc_cg_tolerance,
        "lmrc_cg_max_iterations": lmrc_cg_max_iterations,
        "max_sample_size": max_sample_size, "rank_tol": rank_tol,
        "sample_rank_tol": sample_rank_tol, "seed": seed,
    }
    if sampling == "uniform":
        # Uniform original-item pair sampling is the objective-only mixture.
        # Use the hardened engine for BOTH modes: its finalization protects
        # robust comparisons and has a solver-free numerical fallback.
        objective_mixture_weight, class_mixture_weight, hard_mixture_weight = 1.0, 0.0, 0.0
        max_class_matrix_entries = None
    result = duplicate_search.max_agreement_duplicate_aware_sampled_local_search(
        features, target_order, **options,
        objective_mixture_weight=objective_mixture_weight,
        class_mixture_weight=class_mixture_weight,
        hard_mixture_weight=hard_mixture_weight,
        max_class_matrix_entries=max_class_matrix_entries,
    )
    return wrap_result(
        result, feature_map, "polynomial_sampled_local_search_" + sampling.replace("-", "_"),
        started, feature_runtime,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--sampling", choices=("duplicate-aware", "uniform"), default="duplicate-aware")
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--sample-size", type=none_if_zero, default=200)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--gaussian-interval", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--no-lmrc-seed", action="store_true")
    parser.add_argument("--lmrc-cg-tolerance", type=float, default=1e-6)
    parser.add_argument("--lmrc-cg-max-iterations", type=none_if_zero, default=50)
    parser.add_argument("--objective-mixture-weight", type=float, default=0.5)
    parser.add_argument("--class-mixture-weight", type=float, default=0.25)
    parser.add_argument("--hard-mixture-weight", type=float, default=0.25)
    parser.add_argument("--max-class-matrix-entries", type=none_if_zero, default=4_000_000,
                        help="exact class-table threshold; 0 forces sparse on-demand mode")
    parser.add_argument("--max-sample-size", type=none_if_zero, default=1_000_000)
    parser.add_argument("--sample-rank-tol", type=float, default=0.0)
    args = parser.parse_args()
    try:
        points, common = load_cli_input(args)
        options = vars(args).copy()
        for key in ("points", "delimiter", "output", "target_ranks_file", *common):
            options.pop(key, None)
        options["use_lmrc_seed"] = not options.pop("no_lmrc_seed")
        emit_cli_result(max_agreement_sampled_polynomial_local_search(points, **common, **options), args.output)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
