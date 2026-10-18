"""Polynomial min-disagreement via the linear greedy minimal-IIS approximation.

Lift all nonconstant monomials of total degree <= p, then run the existing
score-factorized Gurobi LP/IIS engine on the entire comparable-pair population.
The factor is r_p+1 (bounded by D_p+1), NOT the original dimension d.
Exact duplicate rows stay tied; target ties are omitted; incompatible strict
target comparisons between duplicates are counted as forced disagreements.
The numerical certificate is for the computed float64 lifted features.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from polynomial_common import (
    DEFAULT_MAX_FEATURE_ENTRIES, DEFAULT_MAX_FEATURES, PolynomialRankingResult,
    add_common_arguments, core, emit_cli_result, load_cli_input, none_if_zero,
    optional_limit, polynomial_features, wrap_result,
)


def closest_polynomial_ranking(
    points: Any, target_order: Any = None, *, target_ranks: Any = None,
    degree: int = 2, feature_basis: str = "monomial", input_scaling: str = "power2",
    max_features: int | None = DEFAULT_MAX_FEATURES,
    max_feature_entries: int | None = DEFAULT_MAX_FEATURE_ENTRIES,
    max_modeled_comparisons: int | None = 2_000_000,
    rank_tol: float | None = None, seed: int = 0, output_flag: bool = False,
    threads: int | None = None, feasibility_tol: float = 1e-8,
    gurobi_params: Any = None,
) -> PolynomialRankingResult:
    """Run full-population IIS deletion, with a preflight LP-size guard.

    Sampling is intentionally not used here: it would lose the full-instance
    min-disagreement certificate. Solver timeouts raise rather than returning
    an uncertified result. The returned observed approximation_factor is the
    maximum minimal-IIS block size (or 1 if no block was deleted).
    """
    started = time.perf_counter()
    features, feature_map = polynomial_features(
        points, degree=degree, feature_basis=feature_basis, input_scaling=input_scaling,
        max_features=max_features, max_feature_entries=max_feature_entries,
    )
    feature_runtime = time.perf_counter() - started
    limit = optional_limit(max_modeled_comparisons, "max_modeled_comparisons")
    ranking_input = core._validate_ranking_input(features, target_order, target_ranks)
    modeled = ranking_input.pair_index.modeled_comparisons
    if limit is not None and modeled > limit:
        raise ValueError(
            f"the full IIS algorithm models {modeled} comparisons, exceeding "
            f"max_modeled_comparisons={limit}; use a smaller dataset or increase "
            "the guard after checking LP memory. Sampled local search is a "
            "scalable heuristic, not a replacement for this certificate"
        )
    result = core.closest_linear_ranking(
        features, target_order, target_ranks=target_ranks, rank_tol=rank_tol,
        seed=seed, output_flag=output_flag, threads=threads,
        feasibility_tol=feasibility_tol, gurobi_params=gurobi_params,
    )
    return wrap_result(result, feature_map, "polynomial_greedy_iis_min_disagreement",
                       started, feature_runtime)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-modeled-comparisons", type=none_if_zero, default=2_000_000)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--feasibility-tol", type=float, default=1e-8)
    parser.add_argument("--time-limit", type=float, help="per Gurobi solve; timeout is an error")
    parser.add_argument("--solver-log", action="store_true", help="solver log may precede JSON")
    args = parser.parse_args()
    try:
        points, common = load_cli_input(args)
        params = None
        if args.time_limit is not None:
            if not 0 < args.time_limit < float("inf"):
                raise ValueError("time_limit must be finite and positive")
            params = {"TimeLimit": args.time_limit}
        result = closest_polynomial_ranking(
            points, **common, max_modeled_comparisons=args.max_modeled_comparisons,
            threads=args.threads, feasibility_tol=args.feasibility_tol,
            output_flag=args.solver_log, gurobi_params=params,
        )
        emit_cli_result(result, args.output)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
