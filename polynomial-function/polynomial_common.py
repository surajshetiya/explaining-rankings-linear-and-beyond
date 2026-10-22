"""Shared polynomial lifting, model serialization, and command-line helpers.

Keep this folder directly below Code: it deliberately reuses the tested linear
ranking engines rather than maintaining a second implementation of weak ranks.
All nonconstant monomials of total degree 1..p are included. Products use
float64; numerical certificates concern the computed lifted instance, not an
exact rational evaluation of the manuscript's polynomial feature map.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CODE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

import closest_linear_ranking_gurobi as core  # noqa: E402
import linear_ranking_baselines as linear  # noqa: E402

DEFAULT_MAX_FEATURES = 10_000
DEFAULT_MAX_FEATURE_ENTRIES = 25_000_000


def optional_limit(value: int | None, name: str) -> int | None:
    return None if value is None else linear._positive_integer(value, name)


def none_if_zero(value: str) -> int | None:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("use a positive integer, or 0 to disable")
    return integer or None


@dataclass(frozen=True)
class PolynomialFeatureMap:
    """Reusable map; term indices are zero-based and may repeat (powers)."""

    degree: int
    input_dimension: int
    terms: tuple[tuple[int, ...], ...]
    input_scale_exponents: np.ndarray
    term_weights: np.ndarray
    feature_basis: str
    input_scaling: str

    @property
    def expanded_dimension(self) -> int:
        return len(self.terms)

    def transform(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        *,
        max_feature_entries: int | None = DEFAULT_MAX_FEATURE_ENTRIES,
    ) -> np.ndarray:
        x = core._validate_points(points)
        if x.shape[1] != self.input_dimension:
            raise ValueError("prediction points have the wrong feature dimension")
        entries = x.shape[0] * self.expanded_dimension
        limit = optional_limit(max_feature_entries, "max_feature_entries")
        if limit is not None and entries > limit:
            raise ValueError(
                f"polynomial expansion needs {entries} float64 entries "
                f"({8 * entries / 1024**2:.1f} MiB), exceeding "
                f"max_feature_entries={limit}; lower degree or increase the "
                "guard only if the features AND working copies fit memory"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = np.ldexp(x, self.input_scale_exponents)
            restored = np.ldexp(scaled, -self.input_scale_exponents)
        if not np.isfinite(scaled).all() or not np.array_equal(x, restored):
            raise ValueError(
                "power-of-two input scaling is not reversible for these "
                "coordinates; rescale the data explicitly or use input_scaling='none'"
            )

        features = np.empty((x.shape[0], self.expanded_dimension), dtype=np.float64)
        prefix_columns: dict[tuple[int, ...], int] = {}
        # Each higher-degree monomial reuses its already computed prefix:
        # O(n D_p), not O(n p D_p), floating-point multiplications.
        for column, term in enumerate(self.terms):
            if len(term) == 1:
                features[:, column] = scaled[:, term[0]]
            else:
                left = features[:, prefix_columns[term[:-1]]]
                right = scaled[:, term[-1]]
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    np.multiply(left, right, out=features[:, column])
                product = features[:, column]
                if not np.isfinite(product).all():
                    raise ValueError("polynomial products overflowed; enable power2 scaling")
                if np.any((product == 0.0) & (left != 0.0) & (right != 0.0)):
                    raise ValueError(
                        "a nonzero polynomial monomial underflowed to zero; "
                        "lower the degree or explicitly rescale the data"
                    )
            prefix_columns[term] = column
        # Apply kernel feature weights after raw products: prefixes must NOT
        # include the kernel weights of their lower-degree monomials.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            features *= self.term_weights
        if not np.isfinite(features).all():
            raise ValueError("weighted polynomial features overflowed")
        if not np.all(np.isfinite(self.term_weights) & (self.term_weights > 0.0)):
            raise ValueError("polynomial feature weights must be finite and positive")
        # Degree-one columns are losslessly scaled copies of the input, so no
        # distinct input rows can collapse into a new duplicate class.
        if self.feature_basis == "monomial":
            if not np.array_equal(features[:, : self.input_dimension], scaled):
                raise RuntimeError("degree-one features failed to preserve input rows")
        else:
            # Even kernel weights round: check injectivity of each original
            # coordinate, without building a second full duplicate-class table.
            for j in range(self.input_dimension):
                if np.unique(scaled[:, j]).size != np.unique(features[:, j]).size:
                    raise ValueError("kernel feature weighting collapsed distinct coordinates")
        return features

    def to_model(self, coefficients: np.ndarray) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "input_dimension": self.input_dimension,
            "expanded_dimension": self.expanded_dimension,
            "feature_basis": self.feature_basis,
            "input_scaling": self.input_scaling,
            "input_scale_exponents": self.input_scale_exponents.tolist(),
            "monomial_indices": [list(term) for term in self.terms],
            "feature_weights": self.term_weights.tolist(),
            "coefficients": np.asarray(coefficients, dtype=np.float64).tolist(),
            "constant": 0.0,
            "score_convention": (
                "u[j] = ldexp(x[j], input_scale_exponents[j]); "
                "score = sum(coefficients[t] * feature_weights[t] * "
                "product(u[j] for j in monomial_indices[t]))"
            ),
        }


def polynomial_features(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    degree: int = 2,
    feature_basis: str = "monomial",
    input_scaling: str = "power2",
    max_features: int | None = DEFAULT_MAX_FEATURES,
    max_feature_entries: int | None = DEFAULT_MAX_FEATURE_ENTRIES,
) -> tuple[np.ndarray, PolynomialFeatureMap]:
    """Lift items FIRST, then let ranking engines take feature differences.

    ``polynomial-kernel`` weights realize (1 + <u,v>)**p - 1. The omitted
    constant cancels from rankings. Both bases span exactly the same
    unrestricted degree-at-most-p polynomial family in exact arithmetic;
    their L2 regularization geometries differ.
    """
    x = core._validate_points(points)
    p = linear._positive_integer(degree, "degree")
    if feature_basis not in {"monomial", "polynomial-kernel"}:
        raise ValueError("feature_basis must be monomial or polynomial-kernel")
    if input_scaling not in {"power2", "none"}:
        raise ValueError("input_scaling must be power2 or none")
    d = x.shape[1]
    count = math.comb(d + p, p) - 1
    limit = optional_limit(max_features, "max_features")
    if limit is not None and count > limit:
        raise ValueError(
            f"degree {p} in dimension {d} needs {count} monomials, exceeding "
            f"max_features={limit}; lower degree/dimension or increase the guard"
        )
    entries_limit = optional_limit(max_feature_entries, "max_feature_entries")
    if entries_limit is not None and x.shape[0] * count > entries_limit:
        raise ValueError(
            f"polynomial expansion needs {x.shape[0] * count} entries, "
            f"exceeding max_feature_entries={entries_limit}; lower degree or "
            "increase the guard after checking memory"
        )
    shifts = np.zeros(d, dtype=np.int64)
    if input_scaling == "power2":
        maxima = np.max(np.abs(x), axis=0)
        _, exponents = np.frexp(maxima)
        shifts[maxima > 0] = -exponents[maxima > 0]
    terms = tuple(
        term
        for total_degree in range(1, p + 1)
        for term in itertools.combinations_with_replacement(range(d), total_degree)
    )
    weights = np.ones(count, dtype=np.float64)
    if feature_basis == "polynomial-kernel":
        for t, term in enumerate(terms):
            multiplicities = [len(list(group)) for _, group in itertools.groupby(term)]
            log_weight = (
                math.lgamma(p + 1) - math.lgamma(p - len(term) + 1)
                - sum(math.lgamma(value + 1) for value in multiplicities)
            )
            try:
                weights[t] = math.exp(0.5 * log_weight)
            except OverflowError as exc:
                raise ValueError("polynomial kernel feature weights overflowed") from exc
    feature_map = PolynomialFeatureMap(p, d, terms, shifts, weights, feature_basis, input_scaling)
    return feature_map.transform(x, max_feature_entries=entries_limit), feature_map


@dataclass(frozen=True)
class PolynomialRankingResult:
    """Preserve existing result attributes and add a reusable polynomial model."""

    linear_result: Any
    feature_map: PolynomialFeatureMap
    algorithm: str
    feature_construction_runtime_seconds: float
    total_runtime_seconds: float

    def __getattr__(self, name: str) -> Any:
        return getattr(self.linear_result, name)

    @property
    def kendall_agreement(self) -> int:
        return int(self.linear_result.total_comparisons - self.linear_result.kendall_tau)

    @property
    def expanded_dimension(self) -> int:
        return self.feature_map.expanded_dimension

    @property
    def degree(self) -> int:
        return self.feature_map.degree

    @property
    def input_dimension(self) -> int:
        return self.feature_map.input_dimension

    @property
    def polynomial_model(self) -> dict[str, Any]:
        return self.feature_map.to_model(self.linear_result.direction)

    def to_dict(self) -> dict[str, Any]:
        result = self.linear_result.to_dict()
        result.update({
            "algorithm": self.algorithm,
            "kendall_agreement": int(result["total_comparisons"] - result["kendall_tau"]),
            "degree": self.feature_map.degree,
            "input_dimension": self.feature_map.input_dimension,
            "expanded_dimension": self.feature_map.expanded_dimension,
            "polynomial_model": self.feature_map.to_model(self.linear_result.direction),
            "polynomial_feature_arithmetic": "float64",
            "certificate_instance": "computed_float64_polynomial_features",
            "feature_construction_runtime_seconds": self.feature_construction_runtime_seconds,
            "ranking_engine_runtime_seconds": self.total_runtime_seconds - self.feature_construction_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
        })
        if "approximation_factor" in result:
            result["theoretical_approximation_factor_upper_bound"] = result["dimension_bound"] + 1
        return result


def wrap_result(result: Any, feature_map: PolynomialFeatureMap, algorithm: str,
                started: float, feature_runtime: float) -> PolynomialRankingResult:
    return PolynomialRankingResult(result, feature_map, algorithm, feature_runtime,
                                   time.perf_counter() - started)


def predict_polynomial_scores(
    points: Sequence[Sequence[float]] | np.ndarray,
    result_or_model: PolynomialRankingResult | dict[str, Any],
    *,
    max_feature_entries: int | None = DEFAULT_MAX_FEATURE_ENTRIES,
) -> np.ndarray:
    """Evaluate the serialized model on new points using its training scaling.

    On cancellation-heavy data a direct float dot product can appear tied;
    use conditioned_scores/ranking_tie_ids for the certified training order.
    """
    if isinstance(result_or_model, PolynomialRankingResult):
        model = result_or_model.feature_map.to_model(result_or_model.direction)
    else:
        model = result_or_model.get("polynomial_model", result_or_model)
    feature_map = PolynomialFeatureMap(
        int(model["degree"]), int(model["input_dimension"]),
        tuple(tuple(int(j) for j in term) for term in model["monomial_indices"]),
        np.asarray(model["input_scale_exponents"], dtype=np.int64),
        np.asarray(model["feature_weights"], dtype=np.float64),
        str(model["feature_basis"]), str(model["input_scaling"]),
    )
    features = feature_map.transform(points, max_feature_entries=max_feature_entries)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1],) or not np.isfinite(coefficients).all():
        raise ValueError("invalid polynomial coefficient vector")
    scores = features @ coefficients + float(model.get("constant", 0.0))
    if not np.isfinite(scores).all():
        raise RuntimeError("polynomial prediction scores became nonfinite")
    return scores


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("points", type=Path, help="headerless CSV, whitespace text, or .npy features")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-order", type=core._parse_target_order, metavar="I,J,...")
    target.add_argument("--target-ranks", type=core._parse_target_ranks, metavar="R,R,...")
    target.add_argument("--target-ranks-file", type=Path)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--feature-basis", choices=("monomial", "polynomial-kernel"), default="monomial")
    parser.add_argument("--input-scaling", choices=("power2", "none"), default="power2")
    parser.add_argument("--max-features", type=none_if_zero, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--max-feature-entries", type=none_if_zero, default=DEFAULT_MAX_FEATURE_ENTRIES)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--rank-tol", type=float, default=None, help="optional SVD cutoff; changes the comparison space")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)


def load_cli_input(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    points = core._load_points(args.points, args.delimiter)
    ranks = args.target_ranks
    if args.target_ranks_file is not None:
        ranks = core._load_target_ranks(args.target_ranks_file, args.delimiter)
    return points, {
        "target_order": args.target_order, "target_ranks": ranks,
        "degree": args.degree, "feature_basis": args.feature_basis,
        "input_scaling": args.input_scaling, "max_features": args.max_features,
        "max_feature_entries": args.max_feature_entries,
        "rank_tol": args.rank_tol, "seed": args.seed,
    }


def emit_cli_result(result: PolynomialRankingResult, output: Path | None) -> None:
    rendered = json.dumps(result.to_dict(), indent=2, allow_nan=False)
    if output is None:
        print(rendered)
    else:
        output.write_text(rendered + "\n", encoding="utf-8")
