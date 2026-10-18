"""Optimized implementation of Theorem ``thm:main-kt`` with Gurobi.

The theorem repeatedly removes an inclusion-minimal infeasible subsystem
(IIS) from the target pair comparisons until the remaining comparisons are
linearly realizable.  Gurobi's native IIS routine finds a sparse candidate in
the score-factorized model; a small deletion-filter model then refines it to
the comparison-irreducible subsystem required by the proof.

The public function is :func:`closest_linear_ranking`.  A strict target may be
given as the ``target_order`` permutation; a weak target is given by one
numeric ``target_ranks`` value per original row (smaller is higher, equality is
a tie).  The returned flattened ``ranking`` uses original row indices, while
the aligned tie-ID arrays carry the weak-order semantics.

Performance-oriented implementation details
-------------------------------------------
* The certified default uses exact translations and reversible power-of-two
  column scaling; optional SVD projection is available when the caller
  explicitly supplies a rank cutoff.
* Scores are factorized as ``score = B @ q``.  Pair rows then contain only
  two nonzeros (``score[i] - score[j] >= 1``), avoiding a dense
  binomial(n, 2)-by-r pair-difference matrix.
* One persistent Gurobi model is reoptimized after every batch deletion.
* A target-adjacent fast path proves a realizable target using only n-1 rows.
* When at least one comparison is optimizable, the final weak score order is
  refined generically and then certified by a margin-one adjacent-class model.
  With no strict target pair, the zero direction returns one all-tied group.
  Exact duplicate rows remain tied in either case.

Dependencies: NumPy, gurobipy, and a working Gurobi license.
Install the Python packages with ``python -m pip install numpy gurobipy``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # Keep --help and importing diagnostics informative.
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]
    _GUROBI_IMPORT_ERROR: ImportError | None = exc
else:
    _GUROBI_IMPORT_ERROR = None


@dataclass(frozen=True)
class ClosestLinearRankingResult:
    """Result and certificate produced by :func:`closest_linear_ranking`.

    Attributes
    ----------
    ranking:
        Original row indices, ordered from highest to lowest returned score.
    direction:
        An ambient-dimensional scoring direction realizing ``ranking``.
    conditioned_scores:
        Numerically stable conditioned scores in original row order.  The
        exact feature scaling and translation do not affect their order.
    target_order:
        Validated input permutation, highest to lowest.
    target_tie_ids:
        Target tie-group IDs aligned with the original point rows.
    ranking_tie_ids:
        Returned score-tie IDs aligned with the original point rows.
    deleted_comparisons:
        Shape ``(k, 2)`` array of original row-index pairs.  Each row is
        oriented from the higher to the lower item in ``target_order``.  Forced
        cross-target duplicate ties are counted separately rather than stored.
    iis_blocks:
        The same deleted pairs partitioned by IIS-removal iteration.
    kendall_tau:
        Weak Kendall disagreement count.  Target ties are excluded and a
        returned tie on a target-strict pair is a disagreement.
    dimension_bound:
        Safe dimension bound used by the model: at most the number of varying
        features and one less than the number of distinct point rows in ambient
        mode, or the selected SVD dimension in projected mode.  The observed-
        factor certificate does not rely on an expensive rank estimate.
    approximation_factor:
        Observed certificate factor: the largest removed IIS size (or one when
        no IIS is removed).  It is at most the theorem's r+1 in exact arithmetic.
    projected_instance:
        Whether an explicitly requested SVD cutoff modeled a projection rather
        than the default ambient comparison space.
    solver_runtime_seconds:
        Wall time spent constructing and solving the Gurobi models.
    """

    ranking: np.ndarray
    direction: np.ndarray
    conditioned_scores: np.ndarray
    target_order: np.ndarray
    target_tie_ids: np.ndarray
    ranking_tie_ids: np.ndarray
    deleted_comparisons: np.ndarray
    iis_blocks: tuple[np.ndarray, ...]
    kendall_tau: int
    total_comparisons: int
    forced_tie_disagreements: int
    optimizable_comparisons: int
    dimension_bound: int
    approximation_factor: int
    projected_instance: bool
    solver_runtime_seconds: float

    @property
    def deleted_count(self) -> int:
        """Number of comparisons deleted by the theorem's algorithm."""

        return self.forced_tie_disagreements + int(self.deleted_comparisons.shape[0])

    @property
    def modeled_deleted_count(self) -> int:
        """Deleted optimizable comparisons, excluding structurally forced ties."""

        return int(self.deleted_comparisons.shape[0])

    @property
    def iterations(self) -> int:
        """Number of IIS blocks removed."""

        return len(self.iis_blocks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "ranking": self.ranking.tolist(),
            "direction": self.direction.tolist(),
            "conditioned_scores": self.conditioned_scores.tolist(),
            "target_order": self.target_order.tolist(),
            "target_tie_ids": self.target_tie_ids.tolist(),
            "ranking_tie_ids": self.ranking_tie_ids.tolist(),
            "deleted_comparisons": self.deleted_comparisons.tolist(),
            "iis_blocks": [block.tolist() for block in self.iis_blocks],
            "deleted_count": self.deleted_count,
            "iterations": self.iterations,
            "kendall_tau": self.kendall_tau,
            "total_comparisons": self.total_comparisons,
            "forced_tie_disagreements": self.forced_tie_disagreements,
            "optimizable_comparisons": self.optimizable_comparisons,
            "modeled_deleted_count": self.modeled_deleted_count,
            "dimension_bound": self.dimension_bound,
            "approximation_factor": self.approximation_factor,
            "projected_instance": self.projected_instance,
            "solver_runtime_seconds": self.solver_runtime_seconds,
        }


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is required. Install numpy and gurobipy, then configure "
            "a valid Gurobi license before calling closest_linear_ranking()."
        ) from _GUROBI_IMPORT_ERROR


@dataclass(frozen=True)
class _ComparablePairIndex:
    """Compact index of target-comparable, nonduplicate item pairs."""

    row_starts: np.ndarray
    lower_starts: np.ndarray
    point_class_ids: np.ndarray
    class_positions: np.ndarray
    class_offsets: np.ndarray
    modeled_comparisons: int
    target_comparisons: int
    forced_tie_disagreements: int

    @property
    def n(self) -> int:
        return int(self.point_class_ids.size)

    def decode(self, pair_ids: np.ndarray) -> np.ndarray:
        """Decode IDs to target positions without materializing all pairs."""

        ids = np.asarray(pair_ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.modeled_comparisons):
            raise ValueError("pair ID lies outside the modeled comparison population")

        first = np.searchsorted(self.row_starts, ids, side="right") - 1
        offsets = ids - self.row_starts[first]
        lower = self.lower_starts[first]
        classes = self.point_class_ids[first]
        seconds = lower + offsets

        class_sizes = np.diff(self.class_offsets)
        duplicate_mask = class_sizes[classes] > 1
        if np.any(duplicate_mask):
            duplicate_classes = np.unique(classes[duplicate_mask])
            for class_id_raw in duplicate_classes:
                class_id = int(class_id_raw)
                selected = np.flatnonzero(classes == class_id)
                begin = int(self.class_offsets[class_id])
                end = int(self.class_offsets[class_id + 1])
                excluded_positions = self.class_positions[begin:end]
                selected_lower = lower[selected]
                selected_offsets = offsets[selected]
                excluded_before = np.searchsorted(
                    excluded_positions, selected_lower, side="left"
                )
                candidate = selected_lower + selected_offsets
                while True:
                    excluded_through = np.searchsorted(
                        excluded_positions, candidate, side="right"
                    )
                    updated = (
                        selected_lower
                        + selected_offsets
                        + excluded_through
                        - excluded_before
                    )
                    if np.array_equal(updated, candidate):
                        break
                    candidate = updated
                seconds[selected] = candidate

        if np.any(seconds >= self.n) or np.any(
            self.point_class_ids[first] == self.point_class_ids[seconds]
        ):
            raise RuntimeError("internal comparable-pair decoding failed")
        return np.column_stack((first, seconds)).astype(np.int64, copy=False)


@dataclass(frozen=True)
class _RankingInput:
    """Validated points, weak target order, and exact point equivalence classes."""

    points: np.ndarray
    target_order: np.ndarray
    target_tie_ids: np.ndarray
    target_tie_ids_order: np.ndarray
    point_class_ids: np.ndarray
    point_class_ids_order: np.ndarray
    representative_target_positions: np.ndarray
    pair_index: _ComparablePairIndex

    @property
    def point_class_count(self) -> int:
        return int(self.representative_target_positions.size)

    @property
    def target_tie_count(self) -> int:
        return int(self.target_tie_ids_order[-1]) + 1


def _validate_points(
    points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return a finite float64 matrix while preserving exact duplicate rows."""

    try:
        raw_points = np.asarray(points)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("points must be a rectangular numeric matrix") from exc

    if raw_points.ndim != 2:
        raise ValueError(f"points must have shape (n, d), got {raw_points.shape}")
    if raw_points.dtype.kind not in "biuf":
        raise ValueError(
            "points must be a real numeric matrix exactly representable as float64"
        )
    x = raw_points.astype(np.float64, copy=False)
    if raw_points.dtype.kind in "biu" or raw_points.dtype.itemsize > 8:
        with np.errstate(over="ignore", invalid="ignore"):
            restored = x.astype(raw_points.dtype)
        if not np.array_equal(restored, raw_points):
            raise ValueError(
                "conversion to float64 would alter a coordinate; explicitly "
                "rescale or pass the intended float64 instance"
            )

    n, d = x.shape
    if n == 0:
        raise ValueError("points must contain at least one row")
    if d == 0:
        raise ValueError("points must contain at least one feature column")
    if not np.isfinite(x).all():
        raise ValueError("points must contain only finite values")
    return np.ascontiguousarray(x)


def _validate_target_order(
    target_order: Sequence[int] | np.ndarray | None,
    n: int,
) -> np.ndarray:
    if target_order is None:
        return np.arange(n, dtype=np.int64)
    raw = np.asarray(target_order)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError(
            f"target_order must be a length-{n} permutation, got shape {raw.shape}"
        )
    if np.issubdtype(raw.dtype, np.integer):
        order = raw.astype(np.int64, copy=True)
    elif np.issubdtype(raw.dtype, np.floating):
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("target_order entries must be integer row indices")
        order = raw.astype(np.int64)
    else:
        raise ValueError("target_order entries must be integer row indices")
    if not np.array_equal(np.sort(order), np.arange(n, dtype=np.int64)):
        raise ValueError(
            f"target_order must contain each row index 0,...,{n - 1} exactly once"
        )
    return order


def _target_from_ranks(
    target_ranks: Sequence[float] | np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw = np.asarray(target_ranks)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("target_ranks must be a numeric vector") from exc
    if raw.ndim != 1 or raw.size != n or raw.dtype.kind not in "biuf":
        raise ValueError(f"target_ranks must be a length-{n} numeric vector")
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("target_ranks must contain only finite values")
    if raw.dtype.kind in "biu" or raw.dtype.itemsize > 8:
        with np.errstate(over="ignore", invalid="ignore"):
            restored = values.astype(raw.dtype)
        if not np.array_equal(restored, raw):
            raise ValueError("conversion to float64 would merge or alter target ranks")

    order = np.argsort(values, kind="stable").astype(np.int64, copy=False)
    ordered_values = values[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = ordered_values[1:] != ordered_values[:-1]
    ids_order = np.cumsum(new_group, dtype=np.int64) - 1
    ids = np.empty(n, dtype=np.int64)
    ids[order] = ids_order
    return order, ids


def _build_comparable_pair_index(
    target_tie_ids_order: np.ndarray,
    point_class_ids_order: np.ndarray,
    point_class_count: int,
) -> _ComparablePairIndex:
    """Index strict target comparisons between distinct point classes."""

    n = int(target_tie_ids_order.size)
    changes = np.flatnonzero(target_tie_ids_order[1:] != target_tie_ids_order[:-1]) + 1
    group_ends = np.concatenate((changes, np.array([n], dtype=np.int64)))
    group_end_by_id = group_ends
    lower_starts = group_end_by_id[target_tie_ids_order]

    class_sizes = np.bincount(point_class_ids_order, minlength=point_class_count)
    class_offsets = np.empty(point_class_count + 1, dtype=np.int64)
    class_offsets[0] = 0
    np.cumsum(class_sizes, dtype=np.int64, out=class_offsets[1:])
    class_positions = np.argsort(point_class_ids_order, kind="stable").astype(
        np.int64, copy=False
    )

    same_class_lower = np.zeros(n, dtype=np.int64)
    for class_id_raw in np.flatnonzero(class_sizes > 1):
        class_id = int(class_id_raw)
        begin = int(class_offsets[class_id])
        end = int(class_offsets[class_id + 1])
        positions = class_positions[begin:end]
        group_ids = target_tie_ids_order[positions]
        same_class_lower[positions] = (
            end - begin - np.searchsorted(group_ids, group_ids, side="right")
        )

    target_row_counts = n - lower_starts
    modeled_row_counts = target_row_counts - same_class_lower
    if np.any(modeled_row_counts < 0):
        raise RuntimeError("internal comparable-pair counts became negative")
    row_starts = np.empty(n + 1, dtype=np.int64)
    row_starts[0] = 0
    np.cumsum(modeled_row_counts, dtype=np.int64, out=row_starts[1:])
    target_comparisons = int(np.sum(target_row_counts, dtype=np.int64))
    forced = int(np.sum(same_class_lower, dtype=np.int64))
    modeled = int(row_starts[-1])
    if modeled + forced != target_comparisons:
        raise RuntimeError("comparable-pair partition accounting failed")
    return _ComparablePairIndex(
        row_starts=row_starts,
        lower_starts=lower_starts,
        point_class_ids=point_class_ids_order,
        class_positions=class_positions,
        class_offsets=class_offsets,
        modeled_comparisons=modeled,
        target_comparisons=target_comparisons,
        forced_tie_disagreements=forced,
    )


def _validate_ranking_input(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None,
    target_ranks: Sequence[float] | np.ndarray | None,
) -> _RankingInput:
    x = _validate_points(points)
    n = x.shape[0]
    if target_order is not None and target_ranks is not None:
        raise ValueError("target_order and target_ranks are mutually exclusive")
    if target_ranks is None:
        order = _validate_target_order(target_order, n)
        target_tie_ids = np.empty(n, dtype=np.int64)
        target_tie_ids[order] = np.arange(n, dtype=np.int64)
    else:
        order, target_tie_ids = _target_from_ranks(target_ranks, n)

    _, representative_original, point_class_ids = np.unique(
        x, axis=0, return_index=True, return_inverse=True
    )
    point_class_ids = point_class_ids.astype(np.int64, copy=False)
    point_class_count = int(representative_original.size)
    point_class_ids_order = point_class_ids[order]
    _, representative_target_positions = np.unique(
        point_class_ids_order, return_index=True
    )
    representative_target_positions = representative_target_positions.astype(
        np.int64, copy=False
    )
    target_tie_ids_order = target_tie_ids[order]
    pair_index = _build_comparable_pair_index(
        target_tie_ids_order, point_class_ids_order, point_class_count
    )
    return _RankingInput(
        points=x,
        target_order=order,
        target_tie_ids=target_tie_ids,
        target_tie_ids_order=target_tie_ids_order,
        point_class_ids=point_class_ids,
        point_class_ids_order=point_class_ids_order,
        representative_target_positions=representative_target_positions,
        pair_index=pair_index,
    )


def _prediction_from_class_order(
    ranking_input: _RankingInput,
    class_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a strict point-class order to a stable weak item ranking.

    Returns the flattened target-position order, prediction tie IDs aligned to
    original rows, and those IDs in target-position order.
    """

    order = np.asarray(class_order, dtype=np.int64)
    class_count = ranking_input.point_class_count
    if not np.array_equal(np.sort(order), np.arange(class_count, dtype=np.int64)):
        raise RuntimeError("class_order is not a permutation of point classes")
    class_ranks = np.empty(class_count, dtype=np.int64)
    class_ranks[order] = np.arange(class_count, dtype=np.int64)
    prediction_order = class_ranks[ranking_input.point_class_ids_order]
    flattened = np.argsort(prediction_order, kind="stable").astype(np.int64, copy=False)
    prediction_ids = np.empty(ranking_input.points.shape[0], dtype=np.int64)
    prediction_ids[ranking_input.target_order] = prediction_order
    return flattened, prediction_ids, prediction_order


def _all_tied_prediction(
    ranking_input: _RankingInput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the zero-direction weak order with every item tied.

    The flattened permutation follows the stable target order solely to keep
    JSON output deterministic.  Its within-tie order has no rank semantics.
    This representation is useful when the target contains no strict pair at
    all, and for a single exact point class.
    """

    n = ranking_input.points.shape[0]
    flattened = np.arange(n, dtype=np.int64)
    prediction_ids = np.zeros(n, dtype=np.int64)
    prediction_order = np.zeros(n, dtype=np.int64)
    return flattened, prediction_ids, prediction_order


def _weak_order_counts(
    target_tie_ids_order: np.ndarray,
    prediction_tie_ids_order: np.ndarray,
) -> tuple[int, int, int]:
    """Count agreement/disagreement on strict target pairs in O(n log n).

    A predicted tie is a disagreement.  Items tied by the target are queried
    before any member of their target group is inserted, so they are excluded.
    """

    n = int(target_tie_ids_order.size)
    if prediction_tie_ids_order.shape != (n,):
        raise RuntimeError("prediction tie IDs have inconsistent dimensions")
    prediction_count = int(np.max(prediction_tie_ids_order)) + 1 if n else 0
    tree = np.zeros(prediction_count + 1, dtype=np.int64)

    def prefix(count: int) -> int:
        subtotal = 0
        while count > 0:
            subtotal += int(tree[count])
            count -= count & -count
        return subtotal

    def add(index: int) -> None:
        index += 1
        while index < tree.size:
            tree[index] += 1
            index += index & -index

    agreement = 0
    total = 0
    seen = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and target_tie_ids_order[end] == target_tie_ids_order[start]:
            end += 1
        group_predictions = prediction_tie_ids_order[start:end]
        for prediction_raw in group_predictions:
            agreement += prefix(int(prediction_raw))
        total += seen * (end - start)
        for prediction_raw in group_predictions:
            add(int(prediction_raw))
        seen += end - start
        start = end
    return agreement, total - agreement, total


def _validate_input(
    points: Sequence[Sequence[float]] | np.ndarray,
    target_order: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    validated = _validate_ranking_input(points, target_order, None)
    return validated.points, validated.target_order


@dataclass(frozen=True)
class _ScoreParameterization:
    """Numerically conditioned score space and its ambient back-map."""

    score_basis: np.ndarray
    active_features: np.ndarray
    feature_shifts: np.ndarray
    right_basis: np.ndarray | None
    singular_values: np.ndarray | None
    dimension_bound: int
    projected_instance: bool
    ambient_dimension: int

    def to_ambient(self, reduced_direction: np.ndarray) -> np.ndarray:
        """Map a model direction to a safely normalized ambient direction."""

        if self.right_basis is None:
            feature_direction = np.asarray(reduced_direction, dtype=np.float64)
        else:
            assert self.singular_values is not None
            # V(q/s) is the exact SVD back-map.  Multiplying it by min(s) first
            # is a harmless common positive scaling that avoids overflow.
            scaled_coefficients = reduced_direction * (
                float(np.min(self.singular_values)) / self.singular_values
            )
            feature_direction = self.right_basis @ scaled_coefficients

        # score_basis[:, j] = (x[:, j] - anchor_j) * 2**shift_j, hence the
        # ambient coefficient is feature_direction[j] * 2**shift_j.  Apply the
        # shifts up to a common factor so intermediate values cannot overflow.
        nonzero = feature_direction != 0.0
        if not np.any(nonzero):
            raise RuntimeError("failed to map the model direction to input space")
        mantissas, exponents = np.frexp(feature_direction[nonzero])
        total_exponents = exponents.astype(np.int64) + self.feature_shifts[nonzero]
        common_exponent = int(np.max(total_exponents))
        mapped_active = np.zeros_like(feature_direction)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mapped_active[nonzero] = np.ldexp(
                mantissas, total_exponents - common_exponent
            )
        if np.any(mapped_active[nonzero] == 0.0):
            raise RuntimeError(
                "the model direction has too much dynamic range to represent "
                "as a float64 ambient direction"
            )

        ambient = np.zeros(self.ambient_dimension, dtype=np.float64)
        ambient[self.active_features] = mapped_active
        max_abs = float(np.max(np.abs(ambient)))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            raise RuntimeError("failed to map the model direction to input space")
        return ambient


def _exact_difference(values: np.ndarray, anchor: float) -> np.ndarray | None:
    """Return ``values - anchor`` only when every subtraction is exact.

    Knuth's error-free TwoSum decomposition exposes the rounding residual.  A
    zero residual means the computed float is exactly the real difference of
    the two input floats, which is what the original-instance certificate
    needs.  Returning ``None`` makes preprocessing try another translation.
    """

    if anchor == 0.0:
        return values.copy()
    negative_anchor = -anchor
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        difference = values - anchor
        virtual_anchor = difference - values
        residual = (values - (difference - virtual_anchor)) + (
            negative_anchor - virtual_anchor
        )
    if not np.isfinite(difference).all() or np.any(residual != 0.0):
        return None
    return difference


def _exactly_condition_column(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Choose an exact translation and reversible power-of-two scaling.

    Candidate data anchors improve offset-heavy columns without changing their
    exact geometry.  Among exact candidates, the one with the smallest binary
    dynamic range is selected.  The range guard keeps every nonzero model
    coefficient away from Gurobi's tiny-coefficient cutoff and from huge
    magnitudes; extremely ill-conditioned columns must be rescaled upstream.
    """

    candidate_anchors = (
        0.0,
        float(values[0]),
        float(np.min(values)),
        float(np.max(values)),
    )
    best: np.ndarray | None = None
    best_min_exponent = 0
    best_max_exponent = 0
    best_span: int | None = None
    seen: set[float] = set()
    unique_values = np.unique(values)
    with np.errstate(over="ignore", invalid="ignore"):
        adjacent_gaps = np.diff(unique_values)
    adjacent_gaps = adjacent_gaps[np.isfinite(adjacent_gaps) & (adjacent_gaps > 0.0)]
    if adjacent_gaps.size:
        _, gap_exponents = np.frexp(adjacent_gaps)
        gap_minimum = int(np.min(gap_exponents))
        gap_maximum = int(np.max(gap_exponents))
    else:
        gap_minimum = None
        gap_maximum = None

    for anchor in candidate_anchors:
        if anchor in seen:
            continue
        seen.add(anchor)
        translated = _exact_difference(values, anchor)
        if translated is None:
            continue
        magnitudes = np.abs(translated[translated != 0.0])
        if magnitudes.size == 0:
            continue
        _, exponents = np.frexp(magnitudes)
        minimum = int(np.min(exponents))
        maximum = int(np.max(exponents))
        if gap_minimum is not None and gap_maximum is not None:
            minimum = min(minimum, gap_minimum)
            maximum = max(maximum, gap_maximum)
        span = maximum - minimum
        if best_span is None or span < best_span:
            best = translated
            best_min_exponent = minimum
            best_max_exponent = maximum
            best_span = span

    if best is None or best_span is None:
        raise ValueError(
            "no exact, finite coordinate translation was available; rescale "
            "the input coordinates"
        )
    # Centering within roughly [-35, 35] keeps model coefficients above
    # Gurobi's tiny-coefficient cutoff and away from extreme magnitudes.  A
    # wider intrinsic range cannot be repaired by a common feature scaling.
    if best_span > 70:
        raise ValueError(
            "a feature has more than 70 bits of dynamic range after every "
            "exact data-anchor translation; rescale or re-express that feature"
        )

    shift = -((best_min_exponent + best_max_exponent) // 2)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        conditioned = np.ldexp(best, shift)
        restored = np.ldexp(conditioned, -shift)
    if (
        not np.isfinite(conditioned).all()
        or not np.array_equal(restored, best)
        or np.any((best != 0.0) & (conditioned == 0.0))
    ):
        raise ValueError(
            "power-of-two coordinate scaling was not exactly reversible; "
            "rescale the input coordinates"
        )
    return np.ascontiguousarray(conditioned), shift


def _score_parameterization(
    x: np.ndarray,
    target_order: np.ndarray,
    rank_tol: float | None,
    point_class_ids_target: np.ndarray | None = None,
) -> _ScoreParameterization:
    """Build a theorem-faithful ambient or optional projected score space.

    Each nonconstant feature is translated only when its floating-point
    subtraction is exact, then scaled by an exactly reversible power of two.
    This affine diagonal change of coordinates preserves *every* linear
    ranking of the supplied float values.  Constant features are removed
    because they cancel from all comparisons.

    With ``rank_tol is None`` (the default), no rank truncation occurs.  If a
    cutoff is supplied, an economy SVD optionally reduces the conditioned
    feature space; that mode is explicitly a projected-instance heuristic.
    """

    active_features = np.flatnonzero(np.any(x != x[:1], axis=0))
    if active_features.size == 0:
        raise ValueError("all feature columns are constant")

    base = np.empty((x.shape[0], active_features.size), dtype=np.float64)
    feature_shifts = np.empty(active_features.size, dtype=np.int64)
    for output_column, input_column in enumerate(active_features):
        ordered_column = np.ascontiguousarray(x[target_order, input_column])
        base[:, output_column], feature_shifts[output_column] = (
            _exactly_condition_column(ordered_column)
        )
    base = np.ascontiguousarray(base)
    if point_class_ids_target is None:
        class_ids = np.arange(base.shape[0], dtype=np.int64)
    else:
        class_ids = np.asarray(point_class_ids_target, dtype=np.int64)
        if class_ids.shape != (base.shape[0],):
            raise RuntimeError("point-class IDs have inconsistent dimensions")
    class_count = int(np.max(class_ids)) + 1
    if class_count <= 1:
        raise ValueError("at least two distinct point rows are required here")
    _, representative_positions = np.unique(class_ids, return_index=True)
    dimension_bound = min(base.shape[1], class_count - 1)

    if rank_tol is None:
        return _ScoreParameterization(
            score_basis=base,
            active_features=active_features,
            feature_shifts=feature_shifts,
            right_basis=None,
            singular_values=None,
            dimension_bound=dimension_bound,
            projected_instance=False,
            ambient_dimension=x.shape[1],
        )

    tolerance = float(rank_tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rank_tol must be a finite nonnegative number or None")
    _, singular_values, vt = np.linalg.svd(base, full_matrices=False)
    r = min(
        int(np.count_nonzero(singular_values > tolerance)),
        class_count - 1,
    )
    if r == 0:
        raise ValueError(
            "The points have numerical affine dimension zero at the selected "
            "rank tolerance; lower rank_tol."
        )
    # Computing U through B V / s makes exact duplicate B rows follow exactly
    # the same floating-point operations and therefore remain tied bit-for-bit.
    y = np.ascontiguousarray(
        (base @ np.ascontiguousarray(vt[:r].T)) / singular_values[:r]
    )
    if np.unique(y[representative_positions], axis=0).shape[0] != class_count:
        raise ValueError("SVD projection collapsed two distinct points; lower rank_tol")
    return _ScoreParameterization(
        score_basis=y,
        active_features=active_features,
        feature_shifts=feature_shifts,
        right_basis=np.ascontiguousarray(vt[:r].T),
        singular_values=np.ascontiguousarray(singular_values[:r]),
        dimension_bound=r,
        projected_instance=True,
        ambient_dimension=x.shape[1],
    )


def _pair_row_starts(n: int) -> np.ndarray:
    """Lexicographic pair-ID start for rows (i, i+1),...,(i,n-1)."""

    i = np.arange(n, dtype=np.int64)
    # starts[n-1] is M, serving as the search sentinel after the final
    # nonempty upper-triangle row i=n-2.
    return i * (2 * n - i - 1) // 2


def _decode_pair_ids(
    pair_ids: np.ndarray,
    n_or_index: int | _ComparablePairIndex,
) -> np.ndarray:
    """Decode strict legacy or generalized comparable-pair IDs."""

    if isinstance(n_or_index, _ComparablePairIndex):
        return n_or_index.decode(pair_ids)

    ids = np.asarray(pair_ids, dtype=np.int64)
    if ids.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    n = int(n_or_index)
    starts = _pair_row_starts(n)
    first = np.searchsorted(starts, ids, side="right") - 1
    second = first + 1 + (ids - starts[first])
    return np.column_stack((first, second)).astype(np.int64, copy=False)


def _original_pairs(
    pair_ids: np.ndarray,
    n_or_index: int | _ComparablePairIndex,
    target_order: np.ndarray,
) -> np.ndarray:
    target_pairs = _decode_pair_ids(pair_ids, n_or_index)
    if target_pairs.size == 0:
        return target_pairs
    return target_order[target_pairs]


def _set_model_parameters(
    model: Any,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> None:
    if not (1e-9 <= feasibility_tol <= 1e-2):
        raise ValueError("feasibility_tol must lie in Gurobi's [1e-9, 1e-2] range")
    if threads is not None and threads < 0:
        raise ValueError("threads must be nonnegative or None")

    model.setParam("OutputFlag", int(output_flag))
    model.setParam("FeasibilityTol", float(feasibility_tol))
    model.setParam("NumericFocus", 1)
    model.setParam("Method", 1)  # Dual simplex reuses bases across row/RHS changes.
    model.setParam("Seed", int(seed) % 2_000_000_000)
    model.setParam("IgnoreNames", 1)
    if threads is not None:
        model.setParam("Threads", int(threads))
    if gurobi_params:
        for name, value in gurobi_params.items():
            model.setParam(str(name), value)


def _optimize_for_status(model: Any) -> int:
    """Optimize and disambiguate the rare INF_OR_UNBD presolve status."""

    model.optimize()
    status = int(model.Status)
    if status == GRB.INF_OR_UNBD:
        model.setParam("DualReductions", 0)
        model.optimize()
        status = int(model.Status)
    return status


def _status_error(model: Any, context: str) -> RuntimeError:
    status = int(model.Status)
    status_name = "UNKNOWN"
    for candidate in (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    ):
        if getattr(GRB, candidate, None) == status:
            status_name = candidate
            break
    return RuntimeError(
        f"Gurobi could not certify {context}; status={status_name} ({status}). "
        "No approximation guarantee is claimed for an interrupted, limited, "
        "or numerically unresolved solve."
    )


def _minimal_comparison_iis(
    env: Any,
    candidate_ids: np.ndarray,
    score_basis: np.ndarray,
    pair_index: _ComparablePairIndex,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> np.ndarray:
    """Shrink an infeasible candidate to a comparison-minimal subsystem.

    The full model uses structural ``score = B q`` equations.  Gurobi's Force
    attributes are ignored for pure LPs, so an IIS of that extended model need
    not be minimal once every structural equation is restored.  A deletion
    filter on a smaller score-factorized model preserves the exact default
    coefficients, avoids a dense pair-difference matrix, and proves the
    inclusion-minimal comparison subsystem required by the theorem.
    """

    _, q_dimension = score_basis.shape
    pairs = _decode_pair_ids(candidate_ids, pair_index)
    involved, inverse = np.unique(pairs, return_inverse=True)
    local_pairs = inverse.reshape(-1, 2)

    with gp.Model("comparison_iis_refinement", env=env) as direct:
        _set_model_parameters(
            direct,
            output_flag=output_flag,
            threads=threads,
            feasibility_tol=feasibility_tol,
            seed=seed,
            gurobi_params=gurobi_params,
        )
        z = direct.addMVar(q_dimension, lb=-GRB.INFINITY)
        local_score = direct.addMVar(involved.size, lb=-GRB.INFINITY)
        direct.addConstr(local_score == np.ascontiguousarray(score_basis[involved]) @ z)
        comparison_constraints = [
            direct.addConstr(local_score[int(higher)] - local_score[int(lower)] >= 1.0)
            for higher, lower in local_pairs
        ]
        keep = np.ones(candidate_ids.size, dtype=bool)
        direct.update()
        status = _optimize_for_status(direct)
        if status != GRB.INFEASIBLE:
            if status == GRB.OPTIMAL:
                raise RuntimeError(
                    "the extended-model IIS did not remain infeasible after "
                    "restoring all structural score equations"
                )
            raise _status_error(direct, "comparison IIS refinement")

        for index, constraint in enumerate(comparison_constraints):
            # Disabling via RHS preserves the matrix structure and simplex
            # basis, which is materially faster than remove/re-add updates.
            constraint.RHS = -GRB.INFINITY
            direct.update()
            status = _optimize_for_status(direct)
            if status == GRB.INFEASIBLE:
                keep[index] = False
                continue
            if status != GRB.OPTIMAL:
                raise _status_error(direct, "comparison deletion filter")
            constraint.RHS = 1.0

        status = _optimize_for_status(direct)
        if status != GRB.INFEASIBLE:
            if status == GRB.OPTIMAL:
                raise RuntimeError("the comparison deletion filter lost infeasibility")
            raise _status_error(direct, "final comparison IIS verification")

    block = candidate_ids[keep].copy()
    if block.size == 0:
        raise RuntimeError("the comparison deletion filter returned no comparisons")
    point_class_count = int(pair_index.class_offsets.size - 1)
    comparison_dimension_bound = min(q_dimension, point_class_count - 1)
    if block.size > comparison_dimension_bound + 1:
        raise RuntimeError(
            f"The comparison IIS has size {block.size}, exceeding Helly's "
            f"dimension bound + 1={comparison_dimension_bound + 1}; numerical "
            "IIS certification failed."
        )
    return block


def _extract_and_remove_iis(
    model: Any,
    env: Any,
    active_pair_ids: np.ndarray,
    number_of_link_constraints: int,
    score_basis: np.ndarray,
    pair_index: _ComparablePairIndex,
    *,
    output_flag: bool,
    threads: int | None,
    feasibility_tol: float,
    seed: int,
    gurobi_params: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove one comparison-minimal IIS and return (remaining IDs, block IDs)."""

    try:
        model.computeIIS()
    except gp.GurobiError as exc:
        numeric_error = getattr(GRB, "ERROR_NUMERIC", 10014)
        if exc.errno != numeric_error:
            raise
        # Gurobi's automatic IIS strategy can fail on a nearly dependent
        # score-link basis even when the preceding infeasibility solve is
        # certified.  Its alternative method 1 uses a different IIS path and
        # is a reliable, still-certified fallback for that case.  Keep it for
        # later iterations after the first automatic-method failure.
        model.setParam("IISMethod", 1)
        model.computeIIS()
    outer_minimal = int(model.IISMinimal)
    if outer_minimal != 1:
        raise RuntimeError(
            "Gurobi's extended-model IIS computation stopped early. A complete "
            "IIS is required before comparison-only refinement."
        )
    constraints = model.getConstrs()
    comparison_constraints = constraints[number_of_link_constraints:]
    if len(comparison_constraints) != active_pair_ids.size:
        raise RuntimeError("internal comparison-to-constraint mapping was lost")

    flags = np.asarray(
        model.getAttr(GRB.Attr.IISConstr, comparison_constraints), dtype=bool
    )
    selected = np.flatnonzero(flags)
    if selected.size == 0:
        raise RuntimeError(
            "Gurobi reported an IIS containing no deletable comparison; the "
            "always-feasible structural score links appear numerically inconsistent."
        )
    candidate_ids = active_pair_ids[selected]
    block = _minimal_comparison_iis(
        env,
        candidate_ids,
        score_basis,
        pair_index,
        output_flag=output_flag,
        threads=threads,
        feasibility_tol=feasibility_tol,
        seed=seed,
        gurobi_params=gurobi_params,
    )

    # Map the refined IDs back to current scalar constraint positions.
    selected_by_id = {
        int(pair_id): int(position)
        for position, pair_id in zip(selected, candidate_ids)
    }
    refined_positions = np.fromiter(
        (selected_by_id[int(pair_id)] for pair_id in block),
        dtype=np.int64,
        count=block.size,
    )
    model.remove([comparison_constraints[int(k)] for k in refined_positions])
    model.update()
    remaining = np.delete(active_pair_ids, refined_positions)
    return remaining, block


def _add_target_adjacent_comparisons(model: Any, score: Any) -> Any:
    """Add target-adjacent comparisons, in target-position order."""

    if int(score.shape[0]) <= 1:
        return None
    return model.addConstr(score[:-1] - score[1:] >= 1.0)


def _add_nonadjacent_comparisons(
    model: Any, score: Any, starts: np.ndarray
) -> np.ndarray:
    """Add all j >= i+2 pair rows and return their IDs in insertion order."""

    n = int(score.shape[0])
    count = (n - 1) * (n - 2) // 2
    ids = np.empty(count, dtype=np.int64)
    cursor = 0
    for i in range(n - 2):
        row_count = n - i - 2
        model.addConstr(score[i] - score[i + 2 :] >= 1.0)
        ids[cursor : cursor + row_count] = (
            starts[i] + 1 + np.arange(row_count, dtype=np.int64)
        )
        cursor += row_count
    if cursor != count:
        raise RuntimeError("internal pair construction count mismatch")
    model.update()
    return ids


def _add_all_indexed_comparisons(
    model: Any,
    score: Any,
    pair_index: _ComparablePairIndex,
) -> np.ndarray:
    """Add every modeled weak-target comparison in pair-ID order."""

    for first in range(pair_index.n):
        begin = int(pair_index.row_starts[first])
        end = int(pair_index.row_starts[first + 1])
        if begin == end:
            continue
        ids = np.arange(begin, end, dtype=np.int64)
        pairs = pair_index.decode(ids)
        if np.any(pairs[:, 0] != first):
            raise RuntimeError("indexed comparison row lost its first endpoint")
        model.addConstr(score[first] - score[pairs[:, 1]] >= 1.0)
    model.update()
    return np.arange(pair_index.modeled_comparisons, dtype=np.int64)


def _generic_refinement_order(
    weak_direction: np.ndarray,
    score_basis: np.ndarray,
    seed: int,
    max_attempts: int = 128,
) -> np.ndarray:
    """Return a linearly realizable strict refinement of a weak score order.

    A bounded generic perturbation is applied to the *direction itself*, not
    merely used as a sorting key.  Every retained comparison has weak margin
    at least one; bounding the perturbation's complete score range by 1/4
    therefore preserves all retained rows.  Explicitly scoring the perturbed
    direction avoids treating a rounding-scale near-tie as a trustworthy
    strict primary key.
    """

    n, r = score_basis.shape
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    base = np.asarray(weak_direction, dtype=np.float64).reshape(-1)
    if base.shape != (r,) or not np.isfinite(base).all():
        raise RuntimeError("the weak ranking direction is invalid")

    rng = np.random.default_rng(seed)
    best_order: np.ndarray | None = None
    best_margin = -math.inf
    for attempt in range(max_attempts):
        direction = rng.standard_normal(r)
        scale = float(np.max(np.abs(direction), initial=0.0))
        if not np.isfinite(scale) or scale == 0.0:
            continue
        direction /= scale
        secondary = score_basis @ direction
        score_range = float(np.ptp(secondary))
        if not math.isfinite(score_range) or score_range == 0.0:
            continue

        candidate = base + (0.25 / score_range) * direction
        candidate_scale = float(np.max(np.abs(candidate), initial=0.0))
        if not math.isfinite(candidate_scale) or candidate_scale == 0.0:
            continue
        candidate /= candidate_scale
        scores = score_basis @ candidate
        if not np.isfinite(scores).all():
            continue
        order = np.argsort(-scores, kind="stable").astype(np.int64, copy=False)
        gaps = scores[order[:-1]] - scores[order[1:]]
        direct_gaps = (score_basis[order[:-1]] - score_basis[order[1:]]) @ candidate
        if (
            np.any(~np.isfinite(gaps))
            or np.any(~np.isfinite(direct_gaps))
            or np.any(gaps <= 0.0)
            or np.any(direct_gaps <= 0.0)
        ):
            continue
        margin = min(float(np.min(gaps)), float(np.min(direct_gaps)))
        if margin > best_margin:
            best_order = order
            best_margin = margin
        if attempt >= 15 and best_order is not None:
            break

    if best_order is None:
        raise RuntimeError("failed to construct a generic tie-breaking direction")
    return best_order


def _replace_with_strict_order_model(
    model: Any,
    q: Any,
    score: Any,
    rank_in_target_positions: np.ndarray,
    number_of_link_constraints: int,
) -> np.ndarray:
    """Certify a proposed strict order and return its q witness."""

    constraints = model.getConstrs()
    if len(constraints) > number_of_link_constraints:
        model.remove(constraints[number_of_link_constraints:])
        model.update()

    if rank_in_target_positions.size > 1:
        higher = rank_in_target_positions[:-1]
        lower = rank_in_target_positions[1:]
        model.addConstr(score[higher] - score[lower] >= 1.0)
        model.update()

    # A zero-objective LP is enough: adjacent margin-one rows imply the entire
    # proposed strict order.  Keeping this an LP avoids an unnecessary dense
    # norm QP and lets Gurobi reuse the persistent feasibility model.
    model.setObjective(0.0, GRB.MINIMIZE)
    status = _optimize_for_status(model)
    if status != GRB.OPTIMAL:
        raise _status_error(model, "the final generic strict ranking")
    return np.asarray(q.X, dtype=np.float64).reshape(-1)


def _count_inversions(permutation: np.ndarray) -> int:
    """Count inversions of a 0,...,n-1 permutation in O(n log n)."""

    n = int(permutation.size)
    tree = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for value_raw in permutation[::-1]:
        value = int(value_raw)
        k = value
        subtotal = 0
        while k > 0:
            subtotal += int(tree[k])
            k -= k & -k
        total += subtotal
        k = value + 1
        while k <= n:
            tree[k] += 1
            k += k & -k
    return total


def _validate_certificate(
    prediction_tie_ids_target: np.ndarray,
    active_pair_ids: np.ndarray,
    modeled_deleted_count: int,
    final_scores_target_order: np.ndarray,
    ranking_input: _RankingInput,
) -> int:
    pair_index = ranking_input.pair_index
    representatives = ranking_input.representative_target_positions
    class_scores = final_scores_target_order[representatives]
    if np.unique(class_scores).size != ranking_input.point_class_count:
        raise RuntimeError("distinct point classes have a numerical score tie")
    expanded_scores = class_scores[ranking_input.point_class_ids_order]
    if not np.array_equal(expanded_scores, final_scores_target_order):
        raise RuntimeError("score ties do not match the exact duplicate-point classes")

    class_order = np.argsort(-class_scores, kind="stable")
    _, expected_prediction_ids, expected_prediction_order = (
        _prediction_from_class_order(ranking_input, class_order)
    )
    if not np.array_equal(expected_prediction_ids, prediction_tie_ids_target):
        raise RuntimeError("the returned direction does not induce the certified ties")

    # Decode in bounded chunks so verification does not create another full
    # O(n^2)-by-2 pair array at peak memory.
    for begin in range(0, active_pair_ids.size, 1_000_000):
        active_pairs = pair_index.decode(active_pair_ids[begin : begin + 1_000_000])
        if np.any(
            expected_prediction_order[active_pairs[:, 0]]
            >= expected_prediction_order[active_pairs[:, 1]]
        ):
            raise RuntimeError("the final ranking reverses an undeleted comparison")

    agreement, distance, total = _weak_order_counts(
        ranking_input.target_tie_ids_order, expected_prediction_order
    )
    del agreement
    total_deleted = pair_index.forced_tie_disagreements + modeled_deleted_count
    if total != pair_index.target_comparisons:
        raise RuntimeError("weak-ranking comparison count changed during validation")
    if distance > total_deleted:
        raise RuntimeError(
            f"certificate failure: disagreement {distance} exceeds the "
            f"{total_deleted} deleted/forced comparisons"
        )
    return distance


def _verify_original_ranking(
    x: np.ndarray, ranking: np.ndarray, direction: np.ndarray
) -> None:
    """Verify exact strict adjacent gaps in bounded memory.

    A fast forward-error bound certifies ordinary rows.  Only cancellation-heavy
    rows fall back to exact rational arithmetic over the supplied float64
    values, so a large common score offset cannot hide or invent a strict gap.
    """

    if ranking.size <= 1:
        return
    bytes_per_row = max(1, 3 * x.shape[1] * np.dtype(np.float64).itemsize)
    chunk_rows = max(1, min(ranking.size - 1, (64 * 1024**2) // bytes_per_row))
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    operations = 2 * x.shape[1] + 3
    denominator = 1.0 - operations * unit_roundoff
    gamma = operations * unit_roundoff / denominator if denominator > 0.0 else math.inf
    underflow_allowance = operations * np.nextafter(0.0, 1.0)
    absolute_direction = np.abs(direction)
    direction_fractions: list[Fraction | None] | None = None

    for begin in range(0, ranking.size - 1, chunk_rows):
        end = min(begin + chunk_rows, ranking.size - 1)
        higher = x[ranking[begin:end]]
        lower = x[ranking[begin + 1 : end + 1]]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            approximate = higher @ direction - lower @ direction
            absolute_sum = (
                np.abs(higher) @ absolute_direction + np.abs(lower) @ absolute_direction
            )
            if denominator > 0.0:
                magnitude_upper = (
                    np.nextafter(absolute_sum, math.inf) + underflow_allowance
                ) / denominator
                error_bound = np.nextafter(
                    gamma * magnitude_upper + underflow_allowance,
                    math.inf,
                )
            else:
                error_bound = np.full_like(absolute_sum, math.inf)
        certified = (
            np.isfinite(approximate)
            & np.isfinite(error_bound)
            & (approximate > error_bound)
        )
        ambiguous = np.flatnonzero(~certified)
        if ambiguous.size == 0:
            continue
        if direction_fractions is None:
            direction_fractions = [
                None if value == 0.0 else Fraction.from_float(float(value))
                for value in direction
            ]
        for local_index in ambiguous:
            exact_gap = Fraction(0)
            for high_value, low_value, weight in zip(
                higher[local_index], lower[local_index], direction_fractions
            ):
                if weight is not None and high_value != low_value:
                    exact_gap += (
                        Fraction.from_float(float(high_value))
                        - Fraction.from_float(float(low_value))
                    ) * weight
            if exact_gap <= 0:
                raise RuntimeError(
                    "the mapped direction does not strictly realize the ranking "
                    "on the supplied coordinates"
                )


def closest_linear_ranking(
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
) -> ClosestLinearRankingResult:
    """Run the optimized ``thm:main-kt`` approximation algorithm.

    Parameters
    ----------
    points:
        Finite ``(n, d)`` point matrix whose coordinates are losslessly
        representable as float64.  Exact duplicate rows are allowed and must
        receive the same output score.
    target_order:
        A zero-based permutation of original row indices, highest first.
        ``None`` means ``[0, 1, ..., n-1]``.
    target_ranks:
        Optional numeric rank per original row.  Smaller values rank higher and
        equal values form a target tie.  Mutually exclusive with
        ``target_order``.
    rank_tol:
        ``None`` (default) keeps the full, exactly affine-conditioned ambient
        comparison space and certifies the supplied points.  A nonnegative
        value requests optional SVD projection at that absolute cutoff.  The
        latter can be faster, but its guarantee is for the projected instance.
    seed:
        Seed used by Gurobi and by the generic tie-refinement step.
    output_flag:
        Whether to display Gurobi's solver log.
    threads:
        Gurobi worker count; ``None`` leaves its automatic default.
    feasibility_tol:
        Gurobi LP feasibility tolerance.
    gurobi_params:
        Optional additional model parameters.  A time/resource limit that
        interrupts optimization or IIS extraction causes an exception rather
        than an uncertified approximation result.

    Returns
    -------
    ClosestLinearRankingResult
        The weak linear ranking, its direction, the deleted IIS certificate,
        tie metadata, and theorem diagnostics.  Distinct point classes are
        strictly ordered whenever the target has a strict comparison; an
        entirely tied target returns the zero direction and one output tie.
        Exact duplicate rows remain tied.  If ``K`` is the returned
        ``approximation_factor``, the removed blocks certify

        ``kendall_tau <= deleted_count <= K * OPT_KT <= (r+1) * OPT_KT``.

        The packing argument does not require ``OPT_KT`` to be known.  This is a
        certificate for the original input in the default ``rank_tol=None``
        mode, and for the projected instance when an SVD cutoff is supplied.

    Notes
    -----
    The paper assumes exact rational LP feasibility.  Gurobi and the optional
    SVD use floating-point arithmetic, so this implementation checks all
    available numerical certificates and raises rather than claiming a
    guarantee after an unresolved solver status.  Default preprocessing itself
    is restricted to exact affine transformations of the supplied floats.
    """

    ranking_input = _validate_ranking_input(points, target_order, target_ranks)
    x = ranking_input.points
    order = ranking_input.target_order
    n, d = x.shape
    pair_index = ranking_input.pair_index

    if (
        ranking_input.point_class_count == 1
        or pair_index.target_comparisons == 0
    ):
        flattened, prediction_ids, prediction_order = _all_tied_prediction(
            ranking_input
        )
        _, distance, total_comparisons = _weak_order_counts(
            ranking_input.target_tie_ids_order, prediction_order
        )
        return ClosestLinearRankingResult(
            ranking=order[flattened],
            direction=np.zeros(d, dtype=np.float64),
            conditioned_scores=np.zeros(n, dtype=np.float64),
            target_order=order.copy(),
            target_tie_ids=ranking_input.target_tie_ids.copy(),
            ranking_tie_ids=prediction_ids,
            deleted_comparisons=np.empty((0, 2), dtype=np.int64),
            iis_blocks=(),
            kendall_tau=distance,
            total_comparisons=total_comparisons,
            forced_tie_disagreements=pair_index.forced_tie_disagreements,
            optimizable_comparisons=0,
            dimension_bound=0,
            approximation_factor=1,
            projected_instance=False,
            solver_runtime_seconds=0.0,
        )

    _require_gurobi()
    normalized_seed = int(seed) % 2_000_000_000

    parameterization = _score_parameterization(
        x, order, rank_tol, ranking_input.point_class_ids_order
    )
    y = parameterization.score_basis
    class_y = y[ranking_input.representative_target_positions]
    q_dimension = int(y.shape[1])
    r = parameterization.dimension_bound
    starts = _pair_row_starts(n)
    adjacent_ids = starts[:-1].copy()
    active_ids = np.empty(0, dtype=np.int64)
    deleted_blocks: list[np.ndarray] = []
    started = time.perf_counter()

    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", int(output_flag))
            env.start()
            with gp.Model("closest_linear_ranking", env=env) as model:
                _set_model_parameters(
                    model,
                    output_flag=output_flag,
                    threads=threads,
                    feasibility_tol=feasibility_tol,
                    seed=normalized_seed,
                    gurobi_params=gurobi_params,
                )

                q = model.addMVar(q_dimension, lb=-GRB.INFINITY, name="q")
                score = model.addMVar(n, lb=-GRB.INFINITY, name="score")
                model.addConstr(score == y @ q)
                number_of_links = n
                legacy_strict = (
                    ranking_input.target_tie_count == n
                    and ranking_input.point_class_count == n
                )
                if legacy_strict:
                    _add_target_adjacent_comparisons(model, score)
                    active_ids = adjacent_ids
                else:
                    active_ids = _add_all_indexed_comparisons(model, score, pair_index)
                model.update()

                status = _optimize_for_status(model)
                if legacy_strict and status == GRB.INFEASIBLE:
                    # This IIS is found using only adjacent target comparisons,
                    # but it is also a valid minimal infeasible subset of the
                    # theorem's full comparison set.  Removing it before adding
                    # the remaining pairs saves work without changing the proof.
                    active_ids, block = _extract_and_remove_iis(
                        model,
                        env,
                        active_ids,
                        number_of_links,
                        y,
                        pair_index,
                        output_flag=output_flag,
                        threads=threads,
                        feasibility_tol=feasibility_tol,
                        seed=normalized_seed,
                        gurobi_params=gurobi_params,
                    )
                    deleted_blocks.append(block)

                    nonadjacent_ids = _add_nonadjacent_comparisons(model, score, starts)
                    active_ids = np.concatenate((active_ids, nonadjacent_ids))
                    del nonadjacent_ids

                    while True:
                        status = _optimize_for_status(model)
                        if status == GRB.OPTIMAL:
                            break
                        if status != GRB.INFEASIBLE:
                            raise _status_error(model, "comparison feasibility")
                        active_ids, block = _extract_and_remove_iis(
                            model,
                            env,
                            active_ids,
                            number_of_links,
                            y,
                            pair_index,
                            output_flag=output_flag,
                            threads=threads,
                            feasibility_tol=feasibility_tol,
                            seed=normalized_seed,
                            gurobi_params=gurobi_params,
                        )
                        deleted_blocks.append(block)
                elif legacy_strict:
                    if status != GRB.OPTIMAL:
                        raise _status_error(model, "target-order feasibility")
                else:
                    while status == GRB.INFEASIBLE:
                        active_ids, block = _extract_and_remove_iis(
                            model,
                            env,
                            active_ids,
                            number_of_links,
                            y,
                            pair_index,
                            output_flag=output_flag,
                            threads=threads,
                            feasibility_tol=feasibility_tol,
                            seed=normalized_seed,
                            gurobi_params=gurobi_params,
                        )
                        deleted_blocks.append(block)
                        status = _optimize_for_status(model)
                    if status != GRB.OPTIMAL:
                        raise _status_error(model, "weak-target feasibility")

                weak_q = np.asarray(q.X, dtype=np.float64).reshape(-1)
                class_order = _generic_refinement_order(
                    weak_q, class_y, normalized_seed
                )
                ordered_representatives = ranking_input.representative_target_positions[
                    class_order
                ]

                final_q = _replace_with_strict_order_model(
                    model,
                    q,
                    score,
                    ordered_representatives,
                    number_of_links,
                )

                ambient = parameterization.to_ambient(final_q)
                # Model scores are stable under the exact affine conditioning.
                mapped_scores_target = y @ final_q
                mapped_class_scores = mapped_scores_target[
                    ranking_input.representative_target_positions
                ]
                mapped_class_order = np.argsort(-mapped_class_scores, kind="stable")
                expanded_scores = mapped_class_scores[
                    ranking_input.point_class_ids_order
                ]
                if (
                    np.unique(mapped_class_scores).size
                    != ranking_input.point_class_count
                    or not np.array_equal(mapped_class_order, class_order)
                    or not np.array_equal(expanded_scores, mapped_scores_target)
                ):
                    raise RuntimeError(
                        "mapping the reduced witness back to the original feature "
                        "space changed the certified order; adjust rank_tol/scaling"
                    )

                # Fail closed against the supplied coordinates as well.  The
                # verifier uses bounded chunks and exact fallback arithmetic.
                flattened, prediction_ids, _ = _prediction_from_class_order(
                    ranking_input, class_order
                )
                ranking_original = order[flattened]
                _verify_original_ranking(x, order[ordered_representatives], ambient)

                all_deleted_ids = (
                    np.concatenate(deleted_blocks)
                    if deleted_blocks
                    else np.empty(0, dtype=np.int64)
                )
                distance = _validate_certificate(
                    prediction_ids,
                    active_ids,
                    int(all_deleted_ids.size),
                    mapped_scores_target,
                    ranking_input,
                )
                comparison_partition_complete = (
                    active_ids.size + all_deleted_ids.size
                    == pair_index.modeled_comparisons
                )
                implicit_target_certificate = (
                    legacy_strict and not deleted_blocks and distance == 0
                )
                if (
                    not comparison_partition_complete
                    and not implicit_target_certificate
                ):
                    raise RuntimeError(
                        "active/deleted comparison partition is incomplete"
                    )

    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi error {exc.errno}: {exc}") from exc

    elapsed = time.perf_counter() - started
    ranking = ranking_original
    conditioned_scores = np.empty(n, dtype=np.float64)
    conditioned_scores[order] = mapped_scores_target
    decoded_blocks = tuple(
        _original_pairs(block, pair_index, order) for block in deleted_blocks
    )
    deleted_pairs = _original_pairs(all_deleted_ids, pair_index, order)
    observed_factor = max(
        (int(block.size) for block in deleted_blocks),
        default=1,
    )

    return ClosestLinearRankingResult(
        ranking=ranking.astype(np.int64, copy=False),
        direction=ambient.astype(np.float64, copy=False),
        conditioned_scores=conditioned_scores,
        target_order=order.copy(),
        target_tie_ids=ranking_input.target_tie_ids.copy(),
        ranking_tie_ids=prediction_ids,
        deleted_comparisons=deleted_pairs,
        iis_blocks=decoded_blocks,
        kendall_tau=distance,
        total_comparisons=pair_index.target_comparisons,
        forced_tie_disagreements=pair_index.forced_tie_disagreements,
        optimizable_comparisons=pair_index.modeled_comparisons,
        dimension_bound=r,
        approximation_factor=observed_factor,
        projected_instance=parameterization.projected_instance,
        solver_runtime_seconds=elapsed,
    )


def _parse_target_order(text: str | None) -> list[int] | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return [int(token.strip()) for token in stripped.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target order must be comma-separated integer row indices"
        ) from exc


def _parse_target_ranks(text: str | None) -> list[float] | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return []
    try:
        values = [float(token.strip()) for token in stripped.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target ranks must be comma-separated finite numbers"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("target ranks must be finite")
    return values


def _load_points(path: Path, delimiter: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if delimiter != "whitespace" and len(delimiter) != 1:
        raise ValueError("delimiter must be one character or 'whitespace'")
    actual_delimiter = None if delimiter == "whitespace" else delimiter
    return np.loadtxt(path, delimiter=actual_delimiter, ndmin=2)


def _load_target_ranks(path: Path, delimiter: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    else:
        if delimiter != "whitespace" and len(delimiter) != 1:
            raise ValueError("delimiter must be one character or 'whitespace'")
        actual_delimiter = None if delimiter == "whitespace" else delimiter
        values = np.loadtxt(path, delimiter=actual_delimiter)
    array = np.asarray(values)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError("target-ranks file must contain one row or one column")
    return array


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the optimized Gurobi implementation of theorem thm:main-kt "
            "on a CSV/text/NPY point matrix."
        )
    )
    parser.add_argument(
        "points", type=Path, help="CSV, whitespace text, or .npy matrix"
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-order",
        type=_parse_target_order,
        metavar="I,J,...",
        help="zero-based row permutation, highest first (default: file row order)",
    )
    target_group.add_argument(
        "--target-ranks",
        type=_parse_target_ranks,
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
            "optional SVD cutoff after column scaling (default: no projection; "
            "any supplied value models a projected instance)"
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
    args = parser.parse_args()

    try:
        point_matrix = _load_points(args.points, args.delimiter)
        target_ranks = args.target_ranks
        if args.target_ranks_file is not None:
            target_ranks = _load_target_ranks(args.target_ranks_file, args.delimiter)
        result = closest_linear_ranking(
            point_matrix,
            args.target_order,
            target_ranks=target_ranks,
            rank_tol=args.rank_tol,
            seed=args.seed,
            output_flag=args.solver_log,
            threads=args.threads,
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
