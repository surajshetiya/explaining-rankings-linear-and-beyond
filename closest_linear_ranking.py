#!/usr/bin/env python3
"""License-free IIS-removal approximation for closest linear ranking.

Port of the comparison-IIS removal idea in closest_linear_ranking_gurobi.py.
Uses NumPy and SciPy/HiGHS, not gurobipy. It produces an approximation to
MINIMUM KENDALL DISAGREEMENT for UNRESTRICTED SIGNED linear directions,
not Equation (6)'s nonnegative-weight ILP optimum.

For the seed-19 dataset the headerless CSV rows are the strict target ranking.

Usage:
    python closest_linear_ranking_license_free.py dataset_100.csv --output result.json

The key replacement for Gurobi.computeIIS() uses Farkas' alternative:
    D q >= 1 is infeasible iff there is y >= 0, sum(y)=1, D.T @ y = 0.
A basic feasible y has at most r+1 nonzero entries (in exact arithmetic),
which identify a small infeasible comparison subsystem (IIS). Removing these
subsystems until the rest is feasible is the approximation algorithm.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import linprog


@dataclass
class ClosestLinearRankingResult:
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
    def deleted_count(self):
        return self.forced_tie_disagreements + len(self.deleted_comparisons)

    @property
    def modeled_deleted_count(self):
        return len(self.deleted_comparisons)

    @property
    def iterations(self):
        return len(self.iis_blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            'ranking': self.ranking.tolist(),
            'direction': self.direction.tolist(),
            'conditioned_scores': self.conditioned_scores.tolist(),
            'target_order': self.target_order.tolist(),
            'target_tie_ids': self.target_tie_ids.tolist(),
            'ranking_tie_ids': self.ranking_tie_ids.tolist(),
            'deleted_comparisons': self.deleted_comparisons.tolist(),
            'iis_blocks': [b.tolist() for b in self.iis_blocks],
            'deleted_count': self.deleted_count,
            'modeled_deleted_count': self.modeled_deleted_count,
            'iterations': self.iterations,
            'kendall_tau': self.kendall_tau,
            'total_comparisons': self.total_comparisons,
            'forced_tie_disagreements': self.forced_tie_disagreements,
            'optimizable_comparisons': self.optimizable_comparisons,
            'dimension_bound': self.dimension_bound,
            'approximation_factor': self.approximation_factor,
            'projected_instance': self.projected_instance,
            'solver_runtime_seconds': self.solver_runtime_seconds,
            'guarantee': 'kendall_tau <= deleted_count <= approximation_factor * OPT_KT (modulo floating-point feasibility)',
            'solver': 'SciPy HiGHS / Farkas basic solution IIS, no Gurobi',
            'objective_note': 'IIS-removal approximation for signed linear rankings; not Equation (6) exact OPT',
        }


def _feasible_direction(differences: np.ndarray, r: int):
    """Obtain q with D q >= 1 or return None when infeasible."""
    if differences.shape[0] == 0:
        return np.zeros(r, dtype=float)
    res = linprog(np.zeros(r), A_ub=-differences,
                  b_ub=-np.ones(len(differences)),
                  bounds=[(None, None)] * r, method='highs')
    if res.status == 0:
        if np.min(differences @ res.x) < 1 - 1e-5:
            raise RuntimeError('LP primal solution did not pass margin verification')
        return res.x
    if res.status == 2:
        return None
    raise RuntimeError(f'HiGHS feasibility failure: {res.message}')


def _find_minimal_iis(differences: np.ndarray) -> np.ndarray | None:
    """Find <= r+1 comparison IIS indices using alternative polytope.

    Returns None precisely when the comparisons are feasible. Uses at most
    r+1 extra LPs to remove redundant comparisons from a candidate block.
    """
    m, r = differences.shape
    if not m:
        return None
    # Maximise feasibility certificate via y>=0, D.T*y=0, sum(y)=1.
    alternative = linprog(
        np.zeros(m),
        A_eq=np.vstack((differences.T, np.ones(m))),
        b_eq=np.r_[np.zeros(r), 1.0],
        bounds=(0, None),
        method='highs-ds',
    )
    if alternative.status == 2:
        return None
    if alternative.status != 0:
        raise RuntimeError(f'HiGHS alternative-system failure: {alternative.message}')
    y = alternative.x
    chosen = np.flatnonzero(y > 1e-9)
    if chosen.size == 0 or chosen.size > r + 1:
        raise RuntimeError('Farkas certificate has an invalid/basic-support size')
    if np.max(np.abs(differences[chosen].T @ y[chosen])) > 1e-5:
        raise RuntimeError('Farkas certificate residual too large')
    if _feasible_direction(differences[chosen], r) is not None:
        raise RuntimeError('Numerically ambiguous Farkas certificate (not infeasible)')
    # Deletion filtering: certify comparison-irreducibility.
    chosen = chosen.tolist()
    k = 0
    while k < len(chosen):
        reduced = chosen[:k] + chosen[k+1:]
        if _feasible_direction(differences[reduced], r) is None:
            chosen = reduced
        else:
            k += 1
    if not chosen or len(chosen) > r + 1:
        raise RuntimeError('IIS minimality certificate failed')
    return np.asarray(chosen, dtype=np.int64)


def closest_linear_ranking(points, target_order=None, *, target_ranks=None,
                           rank_tol=None, seed=0, output_flag=False,
                           threads=None, feasibility_tol=1e-8,
                           gurobi_params=None):
    """Approximate minimum weak Kendall disagreements with signed directions.

    target_order: permutation of original row IDs; default input row order.
    target_ranks: alternatively, numeric per-row ranks allowing target ties.
    rank_tol: optional SVD cutoff (projects instance, as in the Gurobi code).
    threads: accepted for compatibility; HiGHS chooses its own workers.
    """
    if gurobi_params:
        raise ValueError('gurobi_params cannot be applied by the HiGHS solver')
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('points must be a nonempty finite (n,d) numeric array')
    n, d = x.shape
    if target_order is not None and target_ranks is not None:
        raise ValueError('target_order and target_ranks are mutually exclusive')
    if target_ranks is not None:
        ranks = np.asarray(target_ranks, dtype=float)
        if ranks.shape != (n,) or not np.isfinite(ranks).all():
            raise ValueError('target_ranks must be n finite numbers')
        order = np.argsort(ranks, kind='stable')
        ordered_ranks = ranks[order]
        tie_order = np.cumsum(np.r_[True, ordered_ranks[1:] != ordered_ranks[:-1]]) - 1
    else:
        order = (np.arange(n, dtype=int) if target_order is None
                 else np.asarray(target_order, dtype=int))
        if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
            raise ValueError('target_order must be a permutation of 0,...,n-1')
        tie_order = np.arange(n, dtype=int)
    target_ties = np.empty(n, dtype=int)
    target_ties[order] = tie_order
    start_time = time.perf_counter()
    active_cols = np.flatnonzero(np.any(x != x[0], axis=0))
    if len(active_cols):
        # Invertible translation/positive rescaling preserve the comparison cone.
        anchor = x[0, active_cols]
        scales = np.max(np.abs(x[:, active_cols] - anchor), axis=0)
        b = (x[:, active_cols] - anchor) / scales
        if rank_tol is not None:
            if rank_tol < 0 or not np.isfinite(rank_tol):
                raise ValueError('rank_tol must be nonnegative')
            u, s, vt = np.linalg.svd(b, full_matrices=False)
            k = np.count_nonzero(s > rank_tol)
            if k == 0:
                raise ValueError('rank_tol projects all dimensions away')
            mapping = vt[:k].T / s[:k]  # b @ mapping = first k columns of U
            basis = b @ mapping
            projected = True
        else:
            basis = b
            mapping = np.eye(len(active_cols))
            projected = False
    else:
        basis = np.zeros((n, 0))
        mapping = np.empty((0, 0))
        scales = np.empty(0)
        projected = False

    class_ids = np.unique(x, axis=0, return_inverse=True)[1]
    class_count = int(np.max(class_ids)) + 1
    ordered = order
    ii, jj = np.triu_indices(n, k=1)
    # Only strict target comparisons; pairs are in original row indices.
    higher, lower = ordered[ii], ordered[jj]
    target_mask = tie_order[ii] < tie_order[jj]
    higher, lower = higher[target_mask], lower[target_mask]
    total = len(higher)
    distinct = class_ids[higher] != class_ids[lower]
    forced = int(np.count_nonzero(~distinct))
    higher, lower = higher[distinct], lower[distinct]
    pairs = np.column_stack((higher, lower))
    differences = basis[higher] - basis[lower]
    r = basis.shape[1]
    dimension_bound = min(r, class_count - 1)

    active_ids = np.arange(len(pairs), dtype=int)
    deleted_blocks = []
    if r:
        while True:
            current = differences[active_ids]
            iis_local = _find_minimal_iis(current)
            if iis_local is None:
                q = _feasible_direction(current, r)
                if q is None:
                    raise RuntimeError('Alternative/primal certificates disagree')
                break
            block = active_ids[iis_local]
            deleted_blocks.append(block)
            keep = np.ones(len(active_ids), dtype=bool)
            keep[iis_local] = False
            active_ids = active_ids[keep]
            if output_flag and (len(deleted_blocks) <= 5 or len(deleted_blocks) % 25 == 0):
                print(f'IIS block {len(deleted_blocks)}, size {len(block)}, '
                      f'remaining pairs {len(active_ids)}', file=sys.stderr, flush=True)
    else:
        q = np.zeros(0, dtype=float)
        if len(active_ids):
            raise RuntimeError('Distinct rows cannot have zero-dimensional comparison space')

    # Refine weak output direction to one with generic strict scores for all
    # distinct point classes, without changing any retained comparison.
    if r and len(pairs):
        rng = np.random.default_rng(seed)
        base = np.asarray(q, dtype=float)
        if not np.isfinite(base).all():
            raise RuntimeError('Infeasible/nonfinite direction')
        for attempt in range(256):
            perturb = rng.normal(size=r)
            perturbed_scores = basis @ perturb
            sr = np.ptp(perturbed_scores)
            if sr == 0 or not np.isfinite(sr):
                continue
            candidate = base + (0.2 / sr) * perturb
            sc = basis @ candidate
            distinct_class_scores = np.asarray([sc[np.flatnonzero(class_ids == c)[0]]
                                                for c in range(class_count)])
            if (len(np.unique(distinct_class_scores)) == class_count
                and (not len(active_ids) or np.min(differences[active_ids] @ candidate) > 0)):
                q = candidate
                break
        else:
            raise RuntimeError('Could not certify a strict refinement')

    ambient = np.zeros(d, dtype=float)
    if r:
        ambient[active_cols] = (mapping @ q) / scales
        norm = np.max(np.abs(ambient))
        if norm == 0 or not np.isfinite(norm):
            raise RuntimeError('Invalid ambient direction')
        ambient /= norm
    scores = x @ ambient if r else np.zeros(n)
    if r and len(active_ids):
        margin = (x[higher[active_ids]] - x[lower[active_ids]]) @ ambient
        if np.min(margin) <= 0:
            raise RuntimeError('Original-space ranking certificate failed')
    # A duplicate is necessarily tied; all other point classes have distinct scores.
    class_scores = np.asarray([scores[np.flatnonzero(class_ids == c)[0]]
                               for c in range(class_count)])
    class_order = np.argsort(-class_scores, kind='stable')
    class_rank = np.empty(class_count, dtype=int)
    class_rank[class_order] = np.arange(class_count)
    prediction_tie_ids = class_rank[class_ids]
    ranking = np.argsort(prediction_tie_ids, kind='stable')
    distance = int(np.count_nonzero(prediction_tie_ids[higher] >= prediction_tie_ids[lower])) + forced
    deleted = np.concatenate(deleted_blocks) if deleted_blocks else np.empty(0, dtype=int)
    decoded_blocks = tuple(pairs[block].copy() for block in deleted_blocks)
    if distance > len(deleted) + forced:
        raise RuntimeError('IIS deletion guarantee failed')
    observed_factor = max((len(block) for block in deleted_blocks), default=1)
    if observed_factor > dimension_bound + 1:
        raise RuntimeError('IIS dimension-bound certificate failed')
    return ClosestLinearRankingResult(
        ranking=ranking, direction=ambient, conditioned_scores=scores,
        target_order=order.copy(), target_tie_ids=target_ties,
        ranking_tie_ids=prediction_tie_ids,
        deleted_comparisons=pairs[deleted].copy(), iis_blocks=decoded_blocks,
        kendall_tau=distance, total_comparisons=total,
        forced_tie_disagreements=forced, optimizable_comparisons=len(pairs),
        dimension_bound=dimension_bound, approximation_factor=observed_factor,
        projected_instance=projected,
        solver_runtime_seconds=time.perf_counter() - start_time,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('points', type=Path, help='Headerless, ordered CSV, txt or .npy matrix')
    tg = p.add_mutually_exclusive_group()
    tg.add_argument('--target-order', help='Comma-separated original row indices')
    tg.add_argument('--target-ranks', help='Comma-separated numeric rank per row')
    tg.add_argument('--target-ranks-file', type=Path)
    p.add_argument('--delimiter', default=',')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rank-tol', type=float, default=None)
    p.add_argument('--threads', type=int, default=None, help='Accepted for CLI compatibility')
    p.add_argument('--solver-log', action='store_true')
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()
    delim = None if args.delimiter == 'whitespace' else args.delimiter
    x = (np.load(args.points) if args.points.suffix.lower() == '.npy'
         else np.loadtxt(args.points, delimiter=delim, ndmin=2))
    ranks = None
    if args.target_ranks_file:
        path = args.target_ranks_file
        ranks = (np.load(path) if path.suffix.lower() == '.npy'
                 else np.loadtxt(path, delimiter=delim)).reshape(-1)
    elif args.target_ranks:
        ranks = [float(v) for v in args.target_ranks.split(',')]
    order = [int(v) for v in args.target_order.split(',')] if args.target_order else None
    result = closest_linear_ranking(
        x, order, target_ranks=ranks, rank_tol=args.rank_tol,
        seed=args.seed, threads=args.threads, output_flag=args.solver_log,
    )
    rendered = json.dumps(result.to_dict(), indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + '\n', encoding='utf-8')
        print(f'Wrote {args.output}; Kendall disagreements={result.kendall_tau}, '
              f'deleted comparisons={result.deleted_count}, '
              f'blocks={result.iterations}, factor={result.approximation_factor}')


if __name__ == '__main__':
    main()
