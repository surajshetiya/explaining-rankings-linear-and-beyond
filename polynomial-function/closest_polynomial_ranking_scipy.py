#!/usr/bin/env python3
"""License-free degree-p polynomial closest-ranking IIS approximation.

Equivalent mathematical model to closest_polynomial_ranking_gurobi.py:
- construct all nonconstant monomials up to total degree p from scaled inputs
- use the FULL set of strict target comparisons (row order of headerless CSV)
- repeatedly delete a comparison-minimal infeasible subsystem (IIS) using
  convex-hull certificates from scipy.optimize.linprog (HiGHS)
- return a realized polynomial score order and its Kendall disagreement count.

The IIS removal heuristic is not a minimum-inversion exact optimizer; alternative
solvers can delete different IIS blocks and therefore return different values.
This strict-order version needs distinct input data points; it does not handle
weak target ranking ties or duplicate point rows.
"""
from __future__ import annotations
import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path
import time
import numpy as np
from closest_linear_ranking_scipy import closest_linear_ranking


def polynomial_features(X, degree=2, input_scaling='power2'):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all() or not X.size:
        raise ValueError('Expected finite nonempty (n,d) point matrix')
    if degree < 1:
        raise ValueError('degree must be >=1')
    n, d = X.shape
    scales = np.ones(d)
    if input_scaling == 'power2':
        # Reversible (in reals), per-coordinate power-of-two scaling. The
        # polynomial ranking function class remains identical: each monomial
        # coefficient can independently compensate for these scales.
        max_abs = np.max(np.abs(X), axis=0)
        scales = np.where(max_abs > 0, np.exp2(np.ceil(np.log2(np.maximum(max_abs, np.finfo(float).tiny)))), 1)
    elif input_scaling != 'none':
        raise ValueError('input_scaling must be power2 or none')
    Y = X / scales
    exponents = []
    columns = []
    for k in range(1, degree + 1):
        for inds in combinations_with_replacement(range(d), k):
            exponent = np.bincount(inds, minlength=d)
            vals = np.prod(Y[:, inds], axis=1)
            if np.any(vals != vals[0]):
                columns.append(vals)
                exponents.append(exponent.tolist())
    if not columns:
        raise ValueError('all polynomial features are constant')
    features = np.ascontiguousarray(np.column_stack(columns))
    if not np.isfinite(features).all():
        raise ValueError('nonfinite polynomial features')
    meta = {'degree': degree, 'basis': 'monomial', 'input_scaling': input_scaling,
            'coordinate_scales': scales.tolist(), 'exponents': exponents,
            'lifted_dimension': features.shape[1], 'original_dimension': d}
    return features, meta


def closest_polynomial_ranking(X, degree=2, seed=0, input_scaling='power2', verbose=False):
    start = time.perf_counter()
    features, meta = polynomial_features(X, degree=degree, input_scaling=input_scaling)
    feature_time = time.perf_counter() - start
    if verbose:
        print('polynomial features', features.shape, 'feature_time',round(feature_time, 4), flush=True)
    result = closest_linear_ranking(features, seed=seed, verbose=verbose)
    result['algorithm'] = 'polynomial_greedy_iis_min_disagreement'
    result['polynomial'] = meta
    result['lifted_weights'] = result['direction']
    result['feature_runtime_seconds'] = round(feature_time, 6)
    result['total_runtime_seconds'] = round(time.perf_counter() - start, 3)
    result['original_dataset_shape'] = list(np.asarray(X).shape)
    # Consistency: score by lifted coefficients and independently count strict violations
    scores = features @ np.asarray(result['direction'])
    i, j = np.triu_indices(len(X), 1)
    gap = scores[i] - scores[j]
    if np.any(np.abs(gap) <= 1e-11):
        raise RuntimeError('ambiguous (near-)tie in resulting polynomial score ordering')
    verified = int(np.count_nonzero(gap < 0))
    if verified != result['kendall_tau']:
        raise RuntimeError(f'Kendall count mismatch: {verified} vs {result["kendall_tau"]}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Headerless, ranked CSV with coordinates only')
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input-scaling', choices=['power2','none'],default='power2')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    X = np.loadtxt(args.input, delimiter=',', ndmin=2)
    result = closest_polynomial_ranking(X, args.degree, args.seed, args.input_scaling, args.verbose)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    fields = ['algorithm','kendall_tau','total_comparisons','deleted_count','iterations',
              'approximation_factor','dimension_bound','feature_runtime_seconds',
              'total_runtime_seconds','is_exact_optimum']
    print(json.dumps({k: result[k] for k in fields},indent=2))
    print('normalized lifted weights:', result['lifted_weights'])
    if args.output:
        print('Saved result to', args.output)

if __name__ == '__main__':
    main()
