# Linear Ranking Algorithms with Gurobi

This directory contains optimized Python/Gurobi implementations of three
linear-ranking approximation algorithms from the manuscript, three sampled
hybrids, and five experimental baselines used to evaluate them.

| Goal | Theorem | Python function | Guarantee |
|---|---|---|---|
| Minimize Kendall--Tau disagreement | `thm:main-kt` | `closest_linear_ranking` | Deterministic `(r + 1)` approximation |
| Maximize Kendall agreement | `thm:logdim-half-approximation` | `max_agreement_half_approximation` | Deterministic `1/2` approximation |
| Maximize Kendall agreement | `thm:logdim-similarity-ptas` | `max_agreement_similarity_ptas` | Randomized `(1 - epsilon)` approximation with probability at least `1 - delta` |
| Maximize Kendall agreement | PTAS sampling + weighted greedy IIS | `max_agreement_greedy_iis_hybrid` | Deterministic `1/2` floor; stronger data-dependent bound when the sufficient sample bound is met |
| Maximize Kendall agreement | PTAS sampling + great-circle event search | `max_agreement_sampled_kendall_local_search` | Deterministic `1/2` floor; conditional candidate-quality bound when the sufficient sample bound is met |
| Maximize Kendall agreement | Duplicate-aware class sampling + great-circle search | `max_agreement_duplicate_aware_sampled_local_search` | Deterministic `1/2` floor; inverse-probability-weighted heuristic sample |

The implementations are in:

- [`closest_linear_ranking_gurobi.py`](closest_linear_ranking_gurobi.py)
- [`max_agreement_linear_ranking_gurobi.py`](max_agreement_linear_ranking_gurobi.py)
- [`hybrid_linear_ranking_gurobi.py`](hybrid_linear_ranking_gurobi.py)
- [`sampled_kendall_local_search_gurobi.py`](sampled_kendall_local_search_gurobi.py)
- [`duplicate_aware_sampled_kendall_local_search.py`](duplicate_aware_sampled_kendall_local_search.py)
- [`linear_ranking_baselines.py`](linear_ranking_baselines.py)

The max-agreement, hybrid, sampled-local-search, duplicate-aware, and baseline
modules reuse conditioning and certificate helpers from the closest-ranking
module. The three hybrids and baseline module also reuse class-order and
weak-Kendall evaluation helpers from the max-agreement module, so keep all six
files in the same directory.

The hybrid is intended for samples such as 122 or 200 comparisons for which
the PTAS's exact empirical binary MIP is too slow. It uses the same IID
comparison sampling but replaces the MIP with continuous LP feasibility,
weighted primal--dual IIS deletion, bounded greedy add-back, a final
minimum-norm convex QP, and optional restarts. It is therefore a sample-bounded
experimental heuristic, not another implementation of the exact-empirical
PTAS.

The sampled Kendall local search targets the same 122/200-comparison regime.
It replaces both the PTAS binary MIP and the baseline local search's repeated
full-data objective evaluations with weighted great-circle event sweeps on the
IID sample. In nondegenerate exact arithmetic a sweep enumerates every open
sign cell on its chosen circle. The floating-point implementation orders
zero-crossing boundary rays by half-plane and cross product, then explicitly
recounts its best predicted cells.
Only one finalist per restart, an optional LMRC seed, and their antipodal
reverses enter full-data selection. It is also a heuristic rather than an
implementation of the exact-empirical PTAS.

The five baseline entry points are:

| Baseline | Python function | Optimization target |
|---|---|---|
| Linearized maximum rank correlation | `linearized_mrc_baseline` | Ridge-regularized linearized rank statistic |
| RankSVM | `rank_svm_baseline` | Regularized pairwise hinge loss |
| Linear RankNet | `linear_ranknet_baseline` | Regularized pairwise logistic loss |
| Direct Kendall local search | `kendall_local_search_baseline` | Full-data Kendall--Tau distance |
| Elastic MaxFS | `elastic_maxfs_baseline` | Elastic comparison infeasibility followed by greedy add-back |

These are heuristics rather than theorem-backed approximations. Every baseline
is nevertheless evaluated on all target-comparable pairs and returns the same
flattened ranking, tie metadata, and comparison counts as the manuscript
algorithms.

## Requirements

- Python 3.10 or newer
- NumPy
- `gurobipy` and a working Gurobi license for closest ranking, the similarity
  PTAS, the greedy-IIS hybrid, and Elastic MaxFS. The half approximation,
  both sampled local searches, linearized MRC, RankSVM, RankNet, and full
  Kendall local search use NumPy only.

Install the Python packages with:

```bash
python -m pip install numpy gurobipy
```

Gurobi licensing is configured separately. A trivial optimization is a useful
basic check:

```bash
python -c "import gurobipy as gp; m = gp.Model(); m.optimize()"
```

A restricted or size-limited license can pass this check but still reject a
larger ranking model.

## Input and ranking conventions

`points` is an `n x d` matrix. Row `i` is the feature vector `x_i`. A direction
`w` scores an item by

```text
score(i) = <w, x_i>.
```

There are two mutually exclusive ways to specify the target:

- `target_order` is the legacy strict target: a zero-based permutation of the
  original row indices from highest to lowest. For example, `[3, 1, 2, 0]`
  places item 3 above item 1, then item 2, then item 0.
- `target_ranks` is a length-`n` numeric vector aligned with the original matrix
  rows. Smaller values rank higher and equal values denote a target tie. Only
  the ordering and equality of the values matter; their numerical gaps do not.

Passing both raises an error. If both are `None`, matrix row order is used as a
strict target. When `target_ranks` contains ties, `target_order` in the result
is its stable flattened permutation: target groups are sorted by rank and the
original row order is retained inside each tied group.

Points and target ranks must be finite and losslessly representable as
`float64`. Exact duplicate point rows are accepted. They receive identical
scores under every linear direction and therefore remain tied in every output.
Distinct point classes are ordered strictly when there is an optimizable target
comparison. If every target item is tied, every method returns the zero
direction and one all-tied output group without invoking Gurobi.

The returned `ranking` remains a deterministic permutation of original row
indices for backward compatibility. Its within-tie order has no ranking
meaning. The arrays `target_tie_ids` and `ranking_tie_ids`, both aligned with
the original rows, carry the target and output weak-order semantics. Their IDs
are contiguous integers starting at zero, with smaller IDs ranked higher.

Target-tied pairs are excluded from evaluation. Let:

- `M = total_comparisons` be the number of pairs separated by the target;
- `U = forced_tie_disagreements` be those pairs whose point rows are exact
  duplicates and hence must tie in the output; and
- `P = optimizable_comparisons = M - U` be the remaining comparison population.

A forced output tie counts as a disagreement. Thus, for every returned result:

```text
Kendall agreement + Kendall--Tau distance = M
Kendall--Tau distance >= U
```

## Python usage

Run Python from `Code`, or add `Code` to `PYTHONPATH`.

### Minimum-disagreement approximation

```python
import numpy as np

from closest_linear_ranking_gurobi import closest_linear_ranking

points = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
    ]
)
target_order = [3, 1, 2, 0]

result = closest_linear_ranking(points, target_order)

print(result.ranking)
print(result.direction)
print(result.kendall_tau)
print(result.approximation_factor)
```

For a target with ties, supply labels in original row order:

```python
target_ranks = np.array([2, 0, 1, 1], dtype=float)
result = closest_linear_ranking(points, target_ranks=target_ranks)

print(result.target_order)     # [1, 2, 3, 0]
print(result.target_tie_ids)   # [2, 0, 1, 1]
print(result.ranking_tie_ids)  # output weak ranks, in original row order
```

Duplicate points are represented explicitly as an output tie:

```python
duplicate_points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
duplicate_result = closest_linear_ranking(
    duplicate_points,
    target_ranks=[0, 1, 2],
)

assert duplicate_result.ranking_tie_ids[0] == duplicate_result.ranking_tie_ids[1]
assert duplicate_result.forced_tie_disagreements == 1
```

### Deterministic max-agreement half-approximation

```python
from max_agreement_linear_ranking_gurobi import (
    max_agreement_half_approximation,
)

result = max_agreement_half_approximation(points, target_order)

print(result.ranking)
print(result.kendall_agreement)
print(result.total_comparisons)
```

The implementation constructs one deterministic seeded, directly verified
generic order of the distinct point classes, compares it with its reverse, and
returns the better orientation. It does not invoke Gurobi.
The two orientations divide the `P` optimizable comparisons, so the returned
agreement is at least `P/2`, which is at least half of the unknown maximum.

### Randomized max-agreement scheme

```python
from max_agreement_linear_ranking_gurobi import (
    max_agreement_similarity_ptas,
)

result = max_agreement_similarity_ptas(
    points,
    target_order,
    epsilon=0.25,
    delta=0.05,
)

print(result.ranking)
print(result.kendall_agreement)
print(result.empirical_optimum)
print(result.sampling_bound_met)
```

Omitting `sample_size` uses the sufficient theoretical sample size. That bound
is conservative and the empirical solve is exponential in the worst case, so a
smaller sample is often more useful for experiments:

```python
experimental = max_agreement_similarity_ptas(
    points,
    target_order,
    epsilon=0.25,
    delta=0.05,
    sample_size=500,
)

assert not experimental.sampling_bound_met  # unless 500 meets the bound
```

When `sampling_bound_met` is false, the code still returns an exact empirical
optimizer for the sampled multiset, subject to Gurobi's numerical decisions,
but it does not claim the requested `epsilon, delta` population guarantee.

### Sampled greedy-IIS hybrid

```python
from hybrid_linear_ranking_gurobi import max_agreement_greedy_iis_hybrid

hybrid = max_agreement_greedy_iis_hybrid(
    points,
    target_order,
    epsilon=0.8,
    delta=0.8,
    sample_size=200,
    restarts=3,
    max_add_back=64,
    seed=59,
)

print(hybrid.ranking)
print(hybrid.kendall_agreement)
print(hybrid.empirical_disagreement)
print(hybrid.empirical_disagreement_lower_bound)
print(hybrid.guaranteed_fraction_of_optimum)
```

The hybrid draws the PTAS's IID-with-replacement sample from the `P`
optimizable comparisons and aggregates repeated item and point-class
comparisons into positive integer weights. It then uses continuous LP/IIS
steps and one minimum-norm convex QP: there is no empirical binary MIP.
Different restarts reuse the same sample but vary constraint insertion order.
The bounded add-back pass tests deleted comparisons after each greedy run.

For a still smaller practical run, set `sample_size=122`. Passing
`sample_size=None` instead requests the sufficient theoretical VC sample size,
which can be much larger. A manual size of 122 or 200 commonly leaves
`sampling_bound_met=False`; in that case the deterministic `1/2` agreement
floor still applies, but the stronger `epsilon, delta` population statement
does not.

The hybrid follows the same weak-ranking rules as every other implementation.
Target-tied pairs are not sampled or evaluated, exact duplicate point rows
receive one output tie ID, and target-strict comparisons between duplicate
rows contribute to `U = forced_tie_disagreements` rather than to the sampled
population `P`. Supply a tied target with `target_ranks` in exactly the same
way as for the theorem implementations.

### PTAS-sampled Kendall local search

```python
from sampled_kendall_local_search_gurobi import (
    max_agreement_sampled_kendall_local_search,
)

sampled_search = max_agreement_sampled_kendall_local_search(
    points,
    target_order,
    epsilon=0.8,
    delta=0.8,
    sample_size=200,
    restarts=8,
    max_iterations=250,
    patience=40,
    gaussian_interval=10,
    seed=59,
)

print(sampled_search.ranking)
print(sampled_search.kendall_agreement)
print(sampled_search.best_empirical_agreement)
print(sampled_search.selected_empirical_agreement)
print(sampled_search.guaranteed_fraction_of_optimum)
```

The inner objective is the integer-weighted agreement on the same
IID-with-replacement comparison sample used by the PTAS. Repeated item and
point-class comparisons become integer multiplicities, and a thin SVD reduces
the search to the sampled comparison-difference span. Given the current unit
direction and a tangent, the search parameterizes their great circle and
sweeps the sampled zero-crossing events. This enumerates every open sign cell
in nondegenerate exact arithmetic. Numerically, an angle-free half-plane and
cross-product ordering avoids `atan2` resolution loss; rays that are collinear
in working precision are grouped, and the best eight predicted cells are
explicitly recounted.
Coordinate, violated-pair, aggregated-violation, and periodic Gaussian
tangents provide complementary neighborhoods.

No full Kendall count occurs inside that iterative search. The implementation
fully scores only the optional generically refined LMRC candidate and one
refined finalist per restart; their reverse-orientation agreements are obtained
algebraically.
It then returns the candidate with the best full-data agreement. Set
`sample_size=122` for a smaller experiment or `sample_size=None` to request the
sufficient theoretical sample size. Computing the optional LMRC initializer
does make a separate full-data pass; use `use_lmrc_seed=False` or
`--no-lmrc-seed` when that initialization cost is undesirable.

This path uses the same weak-order conventions as the other methods. It never
samples target-tied pairs or comparisons between identical point classes,
keeps exact duplicate rows tied in its output, and reports target-strict
duplicate comparisons separately as `forced_tie_disagreements`. Both
`target_ranks` and `target_ranks_file` therefore work without preprocessing
ties into an arbitrary strict order.

### Duplicate-aware sampled Kendall local search

```python
from duplicate_aware_sampled_kendall_local_search import (
    max_agreement_duplicate_aware_sampled_local_search,
)

duplicate_aware = max_agreement_duplicate_aware_sampled_local_search(
    points,
    target_order,
    epsilon=0.8,
    delta=0.8,
    sample_size=200,
    restarts=8,
    seed=59,
)

print(duplicate_aware.ranking)
print(duplicate_aware.kendall_agreement)
print(duplicate_aware.sampled_class_coverage)
print(duplicate_aware.effective_sample_size)
```

This variant is designed for a large `n` with substantially fewer than `n`
distinct feature rows. It collapses exact duplicates before sampling and
counts how many original item comparisons each sampled directed point-class
edge represents. Its default proposal is a `0.50/0.25/0.25` mixture of the
original item-level Kendall distribution, class-balanced directed-pair
proposals, and margin-focused proposals under an LMRC pilot. This reduces
collisions on comparisons induced by large duplicate classes while still
deliberately visiting rare classes and difficult boundaries.

The search weight for sampled edge `e` is proportional to
`(W_e/P)/q(e)`, so the biased proposal continues to target the original
item-level Kendall objective rather than a class-balanced surrogate. Weights
are scaled to sum to `sample_size`; this common scaling does not change the
optimizer. The mixture and pilot adaptation make this an experimental
heuristic, so the reported uniform-PTAS flag is false on nonempty inputs. Full
scoring of every finalist and its reverse still gives the deterministic
`1/2` floor.

For small `k`, the method constructs an exact directed class table using
`O(k^2)` int64 memory. `max_class_matrix_entries=4_000_000` is the default
automatic-switch threshold. If `k^2` exceeds it, the method instead uses exact
objective draws and uniformly proposed ordered class pairs whose multiplicity
is calculated only when needed. Non-comparable proposals and rejected hard
proposals are retained as null observations in the importance experiment, so
the proposal probability remains known without enumerating all class edges.
This sparse mode uses `O(n + s)` indexing/sample memory rather than `O(k^2)`,
where `s` is the requested sample size.

### Experimental baselines

All five baselines share the input and output conventions above:

```python
from linear_ranking_baselines import (
    elastic_maxfs_baseline,
    kendall_local_search_baseline,
    linear_ranknet_baseline,
    linearized_mrc_baseline,
    rank_svm_baseline,
)

results = {
    "lmrc": linearized_mrc_baseline(points, target_order),
    "ranksvm": rank_svm_baseline(points, target_order),
    "ranknet": linear_ranknet_baseline(points, target_order),
    "local_search": kendall_local_search_baseline(points, target_order),
    "maxfs": elastic_maxfs_baseline(points, target_order),
}

for name, result in results.items():
    print(name, result.kendall_tau, result.kendall_agreement)
```

RankSVM, RankNet, and MaxFS use at most 100,000 IID draws from the `P`
optimizable comparisons by default when that population is larger. Set
`max_training_pairs=None` in Python to use all `P` comparisons. RankSVM then
uses its implicit full-population gradient; RankNet and MaxFS explicitly
represent the population, so that setting can require quadratic memory.

## Function signatures

### `closest_linear_ranking`

```python
closest_linear_ranking(
    points,
    target_order=None,
    *,
    target_ranks=None,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)
```

### `max_agreement_half_approximation`

```python
max_agreement_half_approximation(
    points,
    target_order=None,
    *,
    target_ranks=None,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)
```

### `max_agreement_similarity_ptas`

```python
max_agreement_similarity_ptas(
    points,
    target_order=None,
    *,
    target_ranks=None,
    epsilon,
    delta,
    sample_size=None,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
    max_sample_size=10_000_000,
    max_unique_comparisons=10_000,
)
```

The sample-bound helper is also public:

```python
vc_uniform_sample_size(vc_dimension, eta, delta)
```

### `max_agreement_greedy_iis_hybrid`

```python
max_agreement_greedy_iis_hybrid(
    points,
    target_order=None,
    *,
    target_ranks=None,
    epsilon,
    delta,
    sample_size=200,
    restarts=3,
    max_add_back=64,
    rank_tol=None,
    sample_rank_tol=0.0,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
    max_sample_size=1_000_000,
    max_unique_comparisons=100_000,
)
```

Set `sample_size=None` in the Python API to use
`vc_uniform_sample_size(dimension_bound, epsilon/4, delta)`. Set
`max_add_back=None` to test every comparison removed by a greedy run.

### `max_agreement_sampled_kendall_local_search`

```python
max_agreement_sampled_kendall_local_search(
    points,
    target_order=None,
    *,
    target_ranks=None,
    epsilon,
    delta,
    sample_size=200,
    restarts=8,
    max_iterations=250,
    patience=40,
    gaussian_interval=10,
    ridge=1e-6,
    use_lmrc_seed=True,
    lmrc_cg_tolerance=1e-6,
    lmrc_cg_max_iterations=50,
    rank_tol=None,
    sample_rank_tol=0.0,
    seed=0,
    max_sample_size=1_000_000,
    max_unique_comparisons=100_000,
)
```

As with the sampled greedy-IIS hybrid, `sample_size=None` requests the
sufficient VC size. Set `use_lmrc_seed=False` to skip the full-data LMRC
initializer. Setting `lmrc_cg_max_iterations=None` uses LMRC's adaptive
iteration limit rather than the local search's deliberately bounded default.

### `max_agreement_duplicate_aware_sampled_local_search`

```python
max_agreement_duplicate_aware_sampled_local_search(
    points,
    target_order=None,
    *,
    target_ranks=None,
    epsilon,
    delta,
    sample_size=200,
    restarts=8,
    max_iterations=250,
    patience=40,
    gaussian_interval=10,
    ridge=1e-6,
    use_lmrc_seed=True,
    lmrc_cg_tolerance=1e-6,
    lmrc_cg_max_iterations=50,
    objective_mixture_weight=0.50,
    class_mixture_weight=0.25,
    hard_mixture_weight=0.25,
    rank_tol=None,
    sample_rank_tol=0.0,
    seed=0,
    max_sample_size=1_000_000,
    max_class_matrix_entries=4_000_000,
)
```

The three mixture weights may be any nonnegative values with positive sum;
they are normalized internally. The LMRC pilot is always computed for the
hard-edge proposal. `use_lmrc_seed=False` only omits it from the search seeds
and finalist set. `sample_size=None` uses the ordinary VC size as a reference
draw count, but the adaptive nonuniform heuristic does not claim the PTAS's
uniform-sampling bound. `max_class_matrix_entries` is an automatic-switch
threshold, not a required table size. Set it to `None` to force sparse mode.

### Experimental baseline functions

```python
linearized_mrc_baseline(
    points,
    target_order=None,
    *,
    target_ranks=None,
    ridge=1e-6,
    cg_tolerance=1e-10,
    cg_max_iterations=None,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)

rank_svm_baseline(
    points,
    target_order=None,
    *,
    target_ranks=None,
    regularization=1e-3,
    max_training_pairs=100_000,
    epochs=100,
    batch_size=None,
    learning_rate=0.03,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)

linear_ranknet_baseline(
    points,
    target_order=None,
    *,
    target_ranks=None,
    regularization=1e-3,
    max_training_pairs=100_000,
    epochs=100,
    batch_size=None,
    learning_rate=0.03,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)

kendall_local_search_baseline(
    points,
    target_order=None,
    *,
    target_ranks=None,
    restarts=8,
    max_iterations=250,
    initial_step=0.25,
    step_decay=0.5,
    patience=25,
    min_step=1e-4,
    ridge=1e-6,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)

elastic_maxfs_baseline(
    points,
    target_order=None,
    *,
    target_ranks=None,
    max_training_pairs=100_000,
    deletion_fraction=0.05,
    max_removal_iterations=1_000,
    max_add_back=1_000,
    elastic_tolerance=1e-8,
    rank_tol=None,
    seed=0,
    output_flag=False,
    threads=None,
    feasibility_tol=1e-8,
    gurobi_params=None,
)
```

### Shared parameters

| Parameter | Meaning |
|---|---|
| `points` | Finite `n x d` real matrix. Exact duplicate rows are allowed and form forced output ties. |
| `target_order` | Legacy strict target as an original-row permutation, highest first. It is mutually exclusive with `target_ranks`; if both are `None`, row order is the strict target. |
| `target_ranks` | Optional length-`n` finite numeric labels aligned with original rows. Smaller is higher and equality denotes a target tie. Mutually exclusive with `target_order`. |
| `rank_tol` | `None` preserves the supplied comparison space. A finite nonnegative value, including `0`, enables optional SVD projection with that absolute cutoff. |
| `seed` | Seed for Gurobi, PTAS/hybrid/baseline sampling or randomized search, and generic class-order perturbations. |
| `output_flag` | Whether to display Gurobi logs for methods that invoke it; accepted as a no-op by NumPy-only baselines for API compatibility. |
| `threads` | Gurobi worker count for solver-backed methods. `None` or `0` uses Gurobi's automatic choice. |
| `feasibility_tol` | Gurobi feasibility tolerance in `[1e-9, 1e-2]` for solver-backed methods. |
| `gurobi_params` | Optional model-level Gurobi parameters for solver-backed methods. The PTAS forcibly resets `MIPGap` and `MIPGapAbs` to zero after applying these values. |

If a time, node, work, memory, or other resource limit interrupts a required
solve, the function raises instead of returning an approximation claim.

### PTAS-specific parameters

| Parameter | Meaning |
|---|---|
| `epsilon` | Requested approximation loss in `(0, 1)`. The code sets `eta = epsilon/4`. |
| `delta` | Requested failure-probability upper bound in `(0, 1)`. |
| `sample_size` | Positive IID sample size drawn from the `P` optimizable comparisons. `None` uses the sufficient VC bound. |
| `max_sample_size` | Guard for the total number of random draws. `None` disables it. |
| `max_unique_comparisons` | Guard for weighted binary variables after repeated sampled pair IDs are collapsed. `None` disables it. |

### Hybrid-specific parameters

| Parameter | Meaning |
|---|---|
| `epsilon`, `delta` | Values in `(0, 1)` used to compute `eta = epsilon/4`, the sufficient VC sample size, and the conditional data-dependent bound. They do not by themselves reduce the default 200-draw experimental sample. |
| `sample_size` | Positive number of IID-with-replacement draws from `P`; the default is `200`. `None` uses the sufficient VC size. A practical experiment may instead use `122`. |
| `restarts` | Positive number of weighted greedy-IIS runs on the same sample; default `3`. Restarts vary insertion order, not the sampled multiset. |
| `max_add_back` | Maximum deleted class comparisons tested for feasible re-insertion per restart; default `64`, `0` skips add-back, and `None` tests all of them. |
| `sample_rank_tol` | Nonnegative cutoff for the thin SVD of sampled comparison differences. The default `0` retains every positive singular value. Truncating a positive sampled direction sets `sample_projection_used=True` and disables the stronger data-dependent bound. |
| `max_sample_size` | Draw-count guard, default `1_000_000`; `None` disables it. |
| `max_unique_comparisons` | Distinct sampled item-pair guard before point-class aggregation, default `100_000`; `None` disables it. |

### Sampled-local-search-specific parameters

| Parameter | Meaning |
|---|---|
| `epsilon`, `delta` | Values in `(0, 1)` used for `eta = epsilon/4`, the sufficient VC size, and the conditional candidate-quality bound. They do not alter the default 200-draw sample. |
| `sample_size` | Positive IID-with-replacement draw count from `P`; default `200`. `None` uses the sufficient VC size, while `122` is a smaller practical experiment. |
| `restarts` | Number of local-search seeds on the one shared sample; default `8`. Seeds include projected LMRC when enabled, the weighted sample centroid, their negatives, and seeded Gaussian directions. |
| `max_iterations` | Maximum numerically verified great-circle event sweeps per restart; default `250`. A restart can stop earlier after `patience` stagnant sweeps. |
| `patience` | Consecutive non-improving/plateau sweep limit; default `40`. |
| `gaussian_interval` | Positive interval between Gaussian tangent neighborhoods; default `10`. Other iterations use coordinate, violated-pair, or aggregate-violation tangents. |
| `ridge` | Nonnegative ridge multiplier for the optional LMRC seed; default `1e-6`. |
| `use_lmrc_seed` | Whether to compute and include the original LMRC direction and its sampled-span projection; default `True`. |
| `lmrc_cg_tolerance` | Positive PCG residual tolerance for the LMRC initializer; default `1e-6`. |
| `lmrc_cg_max_iterations` | Optional LMRC PCG iteration cap; default `50`. `None` uses LMRC's adaptive limit. |
| `sample_rank_tol` | Nonnegative thin-SVD cutoff for sampled comparison differences. Default `0` retains every positive returned singular value; truncation is reported by `sample_projection_used`. |
| `max_sample_size` | Draw-count guard, default `1_000_000`; `None` disables it. |
| `max_unique_comparisons` | Distinct sampled item-pair guard before class aggregation, default `100_000`; `None` disables it. |

### Baseline-specific parameters

| Baseline | Parameter | Meaning |
|---|---|---|
| Linearized MRC | `ridge` | Nonnegative ridge multiplier, scaled by the mean conditioned feature variance. |
| Linearized MRC | `cg_tolerance`, `cg_max_iterations` | Relative residual tolerance and optional iteration limit for the matrix-free preconditioned-CG path. With the default `None`, the limit is `min(2000, max(50, 4d))`; small primal or dual systems are solved densely instead. |
| RankSVM/RankNet | `regularization` | Nonnegative L2 coefficient in the average pairwise surrogate objective. |
| RankSVM/RankNet | `max_training_pairs` | Positive pair cap. If it is smaller than `P`, draw that many optimizable comparisons IID with replacement; `None` uses all `P` pairs. |
| RankSVM/RankNet | `epochs`, `learning_rate` | Adam epoch count and learning rate. Defaults are `100` and `0.03`. The best full training-objective epoch is retained. |
| RankSVM/RankNet | `batch_size` | `None` uses endpoint-aggregated full-batch updates on the chosen training population. A positive value enables shuffled mini-batches for sampled/explicit pairs; full implicit RankSVM always uses its exact full-pair update. |
| Kendall local search | `restarts`, `max_iterations` | Numbers of starts and proposal iterations per start; defaults are `8` and `250`. |
| Kendall local search | `initial_step`, `step_decay`, `patience`, `min_step` | Adaptive pattern-search controls, defaulting to `0.25`, `0.5`, `25`, and `1e-4`. |
| Kendall local search | `ridge` | Ridge multiplier used only for its linearized-MRC initial direction. |
| Elastic MaxFS | `max_training_pairs` | Positive LP comparison cap with the same `P`-population sampling rule as above; `None` builds the full optimizable-comparison LP. |
| Elastic MaxFS | `deletion_fraction` | Fraction of currently positive elastic violations removed per round. The default is `0.05`; `0` removes one. |
| Elastic MaxFS | `max_removal_iterations` | Deletion-round guard, default `1_000`. Hitting it triggers a reported feasibility-preserving fallback that drops the remaining active rows. |
| Elastic MaxFS | `max_add_back` | Maximum reverse-order greedy add-back tests. The default is `1_000`, `0` disables add-back, and `None` tests every removed row. |
| Elastic MaxFS | `elastic_tolerance` | Positive weighted-elastic stopping tolerance, at most `1e-3`; default `1e-8`. |

Except for options explicitly documented as zero-disabling, positive integer
parameters reject booleans, nonintegral values, zero, and negatives.
For `n = 1`, every method returns immediately with the only ranking and a zero
direction. An input containing only one distinct point class likewise has one
forced output tie group and needs no strict-order solve. Method parameters are
still validated. More generally, if `P = 0`, every method returns the all-tied
zero-direction result; sampling algorithms return an empty sample with
`sampling_bound_met=True` because there is no optimizable population.

## Command-line usage

The CLIs accept headerless numeric CSV, delimiter-separated text, whitespace
text, and NumPy `.npy` matrices. Results are printed as JSON unless `--output`
is supplied; an existing output file is overwritten.

### Minimum disagreement

```bash
cd Code

python closest_linear_ranking_gurobi.py points.csv \
  --target-order 3,1,2,0 \
  --threads 4 \
  --output closest-result.json
```

Use `--target-ranks` for a small tied target. The values are aligned with the
original point rows:

```bash
python closest_linear_ranking_gurobi.py points.csv \
  --target-ranks 2,0,1,1 \
  --output closest-tied-result.json
```

For a large dataset, put the same vector in one row or one column of a text
file, or in a one-dimensional `.npy` file:

```bash
python closest_linear_ranking_gurobi.py points.csv \
  --target-ranks-file target-ranks.csv \
  --output closest-tied-result.json
```

Text target-rank files use the same `--delimiter` setting as the point matrix.

### Deterministic max agreement

```bash
python max_agreement_linear_ranking_gurobi.py half points.csv \
  --target-order 3,1,2,0 \
  --threads 4 \
  --output half-result.json
```

### Randomized max agreement

```bash
python max_agreement_linear_ranking_gurobi.py ptas points.csv \
  --target-order 3,1,2,0 \
  --epsilon 0.25 \
  --delta 0.05 \
  --threads 4 \
  --output ptas-result.json
```

For a practical experimental sample:

```bash
python max_agreement_linear_ranking_gurobi.py ptas points.npy \
  --epsilon 0.25 \
  --delta 0.05 \
  --sample-size 500
```

### Sampled greedy-IIS hybrid

The following runs use 122 and 200 IID comparison draws, respectively. Both
replace the PTAS binary MIP with weighted continuous LP/IIS iterations and a
final minimum-norm convex QP:

```bash
python hybrid_linear_ranking_gurobi.py points.csv \
  --epsilon 0.8 \
  --delta 0.8 \
  --sample-size 122 \
  --restarts 3 \
  --max-add-back 64 \
  --seed 59 \
  --output hybrid-122-result.json

python hybrid_linear_ranking_gurobi.py points.csv \
  --epsilon 0.8 \
  --delta 0.8 \
  --sample-size 200 \
  --restarts 5 \
  --max-add-back 128 \
  --seed 59 \
  --output hybrid-200-result.json
```

`--max-add-back 0` skips add-back. The CLI defaults to 200 draws; only the
Python API accepts `sample_size=None` to request the theoretical sample size.
The hybrid also accepts the common target options, so a tied target can be
loaded with `--target-ranks-file target-ranks.csv`.

### PTAS-sampled Kendall local search

```bash
python sampled_kendall_local_search_gurobi.py points.csv \
  --epsilon 0.8 \
  --delta 0.8 \
  --sample-size 200 \
  --restarts 8 \
  --max-iterations 250 \
  --patience 40 \
  --gaussian-interval 10 \
  --seed 59 \
  --output sampled-local-search-result.json
```

For a smaller search, change this to `--sample-size 122`. Use
`--no-lmrc-seed` to avoid computing the initializer. The CLI option
`--sample-size 0` requests the sufficient theoretical size, corresponding to
`sample_size=None` in Python. The CLI option
`--lmrc-cg-max-iterations 0` selects LMRC's adaptive iteration limit; its
default is the deliberately bounded value `50`. Tied targets can be supplied
with `--target-ranks` or `--target-ranks-file`.

### Duplicate-aware sampled Kendall local search

```bash
python duplicate_aware_sampled_kendall_local_search.py points.csv \
  --epsilon 0.8 \
  --delta 0.8 \
  --sample-size 200 \
  --restarts 8 \
  --max-iterations 250 \
  --seed 59 \
  --output duplicate-aware-result.json
```

The default mixture is objective/class/hard = `0.50/0.25/0.25`. Override it
with `--objective-mixture-weight`, `--class-mixture-weight`, and
`--hard-mixture-weight`. `--max-class-matrix-entries` sets the largest exact
`k`-by-`k` table; its default is 4,000,000 entries. Larger instances switch
automatically to sparse on-demand sampling, and `0` forces sparse mode.
Target ties use the same `--target-ranks` or `--target-ranks-file` options as
the other commands.

### Experimental baselines

The baseline CLI has one subcommand per public function:

```bash
# Linearized MRC
python linear_ranking_baselines.py lmrc points.csv \
  --target-order 3,1,2,0 --ridge 1e-6

# RankSVM on every optimizable comparison (0 disables the default 100000-pair cap)
python linear_ranking_baselines.py ranksvm points.csv \
  --max-training-pairs 0 --epochs 100

# Linear RankNet on the default bounded comparison sample
python linear_ranking_baselines.py ranknet points.csv \
  --max-training-pairs 100000 --seed 7

# Full-objective multistart Kendall search
python linear_ranking_baselines.py local-search points.csv \
  --restarts 8 --max-iterations 250

# Elastic MaxFS with at most 1000 reverse-order add-back tests
python linear_ranking_baselines.py maxfs points.csv \
  --max-training-pairs 100000 --max-add-back 1000 \
  --output maxfs-result.json
```

The baseline subcommands share `--target-order`, `--target-ranks`,
`--target-ranks-file`, `--delimiter`, `--rank-tol`, `--seed`, `--threads`,
`--solver-log`, and `--output`. The three target options are mutually
exclusive. For RankSVM, RankNet, and MaxFS, `--max-training-pairs 0` requests
all `P` optimizable comparisons. For RankSVM and RankNet, `--batch-size 0`
(also the default) selects endpoint-aggregated full-batch Adam. For MaxFS,
`--max-add-back 0` disables add-back. The Python API additionally permits
`max_add_back=None` to try every removed comparison.

The minimum-disagreement, max-agreement, and hybrid CLIs use the same mutually
exclusive target options. Common options are `--delimiter`, `--rank-tol`,
`--seed`, `--threads`, `--solver-log`, and `--output`. The PTAS subcommand and
hybrid additionally accept `--sample-size`, `--max-sample-size`, and
`--max-unique-comparisons`; the hybrid also accepts `--restarts`,
`--max-add-back`, and `--sample-rank-tol`. Set either CLI resource guard to `0`
to disable it; the Python API uses `None`. `feasibility_tol` and arbitrary
`gurobi_params` are available only through the Python API.

The sampled-local-search CLI shares the three mutually exclusive target
options plus `--delimiter`, `--rank-tol`, `--sample-rank-tol`, `--seed`, and
`--output`. It has the same sample guards and adds `--restarts`,
`--max-iterations`, `--patience`, `--gaussian-interval`, `--ridge`, and the
LMRC controls shown above. Its search uses NumPy rather than Gurobi, so it has
no thread or solver-log option and reports `solver_runtime_seconds=0`.
The duplicate-aware CLI shares those search controls, replaces the distinct
item-pair guard with `--max-class-matrix-entries`, and adds the three sampling
mixture weights. It also uses NumPy only.

Examples of other input formats:

```bash
# Whitespace-separated matrix; target is file row order
python max_agreement_linear_ranking_gurobi.py half \
  points.txt --delimiter whitespace

# NumPy matrix and projected-instance mode
python max_agreement_linear_ranking_gurobi.py ptas points.npy \
  --epsilon 0.3 --delta 0.1 --rank-tol 1e-10
```

## Result objects

All functions return frozen dataclasses. Their attribute bindings are frozen,
but contained NumPy arrays are not read-only. Call `result.to_dict()` for a
JSON-serializable representation.

The following serialization convention is shared by every result type:

- `ranking` and `target_order` are deterministic flattened permutations of the
  original row indices. They preserve the former strict-order output format.
- `ranking_tie_ids` and `target_tie_ids` are the authoritative weak ranks,
  aligned with original row order. Equal IDs mean tied items; smaller IDs rank
  higher. The order of items with the same ID in a flattened permutation is
  deterministic but semantically irrelevant.
- `direction` and `conditioned_scores` tie exact duplicate point rows and are
  strict between consecutive distinct point classes in `ranking` when `P > 0`.
  When `P = 0`, both are zero and every row has one output tie ID.
- `total_comparisons = optimizable_comparisons + forced_tie_disagreements`.
  Target-tied pairs are absent from all three counts.

### `ClosestLinearRankingResult`

| Field | Meaning |
|---|---|
| `ranking` | Deterministic flattened output permutation in original row indices. |
| `direction` | Ambient scoring direction realizing the output weak order; positive scale is arbitrary. |
| `conditioned_scores` | Stable model scores in original row order; they need not numerically equal `points @ direction`. |
| `target_order` | Validated flattened target permutation. |
| `target_tie_ids`, `ranking_tie_ids` | Target and output weak-rank IDs aligned with original rows. |
| `deleted_comparisons` | Removed optimizable comparisons as original-index pairs `(higher, lower)`; forced duplicate ties are reported separately. |
| `iis_blocks` | The modeled deleted comparisons partitioned by minimal-infeasible-subsystem iteration. |
| `kendall_tau` | Distance from the target. |
| `total_comparisons` | Number `M` of target-comparable pairs. |
| `forced_tie_disagreements` | Number `U` of target-comparable exact-duplicate pairs that must tie in every output. |
| `optimizable_comparisons` | Number `P = M - U` of modeled comparisons between distinct point classes. |
| `dimension_bound` | Safe modeled-dimension upper bound. |
| `approximation_factor` | Largest removed IIS block, or `1` if none was removed. |
| `projected_instance` | Whether SVD projection was explicitly requested. |
| `solver_runtime_seconds` | Solver setup, solve, and certificate time. |

Convenience properties are `deleted_count`, `modeled_deleted_count`, and
`iterations`. `deleted_count` includes `U`, whereas `modeled_deleted_count` is
the number of rows in `deleted_comparisons`.

### `HalfApproximationResult`

| Field | Meaning |
|---|---|
| `ranking`, `direction`, `conditioned_scores`, `target_order`, `target_tie_ids`, `ranking_tie_ids` | As above. |
| `kendall_agreement`, `kendall_tau`, `total_comparisons` | Counts on all `M` target-comparable pairs. |
| `forced_tie_disagreements`, `optimizable_comparisons` | The `U` forced duplicate ties and `P = M - U` optimizable pairs. |
| `base_agreement`, `opposite_agreement` | Agreement of the constructed distinct-class order and its reverse; they sum to `P`. |
| `chosen_orientation` | `1` for the base order and `-1` for its reverse. |
| `numerical_margin` | Smallest checked margin between adjacent distinct point classes in conditioned score space; zero when there is only one class. |
| `dimension_bound`, `projected_instance` | Modeled-space diagnostics. |
| `solver_runtime_seconds`, `total_runtime_seconds` | Solver-only and end-to-end wall times. |

The property `guaranteed_fraction_of_optimum` is always `0.5`.

### `SimilarityPTASResult`

| Field | Meaning |
|---|---|
| `ranking`, `direction`, `conditioned_scores`, `target_order`, `target_tie_ids`, `ranking_tie_ids` | As above. |
| `kendall_agreement`, `kendall_tau`, `total_comparisons` | Counts on all `M` target-comparable pairs. |
| `forced_tie_disagreements`, `optimizable_comparisons` | The `U` forced duplicate ties and `P = M - U` optimization/sample population. |
| `empirical_agreement`, `empirical_optimum` | Agreement on the sampled multiset; the implementation verifies equality. |
| `sample_size`, `unique_sample_size` | Total IID draws from `P` and number of distinct sampled optimizable comparisons. Both are zero when `P = 0`. |
| `sampled_comparisons` | Distinct oriented optimizable comparisons in original row indices; it never contains a target tie or an exact-duplicate pair. |
| `sample_multiplicities` | Positive integer weights corresponding to `sampled_comparisons`. |
| `epsilon`, `delta`, `eta` | Requested parameters and `eta = epsilon/4`. |
| `dimension_bound` | Safe VC-dimension upper bound, not necessarily the exact effective dimension. |
| `theoretical_sample_size` | Sufficient size from the explicit VC inequality. |
| `sampling_bound_met` | Whether the actual sample reaches that sufficient size. |
| `selected_numerical_margin` | Smallest selected-comparison margin in the normalized empirical witness. |
| `numerical_margin` | Smallest twice-checked inter-class margin of the infinity-normalized, generically perturbed final witness. |
| `projected_instance` | Whether SVD projection was explicitly requested. |
| `mip_node_count` | Gurobi branch-and-bound nodes for empirical optimization. |
| `solver_runtime_seconds`, `total_runtime_seconds` | Solver-only and end-to-end wall times. |

`numerical_margin` is measured after conditioning with infinity-normalized
reduced coefficients; it is not a raw `points @ direction` gap. No observed
approximation ratio is reported for the PTAS: the full-data optimum is unknown,
and the theorem's event is probabilistic rather than an instance certificate.

### `GreedyIISHybridResult`

| Field | Meaning |
|---|---|
| `ranking`, `direction`, `conditioned_scores`, `target_order`, `target_tie_ids`, `ranking_tie_ids` | Returned weak order, its ambient linear witness, and tie metadata under the shared conventions. |
| `kendall_agreement`, `kendall_tau`, `total_comparisons` | Counts recomputed on all `M` target-comparable pairs for the selected output. |
| `forced_tie_disagreements`, `optimizable_comparisons` | The `U` forced duplicate ties and sampled population `P = M - U`. |
| `sample_size`, `unique_sample_size`, `unique_class_sample_size` | Total IID draws, distinct sampled item comparisons, and distinct directed point-class comparisons after aggregation. |
| `sampled_comparisons`, `sample_multiplicities` | Distinct sampled item comparisons in original row indices and their positive occurrence counts. Target ties and same-class duplicate pairs are absent. |
| `empirical_agreement`, `empirical_disagreement` | Sample counts for the best certificate-bearing forward greedy candidate. They need not describe the returned orientation. |
| `selected_empirical_agreement`, `selected_empirical_disagreement` | Sample counts for the output selected by full-data agreement from all forward and reverse candidates. |
| `empirical_disagreement_lower_bound` | Weighted IIS dual lower bound on the sample's optimum disagreement in the space identified by `empirical_certificate_scope`. |
| `empirical_deletion_upper_bound` | Deleted sample weight after add-back for the certificate-bearing greedy run; its realized empirical disagreement is no larger. |
| `empirical_optimality_gap_upper_bound`, `normalized_empirical_gap` | Difference between the certificate candidate's empirical disagreement and the dual lower bound, in counts and divided by `sample_size`. |
| `empirical_certificate_scope` | `original`, `projected_global`, or `sample_projected`, identifying which modeled hypothesis family the empirical certificate covers. |
| `greedy_approximation_factor` | Largest IIS size in the certificate run; it is at most `sample_dimension + 1`. |
| `iis_rounds`, `dual_raise_total` | Number of distinct IIS extractions and total weighted dual raise for the certificate run. |
| `restarts`, `lower_bound_restart`, `certificate_restart`, `chosen_restart`, `chosen_orientation` | Requested greedy runs, provenance of the maximum dual lower bound and forward certificate, and the restart/orientation selected by full agreement; orientation is `1` or `-1`. |
| `epsilon`, `delta`, `eta` | Requested parameters and `eta = epsilon/4`. |
| `dimension_bound`, `sample_dimension`, `sample_condition_number` | Global modeled-space bound plus the numerical dimension and retained-singular-value condition number of the sampled comparison-difference span. |
| `sample_rank_tolerance`, `sample_projection_used`, `raw_iis_certified` | Sample SVD cutoff, whether it removed a positive sampled singular direction, and whether every reduced IIS was independently re-certified from precompression comparison rows. |
| `theoretical_sample_size`, `sampling_bound_met` | Sufficient VC size and whether the actual number of draws reaches it. |
| `guaranteed_fraction_of_optimum` | Always at least `0.5`; the stronger data-dependent value is exposed only when the sufficient sample size, no-projection, and raw-IIS-certificate preconditions hold. |
| `numerical_margin`, `projected_instance` | Checked final inter-class margin and whether global score-space projection was requested. |
| `solver_optimizations`, `optimization_runtime_seconds`, `solver_runtime_seconds`, `total_runtime_seconds` | Continuous-solver work and wall-time diagnostics. Solver time is contained in optimization and total time. |
| `restart_diagnostics` | Per-restart IIS sizes, dual/deletion values, add-back counts, empirical/full scores, and solver work. |

The forward-candidate and selected-output empirical fields are intentionally
separate. The algorithm may return a reverse orientation because it has better
full-data agreement, while the weighted IIS primal--dual certificate belongs
to a forward feasible-subsystem run. This prevents the sample certificate from
being incorrectly attributed to a different returned orientation.

### `SampledKendallLocalSearchResult`

| Field | Meaning |
|---|---|
| `ranking`, `direction`, `conditioned_scores`, `target_order`, `target_tie_ids`, `ranking_tie_ids` | Returned strict order of distinct point classes when `P > 0`, or the all-tied zero-direction result when `P = 0`; includes flattened permutations and authoritative tie metadata. |
| `kendall_agreement`, `kendall_tau`, `total_comparisons` | Counts recomputed exactly on all `M` target-comparable pairs for the selected output. |
| `forced_tie_disagreements`, `optimizable_comparisons` | The `U` unavoidable cross-target duplicate ties and sampled population `P = M - U`. |
| `sample_size`, `unique_sample_size`, `unique_class_sample_size` | IID draw count, distinct sampled item comparisons, and directed point-class comparisons after aggregation. |
| `sampled_comparisons`, `sample_multiplicities` | Distinct oriented sampled item pairs in original indices and their positive occurrence counts. Target ties and identical-class pairs are absent. |
| `best_empirical_agreement`, `best_empirical_disagreement` | Sample score of the retained candidate used for the conditional candidate-quality bound. This candidate need not be the returned full-data winner. |
| `selected_empirical_agreement`, `selected_empirical_disagreement` | Sample score of the candidate selected by exact full-data agreement. |
| `initial_lmrc_empirical_agreement`, `initial_lmrc_empirical_disagreement` | Refined LMRC seed's sample counts, or `None` when `use_lmrc_seed=False`. |
| `epsilon`, `delta`, `eta` | Requested values and `eta = epsilon/4`. |
| `dimension_bound`, `sample_dimension`, `sample_condition_number` | Global modeled dimension bound plus the retained sampled-difference dimension and singular-value condition number. |
| `sample_rank_tolerance`, `sample_projection_used` | Sample SVD cutoff and whether it removed a positive sampled singular direction. |
| `theoretical_sample_size`, `sampling_bound_met` | Sufficient VC size and whether the actual draw count reaches it. |
| `restarts`, `max_iterations_per_restart`, `performed_iterations` | Configured and realized search work. |
| `great_circle_sweeps`, `strict_improvements`, `plateau_moves` | Great-circle event neighborhoods explored and accepted move counts across all restarts. |
| `full_candidate_evaluations`, `candidate_count` | Distinct forward class orders fully counted and total forward/reverse candidates considered. Reverse scores are obtained algebraically. |
| `certificate_source`, `certificate_restart`, `certificate_orientation` | Provenance of the candidate with the largest empirical agreement. |
| `chosen_source`, `chosen_restart`, `chosen_orientation` | Provenance of the returned full-data winner; source is `lmrc`, `sample_local_search`, or `empty`. |
| `guaranteed_fraction_of_optimum` | Deterministic `0.5` floor, or the larger conditional candidate bound when the sufficient sample size is met; `1.0` when `P = 0`. |
| `numerical_margin`, `projected_instance` | Checked final inter-class margin and whether global score-space projection was requested. |
| `optimization_runtime_seconds`, `solver_runtime_seconds`, `total_runtime_seconds` | Initialization, search/finalist-selection, and end-to-end timings. This implementation performs no solver optimization, so solver time is zero. |
| `restart_diagnostics` | Per-restart seed type, initial/final empirical score, robustness, neighborhood counts, improvements, stagnation, full score, and runtime. |

The `best_empirical_*` and `selected_empirical_*` fields are deliberately
separate. Full-data selection can prefer a candidate or antipodal orientation
with lower sample agreement. Because its full agreement is at least that of
the empirical certificate candidate, the latter's uniform-convergence lower
bound still transfers to the returned result. For a nonempty population,
`candidate_count` is twice the number of restarts, plus two when LMRC is
enabled; cached duplicate class orders can make `full_candidate_evaluations`
smaller. JSON output adds `"algorithm": "sampled_kendall_local_search"`.

### `DuplicateAwareSampledKendallResult`

The duplicate-aware result preserves the common ranking, full Kendall,
sample/search, finalist provenance, and runtime fields above. Its empirical
scores are floating-point importance-weighted values on a scale whose total is
`sample_size`, rather than raw integer occurrence counts. Additional fields
are:

| Field | Meaning |
|---|---|
| `sampling_mode` | `exact_class_graph`, `sparse_on_demand`, or `empty`. |
| `point_class_count`, `positive_class_comparisons` | Number of exact feature-row classes and number of positive directed class edges. The latter is `null` in sparse mode because computing it would require enumerating the graph. |
| `sample_size`, `accepted_sample_size`, `null_sample_size` | Total proposal attempts, retained optimizable edge draws, and sparse proposals that were non-comparable or rejected by the hard-margin filter. Exact mode has no null draws. |
| `sampled_class_comparisons`, `sampled_comparisons` | Sampled directed class edges and one correctly target-oriented original-row witness for each edge. |
| `sample_multiplicities`, `sample_population_multiplicities` | Proposal draw counts and the full item-level multiplicity `W_e` of each sampled class edge. |
| `sample_proposal_probabilities`, `sample_importance_weights` | Mixture proposal probability `q(e)` and normalized inverse-probability optimization weight for every sampled edge. |
| `objective_mixture_weight`, `class_mixture_weight`, `hard_mixture_weight` | Normalized mixture coefficients actually used. |
| `effective_sample_size` | Standard importance-weight diagnostic, treating repeated proposal draws separately. |
| `sampled_class_coverage`, `sample_collision_rate` | Fraction of point classes incident to at least one sampled edge and fraction of accepted draws that repeat an already sampled class edge. |
| `edge_multiplicity_cache_size` | Number of unordered class pairs whose exact two-way target multiplicities were computed and cached in sparse mode. |
| `pilot_margin_scale` | Median positive absolute LMRC pilot margin used to normalize hard-edge sampling. |
| `class_graph_runtime_seconds`, `sampling_runtime_seconds` | Exact-table construction time (zero in sparse mode) and proposal/aggregation time. |

`sampling_bound_met` is deliberately false for every nonempty run because the
pilot-adaptive mixture is not the theorem's IID uniform sampler.
`guaranteed_fraction_of_optimum` is nevertheless `0.5` because every finalist
is fully evaluated together with its antipodal reverse; it is `1.0` when
`P = 0`. JSON output adds
`"algorithm": "duplicate_aware_sampled_kendall_local_search"`.

### `BaselineRankingResult`

All five functions in `linear_ranking_baselines.py` return this common frozen
dataclass.

| Field | Meaning |
|---|---|
| `algorithm` | One of `linearized_mrc`, `rank_svm`, `linear_ranknet`, `kendall_local_search`, or `elastic_maxfs`. |
| `ranking`, `direction`, `conditioned_scores`, `target_order`, `target_tie_ids`, `ranking_tie_ids` | Flattened weak orders, their ambient witness, and tie metadata as described above. |
| `kendall_agreement`, `kendall_tau`, `total_comparisons` | Counts recomputed on all `M` target-comparable pairs, even when training was sampled. |
| `forced_tie_disagreements`, `optimizable_comparisons` | The `U` forced duplicate ties and `P = M - U` trainable comparisons. |
| `numerical_margin` | Smallest checked margin between adjacent distinct output point classes; zero for one class. |
| `dimension_bound`, `projected_instance` | Modeled-score-space diagnostics shared with the manuscript algorithms. |
| `optimization_iterations` | Method-dependent work count: ridge/CG iterations, Adam epochs, local-search iterations, or MaxFS LP optimizations. |
| `surrogate_objective` | Retained hinge/logistic objective, local-search distance, or MaxFS deleted sample weight; `None` for linearized MRC. |
| `training_sample_size` | Number of optimizable-comparison occurrences used for training. It is `P` in full pairwise/local-search/MaxFS modes and zero for linearized MRC. |
| `unique_training_comparisons` | Number of distinct optimizable training comparisons. |
| `optimization_runtime_seconds` | Time spent constructing/training the heuristic before the common final class-order certificate. |
| `solver_runtime_seconds` | Zero for the four NumPy baselines; for MaxFS, Gurobi time for the elastic/add-back LP solves. MaxFS optimization and solver timings overlap and should not be added. |
| `total_runtime_seconds` | End-to-end wall time. |
| `diagnostics` | JSON-safe method-specific metadata such as optimizer choice, sample mode, local-search moves, or MaxFS deletions/add-backs. |

When `P > 0`, finalization infinity-normalizes the retained baseline direction,
preserves score gaps above a conservative floating-point threshold, and uses a
bounded deterministic generic perturbation to refine numerical ties. The
returned order is verified from both scores and adjacent point differences;
no final witness LP is used. When `P = 0`, it uses the zero-direction fast path.
Consequently, the flattened permutation never invents a comparison unsupported
by the tie IDs and returned direction.

## Guarantees

### Minimum disagreement

Let `K = result.approximation_factor`, and let `OPT_KT(model)` be the optimum in
the score family modeled by the run. When the numerical LP/IIS decisions match
their exact counterparts, the removed disjoint minimal infeasible subsystems
certify

```text
result.kendall_tau
    <= result.deleted_count
    <= K * OPT_KT(model)
    <= (r + 1) * OPT_KT(model).
```

The algorithm is approximate, not an exact Kendall--Tau optimizer. A deleted
comparison can still agree with the returned ranking. Structurally forced ties
are included in `deleted_count` but not in `deleted_comparisons`; since
`approximation_factor >= 1`, adding the same unavoidable `U` term preserves the
displayed approximation inequality.

### Deterministic half approximation

For each of the `P` comparisons between distinct point classes, exactly one of
the constructed class order and its reverse agrees. Therefore

```text
base_agreement + opposite_agreement = P.
```

The returned agreement is at least `P/2`. Since forced ties can never agree,
`OPT_agr <= P`, so the result is at least `OPT_agr/2`. This universal argument
remains valid with exact duplicate rows and target ties. It also remains valid
if optional projection was used: the returned ambient direction is checked to
be strict between original distinct point classes, and its negation reverses
their order while preserving duplicate ties.

### Randomized similarity scheme

The implementation sets `eta = epsilon/4`, `v = dimension_bound`, and chooses
`s` on the decreasing branch `s >= ceil(32 v / eta^2)` such that

```text
8 (2 e s / v)^v exp(-s eta^2 / 32) <= delta.
```

`vc_uniform_sample_size(v, eta, delta)` returns the smallest integer on that
branch satisfying the inequality. This instantiates the universal constant
hidden in the manuscript using a classical VC tail inequality and Sauer's
growth bound; see Theorem 22.18 in
[Wasserman, *All of Statistics*](https://padc-web.obspm.fr/rcsed/data1/springer_books/2004_Book_AllOfStatistics.pdf).

Sampling is independent and uniform with replacement over the `P` optimizable
comparisons. Forced duplicate ties are omitted because no direction can satisfy
them. On the resulting uniform-convergence event, exact empirical optimization
gives

```text
agreement >= OPT_agr - 2 eta P
          >= (1 - epsilon) OPT_agr.
```

The second inequality uses the antipodal lower bound `OPT_agr >= P/2` and
`eta = epsilon/4`.

The event has probability at least `1 - delta` when
`sampling_bound_met=True`, under the theorem's exact assumptions. Since the
same unavoidable `U` term occurs in every distance, the result also implies

```text
Kendall--Tau distance <= OPT_KT + 2 eta P
                     =  OPT_KT + (epsilon/2) P.
```

`sampling_bound_met=True` records only that the sufficient size inequality was
met. It neither certifies that a particular random sample lies in the
uniform-convergence event nor proves that floating-point Gurobi decisions equal
exact rational decisions. In projected mode, the PTAS guarantee is relative to
the projected score family's optimum, not the original family's optimum.

### Sampled greedy-IIS hybrid

Let `S` be the IID comparison multiset of size `s`, let `OPT_dis(S)` be its
minimum realizable disagreement, and abbreviate the hybrid result's empirical
lower bound, forward disagreement, deletion upper bound, and observed IIS
factor by `L`, `D`, `E`, and `K`. The weighted primal--dual IIS accounting and
add-back checks certify

```text
L <= OPT_dis(S) <= D <= E <= K L,
K <= sample_dimension + 1.
```

Consequently `G = D - L = empirical_optimality_gap_upper_bound` upper-bounds
the certificate candidate's additive empirical suboptimality. This is a
sample-level certificate produced by continuous LP/IIS steps; the final
minimum-norm witness uses one continuous convex QP, and the hybrid never solves
the PTAS's empirical binary MIP.
Restarts can improve `D`, `L`, or the full-data candidate but do not change the
shared IID sample.

Every strict forward candidate is evaluated together with its antipodal
reverse. Their agreements sum to `P`, and the output is selected by full-data
agreement, so deterministically

```text
result.kendall_agreement >= P / 2 >= OPT_agr / 2.
```

This `1/2` floor applies for practical manual samples such as 122 or 200 and
does not depend on `epsilon`, `delta`, or empirical optimality. When
`sampling_bound_met=True`, `sample_projection_used=False`,
`projected_instance=False`, `raw_iis_certified=True`, and the VC and numerical
assumptions hold, the empirical gap additionally gives, with probability at
least `1 - delta`, the data-dependent fraction

```text
max(1/2, min(1, 1 - epsilon - 2 G/s)).
```

This is the value returned by `guaranteed_fraction_of_optimum`. If a manual
sample is below `theoretical_sample_size`, or `sample_rank_tol` truncates a
positive sampled direction, the property deliberately returns only `0.5`.
When global `rank_tol` requests projection, the property conservatively returns
only `0.5`; the empirical fields remain qualified by
`empirical_certificate_scope`. The method remains a heuristic rather than the
similarity PTAS because weighted greedy IIS does not exactly optimize the sample
in general.

### PTAS-sampled Kendall local search

Every forward finalist and the optional LMRC candidate are retained together
with their antipodal reverses. For each such pair the two full-data agreements
sum to `P`. Because selection uses the exact full-data agreement of these
finalists, the returned direction deterministically satisfies

```text
result.kendall_agreement >= P / 2 >= OPT_agr / 2.
```

This floor is independent of `epsilon`, `delta`, the sample size, and whether
the local search reaches a sample optimum. It therefore still applies to the
practical 122- and 200-draw configurations.

Let `A_S` be `best_empirical_agreement / sample_size`, the sampled agreement
rate of the explicitly retained empirical-certificate candidate. If
`sampling_bound_met=True`, the uniform-convergence event used by the sampler
holds with probability at least `1 - delta` under its stated assumptions. On
that event the certificate candidate's population agreement rate is at least
`A_S - eta`. This remains valid even though the candidate was chosen after
seeing the sample, because the event is uniform over the modeled direction
family. The final output is then chosen by exact full-data agreement from a
set containing that candidate, so it cannot have lower population agreement.
Since `OPT_agr <= P`, the result property returns the corresponding fraction

```text
max(1/2, min(1, best_empirical_agreement / sample_size - eta)).
```

This is a **candidate generalization bound**, not an empirical-optimality
certificate and not a `(1 - epsilon)` PTAS guarantee. The event says how a
particular retained candidate transfers from sample to population; it does not
say that the bounded multistart search found the best sampled direction. If a
manual sample is smaller than `theoretical_sample_size`, including the common
122- or 200-draw cases, `guaranteed_fraction_of_optimum` deliberately reports
only the deterministic `0.5` floor.

### Experimental baselines

The five baselines provide empirical comparators and do not claim approximation
factors or global surrogate optimality. Adam retains the best observed training
objective, local search retains the best visited full-data Kendall weak order, and
MaxFS returns a feasible subsystem found by elastic deletion and bounded greedy
add-back. In full unweighted MaxFS mode, the implementation checks that the
returned distance after subtracting `U` is no larger than the number of finally
deleted optimizable comparisons. In sampled mode it checks the analogous
weighted disagreement only on the sampled `P`-population multiset. Neither
check makes the deleted subsystem optimal.

## Optimization and complexity notes

### Closest ranking

- Scores are factorized as `score = B @ q`; each comparison has two nonzeros
  rather than a dense difference row.
- Exact duplicate rows are grouped once. Target ties and the `U` structurally
  forced duplicate comparisons never enter the LP.
- A target-adjacent fast path uses only `n - 1` comparisons for the legacy
  strict, duplicate-free target when it is realizable.
- One persistent Gurobi model is warm-started across IIS-deletion rounds.
- A sparse IIS candidate is reduced by a comparison-only deletion filter to
  the inclusion-minimal subsystem needed by the proof.
- The final distinct-point-class order is checked with adjacent margin-one
  constraints; duplicate items inherit their class score.

The generalized full comparison model has `P` rows rather than all
`n(n - 1)/2` pairs. IIS extraction and one LP resolve per candidate comparison
can still dominate runtime.

### Half approximation

- Descending lexicographic order of the `k` distinct point classes in the
  conditioned basis is deterministically linearly realizable.
- One tie-aware Fenwick count obtains its full-data agreement in
  `O(n log k)`; the reverse complements it on `P`. No full pair list is
  materialized.
- One bounded maximum-margin LP with `k - 1` adjacent class inequalities
  constructs a stable ambient witness for the chosen orientation.

Apart from the LP, preprocessing uses `O(n d)` storage and approximately
`O(n d log n)` comparison work.

### Randomized similarity scheme

- Pair IDs are sampled in bounded chunks from `P` without constructing its pair
  list. The compact decoder skips target ties and exact-duplicate pairs.
- Repeated sampled pairs are collapsed into integer multiplicities.
- The empirical model has one binary variable per distinct sampled comparison,
  not per draw.
- Score variables are created only for sampled items and linked through
  `score = B @ q`.
- Gurobi indicators encode `selected => score[higher]-score[lower] >= 1`; the
  weighted objective is solved with zero relative and absolute MIP gaps.
- One bounded maximum-margin LP stabilizes a witness for the selected empirical
  comparisons. Sixteen reproducible generic perturbations are scaled to remain
  inside that selected-comparison cone, and the numerically strongest witness
  that strictly orders distinct classes is returned. If all sixteen trials have
  a numerical class tie, the search expands to at most 128 trials. This avoids
  a second `k - 1`-row class-order LP.
- Full-data weak-order agreement uses one `O(n log k)` count instead of all `M`
  target-comparable pairs.

The indicator MILP is an optimized equivalent of the theorem's `2^s` subset
enumeration, but its worst-case complexity remains exponential. The defaults
guard at most 10,000,000 draws and 10,000 unique comparisons/MILP binaries.
The absolute sample ceiling is `2^53 - 1`, because integer objective weights
must be exactly representable in `float64`. The total guard is checked before
sampling; the unique guard can stop sampling as soon as it is exceeded.
Disabling guards does not make a large computation practical, and returned JSON
can itself be large because it includes the sampled comparison certificate.

For `k` distinct point classes, the default
`dimension_bound = min(number of varying features, k - 1)` safely upper-bounds
the effective dimension without a potentially unsafe numerical-rank decision.
It can exceed the true `r`, making the sufficient sample larger than the
theorem's ideal rank-aware bound.

### Sampled greedy-IIS hybrid

- Pair IDs are drawn IID with replacement from `P` using the PTAS sampler;
  target ties and structurally forced same-point-class comparisons never enter
  the sample.
- Repeated item comparisons are collapsed to multiplicities, then comparisons
  with the same directed point-class endpoints are aggregated again. The LP
  therefore has one weighted row per distinct sampled class comparison.
- A thin SVD represents only the row span of sampled comparison differences,
  so the continuous model dimension is at most the number of distinct sampled
  class comparisons even when the ambient feature dimension is much larger.
  With `sample_rank_tol=0`, every positive returned singular value is retained.
- Each restart maintains one continuous feasibility model. On an infeasible
  solve it obtains a comparison-minimal IIS, raises that IIS's dual weight by
  the smallest residual multiplicity, and disables newly exhausted rows.
  There are no indicator constraints, binary variables, or branch-and-bound
  nodes.
- Without explicit sampled-rank truncation, every reduced-space IIS is
  independently rechecked through the Gram matrix of its precompression rows;
  this uses at most the IIS size in auxiliary variables rather than the ambient
  feature dimension.
- Greedy add-back re-enables up to `max_add_back` deleted comparisons in
  descending multiplicity order, retaining each one only if the subsystem
  remains feasible. Add-back cannot worsen the deletion certificate.
- One final minimum-L2-norm convex QP chooses a stable hard-margin witness for
  the retained sample instead of returning an arbitrary feasibility vertex.
- Restarts reuse the same sample and change only constraint insertion order.
  Every strict forward result is scored on all `M` comparable pairs in
  `O(n log k)`; its reverse score is obtained algebraically, and repeated class
  orders share a cached evaluation. The best full-data candidate is returned.

The default experimental workload is three restarts, 200 draws, and at most 64
add-back attempts per restart. A 122-draw run is smaller still. Runtime can
nevertheless be dominated by repeated IIS extraction on numerically delicate
samples; the hybrid trades the PTAS's worst-case exponential binary search for
a bounded sample of continuous LP/IIS work, not for a universal runtime bound.
The draw and distinct-item guards default to 1,000,000 and 100,000.

### PTAS-sampled Kendall local search

- It uses the PTAS pair-ID sampler, so `s` IID draws are decoded without
  materializing the `P`-pair population. Repeated item comparisons and then
  repeated directed class comparisons are aggregated into integer weights.
- A thin SVD restricts the iterative search to the row span of the `u` unique
  sampled class-comparison differences. Its dimension `r_s` is at most `u`,
  even when the ambient data are high-dimensional.
- Each move computes the two zero-crossing events of every active sampled
  comparison on one chosen great circle, sorts and groups those events, and
  sweeps the resulting open cells by integer objective deltas. In
  nondegenerate exact arithmetic this identifies the best cell on that circle.
  The numerical path orders boundary rays without trigonometric angles, groups
  rays that are collinear in working precision, and explicitly recomputes the
  top eight predicted cells before accepting a move.
- Restarts mix projected-LMRC, weighted-centroid, antipodal, and Gaussian
  seeds. Iterations alternate coordinate, violated-pair,
  aggregate-violation, and periodic Gaussian tangents. Agreement never
  decreases; a lower-decile margin criterion permits numerically stronger
  plateau moves, and `patience` stops a stagnant restart.
- Only the iterative moves are sample-only. When enabled, LMRC first makes a
  separate full-item pass and ridge solve; its dense/dual or matrix-free-CG
  cost can dominate initialization on a large dataset. Full Kendall evaluation
  is then limited to that LMRC candidate and one forward finalist per restart.
  Reverse agreements are complements, and a class-order cache avoids repeated
  full counts. Use `--no-lmrc-seed` to remove both the LMRC initialization and
  its extra finalist.
- The method creates no binary variables, MIP, branch-and-bound tree, or Gurobi
  model. Accordingly `solver_runtime_seconds` is always zero.

Let `r_g` be the global conditioned score dimension. Constructing and
factorizing the sampled difference matrix uses `O(u r_g)` storage and roughly
`O(u r_g min(u, r_g))` thin-SVD work. A great-circle sweep costs
`O(u r_s + u log u)`: matrix-vector products contribute the first term and
sorting at most `2u` events contributes the second. Thus at most `R` restarts
and `I` sweeps per restart cost

```text
O(R I (u r_s + u log u))
```

after sampling and factorization, with early termination often reducing the
realized count. At most `R + 1` forward class orders receive full-data scoring
when LMRC is enabled, each in `O(n log k)` for `k` distinct point classes;
disabling LMRC with `--no-lmrc-seed` removes its initialization and finalist.
The default workload is `R = 8`, `I = 250`, `s = 200`, and patience `40`; a
122-draw run reduces `u` further. The theoretical VC size can still make the
SVD, event sweeps, and JSON sample certificate large. The guards prevent
accidental oversized runs but do not make such runs scalable, and this bounded
local search has no global runtime or empirical-optimality guarantee.

### Experimental baselines

- Linearized MRC assigns a common midrank response within each target tie and
  forms the centered rank statistic without enumerating pairs.
  It solves a dense primal or dual ridge system when the smaller relevant
  dimension is at most 512 and otherwise uses diagonally preconditioned,
  matrix-free conjugate gradients. Each matrix-free multiply costs `O(n d)`.
- Full RankSVM evaluates the exact `P`-population hinge objective and a valid
  subgradient in `O(n d + n log n)` time and `O(n + d)` extra storage per
  epoch. Target-group-batched Fenwick sweeps aggregate endpoints of all
  margin-violating eligible pairs while class counts exclude forced duplicates,
  so neither pair differences nor all `P` pair IDs are materialized.
- Sampled RankSVM and RankNet draw comparisons IID with replacement. With the
  default full-batch update, repeated sampled pair IDs are collapsed to
  multiplicities and gradients are accumulated at comparison endpoints before one matrix
  multiply. Positive `batch_size` instead uses shuffled mini-batches. RankNet's
  stable logistic calculation avoids exponential overflow. Both pairwise
  optimizers start from the better-surrogate choice between zero and a bounded
  linearized-MRC warm start.
- Kendall local search starts from linearized MRC, its reverse, and seeded
  Gaussian directions. Coordinate proposals update scores in `O(n)` before a
  tie-aware `O(n log k)` count; every tenth iteration also tests a Gaussian
  proposal. The unavoidable `U` term is constant. Non-worsening moves can
  traverse objective plateaus, while the stored incumbent never worsens.
- MaxFS keeps one score-factorized elastic LP and changes elastic objective
  coefficients rather than rebuilding rows. It removes the largest positive
  raw slacks in batches, reoptimizes the warm-started LP, and tests deleted rows
  in reverse order. Its LP has one elastic row per distinct selected
  optimizable comparison; repeated sampled comparisons become integer weights.

Every nontrivial baseline finishes with a directly verified generic refinement
of its direction. Complete-data agreement and distance are then computed by a
tie-aware `O(n log k)` count.

## Numerical safeguards and limitations

The manuscript uses exact rational feasibility and optimization; Gurobi uses
floating-point arithmetic. The code validates every available certificate and
raises on unresolved statuses or failed checks:

- Input conversion must preserve coordinates in `float64`.
- Target ranks must be a finite numeric length-`n` vector whose conversion to
  `float64` does not merge or alter labels.
- Default preprocessing uses exactly checked translations and reversible
  powers-of-two feature scaling.
- Constant features are removed. Exact duplicate rows are grouped and checked
  to remain tied; at least two distinct classes are needed only for a nontrivial
  strict class-order solve.
- Features with more than 70 bits of dynamic range after tested exact
  translations are rejected with a rescaling error.
- Per-column scaling cannot remove near-affine dependence between different
  feature columns. If Gurobi reports a numerical certificate failure on such
  an instance, rescale/re-express the features or, when modeling the
  numerically projected score family is acceptable, retry with `rank_tol=0`
  to use SVD coordinates without intentionally truncating a positive singular
  value.
- Returned class orders are checked against the original coordinates, with
  exact rational fallback for cancellation-sensitive adjacent class comparisons.
- Agreement plus distance is checked against `M`.
- Every baseline validates finite optimizer output, preserves forced duplicate
  ties, protects robust score separations, replaces numerical ties by a
  realizable generic perturbation, and directly checks every adjacent class gap.
- Pairwise baselines check finite nonnegative retained surrogate values and
  exact sample-size accounting. MaxFS additionally checks that its final
  ranking preserves every active comparison and that sampled disagreement is
  covered by the finally deleted sample weight.
- The PTAS checks that samples contain neither target ties nor duplicate-point
  pairs, as well as sample multiplicities, an integral optimal objective and
  bound, selected-comparison preservation, inter-class strictness of the
  perturbed witness, and equality of final empirical agreement with the MIP
  optimum.
- The hybrid checks IID sample accounting after both aggregation levels, the
  sampled-difference SVD map, precompression IIS infeasibility, minimal-IIS
  size bounds, weighted primal--dual inequalities, add-back feasibility,
  preservation of every active sampled row, forward/reverse complementarity,
  and the final full-data `1/2` floor.
- The sampled Kendall local search checks both levels of sample accounting,
  the sampled-difference SVD map, finite normalized seeds and margins, direct
  integer recounts of the best event-sweep intervals, monotonicity of the
  retained sampled objective, preservation of active sampled rows during
  strict refinement, forward/reverse complementarity, final full-data counts,
  and the antipodal `1/2` floor.
- Interrupted, limited, unbounded, suboptimal, or numerically unresolved solves
  raise rather than return a claimed guarantee.

These checks do not turn Gurobi into an exact rational solver. In particular,
LP infeasibility/IIS decisions and MIP optimality remain tolerance-based even
with `MIPGap=0`. Verifying a returned direction also cannot prove that numerical
optimization did not miss a superior empirical sign pattern. Interpret the
theorem guarantees conditionally: they apply when the solver decisions match
the corresponding exact systems. Fixed seeds do not ensure identical low-level
results across Gurobi versions, platforms, and thread counts.

The baseline surrogate optimizers are deliberately bounded heuristics: a fixed
epoch, restart, deletion, or add-back limit can stop before a better solution
is found. Their diagnostics report the work performed but do not constitute an
optimality certificate. Linearized MRC, RankSVM, RankNet, and full Kendall
local search now run completely in NumPy; only MaxFS uses Gurobi for its elastic
search.

The hybrid's restarts and add-back limit are likewise heuristic search
controls. Its weighted primal--dual fields are a genuine certificate for the
sampled forward candidate, but they do not assert that the full-data-selected
orientation is sample-optimal. The stronger population bound is exposed only
when the theoretical sample-size condition is met, neither global nor sampled
projection was requested, and the precompression IIS checks succeeded.

The sampled Kendall search's restart, iteration, tangent schedule, and patience
limits are also heuristic controls. A sweep covers all predicted open sign
cells only on its chosen great circle, not all directions. The angle-free
boundary-ray ordering avoids `atan2` resolution loss, but rays collinear in
working precision are grouped and only the best eight predicted cells are
directly recounted. Thus “exact line search” should be understood as the
nondegenerate exact-arithmetic ideal of the numerically verified event sweep.
Neither the sweep nor multistart search
certifies the empirical optimum. Its stronger result property is instead the
candidate-quality uniform-convergence bound described above, and is exposed
only when `sampling_bound_met=True`; a 122- or 200-draw run below the theoretical
size reports the deterministic `0.5` floor.

For theorem-faithful geometry, leave `rank_tol=None`. Supplying `rank_tol`
performs an SVD projection and changes the modeled score family. Closest-ranking
and PTAS guarantees then apply to that projected family, as recorded by
`projected_instance=True`. Projection is rejected if it collapses two originally
distinct point classes; original duplicate classes remain tied. The
half-approximation's universal antipodal guarantee still holds for the original
points once its returned direction has passed strict inter-class
original-coordinate verification. The hybrid has a separate
`sample_rank_tol`; positive-value truncation there is recorded by
`sample_projection_used=True` and suppresses its stronger data-dependent
fraction, while leaving the antipodal `1/2` floor intact. The sampled Kendall
search reports the same sample-SVD diagnostics; its conditional property is a
bound on one retained candidate rather than an empirical-optimality bound and
therefore does not add the hybrid's no-projection/IIS preconditions.

## Validation

The three manuscript implementations have been exercised with Gurobi 13.0.1
on realizable and non-realizable targets, nonidentity strict targets, tied
target ranks, exact duplicate rows within and across target levels,
one-dimensional and collinear data, opt-in projection, duplicate sample draws,
the `P = 0` case, and the `n = 1` case. Small random instances were checked
against independently enumerated feasible-ranking optima, and the sample-size
helper was checked directly against its stated logarithmic tail inequality.

The five baseline paths were also exercised on full and sampled optimizable
comparison sets, nonidentity and tied targets, duplicate rows, projected
instances, and the `n = 1` case. Random cross-method sweeps verified weak-order
ambient realization, tie-ID and permutation validity, forced-disagreement
accounting, JSON serialization, and `agreement + distance = M`; the implicit
full-population hinge and sampled logistic/hinge gradients were separately
checked against explicit eligible-pair enumeration and finite differences on
small instances.


# Polynomial-function ranking experiments

This folder implements the six primary polynomial baselines discussed in the
conversation, an additional inexpensive LMRC baseline, duplicate-aware sampled
Kendall local search, and the greedy-IIS approximation for minimum disagreement.
All methods use a scalar polynomial score of **total degree at most `p`**, with
arbitrary positive or negative coefficients. The default is `p = 2`.

The existing linear ranking engines in the parent `Code` directory provide
weak-ranking evaluation, duplicate handling, robust score refinement, and
Gurobi IIS/MaxFS routines. Keep this folder directly under `Code`; it is not a
standalone distribution. Existing linear implementation files are unchanged.

## Files and dependencies

| File | Purpose |
|---|---|
| `polynomial_common.py` | Polynomial features, memory guards, serialized model, prediction, shared CLI |
| `polynomial_ranking_baselines.py` | RankSVM, polynomial-score RankNet, preference RankRLS, smoothed MRC, full Kendall local search, Elastic MaxFS, LMRC |
| `sampled_polynomial_kendall_local_search.py` | Duplicate-aware importance sampling and sampled great-circle local search; uniform-pair ablation |
| `closest_polynomial_ranking_gurobi.py` | Full-instance minimum-disagreement IIS approximation |
| `test_polynomial_ranking.py` | Feature, gradient, weak-order, sparse-sampling, model, and CLI tests |
| `example_points.csv`, `example_target_ranks.csv` | Small example with both exact duplicates and target ties |

Use Python 3.10+ and NumPy:

```bash
python3 -m pip install numpy
```

Only Elastic MaxFS and the nontrivial minimum-disagreement IIS algorithm need
`gurobipy` and a working Gurobi license:

```bash
python3 -m pip install gurobipy
```

There is no SciPy, scikit-learn, RLScore, or neural-network dependency. NumPy
baselines and sampled search do not invoke Gurobi during finalization.

## Quick start

Run these commands from `Code/polynomial-function`:

```bash
python3 polynomial_ranking_baselines.py ranksvm example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output ranksvm.json

python3 polynomial_ranking_baselines.py ranknet example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output ranknet.json

python3 polynomial_ranking_baselines.py rankrls example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output rankrls.json

python3 polynomial_ranking_baselines.py smoothed-mrc example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output mrc.json

python3 polynomial_ranking_baselines.py local example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output local.json

python3 polynomial_ranking_baselines.py lmrc example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output lmrc.json

# These two require Gurobi for this nontrivial example:
python3 polynomial_ranking_baselines.py maxfs example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output maxfs.json

python3 closest_polynomial_ranking_gurobi.py example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 --output iis.json

python3 sampled_polynomial_kendall_local_search.py example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 \
  --sample-size 200 --restarts 8 --max-iterations 250 --seed 59 \
  --output sampled.json
```

For the existing dataset, starting from the parent `Code` directory:

```bash
python3 polynomial-function/sampled_polynomial_kendall_local_search.py cleaned.csv \
  --degree 2 --sample-size 200 --restarts 8 --max-iterations 250 \
  --max-class-matrix-entries 0 --seed 59 --output polynomial-sampled.json
```

This last command assumes the CSV row order is the desired target ranking.
Supply `--target-ranks-file PATH` if that is not true. It does not infer ranks
from a CSV feature column.

Use `--help` on each script or baseline subcommand for its options.

## Input, duplicates, and ties

Inputs are finite numeric, **headerless** `n x d` feature matrices in CSV,
whitespace text (`--delimiter whitespace`), or `.npy` format. Do not include a
rank/label column among the features. Without a target option, original row
order is the target, highest first. Three mutually exclusive options are
available:

- `--target-order 2,0,1,...`: zero-based permutation, highest first.
- `--target-ranks 2,0,1,...`: numeric rank per original row; smaller is better
  and equality means a target tie.
- `--target-ranks-file PATH`: the same numeric rank vector, aligned to rows.

The conventions exactly match the linear implementations:

- Target-tied pairs are excluded from the comparison population.
- Identical feature rows always get identical scores and prediction tie IDs.
- A target-strict pair of identical rows is an unavoidable disagreement,
  included in `forced_tie_disagreements` and excluded from optimization.
- Duplicate multiplicities are retained: neither training nor evaluation
  silently changes to one vote per distinct point.
- Full-data counts satisfy `kendall_agreement + kendall_tau == total_comparisons`.
- `ranking` is a flattened permutation for compatibility. `ranking_tie_ids`,
  aligned with original rows, is the authoritative weak ranking.
- Entirely tied targets return zero coefficients and one predicted tie group.

Including every degree-one monomial prevents collisions such as `x` and `-x`
becoming identical merely because their squares agree. Numerical ties between
distinct lifted point classes are refined with the existing solver-free,
verified generic-witness routine. Failures are reported, not hidden by random
tie-breaking in the reported objective.

## Polynomial representation and numerical limits

For input dimension `d` and maximum degree `p`, the feature count is

```text
D_p = binomial(d + p, p) - 1.
```

For two input coordinates and `p=2`, the feature order is
`x_0, x_1, x_0^2, x_0*x_1, x_1^2`. In general, increasing total degree is
followed by lexicographic combinations-with-replacement of coordinate indices.
There is no constant feature because it cancels from comparisons.
Items are lifted first; comparisons are `Phi(x_i) - Phi(x_j)`, NOT
`Phi(x_i - x_j)`.

Default `--input-scaling power2` reversibly rescales each original coordinate
by a power of two so that its maximum absolute training value is below one.
This diagonal transformation preserves the unrestricted polynomial family in
exact arithmetic. Scaling that loses input information, nonfinite products,
and monomials underflowing from nonzero to zero are rejected. Use
`--input-scaling none` only when the raw polynomial products are well scaled.
Each monomial reuses a lower-degree prefix, taking `O(n D_p)` multiplications.

Default `--feature-basis monomial` uses unweighted monomials. Optional
`--feature-basis polynomial-kernel` uses positive multinomial weights with

```text
<Phi(u), Phi(v)> = (1 + <u,v>)^p - 1,
```

where `u` and `v` are scaled inputs. Both bases have the same unrestricted
polynomial hypothesis family, but different regularization geometry. RankSVM,
RankNet, and RankRLS preserve L2 regularization in the chosen lifted feature
basis by default; they do not undo the kernel weights by applying further
column normalization. Thus kernel mode implements the polynomial-kernel
surrogate in explicit primal features without an `n x n` kernel matrix.
Scaling inputs changes that kernel's geometry; use consistent scaling across
experiments and report it. Finite optimizer iterations are not an exact
surrogate-optimality certificate.

All products and optimization use **float64**. The manuscript defines exact
rational polynomial products; floating-point multiplication is a numerical
approximation to that map. Ranking certificates here concern the computed
float64 feature vectors, as reported by `certificate_instance`. They are not
exact-rational certificates for symbolic polynomial evaluation. A returned
model is an evaluable polynomial, but on cancellation-heavy data its exact
symbolic order can differ from the rounded-feature order. This distinction
also applies to the IIS approximation guarantee.

## Baseline objectives

Let `m_ij = a dot (Phi(x_i) - Phi(x_j))` for an eligible target preference
`i > j`. Pair averages always retain original-item weights.

| Subcommand | Objective / optimization |
|---|---|
| `ranksvm` | Minimize mean `max(0, 1 - m_ij) + lambda*||a||^2/2`; Adam/subgradient iterations |
| `ranknet` | Minimize mean `log(1 + exp(-m_ij)) + lambda*||a||^2/2`; stable logistic loss and Adam |
| `rankrls` | Minimize mean `(1 - m_ij)^2/2 + lambda*||a||^2/2`; matrix-free preconditioned conjugate gradients |
| `smoothed-mrc` | Maximize mean `sigmoid(m_ij/h)` on the unit sphere; multistart projected ascent, backtracking, decreasing bandwidths |
| `local` | Direct multistart coordinate/random search on full-data Kendall disagreement |
| `maxfs` | Elastic comparison-slack LP, deletion of heavily violated rows, then bounded greedy add-back |
| `lmrc` | Existing ridge-linearized rank-correlation/midrank fit on polynomial features |

RankNet here means its pairwise loss with a **fixed-degree polynomial score**,
not an unrestricted neural network. Smoothed MRC is explicitly a logistic
smoothing adaptation motivated by the literature; it is not a reproduction of
the self-induced Gaussian smoothing/inference algorithm. Unlike RankNet it
maximizes the sum of sigmoids, not the sum of log-sigmoids.

RankRLS uses unit preference differences, not the magnitudes of numeric rank
labels. By default it trains on the full eligible preference graph. Its
Laplacian uses target-group, duplicate-class, and class/target-group sums to
exclude target ties and forced duplicate comparisons in `O(n)` operations.
Each CG matrix-vector product costs `O(n D_p)`; no all-pairs or kernel matrix
is constructed. `normal_equations_converged` and the relative residual reveal
whether the CG budget sufficed.

RankSVM has a full-data hinge mode using the existing tie-aware endpoint-count
gradient, costing `O(n log n + n D_p)` per epoch without enumerating pairs.
RankNet and smoothed MRC default to at most 100,000 uniform eligible original
item-pair draws. Repeated pairs keep their draw counts. For bounded training:

```bash
python3 polynomial_ranking_baselines.py ranknet example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 \
  --max-training-pairs 200 --epochs 100 --regularization 0.001

# Full hinge mode remains implicit rather than allocating all pairs:
python3 polynomial_ranking_baselines.py ranksvm example_points.csv \
  --target-ranks-file example_target_ranks.csv --max-training-pairs 0

# Full RankRLS is already the default; use a sample only for a matched-budget ablation:
python3 polynomial_ranking_baselines.py rankrls example_points.csv \
  --target-ranks-file example_target_ranks.csv --max-training-pairs 200
```

For RankNet/MRC, `--max-training-pairs 0` requests full pair materialization and
is additionally limited by `--max-materialized-pairs` (default 1,000,000).
For RankRLS, zero means the implicit full graph. For MaxFS, zero requests the
full comparison LP and can be expensive. These zeros have no connection to
the sampled search's `--sample-size 0` described below.

## Sampled polynomial local search

The method first builds polynomial features, computes a full-data polynomial
LMRC pilot, and samples eligible preferences. Default proposal weights are:

```text
objective/item-level component: 0.50
class-balanced component:      0.25
pilot-hard component:          0.25
```

The exact small-class mode uses directed class-edge multiplicities. When the
`k x k` class table exceeds `--max-class-matrix-entries` (default 4,000,000),
sampling switches automatically to sparse on-demand multiplicity queries.
`--max-class-matrix-entries 0` forces this mode and avoids the large-table
failure previously encountered on mostly unique datasets. Sparse class/hard
proposals can be null; `accepted_sample_size + null_sample_size == sample_size`.
The item-level component guarantees comparison coverage in the proposal.

Inverse-proposal weighting preserves the original item-level objective; it
does not redefine the score as an unweighted class-level agreement. Aggregated
weights are normalized by one common factor to sum to the requested sample
budget. A thin SVD of the sampled differences reduces the search dimension to
at most the number of distinct sampled preferences. Great-circle event sweeps
search the weighted sampled Kendall objective, with direct numerical checks
before accepting moves. Full-data evaluation is used only for finalists and
the pilot, including antipodal reverses.

The search is a **heuristic, not the similarity PTAS**: local search does not
prove empirical optimality. Comparing each strict class-order finalist with
its reverse gives at least half of the optimizable comparisons, hence at least
half of optimum agreement for the computed feature instance. Target-strict
duplicate comparisons cannot be agreed with in either orientation.

`--sample-size 200` means 200 comparison draws/proposal attempts, not 200
points. `epsilon` and `delta` only supply the reference uniform-sample-size
diagnostic when a fixed sample budget is specified; they do not certify this
heuristic. `sampling_bound_met` is conservatively false on nonempty instances,
including the uniform ablation. `--sample-size 0` requests the sufficient
reference theoretical size, which can be very large and is guarded by
`--max-sample-size` (default 1,000,000).

For a uniform original-item-pair ablation using the same hardened search:

```bash
python3 sampled_polynomial_kendall_local_search.py example_points.csv \
  --target-ranks-file example_target_ranks.csv --degree 2 \
  --sampling uniform --sample-size 200 --seed 59
```

Uniform mode uses the objective-only mixture `(1,0,0)` and sparse decoding;
class/hard mixture flags and the exact-class-table threshold are ignored.
`--no-lmrc-seed` removes the pilot from search seeds/finalist selection, but
the pilot is still computed for the common sampling path.

## Minimum-disagreement approximation: the dimension is lifted

The algorithm runs the parent greedy minimal-IIS deletion method on **all**
eligible lifted comparisons. Its score-factorized LP stores two-nonzero pair
constraints rather than a dense pair-by-polynomial-feature matrix. Persistent
models, an adjacent-target feasibility fast path, minimal-IIS filtering, and
strict-class witness verification are inherited from the linear engine.

The theorem's factor is `r_p + 1`, where `r_p` is the rank of lifted feature
differences, bounded by `min(D_p, k-1)` for `k` distinct input rows. It is NOT
`d`, the original number of attributes. For example, three attributes at
degree two give `D_p=9` and a safe upper bound of 10, not 3.

The implementation avoids an expensive exact rank computation. It reports
`dimension_bound` as a safe bound and `approximation_factor` as the largest
minimal-IIS block actually removed (or one if none). In exact feasibility
arithmetic the packing certificate is

```text
kendall_tau <= deleted_count <= approximation_factor * OPT_KT,
approximation_factor <= r_p + 1 <= D_p + 1.
```

Forced duplicate disagreements are included in `deleted_count` separately
from the explicit IIS blocks. The implementation verifies available numerical
certificates and errors on unresolved solver status or witness failure. See
the float64 feature-map limitation above.

This full-instance algorithm is not a sampled large-data heuristic. Default
`--max-modeled-comparisons 2000000` guards the LP size; zero disables that guard.
`--time-limit SECONDS` sets a per-solve Gurobi limit, not an overall runtime
deadline; an interrupted solve is an error, not an approximation result.
Use sampled local search or bounded MaxFS when this full comparison population
is impractical, and label those results as heuristics.

`--rank-tol` explicitly requests SVD projection and changes the comparison
space. Any approximation certificate then applies only to the projected
instance. Leave it unset for the full computed polynomial family. Similarly,
a positive `--sample-rank-tol` may truncate the sampled search span.

## Comparable JSON output and saved models

Every result includes the linear-compatible fields `ranking`, `direction`,
`conditioned_scores`, `target_order`, `target_tie_ids`, `ranking_tie_ids`,
`kendall_agreement`, `kendall_tau`, `total_comparisons`,
`forced_tie_disagreements`, `optimizable_comparisons`, `dimension_bound`,
`solver_runtime_seconds`, and `total_runtime_seconds`, plus method diagnostics.

Polynomial results add `degree`, `input_dimension`, `expanded_dimension`,
`feature_construction_runtime_seconds`, `ranking_engine_runtime_seconds`,
`polynomial_feature_arithmetic`, `certificate_instance`, and `polynomial_model`.
Total time includes feature construction and algorithm execution, but excludes
CLI file loading. JSON is printed when `--output` is
omitted; `--output PATH` saves it. Solver logs are disabled by default so stdout
is parseable JSON; enabling `--solver-log` can prepend a solver log.

**`direction` has length `D_p`, not `d`.** Its entries multiply the model's
lifted/scaled features. `polynomial_model` records coefficient values, repeated
coordinate indices for each monomial, kernel weights, and training input
power-of-two scale exponents. A coefficient on the original raw monomial is

```text
direction[t] * feature_weights[t] * 2**sum(input_scale_exponents[j] for j in term[t]).
```

Keeping the factored representation avoids coefficient overflow/underflow.
To predict using exactly that representation:

```python
import json
import numpy as np
from polynomial_common import predict_polynomial_scores
from polynomial_ranking_baselines import polynomial_rank_rls_baseline

points = np.loadtxt("example_points.csv", delimiter=",")
ranks = np.loadtxt("example_target_ranks.csv", delimiter=",")
result = polynomial_rank_rls_baseline(points, target_ranks=ranks, degree=2)
scores = predict_polynomial_scores(points, result)
saved = result.to_dict()
# The same helper accepts a dictionary loaded from a result JSON file.
```

For cancellation-heavy training data, use `conditioned_scores` and
`ranking_tie_ids` for the verified lifted-feature order; a direct float dot
product can appear tied. Prediction points reuse training scaling and can
still overflow/underflow if far outside the training range; such errors are
reported explicitly.

## Memory and experiment recommendations

Sampling bounds the comparison-search problem, not feature construction.
The full `n x D_p` matrix is still built to support the pilot and full-data
evaluation. Default guards are `--max-features 10000` and
`--max-feature-entries 25000000`. The latter is approximately 191 MiB for the
feature matrix alone; conditioning, indexing, and optimization need additional
working memory. Zero disables either guard. Avoid increasing them solely to
silence an error without checking available memory.

Start with degrees `1, 2, 3`; degree one is the linear-family control. Use the
same degree, scaling, feature basis, duplicate/tie convention, and full-data
evaluator for all methods. Tune regularization and search budgets, report
multiple seeds, and compare both matched-pair-budget and full-data baselines.
Report end-to-end runtime including feature construction, agreement,
disagreement, forced duplicate count, sampled effective sample size, and
accepted/null proposal counts. Full local search is an important sampling
ablation but makes repeated `O(n log n)` full-ranking evaluations.

TensorSketch and wider nonlinear comparators (neural RankNet, conventional
RankBoost) are not included here: they were optional additional comparators,
not the six same-family baselines. A sketch restricts/approximates the model
and would need separately labeled experiments; trees/neural networks need not
be polynomial scores of the matched degree.

## Tests

```bash
python3 -m unittest discover -s . -p 'test_polynomial_ranking.py' -v
```

Tests cover monomial order, kernel identity, degree-one equivalence, power-of-two
scaling, feature guards, underflow rejection, implicit preference-graph counts,
RankRLS normal equations, MRC gradients, duplicates/target ties, sparse versus
exact sampling, the antipodal floor, model round trips, and CLI help. The
nontrivial Gurobi test is skipped if the package or license is unavailable.
Zero-comparison Gurobi-algorithm fast paths are tested without a solver.

## Literature and attribution

- Polynomial RankSVM is an established nonlinear ranking baseline:
  [Joachims's SVM-rank documentation](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)
  supports polynomial kernels; [Raykar, Duraiswami, and Krishnapuram (AISTATS
  2007)](https://proceedings.mlr.press/v2/raykar07a.html) includes quadratic
  RankSVM experiments. Here its pairwise hinge objective is fitted in explicit
  features with the parent optimizer, not by calling SVM-rank's executable.
- [Burges et al., *Learning to Rank using Gradient Descent* (ICML
  2005)](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)
  introduces the RankNet pairwise probabilistic loss. Restricting its score to
  `a dot Phi_p(x)` is the fixed-polynomial adaptation used here.
- [Pahikkala et al., *Learning to Rank with Pairwise Regularized Least-Squares*
  (SIGIR LTR workshop 2007)](https://staff.cs.utu.fi/~aatapa/publications/inpPaTsAiBoSa07a.pdf)
  and the [authors' RLScore ranking tutorial](https://staff.cs.utu.fi/~aatapa/software/RLScore/tutorial_ranking.html)
  describe RankRLS and its preference-graph variant. This implementation uses
  unit directed preferences and our existing full-data tie-aware evaluator,
  not RLScore's random prediction-tie convention.
- [Zhang, Jin, Shao, and Ying, *Statistical Inference on Transformation Models:
  a Self-induced Smoothing Approach*](https://www.columbia.edu/~zj7/sitmmmt-arxiv.pdf)
  is representative smoothed-MRC literature. Our logistic sphere-constrained
  continuation heuristic is a related adaptation; its statistical inference
  guarantees are not claimed.
- Polynomial full Kendall local search, polynomial LMRC, and polynomial
  Elastic MaxFS are adaptations of this repository's linear experiment
  baselines. The sampled polynomial method similarly lifts the existing
  duplicate-aware search; it is not claimed as a published polynomial PTAS.
- The min-disagreement construction follows the manuscript's
  `thm:polynomial-min-disagreement` extension of the linear minimal-IIS method.


