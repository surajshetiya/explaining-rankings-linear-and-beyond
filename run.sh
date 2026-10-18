#!/usr/bin/env bash
# run_kt_experiments.sh -- Kendall-Tau (KT) error experiments for every linear
# and polynomial ranking method in the codebase, on one dataset.
#
# Place this file in the `Code` directory (next to closest_linear_ranking_gurobi.py,
# with `polynomial-function/` beneath it) and run:   bash run_kt_experiments.sh
#
# Flags:
#   --quick        one seed, reduced sweeps (smoke test)
#   --dry-run      print the commands without running them
#   --no-resume    re-run jobs even when their JSON result already exists
#   --linear-only  skip the polynomial experiments
#   --poly-only    skip the linear experiments
#   -h, --help     show this help
#
# Environment variables (all optional):
#   DATA            dataset path               (default: <Code>/dataset_100.csv)
#   TARGET_ARGS     target spec, ABSOLUTE paths only, e.g.
#                   "--target-ranks-file /abs/ranks.csv" or "--target-order 3,1,2,0"
#                   (default: empty = CSV row order is the strict target)
#   OUT_DIR         results directory          (default: <Code>/kt_experiments)
#   SEEDS           seeds for randomized methods (default: "59 60 61")
#   DEGREES         polynomial degrees          (default: "1 2 3")
#   THREADS         Gurobi threads              (default: 4)
#   RUN_TIMEOUT     per-run timeout in seconds  (default: 1800)
#   POLY_MAX_FEATURES  skip a degree whose expanded dimension exceeds this (default: 5000)
#   REF_OPT_LINEAR  known optimal KT for the linear family (optional)
#   REF_OPT_POLY    known optimal KT for the polynomial family (optional)
#   PYTHON          interpreter                 (default: python3)
#
# Outputs (in OUT_DIR):
#   json/*.json          raw result of every run
#   logs/*.log           stdout/stderr of every run
#   manifest.tsv         one line per run (status, exit code, wall time)
#   results.csv          one row per run: KT error, ratio/gap to reference, ...
#   summary_by_config.csv  per configuration, aggregated over seeds
#
# Aggregation is done by aggregate_kt_results.py (keep it next to this script).

set -u -o pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# POLY_DIR="${POLY_DIR:-$CODE_DIR/polynomial-function}"
POLY_DIR=$CODE_DIR
DATA="${DATA:-$CODE_DIR/dataset_100.csv}"
OUT_DIR="${OUT_DIR:-$CODE_DIR/kt_experiments}"
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-59 60 61}"
DEGREES="${DEGREES:-1 2 3}"
THREADS="${THREADS:-4}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"
POLY_MAX_FEATURES="${POLY_MAX_FEATURES:-5000}"
TARGET_ARGS="${TARGET_ARGS:-}"
REF_OPT_LINEAR="${REF_OPT_LINEAR:-}"
REF_OPT_POLY="${REF_OPT_POLY:-}"

QUICK=0; DRY_RUN=0; RESUME=1; RUN_LINEAR=1; RUN_POLY=1

usage() { awk 'NR>1 && /^#/ {print substr($0,3); next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --no-resume) RESUME=0 ;;
    --linear-only) RUN_POLY=0 ;;
    --poly-only) RUN_LINEAR=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------------
[[ -f "$DATA" ]] || { echo "Dataset not found: $DATA (set DATA=...)" >&2; exit 1; }
[[ -f "$CODE_DIR/aggregate_kt_results.py" ]] || { echo "aggregate_kt_results.py not found in $CODE_DIR" >&2; exit 1; }
[[ -f "$CODE_DIR/closest_linear_ranking_gurobi.py" ]] || {
  echo "Run this script from the Code directory (closest_linear_ranking_gurobi.py not found in $CODE_DIR)." >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }

read -r -a TARGET_ARR <<< "$TARGET_ARGS"
read -r -a SEED_ARR <<< "$SEEDS"
[[ $QUICK == 1 ]] && SEED_ARR=("${SEED_ARR[0]}")
FIRST_SEED="${SEED_ARR[0]}"

read -r N_ROWS N_COLS < <("$PYTHON" - "$DATA" <<'PY'
import sys
import numpy as np
p = sys.argv[1]
a = np.load(p) if p.endswith(".npy") else np.loadtxt(p, delimiter=",", ndmin=2)
print(a.shape[0], a.shape[1])
PY
) || true
if [[ -z "${N_ROWS:-}" ]]; then
  echo "Could not read $DATA as a headerless numeric CSV (n x d). Remove header/label columns." >&2
  exit 1
fi

HAVE_GUROBI=0
"$PYTHON" -c "import gurobipy" >/dev/null 2>&1 && HAVE_GUROBI=1
[[ $HAVE_GUROBI == 0 ]] && echo "WARNING: gurobipy not importable; Gurobi-based methods will be recorded as failed." >&2

TO_CMD=()
if TB="$(command -v timeout || command -v gtimeout || true)"; [[ -n "$TB" ]]; then
  TO_CMD=("$TB" "$RUN_TIMEOUT")
else
  echo "WARNING: no 'timeout' command found; runs have no time limit." >&2
fi

JSON_DIR="$OUT_DIR/json"; LOG_DIR="$OUT_DIR/logs"; MANIFEST="$OUT_DIR/manifest.tsv"
mkdir -p "$JSON_DIR" "$LOG_DIR"
[[ -f "$MANIFEST" ]] || printf 'id\tfamily\tmethod\tdegree\tparams\tseed\tstatus\texit_code\twall_seconds\n' > "$MANIFEST"

{
  echo "date: $(date)"
  echo "data: $DATA  (n=$N_ROWS, d=$N_COLS)"
  echo "target_args: ${TARGET_ARGS:-<row order>}"
  echo "seeds: ${SEED_ARR[*]}  degrees: $DEGREES  threads: $THREADS  timeout: ${RUN_TIMEOUT}s  quick: $QUICK"
} > "$OUT_DIR/config.txt"
cat "$OUT_DIR/config.txt"

# ----------------------------------------------------------------------------
# Job runner
#   run_job FAMILY METHOD DEGREE PARAMS SEED CWD CMD...
# The target options and --output <json> are appended automatically.
# ----------------------------------------------------------------------------
JOB_COUNT=0; FAIL_COUNT=0

run_job() {
  local family="$1" method="$2" degree="$3" params="$4" seed="$5" cwd="$6"
  shift 6
  local id out log status code=0 t0 t1
  id="$(printf '%s__%s__%s__s%s' "${family}${degree:+_d$degree}" "$method" "$params" "$seed" \
        | tr -c 'A-Za-z0-9._=-' '_')"
  out="$JSON_DIR/$id.json"; log="$LOG_DIR/$id.log"
  JOB_COUNT=$((JOB_COUNT + 1))

  if [[ $DRY_RUN == 1 ]]; then
    printf '[dry-run] (cd %s && ' "$cwd"; printf '%q ' "$@" ${TARGET_ARR[@]+"${TARGET_ARR[@]}"}; printf -- '--output %q)\n' "$out"
    return 0
  fi

  if [[ $RESUME == 1 && -s "$out" ]]; then
    status="cached"; t0=0; t1=0
  else
    rm -f "$out"
    { printf '# cmd: '; printf '%q ' "$@" ${TARGET_ARR[@]+"${TARGET_ARR[@]}"}; printf -- '--output %q\n' "$out"; } > "$log"
    t0=$(date +%s)
    (
      cd "$cwd" || exit 97
      export PYTHONPATH="$CODE_DIR:$POLY_DIR${PYTHONPATH:+:$PYTHONPATH}"
      ${TO_CMD[@]+"${TO_CMD[@]}"} "$@" ${TARGET_ARR[@]+"${TARGET_ARR[@]}"} --output "$out"
    ) >> "$log" 2>&1
    code=$?
    t1=$(date +%s)
    if [[ $code -eq 0 && -s "$out" ]]; then status="ok"
    elif [[ $code -eq 124 ]]; then status="timeout"; rm -f "$out"
    else status="failed"; rm -f "$out"; fi
  fi
  [[ $status == ok || $status == cached ]] || FAIL_COUNT=$((FAIL_COUNT + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$id" "$family" "$method" "$degree" "$params" "$seed" "$status" "$code" "$((t1 - t0))" >> "$MANIFEST"
  printf '[%4d] %-8s %s\n' "$JOB_COUNT" "$status" "$id"
}

# ----------------------------------------------------------------------------
# Sweep definitions (shrunk in --quick mode)
# ----------------------------------------------------------------------------
# PTAS: the exact empirical MIP is exponential, so keep samples small.
# NOTE: with an explicit --sample-size, epsilon/delta only change the reported
# guarantee flag (sampling_bound_met), not the sampled instance; the second
# (eps,delta) pair is therefore run with the first seed only.
PTAS_SS=(25 50 75 100)
PTAS_ED=("0.8:0.8" "0.25:0.05")
# "sample_size:restarts:max_add_back"
HYB_CFGS=("122:3:64" "200:3:64" "200:5:128" "400:5:128")
# "sample_size:restarts:lmrc|nolmrc"
SLS_CFGS=("122:8:lmrc" "200:8:lmrc" "500:8:lmrc" "200:16:lmrc" "200:8:nolmrc")
# theory sweep for sampled local search: --sample-size 0 -> sufficient VC size
SLS_THEORY_EPS=(0.8 0.5 0.25)
SLS_THEORY_DELTA=(0.8 0.1)
# "sample_size:restarts:w_objective:w_class:w_hard:max_class_matrix_entries"
DUP_CFGS=("122:8:0.5:0.25:0.25:4000000" "200:8:0.5:0.25:0.25:4000000" "500:8:0.5:0.25:0.25:4000000"
          "200:8:1:0:0:4000000" "200:8:0.25:0.5:0.25:4000000" "200:8:0.25:0.25:0.5:4000000"
          "200:8:0.5:0.25:0.25:0")
LMRC_RIDGE=(1e-8 1e-6 1e-3)
PAIRWISE_REG=(1e-4 1e-3 1e-2)
LS_RESTARTS=(8 32)
MAXFS_DELFRAC=(0.01 0.05 0.2)
POLY_SLS_SS=(122 200 500)

if [[ $QUICK == 1 ]]; then
  PTAS_SS=(25 50); PTAS_ED=("0.8:0.8")
  HYB_CFGS=("122:3:64"); SLS_CFGS=("122:8:lmrc"); DUP_CFGS=("122:8:0.5:0.25:0.25:4000000")
  SLS_THEORY_EPS=(0.8); SLS_THEORY_DELTA=(0.8)
  LMRC_RIDGE=(1e-6); PAIRWISE_REG=(1e-3); LS_RESTARTS=(8); MAXFS_DELFRAC=(0.05)
  POLY_SLS_SS=(122); DEGREES="1 2"
fi

# ----------------------------------------------------------------------------
# LINEAR experiments
# ----------------------------------------------------------------------------
run_linear() {
  local s ss cfg ed eps delta r a lmrc idx ridge reg frac wo wc wh mc
  echo; echo "=== LINEAR experiments ==="

  # Theorem 1: (r+1)-approximation for minimum KT (deterministic -> one seed)
  run_job linear closest "" "default" "$FIRST_SEED" "$CODE_DIR" \
    "$PYTHON" closest_linear_ranking_gurobi.py "$DATA" --seed "$FIRST_SEED" --threads "$THREADS"

  # Theorem 2: deterministic 1/2-approximation for max agreement
  for s in "${SEED_ARR[@]}"; do
    run_job linear half "" "default" "$s" "$CODE_DIR" \
      "$PYTHON" max_agreement_linear_ranking_gurobi.py half "$DATA" --seed "$s" --threads "$THREADS"
  done

  # Theorem 3: randomized (1-eps) PTAS, vary sample size and (epsilon, delta)
  for ss in "${PTAS_SS[@]}"; do
    idx=0
    for ed in "${PTAS_ED[@]}"; do
      IFS=: read -r eps delta <<< "$ed"
      for s in "${SEED_ARR[@]}"; do
        if [[ $idx -gt 0 && "$s" != "$FIRST_SEED" ]]; then continue; fi
        run_job linear ptas "" "ss=$ss,eps=$eps,delta=$delta" "$s" "$CODE_DIR" \
          "$PYTHON" max_agreement_linear_ranking_gurobi.py ptas "$DATA" \
          --epsilon "$eps" --delta "$delta" --sample-size "$ss" --seed "$s" --threads "$THREADS"
      done
      idx=$((idx + 1))
    done
  done

  # Sampled greedy-IIS hybrid
  for cfg in "${HYB_CFGS[@]}"; do
    IFS=: read -r ss r a <<< "$cfg"
    for s in "${SEED_ARR[@]}"; do
      run_job linear hybrid "" "ss=$ss,restarts=$r,addback=$a" "$s" "$CODE_DIR" \
        "$PYTHON" hybrid_linear_ranking_gurobi.py "$DATA" \
        --epsilon 0.8 --delta 0.8 --sample-size "$ss" --restarts "$r" --max-add-back "$a" \
        --seed "$s" --threads "$THREADS"
    done
  done

  # PTAS-sampled Kendall local search (NumPy only)
  for cfg in "${SLS_CFGS[@]}"; do
    IFS=: read -r ss r lmrc <<< "$cfg"
    for s in "${SEED_ARR[@]}"; do
      local extra=()
      [[ $lmrc == nolmrc ]] && extra=(--no-lmrc-seed)
      run_job linear sampled_local_search "" "ss=$ss,restarts=$r,$lmrc" "$s" "$CODE_DIR" \
        "$PYTHON" sampled_kendall_local_search_gurobi.py "$DATA" \
        --epsilon 0.8 --delta 0.8 --sample-size "$ss" --restarts "$r" \
        --max-iterations 250 --patience 40 --gaussian-interval 10 --seed "$s" ${extra[@]+"${extra[@]}"}
    done
  done
  # ... and with the theoretical VC sample size, where epsilon/delta DO change the sample
  for eps in "${SLS_THEORY_EPS[@]}"; do
    for delta in "${SLS_THEORY_DELTA[@]}"; do
      for s in "${SEED_ARR[@]}"; do
        run_job linear sampled_local_search_theory "" "eps=$eps,delta=$delta,restarts=8" "$s" "$CODE_DIR" \
          "$PYTHON" sampled_kendall_local_search_gurobi.py "$DATA" \
          --epsilon "$eps" --delta "$delta" --sample-size 0 --restarts 8 --seed "$s"
      done
    done
  done

  # Duplicate-aware sampled local search (NumPy only)
  for cfg in "${DUP_CFGS[@]}"; do
    IFS=: read -r ss r wo wc wh mc <<< "$cfg"
    for s in "${SEED_ARR[@]}"; do
      run_job linear dup_aware_local_search "" "ss=$ss,restarts=$r,mix=$wo/$wc/$wh,maxcls=$mc" "$s" "$CODE_DIR" \
        "$PYTHON" duplicate_aware_sampled_kendall_local_search.py "$DATA" \
        --epsilon 0.8 --delta 0.8 --sample-size "$ss" --restarts "$r" \
        --objective-mixture-weight "$wo" --class-mixture-weight "$wc" --hard-mixture-weight "$wh" \
        --max-class-matrix-entries "$mc" --seed "$s"
    done
  done

  # Experimental baselines
  for ridge in "${LMRC_RIDGE[@]}"; do
    run_job linear lmrc "" "ridge=$ridge" "$FIRST_SEED" "$CODE_DIR" \
      "$PYTHON" linear_ranking_baselines.py lmrc "$DATA" --ridge "$ridge" --seed "$FIRST_SEED"
  done

  for m in ranksvm ranknet; do
    # default hyper-parameters, then a regularization sweep on all P comparisons
    run_job linear "$m" "" "default" "$FIRST_SEED" "$CODE_DIR" \
      "$PYTHON" linear_ranking_baselines.py "$m" "$DATA" --seed "$FIRST_SEED"
    for reg in "${PAIRWISE_REG[@]}"; do
      run_job linear "$m" "" "reg=$reg,epochs=200,allpairs" "$FIRST_SEED" "$CODE_DIR" \
        "$PYTHON" linear_ranking_baselines.py "$m" "$DATA" \
        --regularization "$reg" --max-training-pairs 0 --epochs 200 --seed "$FIRST_SEED"
    done
  done

  for r in "${LS_RESTARTS[@]}"; do
    for s in "${SEED_ARR[@]}"; do
      run_job linear local_search "" "restarts=$r" "$s" "$CODE_DIR" \
        "$PYTHON" linear_ranking_baselines.py local-search "$DATA" \
        --restarts "$r" --max-iterations 250 --seed "$s"
    done
  done

  run_job linear maxfs "" "default" "$FIRST_SEED" "$CODE_DIR" \
    "$PYTHON" linear_ranking_baselines.py maxfs "$DATA" --seed "$FIRST_SEED" --threads "$THREADS"
  for frac in "${MAXFS_DELFRAC[@]}"; do
    run_job linear maxfs "" "delfrac=$frac,allpairs" "$FIRST_SEED" "$CODE_DIR" \
      "$PYTHON" linear_ranking_baselines.py maxfs "$DATA" \
      --max-training-pairs 0 --deletion-fraction "$frac" --max-add-back 1000 \
      --seed "$FIRST_SEED" --threads "$THREADS"
  done
}

# ----------------------------------------------------------------------------
# POLYNOMIAL experiments
# ----------------------------------------------------------------------------
run_poly() {
  local deg dp m reg s ss kb
  echo; echo "=== POLYNOMIAL experiments ==="
  for deg in $DEGREES; do
    dp="$("$PYTHON" -c "import math; print(math.comb($N_COLS + $deg, $deg) - 1)")"
    if [[ "$dp" -gt "$POLY_MAX_FEATURES" ]]; then
      echo "Skipping degree $deg: expanded dimension $dp > POLY_MAX_FEATURES=$POLY_MAX_FEATURES"
      continue
    fi
    echo "--- degree $deg (expanded dimension $dp) ---"

    # Pairwise-surrogate baselines: default, regularization sweep, polynomial-kernel basis
    for m in ranksvm ranknet rankrls; do
      run_job poly "$m" "$deg" "deg=$deg,default" "$FIRST_SEED" "$POLY_DIR" \
        "$PYTHON" polynomial_ranking_baselines.py "$m" "$DATA" --degree "$deg"
      for reg in "${PAIRWISE_REG[@]}"; do
        run_job poly "$m" "$deg" "deg=$deg,reg=$reg" "$FIRST_SEED" "$POLY_DIR" \
          "$PYTHON" polynomial_ranking_baselines.py "$m" "$DATA" --degree "$deg" --regularization "$reg"
      done
      run_job poly "$m" "$deg" "deg=$deg,reg=1e-3,kernel" "$FIRST_SEED" "$POLY_DIR" \
        "$PYTHON" polynomial_ranking_baselines.py "$m" "$DATA" --degree "$deg" \
        --regularization 1e-3 --feature-basis polynomial-kernel
    done

    # Smoothed MRC and full Kendall local search (randomized -> all seeds)
    for s in "${SEED_ARR[@]}"; do
      run_job poly smoothed_mrc "$deg" "deg=$deg" "$s" "$POLY_DIR" \
        "$PYTHON" polynomial_ranking_baselines.py smoothed-mrc "$DATA" --degree "$deg" --seed "$s"
      run_job poly local_search "$deg" "deg=$deg" "$s" "$POLY_DIR" \
        "$PYTHON" polynomial_ranking_baselines.py local "$DATA" --degree "$deg" --seed "$s"
    done

    run_job poly lmrc "$deg" "deg=$deg" "$FIRST_SEED" "$POLY_DIR" \
      "$PYTHON" polynomial_ranking_baselines.py lmrc "$DATA" --degree "$deg"

    # Gurobi-based: Elastic MaxFS and the minimum-disagreement IIS approximation
    run_job poly maxfs "$deg" "deg=$deg" "$FIRST_SEED" "$POLY_DIR" \
      "$PYTHON" polynomial_ranking_baselines.py maxfs "$DATA" --degree "$deg"
    run_job poly closest "$deg" "deg=$deg" "$FIRST_SEED" "$POLY_DIR" \
      "$PYTHON" closest_polynomial_ranking_gurobi.py "$DATA" --degree "$deg"

    # Sampled polynomial local search: importance sampling (default) and uniform ablation
    for ss in "${POLY_SLS_SS[@]}"; do
      for s in "${SEED_ARR[@]}"; do
        run_job poly sampled_local_search "$deg" "deg=$deg,ss=$ss,importance" "$s" "$POLY_DIR" \
          "$PYTHON" sampled_polynomial_kendall_local_search.py "$DATA" --degree "$deg" \
          --sample-size "$ss" --restarts 8 --max-iterations 250 --seed "$s"
      done
    done
    for s in "${SEED_ARR[@]}"; do
      run_job poly sampled_local_search "$deg" "deg=$deg,ss=200,uniform" "$s" "$POLY_DIR" \
        "$PYTHON" sampled_polynomial_kendall_local_search.py "$DATA" --degree "$deg" \
        --sampling uniform --sample-size 200 --restarts 8 --max-iterations 250 --seed "$s"
    done
  done
}

[[ $RUN_LINEAR == 1 ]] && run_linear
[[ $RUN_POLY == 1 ]] && run_poly

echo
echo "Finished: $JOB_COUNT jobs, $FAIL_COUNT not ok."
[[ $DRY_RUN == 1 ]] && exit 0

# ----------------------------------------------------------------------------
# Aggregate KT errors and compare against the reference (optimal) value
# ----------------------------------------------------------------------------
export REF_OPT_LINEAR REF_OPT_POLY
"$PYTHON" "$CODE_DIR/aggregate_kt_results.py" "$OUT_DIR"
