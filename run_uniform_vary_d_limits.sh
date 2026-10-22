#!/bin/bash
# Find the largest sample size (multiple of STEP) that finishes within TIMEOUT.
#   Phase 1: start at START, double while it succeeds, until a run fails.
#   Phase 2: binary search between last success (lo) and first failure (hi),
#            with midpoints rounded down to a multiple of STEP, until the
#            midpoint can no longer be separated from lo/hi.
# If START itself fails: lo=0, hi=START and only phase 2 runs.

TIMEOUT=1800   # 30 minutes = 1800 seconds

data_seeds=(19 42 59 93 123)
dims=(3 7 11 15)


# ---- configuration (set these or export them before running) ----
seed=19
DATASET=uniform_1M_3D_100_05_19
DATASETPATH=data/full/${DATASET}.csv
START=2000
STEP=100

mkdir -p "bin_src_output/${seed}"


#!/usr/bin/env bash

# Assumes DATASET, DATASETPATH, seed, and TIMEOUT are already set.
# Example: TIMEOUT="30m"

START_SAMPLE=2000
GRANULARITY=100

last_passed_item=0
first_failed_item=0

run_sample() {
    local sample="$1"
    local outfile="bin_src_output/${seed}/${DATASET}_duplicate_sampled_full_${sample}.csv"

    mkdir -p "full_output/${seed}" || return 1

    echo "========================================"
    echo "Sampling duplicate aware: ${sample}..."
    echo "Running: python duplicate_aware_sampled_kendall_local_search.py --epsilon 0.8 --delta 0.8 --sample-size ${sample} --seed ${seed} ${DATASETPATH}"
    echo "Started: $(date)"
    echo "========================================"

    timeout --signal=TERM --kill-after=10s "$TIMEOUT" \
        python duplicate_aware_sampled_kendall_local_search.py \
        --epsilon 0.8 \
        --delta 0.8 \
        --sample-size "$sample" \
        --seed 59 \
        "$DATASETPATH" > "$outfile" 2>&1

    local status=$?

    if [ "$status" -eq 0 ]; then
        echo "COMPLETED successfully: sample=${sample}"
        return 0
    elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
        echo "TIMEOUT: sample=${sample} exceeded the time limit"
    else
        echo "FAILED: sample=${sample}, exit code=${status}"
    fi

    return 1
}

# --------------------------------------------------
# Phase 1: Exponential (doubling) search
# --------------------------------------------------

sample=$START_SAMPLE

if run_sample "$sample"; then
    last_passed_item=$sample

    while true; do
        sample=$((sample * 2))

        if run_sample "$sample"; then
            last_passed_item=$sample
        else
            first_failed_item=$sample
            break
        fi
    done
else
    # The very first sample failed.
    last_passed_item=0
    first_failed_item=$START_SAMPLE
fi

echo
echo "Doubling search finished."
echo "Last successful sample: ${last_passed_item}"
echo "First failed sample:    ${first_failed_item}"

# --------------------------------------------------
# Phase 2: Binary search
# --------------------------------------------------

while (( first_failed_item - last_passed_item > GRANULARITY )); do

    # Midpoint rounded down to the nearest 100.
    sample=$(( ((last_passed_item + first_failed_item) / 2 / GRANULARITY) * GRANULARITY ))

    # Ensure the midpoint lies strictly between the bounds.
    if (( sample <= last_passed_item || sample >= first_failed_item )); then
        break
    fi

    if run_sample "$sample"; then
        last_passed_item=$sample
    else
        first_failed_item=$sample
    fi

    echo "Current bounds: successful=${last_passed_item}, failed=${first_failed_item}"
done

# --------------------------------------------------
# Final result
# --------------------------------------------------

echo
echo "========================================"
echo "Search completed."
echo "Largest successful sample found: ${last_passed_item}"
echo "Smallest failed bound found:     ${first_failed_item}"
echo "Remaining interval:              $((first_failed_item - last_passed_item))"
echo "========================================"







## Runs one experiment. Returns 0 on success, non-zero on failure/timeout.
#run_one() {
#    local sample="$1"
#    local cmd="python duplicate_aware_sampled_kendall_local_search.py --epsilon 0.8 --delta 0.8 --sample-size ${sample} --seed 59 ${DATASETPATH} > bin_src_output/${seed}/${DATASET}_duplicate_sampled_full_${sample}.csv 2>&1;"
#
#    echo "========================================"
#    echo "Sampling duplicate aware : ${sample}..."
#    echo "Running: $cmd"
#    echo "Started: $(date)"
#    echo "========================================"
#
#    timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
#    local status=$?
#
#    if [ $status -eq 124 ] || [ $status -eq 137 ]; then
#        echo "TIMEOUT: ${sample} exceeded ${TIMEOUT}"
#        return 1
#    elif [ $status -ne 0 ]; then
#        echo "FAILED: ${sample} exit code $status"
#        return 1
#    fi
#    echo "COMPLETED successfully: ${sample}"
#    return 0
#}
#
#lo=0          # largest known-good sample size (0 = none yet)
#hi=""         # smallest known-bad sample size (empty = none yet)
#
## ---------- Phase 1: doubling ----------
#sample=$START
#while true; do
#    if run_one "$sample"; then
#        lo=$sample
#        sample=$((sample * 2))
#    else
#        hi=$sample
#        break
#    fi
#done
#
#echo "Phase 1 done: lo=${lo} (pass), hi=${hi} (fail)"
#
## ---------- Phase 2: binary search on multiples of STEP ----------
#while true; do
#    # midpoint rounded down to a multiple of STEP
#    mid=$(( (lo + hi) / 2 / STEP * STEP ))
#
#    # stop when mid can't land strictly between lo and hi
#    if [ "$mid" -le "$lo" ] || [ "$mid" -ge "$hi" ]; then
#        break
#    fi
#
#    if run_one "$mid"; then
#        lo=$mid
#    else
#        hi=$mid
#    fi
#done
#
