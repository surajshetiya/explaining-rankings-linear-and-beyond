#!/bin/bash

TIMEOUT=1800   # 30 minutes = 1800 seconds

DATASET="new_adult"
DATASETPATH="data/full/${DATASET}.csv"

samples=(2000 20000 40000 60000 80000 100000)
samples=(120000 150000 175000 200000)

#last_passed_item=0
#
#for sample in "${samples[@]}"; do
#
#    cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --sample-size ${sample} --seed 59 ${DATASETPATH} > full_output/${DATASET}_duplicate_sampled_full_${sample}.csv 2>&1;"
#    echo "========================================"
#    echo "Sampling duplicate aware : ${sample}..."
#    echo "Running: $cmd"
#    echo "Started: $(date)"
#    echo "========================================"
#    timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
#    status=$?
#
#    if [ $status -eq 124 ]; then
#        echo "TIMEOUT: command exceeded 30 minutes"
#	break
#    elif [ $status -ne 0 ]; then
#        echo "FAILED: exit code $status"
#	break
#    else
#        echo "COMPLETED successfully"
#        last_passed_item="$sample"
#    fi
#
#done
#
add_samples=(5000 10000 15000)

#if [ $last_passed_item > 0 ]; then
#    for add_sample in "${add_samples[@]}"; do
#	sample=$((add_sample + last_passed_item))
#        cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --sample-size ${sample} --seed 59 ${DATASETPATH} > full_output/${DATASET}_duplicate_sampled_full_${sample}.csv 2>&1;"
#        echo "========================================"
#        echo "Sampling : ${sample}..."
#        echo "Running: $cmd"
#        echo "Started: $(date)"
#        echo "========================================"
#        timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
#        status=$?
#
#        if [ $status -eq 124 ]; then
#            echo "TIMEOUT: command exceeded 30 minutes"
#            break
#        elif [ $status -ne 0 ]; then
#            echo "FAILED: exit code $status"
#            break
#        else
#            echo "COMPLETED successfully"
#        fi
#    done
#fi

last_passed_item=0

for sample in "${samples[@]}"; do

    cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --sample-size ${sample} --restarts 8 --max-iterations 250 --seed 59 ${DATASETPATH} > full_output/${DATASET}_duplicate_sampled_restarts_8_max_iter_250_${sample}.csv 2>&1;"
    echo "========================================"
    echo "Sampling duplicate aware : ${sample}..."
    echo "Running: $cmd"
    echo "Started: $(date)"
    echo "========================================"
    timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
    status=$?

    if [ $status -eq 124 ]; then
        echo "TIMEOUT: command exceeded 30 minutes"
	break
    elif [ $status -ne 0 ]; then
        echo "FAILED: exit code $status"
	break
    else
        echo "COMPLETED successfully"
        last_passed_item="$sample"
    fi

done

add_samples=(5000 10000 15000)

if [ $last_passed_item > 0 ]; then
    for add_sample in "${add_samples[@]}"; do
	sample=$((add_sample + last_passed_item))
        cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --restarts 8 --max-iterations 250 --sample-size ${sample} --seed 59 ${DATASETPATH} > full_output/${DATASET}_duplicate_sampled_restarts_8_max_iter_250_${sample}.csv 2>&1;"
        echo "========================================"
        echo "Sampling : ${sample}..."
        echo "Running: $cmd"
        echo "Started: $(date)"
        echo "========================================"
        timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
        status=$?

        if [ $status -eq 124 ]; then
            echo "TIMEOUT: command exceeded 30 minutes"
            break
        elif [ $status -ne 0 ]; then
            echo "FAILED: exit code $status"
            break
        else
            echo "COMPLETED successfully"
        fi
    done
fi
