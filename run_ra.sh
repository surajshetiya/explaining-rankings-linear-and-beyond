#!/bin/bash

TIMEOUT=1800   # 30 minutes = 1800 seconds

#DATASET="jee_additive_fixed"
DATASET="new_adult"
DATASETPATH="data/full/${DATASET}.csv"

seeds=(19 42 59 93 123)

for seed in "${seeds[@]}"; do

    mkdir full_output/${seed}
    samples=(2000 20000 60000 10000 140000 160000 180000)


    commands=(
        "python3 linear_ranking_baselines.py ranksvm --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_ranksvm.out"
        "python3 linear_ranking_baselines.py ranknet --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_ranknet.out"
        "python3 linear_ranking_baselines.py lmrc --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_lmrc.out"
    )

    for cmd in "${commands[@]}"; do
        echo "========================================"
        echo "Running: $cmd"
        echo "Started: $(date)"
        echo "========================================"
    
        timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
        status=$?
    
        if [ $status -eq 124 ]; then
            echo "TIMEOUT: command exceeded 30 minutes"
        elif [ $status -ne 0 ]; then
            echo "FAILED: exit code $status"
        else
           echo "COMPLETED successfully"
        fi
   
        echo
    done
done
