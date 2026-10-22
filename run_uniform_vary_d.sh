#!/bin/bash

TIMEOUT=1800   # 30 minutes = 1800 seconds

data_seeds=(19 42 59 93 123)
dims=(3 7 11 15)

for dseed in "${data_seeds[@]}"; do
    for dim in "${dims[@]}"; do

        DATASET="uniform_1M_${dim}D_100_05_${dseed}"
        DATASETPATH="data/full/${DATASET}.csv"
        
        seeds=(19 42 59 93 123)
        
        for seed in "${seeds[@]}"; do
        
            mkdir full_output/${seed}
            samples=(2000 20000 60000 10000 140000 160000 180000)
        
        
            commands=(
                "python3 linear_ranking_baselines.py ranksvm --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_ranksvm.out"
                "python3 linear_ranking_baselines.py ranknet --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_ranknet.out"
                "python3 linear_ranking_baselines.py maxfs --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_maxfs.out"
                "python3 linear_ranking_baselines.py local-search --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_ls.out"
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
             
             commands=(
                 "python3 max_agreement_linear_ranking_gurobi.py half --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_max.out"
                 "python3 max_agreement_linear_ranking_gurobi.py ptas --seed ${seed} --epsilon=0.1 --delta=0.1 --sample-size 200 ${DATASETPATH} > full_output/${seed}/${DATASET}_ptas_200.out"
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
             
             # 2. Use a regular variable (instead of an array) to store the last passed item
             last_passed_item=0
             
              3. Loop over the items
             for sample in "${samples[@]}"; do
             
                 cmd="python3 hybrid_linear_ranking_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${seed}/${DATASET}_hybrid_${sample}.csv 2>&1"
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
                     last_passed_item="$sample"
                 fi
             
                 # 4. Run the command and redirect output to a file
                 # if  "$item" > output.txt 2>&1; then
                 #if python3 hybrid_linear_ranking_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1; then
                 #    # Overwrite the variable so it only holds the LAST passed item
                 #    last_passed_item="$item"
                 #else
                 #    echo "   -> Failed! Breaking out of the loop immediately."
                 #    
                 #    # 5. Stop the loop right here and skip any remaining items
                 #    break
                 #fi
             
             done
             
             add_samples=(5000 10000 15000)
             
             if [ $last_passed_item > 0 ]; then
                 for add_sample in "${add_samples[@]}"; do
             	sample=$((add_sample + last_passed_item))
                     cmd="python3 hybrid_linear_ranking_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${seed}/${DATASET}_hybrid_${sample}.csv 2>&1;"
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
             # 2. Use a regular variable (instead of an array) to store the last passed item
             last_passed_item=0
             
             # 3. Loop over the items
             for sample in "${samples[@]}"; do
             
                 #cmd="python3 hybrid_linear_ranking_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${seed}/${DATASET}_hybrid_${sample}.csv 2>&1"
                 cmd="python3 sampled_kendall_local_search_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${seed}/${DATASET}_sampled_${sample}.csv 2>&1"
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
                     last_passed_item="$sample"
                 fi
             
                 # 4. Run the command and redirect output to a file
                 # if  "$item" > output.txt 2>&1; then
                 #if python3 hybrid_linear_ranking_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1; then
                 #    # Overwrite the variable so it only holds the LAST passed item
                 #    last_passed_item="$item"
                 #else
                 #    echo "   -> Failed! Breaking out of the loop immediately."
                 #    
                 #    # 5. Stop the loop right here and skip any remaining items
                 #    break
                 #fi
             
             done
             
             add_samples=(5000 10000 15000)
             
             if [ $last_passed_item > 0 ]; then
                 for add_sample in "${add_samples[@]}"; do
             	sample=$((add_sample + last_passed_item))
                     #cmd="python3 hybrid_linear_ranking_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1"
                     cmd="python3 sampled_kendall_local_search_gurobi.py --seed ${seed} --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${seed}/${DATASET}_sampled_${sample}.csv 2>&1"
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
             
             last_passed_item=0
             
             for sample in "${samples[@]}"; do
             
                 cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --sample-size ${sample} --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_duplicate_sampled_full_${sample}.csv 2>&1;"
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
                     cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --sample-size ${sample} --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_duplicate_sampled_full_${sample}.csv 2>&1;"
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
             
             last_passed_item=0
             
             for sample in "${samples[@]}"; do
             
                 cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --sample-size ${sample} --restarts 8 --max-iterations 250 --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_duplicate_sampled_restarts_8_max_iter_250_${sample}.csv 2>&1;"
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
                     cmd="python duplicate_aware_sampled_kendall_local_search.py  --epsilon 0.8 --delta 0.8 --restarts 8 --max-iterations 250 --sample-size ${sample} --seed ${seed} ${DATASETPATH} > full_output/${seed}/${DATASET}_duplicate_sampled_restarts_8_max_iter_250_${sample}.csv 2>&1;"
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
        done
    done
done
