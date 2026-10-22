#!/bin/bash

TIMEOUT=1800   # 30 minutes = 1800 seconds

DATASET="new_adult"
DATASETPATH="data/full/${DATASET}.csv"

samples=(2000 20000 40000 60000 80000 10000)

#commands=(
#    "python3 linear_ranking_baselines.py ranksvm  --seed 59 ${DATASETPATH} > full_output/${DATASET}_ranksvm.out"
#    "python3 linear_ranking_baselines.py ranknet  --seed 59 ${DATASETPATH} > full_output/${DATASET}_ranknet.out"
#    "python3 linear_ranking_baselines.py maxfs  --seed 59 ${DATASETPATH} > full_output/${DATASET}_maxfs.out"
#    "python3 linear_ranking_baselines.py local-search  --seed 59 ${DATASETPATH} > full_output/${DATASET}_ls.out"
#    "python3 linear_ranking_baselines.py lmrc  --seed 59 ${DATASETPATH} > full_output/${DATASET}_lmrc.out"
#    "python3 max_agreement_linear_ranking_gurobi.py half --seed 59 ${DATASETPATH} > full_output/${DATASET}_max.out"
#    "python3 max_agreement_linear_ranking_gurobi.py ptas --seed 59 --epsilon=0.1 --delta=0.1 --sample-size 200 ${DATASETPATH} > full_output/${DATSET}_ptas_2000.out"
#)
#
#for cmd in "${commands[@]}"; do
#    echo "========================================"
#    echo "Running: $cmd"
#    echo "Started: $(date)"
#    echo "========================================"
#
#    timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
#    status=$?
#
#    if [ $status -eq 124 ]; then
#        echo "TIMEOUT: command exceeded 30 minutes"
#    elif [ $status -ne 0 ]; then
#        echo "FAILED: exit code $status"
#    else
#        echo "COMPLETED successfully"
#    fi
#
#    echo
#done

# 2. Use a regular variable (instead of an array) to store the last passed item
last_passed_item=0

# 3. Loop over the items
for sample in "${samples[@]}"; do

    cmd="python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${DATASET}_hybrid_${sample}.csv 2>&1"
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
    #if python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1; then
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
        cmd="python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1;"
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

    #cmd="python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${DATASET}_hybrid_${sample}.csv 2>&1"
    cmd="python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${DATASET}_sampled_${sample}.csv 2>&1"
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
    #if python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1; then
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
        #cmd="python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_hybrid_${sample}.csv 2>&1"
        cmd="python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > full_output/${DATASET}_sampled_${sample}.csv 2>&1"
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
## 2. Use a regular variable (instead of an array) to store the last passed item
#last_passed_item=0
#
## 3. Loop over the items
#for sample in "${samples[@]}"; do
#
#    echo "Sampling : $item..."
#
#    # 4. Run the command and redirect output to a file
#    # if  "$item" > output.txt 2>&1; then
#    if python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_sampled_${sample}.csv 2>&1; then
#        # Overwrite the variable so it only holds the LAST passed item
#        last_passed_item="$item"
#    else
#        echo "   -> Failed! Breaking out of the loop immediately."
#        
#        # 5. Stop the loop right here and skip any remaining items
#        break
#    fi
#
#done
#
#add_samples = (5000 10000 15000)
#
#if((last_passed_item > 0)); then
#    for add_sample in "${add_samples[@]}"; do
#	sample=$((add_sample + last_passed_item))
#        if python3 sampled_kendall_local_search_gurobi.py py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size ${sample} ${DATASETPATH} > ${DATASET}_sampled_${sample}.csv 2>&1; then
#		# Done
#	else
#		break
#	fi
#    done
#fi
##
##commands_old=(
##    "python3 max_agreement_linear_ranking_gurobi.py half --seed 59 data/cleaned/adult_cleaned.csv > output/adult_max.out"
##    "python3 max_agreement_linear_ranking_gurobi.py ptas --seed 59 --epsilon=0.1 --delta=0.1 --sample-size 200 data/cleaned/adult_cleaned.csv > output/adult_ptas_2000.out"
##    "python3 closest_linear_ranking_gurobi.py --seed 59 data/cleaned/adult_cleaned.csv > output/adult_closest.out"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 2000 data/cleaned/adult_cleaned.csv > adult_hybrid_2000.csv"
##    "python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 2000 data/cleaned/adult_cleaned.csv > adult_sampled_ls_2000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 20000 data/cleaned/adult_cleaned.csv > adult_hybrid_20000.csv"
##    "python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 20000 data/cleaned/adult_cleaned.csv > adult_sampled_ls_20000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 40000 data/cleaned/adult_cleaned.csv > adult_hybrid_40000.csv"
##    "python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 40000 data/cleaned/adult_cleaned.csv > adult_sampled_ls_40000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 60000 data/cleaned/adult_cleaned.csv > adult_hybrid_60000.csv"
##    "python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 60000 data/cleaned/adult_cleaned.csv > adult_sampled_ls_60000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 80000 data/cleaned/adult_cleaned.csv > adult_hybrid_80000.csv"
##    "python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 80000 data/cleaned/adult_cleaned.csv > adult_sampled_ls_80000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 100000 data/cleaned/adult_cleaned.csv > adult_hybrid_100000.csv"
##    "python3 sampled_kendall_local_search_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 100000 data/cleaned/adult_cleaned.csv > adult_sampled_ls_100000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 90000 data/cleaned/adult_cleaned.csv > adult_hybrid_90000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 95000 data/cleaned/adult_cleaned.csv > adult_hybrid_95000.csv"
##    "python3 hybrid_linear_ranking_gurobi.py --seed 59 --epsilon 0.1 --delta 0.1 --sample-size 85000 data/cleaned/adult_cleaned.csv > adult_hybrid_85000.csv"
##)
##
##for cmd in "${commands[@]}"; do
##    echo "========================================"
##    echo "Running: $cmd"
##    echo "Started: $(date)"
##    echo "========================================"
##
##    timeout --signal=TERM --kill-after=10s "$TIMEOUT" bash -c "$cmd"
##    status=$?
##
##    if [ $status -eq 124 ]; then
##        echo "TIMEOUT: command exceeded 30 minutes"
##    elif [ $status -ne 0 ]; then
##        echo "FAILED: exit code $status"
##    else
##        echo "COMPLETED successfully"
##    fi
##
##    echo
##done
