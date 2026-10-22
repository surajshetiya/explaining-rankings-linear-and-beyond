#!/bin/bash

for seed in 19 42 59 93 123
    do
	for dis in 10 15 20
	do
            disagreement=$(echo "scale=2; $dis / 100" | bc)

            python generate_mallows_dataset.py \
                --n 1000000 \
                --d 3 \
                --radius 100 \
                --disagreement $disagreement \
                --output uniform_1M_3D_100_${dis}_${seed}.csv \
                --seed $seed
	done
done

"""
for d in 3
do
    for seed in 19 42 59 93 123
    do
	for data in 100000 200000 400000
	do
            python generate_mallows_dataset.py \
                --n $data \
                --d $d \
                --radius 100 \
                --disagreement 0.05 \
                --output uniform_${data}_${d}D_100_05_${seed}.csv \
                --seed $seed
        done
    done
done
for d in 3 7 11 15
do
    for seed in 19 42 59 93 123
    do
        python generate_mallows_dataset.py \
            --n 1000000 \
            --d $d \
            --radius 100 \
            --disagreement 0.05 \
            --output uniform_1M_${d}D_100_05_${seed}.csv \
            --seed $seed
    done
done
"""
