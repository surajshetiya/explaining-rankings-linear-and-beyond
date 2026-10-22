# Explaining Rankings: Linear and Beyond

This project runs experiments for two ranking methods: the hybrid method and
sampled Kendall local search.

## Before you run it

You need Python, NumPy, Gurobi, and a working Gurobi license:

```bash
python3 -m pip install numpy gurobipy
```

The runner is configured near the top of `run.sh`. Choose the dataset name and
path there. The included `new_adult` dataset is at:

```bash
DATASETPATH="data/new_adult.csv"
```

Create the folder used for results once:

```bash
mkdir -p full_output
```

Each dataset should be a headerless CSV file: one item per row and numeric
features in the columns. The row order is used as the target ranking.

## Run the experiments

From the project folder, run:

```bash
bash run.sh
```

The script tries a series of sample sizes for the hybrid method first, then
does the same for sampled local search. Each run has a 30-minute limit. If a
run fails or times out, that method stops and moves on.

Results are saved in `full_output/`. Files beginning with `*_hybrid_` belong
to the hybrid method; files beginning with `*_sampled_` belong to sampled local
search.

To use another dataset or change the sample sizes, edit `DATASET`,
`DATASETPATH`, or `samples` at the top of `run.sh` before running it.
