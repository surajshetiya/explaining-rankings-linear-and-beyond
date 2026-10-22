# Explaining Rankings: Linear and Beyond

This repository contains the code used for the ranking experiments. It includes
linear and polynomial methods, along with a small example dataset.

## Run the code

You will need Python 3.10 or newer. From this folder, install the required
packages:

```bash
python3 -m pip install numpy scipy gurobipy
```

Some methods use Gurobi, so you will also need a working Gurobi license. If you
only want to try the non-Gurobi methods, the script will still run those and
report the Gurobi-based runs as unavailable.

Start with the short test run:

```bash
bash run.sh --quick
```

This uses the included `dataset_100.csv` and saves results in
`kt_experiments/`. The main summary is written to
`kt_experiments/summary_by_config.csv`.

To run the full set of experiments instead, use:

```bash
bash run.sh
```

The full run can take a while. You can preview the commands without running
anything with:

```bash
bash run.sh --dry-run
```

## Use your own data

Provide a headerless CSV file where each row is one item and each column is a
numeric feature:

```bash
DATA=/full/path/to/your-data.csv bash run.sh --quick
```

By default, the CSV row order is treated as the target ranking. Run
`bash run.sh --help` to see the available options, including how to supply a
different target ranking.
