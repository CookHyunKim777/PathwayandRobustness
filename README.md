# Synchronization Pathways and Robustness in Power Grids

This repository contains the simulation and analysis code associated with the
manuscript *Synchronization Pathways and Robustness in Power Grids* by C. H.
Kim, J. Kim, S. Park, and B. Kahng.

The code covers synthetic power-grid ensembles, finite-perturbation
experiments, cluster-resolved post-processing, ad hoc potential calculations,
and data-derived representations of four European transmission grids.

## Repository structure

- `core/`: numerical integration, sampling, and synchronized-cluster detection.
- `simulations/`: synthetic-grid and finite-perturbation simulations.
- `analysis/`: cluster, convergence, perturbation, and ad hoc potential analyses.
- `european_grids/`: preprocessing, C simulations, aggregation, and analysis for
  the French, German, Spanish, and British grid representations.

Each major directory contains a dedicated README with commands and input/output
details.

## Python environment

Python 3.10 or later is recommended. Create an isolated environment and install
the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run Python entry points from the repository root with module syntax. For
example:

```bash
python -m simulations.synthetic.main_ensemble 3.0 1.0 10.0 0
python -m analysis.clusters.Cluster_R_Analysis 3.0 1.0 10.0
```

## Data and output paths

Simulation outputs are written beneath `results/` by default, and analysis
scripts read from the same location. Set `SYNC_PATHS_DATA_DIR` to use another
data root:

```bash
export SYNC_PATHS_DATA_DIR=/path/to/results
```

Generated results and empirical grid inputs are not included in this
repository. The European-grid workflow requires separately obtained node and
link data; see `european_grids/README.md` for the expected formats.

## Reproducibility note

The scientific algorithms, numerical parameters, cluster criteria, and output
schemas follow the original research code. Repository-specific changes are
limited to package imports, portable data paths, and English documentation.

