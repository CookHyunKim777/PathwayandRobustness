# Analysis

This directory contains the post-processing scripts used to aggregate the
simulation outputs and evaluate the synchronization pathways.

## Directory structure

- `clusters/`: cluster order parameter, mean natural frequency, and mean
  angular-velocity analyses.
- `convergence/`: convergence-time and convergence-rate statistics.
- `perturbation/`: robustness and recovery analysis following finite
  perturbations.
- `ad_hoc_potential/`: ad hoc potential functions for the massive and massless
  symmetric and asymmetric models.

Run command-line analysis scripts from the repository root. For example:

```bash
python -m analysis.clusters.Cluster_R_Analysis 3.0 1.0 10.0
python -m analysis.convergence.Convergence_Time_Collect 3.0 1.0 10.0
python -m analysis.perturbation.Cascading_Analysis 3.0 1.0 10.0
```

The scripts read simulation results from `results/` by default. Set
`SYNC_PATHS_DATA_DIR` to use another data location. The analysis algorithms,
classification criteria, and statistical definitions are unchanged from the
original research code.
