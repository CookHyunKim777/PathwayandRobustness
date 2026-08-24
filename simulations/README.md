# Simulations

This directory contains the simulation entry points used for the synthetic
power-grid models and the finite-perturbation experiments.

## Directory structure

- `synthetic/`: ensemble simulations for symmetric and asymmetric power
  distributions, with and without inertia.
- `perturbation/`: finite-perturbation simulations used to measure robustness
  and recovery.

The simulation algorithms and command-line parameters are unchanged from the
original research code. Run the scripts from the repository root with Python's
module syntax so that imports from `core/` resolve correctly.

Examples:

```bash
python -m simulations.synthetic.main_ensemble 3.0 1.0 10.0 0
python -m simulations.synthetic.main_ensemble_massless 3.0 1.0 0
python -m simulations.perturbation.main_Cascading_Process 3.0 1.0 10.0 0
```

By default, output is written beneath `results/` in the repository root. To
use a different location, set the `SYNC_PATHS_DATA_DIR` environment variable:

```bash
export SYNC_PATHS_DATA_DIR=/path/to/results
```

The symmetric cases use the `DF_double_peak` subdirectory, whereas the
asymmetric cases use `DF_single_peak`.
