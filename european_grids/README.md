# European grid representations

This directory contains preprocessing, simulation, result aggregation, and ad
hoc potential code for the data-derived representations of the French, German,
Spanish, and British transmission grids.

## Directory structure

- `preprocessing/Prepare_Grid_Knobs.py`: remaps empirical power values and
  constructs normalized per-node coupling weights.
- `simulation/Kuramoto_RWG_Annealed.c`: second-order simulation with inertia.
- `simulation/Kuramoto_RWG_Annealed_massless.c`: first-order comparison without
  inertia.
- `analysis/grid_action_dualpeak.py`: ad hoc potential functions for negative-
  and positive-side seed clusters.
- `analysis/EU_Grid_Collect.py`: aggregates simulation output by coupling value
  and collects system, cluster, and peak-restricted cluster statistics.
- `hpc/submit.py`: optional SLURM submission helper. It expects the corresponding
  `run_COUNTRY_VARIANT_INDEX.sh` files to be supplied by the user.

## Preprocessing

```bash
python -m european_grids.preprocessing.Prepare_Grid_Knobs COUNTRY \
  --alpha 3.0 --node NODE_FILE --link LINK_FILE \
  --deg-exp 3.0 --outdir PREPARED_DIRECTORY
```

The node input must contain three columns: inertia, damping, and power. The link
input must contain source index, target index, and coupling weight. The script
produces `COUNTRY_node_rescale.txt` and `COUNTRY_link_rescale.txt`.

## Compilation

```bash
gcc -O3 -ffast-math -march=native \
  -o Kuramoto_RWG_Annealed \
  european_grids/simulation/Kuramoto_RWG_Annealed.c -lm

gcc -O3 -ffast-math -march=native \
  -o Kuramoto_RWG_Annealed_massless \
  european_grids/simulation/Kuramoto_RWG_Annealed_massless.c -lm
```

The executables read `COUNTRY_node_rescale.txt` from the current directory. A
per-node coupling-weight file may be supplied as the optional fourth argument:

```bash
./Kuramoto_RWG_Annealed FR 0 3.0 FR_link_rescale.txt
./Kuramoto_RWG_Annealed_massless FR 0 3.0 FR_link_rescale.txt
```

Outputs are written beneath `results/DF/` by default. Set
`SYNC_PATHS_DATA_DIR` to select a different output root.

## Result aggregation

Install NumPy, pandas, and SciPy, then run the collector from the repository
root:

```bash
python -m european_grids.analysis.EU_Grid_Collect FR Annealed 3.0
```

Supported country labels are `DE`, `ES`, `FR`, and `UK`. Supported simulation
types are `Annealed`, `Quenched`, `Annealed_massless`, and
`Quenched_massless`. The collector reads the corresponding simulation tree
beneath `results/DF/` and writes CSV, NPZ, NPY, and JSON summaries to the
matching `<simulation-type>_collect` directory. The same
`SYNC_PATHS_DATA_DIR` setting controls both its input and output roots.

The empirical input data are not included in this code bundle. Their source and
preparation procedure should be documented in the repository-level README.
