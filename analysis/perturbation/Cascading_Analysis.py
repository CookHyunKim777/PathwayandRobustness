"""Analyze a perturbation K sweep for parallel SLURM runs.

The script computes seed-ensemble statistics, including Q5-Q95, at every K
without K binning. It loads all (step, K, seed) combinations with a ThreadPool.
Convergence-time seed counts are stored at the 1000-second cluster-measurement
resolution; peak detection and rebinning are deferred to plotting.

Usage:
    python Cascading_Analysis.py <degree_exponent> <exponent> <m>
    Example: python Cascading_Analysis.py 3.0 1.00 5.0
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

N_WORKERS = 32  # Adjust between 16 and 64 according to NFS performance.

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <degree_exponent> <exponent> <m>")
    sys.exit(1)

degree_exponent = float(sys.argv[1])
exponent        = float(sys.argv[2])
m               = float(sys.argv[3])

print(f"Arguments: degree_exponent={degree_exponent}, exponent={exponent}, m={m}")

N        = 1024
max_seed = 64
seeds    = list(range(max_seed))

step_names = ['step1', 'step2', 'step3', 'step4']

K_values = np.round(np.arange(0.0, 20.0, 0.1), 2)

repo_root = Path(__file__).resolve().parents[2]
data_root = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(repo_root / "results"))).expanduser()
base_data_dir = str(data_root / "DF_double_peak" / "perturbation_experiment")
save_base_dir = str(data_root / "DF_double_peak" / "perturbation_K_sweep_analysis")

# ==================== Convergence-time counts (1000-second bins) ====================
# bin j: t in [j*1000, (j+1)*1000), j=0,...,359, for converged seeds.
# bin 360: nonconverged seeds and any converged seed with t >= 360000.
# Each row sums to the valid seeds for (step, K); bins[0:360] count converged seeds.
# Peak detection and rebinning are performed during plotting.
CT_BIN_WIDTH = 1000.0
CT_MAX_TIME  = 360000.0
CT_N_BINS    = int(CT_MAX_TIME // CT_BIN_WIDTH) + 1                           # 361
CT_bin_edges = np.arange(0.0, (CT_N_BINS + 1) * CT_BIN_WIDTH, CT_BIN_WIDTH)   # (362,)
CT_bin_lefts = CT_bin_edges[:-1]                                              # (361,)
CT_bin_centers = CT_bin_lefts + CT_BIN_WIDTH / 2                              # (361,)

n_ct_nan_dropped = 0  # Total number of NaN/inf convergence times discarded.


def conv_time_bin_counts(conv_times, n_not_converged=0,
                         bin_width=CT_BIN_WIDTH, n_bins=CT_N_BINS):
    """
    Count precise convergence times in 1000-second bins and place all
    nonconverged seeds in the final bin. Return counts with shape (n_bins,).
    """
    global n_ct_nan_dropped
    counts = np.zeros(n_bins, dtype=int)

    arr = np.asarray(conv_times, dtype=float)
    if arr.size:
        good = np.isfinite(arr)
        n_ct_nan_dropped += int(arr.size - good.sum())
        arr = arr[good]
        if arr.size:
            idx = np.floor(arr / bin_width).astype(int)
            np.clip(idx, 0, n_bins - 1, out=idx)
            counts += np.bincount(idx, minlength=n_bins).astype(int)

    counts[n_bins - 1] += int(n_not_converged)
    return counts


def compute_statistics(values):
    if len(values) == 0:
        return {k: np.nan for k in
                ['mean','std','median','q1','q3','q5','q10','q90','q95','min','max']} | {'n_samples': 0}
    values = np.array(values)
    return {
        'mean':      np.mean(values),
        'std':       np.std(values),
        'median':    np.median(values),
        'q1':        np.percentile(values, 25),
        'q3':        np.percentile(values, 75),
        'q5':        np.percentile(values, 5),
        'q10':       np.percentile(values, 10),
        'q90':       np.percentile(values, 90),
        'q95':       np.percentile(values, 95),
        'min':       np.min(values),
        'max':       np.max(values),
        'n_samples': len(values),
    }

def collect_seed_data(data_dir, step_name):
    try:
        initial_cluster_sizes = np.load(f"{data_dir}/initial_cluster_sizes.npy")
        result = {
            'initial': {
                'mean_R':               np.load(f"{data_dir}/initial_mean_R.npy")[0],
                'num_clusters':         np.load(f"{data_dir}/initial_num_clusters.npy")[0],
                'largest_cluster_size': int(max(initial_cluster_sizes)) if len(initial_cluster_sizes) > 0 else 0,
                'mean_E':               np.load(f"{data_dir}/initial_mean_E.npy")[0],
            }
        }

        step_dir = f"{data_dir}/{step_name}"
        if not os.path.exists(step_dir):
            return None

        cluster_sizes = np.load(f"{step_dir}/cluster_sizes.npy")
        energy_data   = np.load(f"{step_dir}/energy_rate.npy")
        converged     = np.load(f"{step_dir}/converged.npy")[0]

        result['step'] = {
            'converged':                converged,
            'total_simulated_time':     np.load(f"{step_dir}/total_simulated_time.npy")[0],
            'precise_convergence_time': np.load(f"{step_dir}/precise_convergence_time.npy")[0],
            'mean_R':                   np.load(f"{step_dir}/mean_R.npy")[0],
            'var_R':                    np.load(f"{step_dir}/var_R.npy")[0],
            'num_clusters':             np.load(f"{step_dir}/num_clusters.npy")[0],
            'largest_cluster_size':     int(max(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
            'energy_rate':              energy_data[0],
            'E_initial':                energy_data[1],
            'E_final_avg':              energy_data[2],
            'delta_E':                  energy_data[2] - energy_data[1],
        }
        return result

    except Exception:
        return None


# ==================== Worker: load one (step, K, seed) result ====================
def load_one(step_name, K, seed):
    data_dir = (
        f"{base_data_dir}/degree_exponent_{degree_exponent:.1f}"
        f"/exponent_{exponent:.2f}/m_{m:.1f}"
        f"/K_{K:.2f}/seed_{seed}"
    )
    if not os.path.exists(data_dir):
        return (step_name, round(float(K), 2), seed, None)
    data = collect_seed_data(data_dir, step_name)
    return (step_name, round(float(K), 2), seed, data)


print("=" * 80)
print(f"degree_exponent={degree_exponent}, exponent={exponent}, m={m}")
print(f"K range: {K_values[0]:.2f} ~ {K_values[-1]:.2f}  ({len(K_values)} points)")
print("=" * 80)

stat_fields = [
    'initial_mean_R', 'initial_num_clusters', 'initial_largest_cluster',
    'convergence_time',
    'final_mean_R', 'final_num_clusters', 'final_largest_cluster',
    'delta_R', 'energy_rate', 'delta_E', 'E_initial', 'E_final_avg',
]

# ---------- Load all (step, K, seed) tasks in one ThreadPool ----------
K_to_idx = {round(float(K), 2): i for i, K in enumerate(K_values)}
# data_by_step[step][k_idx][seed] = data(dict) | None
data_by_step = {sn: [dict() for _ in range(len(K_values))] for sn in step_names}

tasks = [(sn, round(float(K), 2), seed)
         for sn in step_names
         for K in K_values
         for seed in seeds]
n_total = len(tasks)

print(f"Loading {n_total} (step, K, seed) tasks with {N_WORKERS} threads...", flush=True)
t0 = time.time()
progress_every = max(1, n_total // 20)
done = 0

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(load_one, sn, K, seed) for (sn, K, seed) in tasks]
    for fut in as_completed(futures):
        step_name, K, seed, data = fut.result()
        if data is not None:
            data_by_step[step_name][K_to_idx[K]][seed] = data
        done += 1
        if done % progress_every == 0 or done == n_total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            print(f"  [{100*done/n_total:5.1f}%] {done}/{n_total}  "
                  f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  "
                  f"({rate:.0f} tasks/s)", flush=True)

# ---------- Aggregate by step in ascending seed order ----------
for step_name in step_names:
    print(f"\n--- {step_name} ---")

    results_by_K = {
        'K_values':        [],
        'converged_ratio': [],
        'n_seeds':         [],
        'conv_time_hist':  [],   # (n_valid_K, CT_N_BINS) seed counts per 1000 seconds
    }
    for f in stat_fields:
        results_by_K[f] = {k: [] for k in
                           ['mean','std','median','q1','q3','q5','q10','q90','q95','n_samples']}

    for k_idx, K in enumerate(K_values):
        lists = {
            'initial_mean_R': [], 'initial_num_clusters': [],
            'initial_largest_cluster': [],
            'converged': [], 'convergence_time': [],
            'final_mean_R': [], 'final_num_clusters': [],
            'final_largest_cluster': [], 'delta_R': [],
            'energy_rate': [], 'delta_E': [],
            'E_initial': [], 'E_final_avg': [],
        }

        seed_data = data_by_step[step_name][k_idx]
        for seed in sorted(seed_data.keys()):
            data = seed_data[seed]
            if data is None:
                continue

            lists['initial_mean_R'].append(data['initial']['mean_R'])
            lists['initial_num_clusters'].append(data['initial']['num_clusters'])
            lists['initial_largest_cluster'].append(data['initial']['largest_cluster_size'])

            lists['converged'].append(data['step']['converged'])
            lists['final_mean_R'].append(data['step']['mean_R'])
            lists['final_num_clusters'].append(data['step']['num_clusters'])
            lists['final_largest_cluster'].append(data['step']['largest_cluster_size'])
            lists['delta_R'].append(data['step']['mean_R'] - data['initial']['mean_R'])

            if data['step']['converged']:
                lists['convergence_time'].append(data['step']['precise_convergence_time'])

            lists['energy_rate'].append(data['step']['energy_rate'])
            lists['delta_E'].append(data['step']['delta_E'])
            lists['E_initial'].append(data['step']['E_initial'])
            lists['E_final_avg'].append(data['step']['E_final_avg'])

        if len(lists['converged']) == 0:
            continue

        results_by_K['K_values'].append(K)
        results_by_K['converged_ratio'].append(np.mean(lists['converged']))
        results_by_K['n_seeds'].append(len(lists['converged']))

        # Count convergence times in 1000-second bins; nonconverged seeds use the last bin.
        conv_flags = np.asarray(lists['converged'], dtype=bool)
        n_not_conv = int(conv_flags.size - conv_flags.sum())
        results_by_K['conv_time_hist'].append(
            conv_time_bin_counts(lists['convergence_time'], n_not_conv))

        for f in stat_fields:
            stats = compute_statistics(lists[f])
            for k in ['mean','std','median','q1','q3','q5','q10','q90','q95','n_samples']:
                results_by_K[f][k].append(stats[k])

    results_by_K['K_values']        = np.array(results_by_K['K_values'])
    results_by_K['converged_ratio'] = np.array(results_by_K['converged_ratio'])
    results_by_K['n_seeds']         = np.array(results_by_K['n_seeds'])
    results_by_K['conv_time_hist']  = (np.array(results_by_K['conv_time_hist'], dtype=int)
                                       if results_by_K['conv_time_hist']
                                       else np.zeros((0, CT_N_BINS), dtype=int))

    # Include bin definitions for direct use during plotting.
    results_by_K['conv_time_bin_width']   = CT_BIN_WIDTH
    results_by_K['conv_time_bin_edges']   = CT_bin_edges
    results_by_K['conv_time_bin_lefts']   = CT_bin_lefts
    results_by_K['conv_time_bin_centers'] = CT_bin_centers

    # Sanity check: the count sum equals the number of seeds.
    if results_by_K['conv_time_hist'].size:
        _bad = np.where(results_by_K['conv_time_hist'].sum(axis=1)
                        != results_by_K['n_seeds'])[0]
        if len(_bad) > 0:
            print(f"  [WARN] conv_time hist count mismatch at {len(_bad)} K points, "
                  f"e.g. K={results_by_K['K_values'][_bad[0]]:.2f}")

    save_dir = (
        f"{save_base_dir}/degree_exponent_{degree_exponent:.1f}"
        f"/exponent_{exponent:.2f}/m_{m:.1f}/{step_name}"
    )
    os.makedirs(save_dir, exist_ok=True)

    final_results = {
        'N': N, 'degree_exponent': degree_exponent,
        'exponent': exponent, 'm': m,
        'step_name': step_name, 'max_seed': max_seed,
        'data': results_by_K,
        'conv_time_hist_info': {
            'bin_width': CT_BIN_WIDTH,
            'n_bins': CT_N_BINS,
            'layout': 'bin j = [j*1000, (j+1)*1000) for j<360; bin 360 = non-converged (+ t>=360000)',
            'row_sum': 'total valid seeds at that K',
            'note': 'peak detection / re-binning deferred to plotting stage',
        },
    }
    np.save(f"{save_dir}/K_sweep_results.npy", final_results)

    # Save counts separately as NPY for loading without opening the dictionary.
    np.save(f"{save_dir}/conv_time_hist_counts.npy", results_by_K['conv_time_hist'])
    np.save(f"{save_dir}/conv_time_hist_K_values.npy", results_by_K['K_values'])
    np.save(f"{save_dir}/conv_time_hist_bin_edges.npy", CT_bin_edges)
    np.save(f"{save_dir}/conv_time_hist_bin_lefts.npy", CT_bin_lefts)
    np.save(f"{save_dir}/conv_time_hist_bin_centers.npy", CT_bin_centers)

    with open(f"{save_dir}/K_sweep_summary.txt", 'w') as f:
        f.write(f"=== Per-K K-sweep: {step_name} ===\n")
        f.write(f"N={N}, degree_exponent={degree_exponent}, exponent={exponent}, m={m}\n")
        n_valid = len(results_by_K['K_values'])
        f.write(f"Valid K points: {n_valid}/{len(K_values)}\n")
        if n_valid > 0:
            f.write(f"Avg seeds per K: {results_by_K['n_seeds'].mean():.1f}\n")
        f.write(f"conv_time hist: bin_width={CT_BIN_WIDTH:.0f}s, n_bins={CT_N_BINS} "
                f"(bin {CT_N_BINS-1} = non-converged)\n")

    print(f"  ✓ {save_dir}")
    print(f"  valid K points: {len(results_by_K['K_values'])}/{len(K_values)}")
    if len(results_by_K['n_seeds']) > 0:
        print(f"  avg seeds/K: {results_by_K['n_seeds'].mean():.1f}")
    print(f"  conv_time_hist: {results_by_K['conv_time_hist'].shape} "
          f"@ {CT_BIN_WIDTH:.0f}s bins")

if n_ct_nan_dropped > 0:
    print(f"\n[WARN] {n_ct_nan_dropped} non-finite convergence_time values dropped from histogram")

print(f"\n{'='*80}")
print(f"Done: degree_exponent={degree_exponent}, exponent={exponent}, m={m}")
print(f"{'='*80}")
