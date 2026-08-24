"""
[SINGLE PEAK] Collect per-K convergence statistics without K binning, using parallel I/O.
- Convergence-time statistics.
- Convergence-rate statistics.
- Separate output for every K value.
- Trivial seeds have conv_time=0; nonconverged seeds use CENSORED_TIME=360000 s.

Relative to the double-peak version, only base_path and the stored
omega_setup='single_peak' tag differ; the analysis is otherwise identical.

[I/O] Load all (K, seed) pairs in a ThreadPool, then aggregate them in ascending
seed order to preserve the original ordering.

[Distribution] Store seed counts at the 1000-second cluster-measurement
resolution. Peak detection and rebinning are deferred to the plotting stage.

Usage: python Convergence_Time_Collect_Single.py <degree_exponent> <exponent> <m>
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json
from scipy import stats as sp_stats

N_WORKERS = 32  # Adjust between 16 and 64 according to NFS performance.

if len(sys.argv) < 4:
    print("Usage: python Convergence_Time_Collect_Single.py <degree_exponent> <exponent> <m>")
    sys.exit(1)

degree_exponent = float(sys.argv[1])
exponent = float(sys.argv[2])
m = float(sys.argv[3])

print(f"[single_peak] Processing: deg={degree_exponent:.1f}, exp={exponent:.2f}, m={m:.1f}")

CENSORED_TIME = 360000.0
TRIVIAL_TIME  = 0.0

K_array = np.round(np.arange(0.0, 20.0, 0.1), 2)
n_K = len(K_array)
max_seed = 64

repo_root = Path(__file__).resolve().parents[2]
data_root = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(repo_root / "results"))).expanduser()
base_path = data_root / "DF_single_peak"
data_base = base_path / "convergence_info"
save_base = base_path / "convergence_statistics_cluster"

def load_npy_scalar(path):
    try:
        t = np.load(path, allow_pickle=True)
        if t.shape == ():
            return float(t)
        elif t.size == 1:
            return float(t.flat[0])
        return None
    except Exception:
        return None

def load_npy_array(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None

def compute_mode(values):
    if len(values) == 0:
        return np.nan
    arr = np.array(values)
    mode_result = sp_stats.mode(arr, keepdims=True)
    return float(mode_result.mode[0])

# ==================== Convergence-time counts (1000-second bins) ====================
# Count all samples, including trivial=0 and nonconverged=CENSORED_TIME cases.
# bin j  : t in [j*1000, (j+1)*1000)   (j = 0 ... 359)
# bin 360: t = CENSORED_TIME, reserved for nonconverged samples.
# bin 0 contains only trivial samples because actual convergence starts at 1000 s.
# Peak detection and rebinning are performed during plotting.
CT_BIN_WIDTH = 1000.0
CT_N_BINS = int(CENSORED_TIME // CT_BIN_WIDTH) + 1                            # 361
CT_bin_edges = np.arange(0.0, (CT_N_BINS + 1) * CT_BIN_WIDTH, CT_BIN_WIDTH)   # (362,)
CT_bin_lefts = CT_bin_edges[:-1]                                              # (361,)
CT_bin_centers = CT_bin_lefts + CT_BIN_WIDTH / 2                              # (361,)


def conv_time_bin_counts(values, bin_width=CT_BIN_WIDTH, n_bins=CT_N_BINS):
    """
    Count convergence-time samples, including censored values, in 1000-second bins.
    Return counts with shape (n_bins,); the sum equals the number of seeds at K.
    """
    if len(values) == 0:
        return np.zeros(n_bins, dtype=int)
    arr = np.asarray(values, dtype=float)
    idx = np.floor(arr / bin_width).astype(int)
    np.clip(idx, 0, n_bins - 1, out=idx)
    return np.bincount(idx, minlength=n_bins).astype(int)


# ==================== Worker: load scalars for one (K, seed) pair ====================
def load_one_seed(k_idx, K, seed):
    """
    Load the required scalars from one (K, seed) directory.
    Branching and aggregation are performed in ascending seed order by the caller.

    Returned dictionary:
      status : 'ok' | 'missing'
      k_idx, seed
      converged/trivial/conv_time/num_cl/total_time (scalar or None)
    """
    seed_dir = (data_base /
                f"degree_exponent_{degree_exponent:.1f}" /
                f"exponent_{exponent:.2f}" /
                f"m_{m:.1f}" /
                f"K_{K:.2f}" /
                f"seed_{seed}")

    if not seed_dir.exists():
        return {'status': 'missing', 'k_idx': k_idx, 'seed': seed}

    converged = load_npy_scalar(seed_dir / "converged.npy")
    if converged is None:
        return {'status': 'missing', 'k_idx': k_idx, 'seed': seed}

    trivial = load_npy_scalar(seed_dir / "trivial_state.npy")
    conv_time = load_npy_scalar(seed_dir / "precise_convergence_time.npy")
    num_cl = load_npy_scalar(seed_dir / "num_clusters.npy")
    total_time = load_npy_scalar(seed_dir / "total_simulated_time.npy")

    return {
        'status': 'ok', 'k_idx': k_idx, 'seed': seed,
        'converged': converged, 'trivial': trivial,
        'conv_time': conv_time, 'num_cl': num_cl, 'total_time': total_time,
    }


# ==================== Per-K collection containers ====================
conv_time_by_K = [[] for _ in range(n_K)]
not_conv_time_by_K = [[] for _ in range(n_K)]
num_clusters_by_K = [[] for _ in range(n_K)]

n_seeds_by_K = np.zeros(n_K, dtype=int)
n_converged_by_K = np.zeros(n_K, dtype=int)
n_not_converged_by_K = np.zeros(n_K, dtype=int)
n_trivial_by_K = np.zeros(n_K, dtype=int)

detailed_conv_times = {}
detailed_not_conv_times = {}
detailed_num_clusters = {}
detailed_converged_seeds = {}
detailed_not_converged_seeds = {}
detailed_trivial_seeds = {}

per_K_rates = []
per_K_conv_stats = []

n_loaded = 0
n_missing = 0

# ---------- Parallel loading of raw scalars for every (K, seed) pair ----------
# per_K_raw[k_idx][seed] = result dictionary; only status='ok' is retained.
per_K_raw = [dict() for _ in range(n_K)]

tasks = [(k_idx, K, seed)
         for k_idx, K in enumerate(K_array)
         for seed in range(max_seed)]
n_total = len(tasks)

print(f"Loading {n_total} (K, seed) tasks with {N_WORKERS} threads...", flush=True)
t0 = time.time()
progress_every = max(1, n_total // 20)
done = 0

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(load_one_seed, k_idx, round(float(K), 2), seed)
               for (k_idx, K, seed) in tasks]
    for fut in as_completed(futures):
        r = fut.result()
        if r['status'] == 'missing':
            n_missing += 1
        else:
            per_K_raw[r['k_idx']][r['seed']] = r
        done += 1
        if done % progress_every == 0 or done == n_total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            print(f"  [{100*done/n_total:5.1f}%] {done}/{n_total}  "
                  f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  "
                  f"({rate:.0f} tasks/s)", flush=True)

# ---------- Aggregate in ascending seed order to preserve ordering and counts ----------
for k_idx, K in enumerate(K_array):
    K_key = f"{K:.2f}"

    k_conv_times = []
    k_not_conv_times = []
    k_num_clusters = []
    k_converged_seeds = []
    k_not_converged_seeds = []
    k_trivial_seeds = []
    k_conv_total_sim_times = []

    n_seeds_this_K = 0

    seed_raw = per_K_raw[k_idx]
    for seed in sorted(seed_raw.keys()):
        r = seed_raw[seed]
        converged = r['converged']

        is_converged = bool(converged)
        n_loaded += 1
        n_seeds_by_K[k_idx] += 1
        n_seeds_this_K += 1

        trivial = r['trivial']
        is_trivial = bool(trivial) if trivial is not None else False
        if is_trivial:
            n_trivial_by_K[k_idx] += 1
            k_trivial_seeds.append(seed)

        conv_time = r['conv_time']
        num_cl = r['num_cl']
        total_time = r['total_time']

        if is_trivial:
            t = TRIVIAL_TIME
            if num_cl is not None and not np.isnan(num_cl):
                num_clusters_by_K[k_idx].append(num_cl)
                k_num_clusters.append(num_cl)
            if total_time is not None and not np.isnan(total_time):
                k_conv_total_sim_times.append(total_time)
            n_converged_by_K[k_idx] += 1
            k_converged_seeds.append(seed)
        elif is_converged:
            t = conv_time if (conv_time is not None and not np.isnan(conv_time)) else CENSORED_TIME
            if num_cl is not None and not np.isnan(num_cl):
                num_clusters_by_K[k_idx].append(num_cl)
                k_num_clusters.append(num_cl)
            if total_time is not None and not np.isnan(total_time):
                k_conv_total_sim_times.append(total_time)
            n_converged_by_K[k_idx] += 1
            k_converged_seeds.append(seed)
        else:
            t = CENSORED_TIME
            n_not_converged_by_K[k_idx] += 1
            k_not_converged_seeds.append(seed)
            if total_time is not None and not np.isnan(total_time):
                not_conv_time_by_K[k_idx].append(total_time)
                k_not_conv_times.append(total_time)

        conv_time_by_K[k_idx].append(t)
        k_conv_times.append(t)

    if k_conv_times:
        detailed_conv_times[K_key] = k_conv_times
    if k_not_conv_times:
        detailed_not_conv_times[K_key] = k_not_conv_times
    if k_num_clusters:
        detailed_num_clusters[K_key] = k_num_clusters
    if k_converged_seeds:
        detailed_converged_seeds[K_key] = k_converged_seeds
    if k_not_converged_seeds:
        detailed_not_converged_seeds[K_key] = k_not_converged_seeds
    if k_trivial_seeds:
        detailed_trivial_seeds[K_key] = k_trivial_seeds

    if n_seeds_this_K > 0:
        n_conv = len(k_converged_seeds)
        n_not_conv = len(k_not_converged_seeds)
        n_triv = len(k_trivial_seeds)

        per_K_rates.append({
            'K': K,
            'N_seeds': n_seeds_this_K,
            'Converged_Count': n_conv,
            'Converged_Rate': n_conv / n_seeds_this_K,
            'NotConverged_Count': n_not_conv,
            'NotConverged_Rate': n_not_conv / n_seeds_this_K,
            'Trivial_Count': n_triv,
            'Trivial_Rate': n_triv / n_seeds_this_K,
        })

        arr_ct = np.array(k_conv_times)
        per_K_conv_stats.append({
            'K': K,
            'Count': n_seeds_this_K,
            'Converged_Count': n_conv,
            'Converged_Rate': n_conv / n_seeds_this_K,
            'PreciseConvTime_Mean': float(np.mean(arr_ct)),
            'PreciseConvTime_Median': float(np.median(arr_ct)),
            'PreciseConvTime_Std': float(np.std(arr_ct)),
            'PreciseConvTime_Q5': float(np.percentile(arr_ct, 5)),
            'PreciseConvTime_Q95': float(np.percentile(arr_ct, 95)),
            'TotalSimTime_Mean': float(np.mean(k_conv_total_sim_times)) if k_conv_total_sim_times else np.nan,
            'NumClusters_Mean': float(np.mean(k_num_clusters)) if k_num_clusters else np.nan,
            'NumClusters_Mode': compute_mode(k_num_clusters),
        })

print(f"\nLoaded: {n_loaded}, Missing: {n_missing}")

def compute_stats(values):
    if len(values) == 0:
        return {k: np.nan for k in
                ['mean', 'var', 'std', 'median',
                 'q5', 'q25', 'q75', 'q95', 'min', 'max', 'count']}
    arr = np.array(values)
    return {
        'count': len(arr),
        'mean': float(np.mean(arr)),
        'var': float(np.var(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'q5': float(np.percentile(arr, 5)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'q95': float(np.percentile(arr, 95)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }

ct_stats = [compute_stats(conv_time_by_K[i]) for i in range(n_K)]
nc_stats = [compute_stats(num_clusters_by_K[i]) for i in range(n_K)]

# Per-K convergence-time counts in 1000-second bins, including censored samples.
ct_hist_counts = np.zeros((n_K, CT_N_BINS), dtype=int)
for i in range(n_K):
    ct_hist_counts[i] = conv_time_bin_counts(conv_time_by_K[i])

# Sanity check: total counts should equal the number of seeds; save regardless.
_bad = np.where(ct_hist_counts.sum(axis=1) != n_seeds_by_K)[0]
if len(_bad) > 0:
    print(f"[WARN] conv_time hist count mismatch at {len(_bad)} K points, "
          f"e.g. K={K_array[_bad[0]]:.2f}")

with np.errstate(divide='ignore', invalid='ignore'):
    converged_rate = np.where(n_seeds_by_K > 0,
                              n_converged_by_K / n_seeds_by_K, np.nan)
    not_converged_rate = np.where(n_seeds_by_K > 0,
                                  n_not_converged_by_K / n_seeds_by_K, np.nan)
    trivial_rate = np.where(n_seeds_by_K > 0,
                            n_trivial_by_K / n_seeds_by_K, np.nan)

print(f"\n=== Per-K summary ===")
print(f"K points with data: {np.sum(n_seeds_by_K > 0)} / {n_K}")
for i in range(n_K):
    if n_seeds_by_K[i] > 0:
        print(f"  K={K_array[i]:.2f}: n_seeds={n_seeds_by_K[i]}, "
              f"conv_rate={converged_rate[i]:.2f}, "
              f"conv_time_median={ct_stats[i]['median']:.1f}")

save_dir = (save_base /
            f"degree_exponent_{degree_exponent:.1f}" /
            f"exponent_{exponent:.2f}" /
            f"m_{m:.1f}" /
            "per_K_statistics")
save_dir.mkdir(parents=True, exist_ok=True)

np.save(save_dir / "K_array.npy", K_array)

stat_keys = ['mean', 'var', 'std', 'median', 'q5', 'q25', 'q75', 'q95', 'min', 'max', 'count']
for key in stat_keys:
    np.save(save_dir / f"conv_time_{key}.npy", np.array([s[key] for s in ct_stats]))
    np.save(save_dir / f"num_clusters_{key}.npy", np.array([s[key] for s in nc_stats]))

np.save(save_dir / "rate_n_seeds.npy", n_seeds_by_K)
np.save(save_dir / "rate_converged_count.npy", n_converged_by_K)
np.save(save_dir / "rate_not_converged_count.npy", n_not_converged_by_K)
np.save(save_dir / "rate_trivial_count.npy", n_trivial_by_K)
np.save(save_dir / "rate_converged.npy", converged_rate)
np.save(save_dir / "rate_not_converged.npy", not_converged_rate)
np.save(save_dir / "rate_trivial.npy", trivial_rate)

# Convergence-time counts in 1000-second bins.
np.save(save_dir / "conv_time_hist_counts.npy", ct_hist_counts)       # (n_K, 361)
np.save(save_dir / "conv_time_hist_bin_edges.npy", CT_bin_edges)      # (362,)
np.save(save_dir / "conv_time_hist_bin_lefts.npy", CT_bin_lefts)      # (361,)
np.save(save_dir / "conv_time_hist_bin_centers.npy", CT_bin_centers)  # (361,)

# Wide count CSV; columns are the left edges of the time bins.
_hist_cols = [f"t_{int(l)}" for l in CT_bin_lefts]
df_hist = pd.DataFrame(ct_hist_counts, columns=_hist_cols)
df_hist.insert(0, 'K', K_array)
df_hist.to_csv(save_dir / "per_K_conv_time_hist.csv", index=False)

df_ct = pd.DataFrame({
    'K': K_array,
    'count': [int(s['count']) if not np.isnan(s['count']) else 0 for s in ct_stats],
    **{k: [s[k] for s in ct_stats] for k in ['mean','var','std','median','q5','q25','q75','q95','min','max']},
})
df_ct.to_csv(save_dir / "per_K_conv_time.csv", index=False)

df_nc = pd.DataFrame({
    'K': K_array,
    'count': [int(s['count']) if not np.isnan(s['count']) else 0 for s in nc_stats],
    **{k: [s[k] for s in nc_stats] for k in ['mean','var','std','median','q5','q25','q75','q95','min','max']},
})
df_nc.to_csv(save_dir / "per_K_num_clusters.csv", index=False)

df_rate = pd.DataFrame({
    'K': K_array,
    'n_seeds': n_seeds_by_K,
    'converged_count': n_converged_by_K,
    'not_converged_count': n_not_converged_by_K,
    'trivial_count': n_trivial_by_K,
    'converged_rate': converged_rate,
    'not_converged_rate': not_converged_rate,
    'trivial_rate': trivial_rate,
})
df_rate.to_csv(save_dir / "per_K_rate.csv", index=False)

metadata = {
    'parameters': {'degree_exponent': degree_exponent, 'exponent': exponent, 'm': m},
    'binning': {'K_binning': False, 'K_step': 0.1, 'n_K': n_K},
    'censoring': {
        'censored_time': CENSORED_TIME,
        'trivial_time': TRIVIAL_TIME,
        'description': 'Trivial seeds assigned 0, non-converged seeds assigned CENSORED_TIME in conv_time statistics',
    },
    'conv_time_stats': {
        'n_K_with_data': int(sum(1 for s in ct_stats if not np.isnan(s['count']) and s['count'] > 0)),
        'total_samples': int(sum(s['count'] for s in ct_stats if not np.isnan(s['count']))),
    },
    'conv_time_hist': {
        'bin_width': CT_BIN_WIDTH,
        'n_bins': CT_N_BINS,
        'layout': 'bin j = [j*1000, (j+1)*1000) for j<360; bin 360 = censored (t=CENSORED_TIME)',
        'sample': 'censored-included (trivial=0 -> bin 0, non-converged -> bin 360)',
        'note': 'peak detection / re-binning deferred to plotting stage',
    },
    'rate_stats': {
        'n_K_with_data': int(np.sum(n_seeds_by_K > 0)),
        'total_seeds': int(np.sum(n_seeds_by_K)),
    },
    'source': {
        'data_path': str(data_base),
        'K_range': [float(K_array[0]), float(K_array[-1])],
        'K_step': 0.1,
        'max_seed': max_seed,
        'n_loaded': n_loaded,
        'n_missing': n_missing,
    },
    'omega_setup': 'single_peak',
    'description': 'Per-K convergence time statistics. Trivial seeds: t=0, Non-converged seeds: t=CENSORED_TIME.',
}
with open(save_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

with open(save_dir / "detailed_data.json", 'w') as f:
    json.dump({
        'all_conv_times_censored': detailed_conv_times,
        'not_converged_sim_times': detailed_not_conv_times,
        'num_clusters': detailed_num_clusters,
        'converged_seeds': detailed_converged_seeds,
        'not_converged_seeds': detailed_not_converged_seeds,
        'trivial_seeds': detailed_trivial_seeds,
    }, f)

per_K_dir = (save_base /
             f"degree_exponent_{degree_exponent:.1f}" /
             f"exponent_{exponent:.2f}" /
             f"m_{m:.1f}")
per_K_dir.mkdir(parents=True, exist_ok=True)

if per_K_rates:
    pd.DataFrame(per_K_rates).to_csv(per_K_dir / "convergence_rates.csv", index=False)
if per_K_conv_stats:
    pd.DataFrame(per_K_conv_stats).to_csv(per_K_dir / "convergence_stats_converged.csv", index=False)
with open(per_K_dir / "seed_lists.json", "w") as f:
    json.dump({
        'converged_seeds': detailed_converged_seeds,
        'not_converged_seeds': detailed_not_converged_seeds,
        'trivial_seeds': detailed_trivial_seeds,
    }, f)

print(f"\n{'='*60}")
print(f"Results saved:")
print(f"  Per-K stats: {save_dir}")
print(f"  Per-K CSV:   {per_K_dir}")
print(f"{'='*60}")
print(f"  [Per-K npy] conv_time_*.npy, num_clusters_*.npy, rate_*.npy, K_array.npy")
print(f"  [Per-K npy] conv_time_hist_counts.npy ({n_K}, {CT_N_BINS}) @ {CT_BIN_WIDTH:.0f}s bins")
print(f"  [Per-K CSV] per_K_conv_time.csv, per_K_num_clusters.csv, per_K_rate.csv")
print(f"  [Per-K CSV] per_K_conv_time_hist.csv")
print(f"  [Per-K CSV] convergence_rates.csv, convergence_stats_converged.csv")
print(f"  [JSON]      seed_lists.json, detailed_data.json, metadata.json")
print(f"  [Note]      Trivial seeds: t={TRIVIAL_TIME:.0f}s, Non-converged seeds: t={CENSORED_TIME:.0f}s")
print(f"  [Note]      Hist bin 0 = trivial, bin {CT_N_BINS-1} = censored, bins 1..{CT_N_BINS-2} = actual convergence")
