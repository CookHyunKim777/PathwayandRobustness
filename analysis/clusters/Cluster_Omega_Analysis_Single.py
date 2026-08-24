"""Aggregate cluster mean frequencies for the single-peak case.

At every K, collect cluster mean intrinsic frequency and size directly and
summarize them by mean, standard deviation, median, Q5, Q25, Q75, and Q95.
Classification is identical to Cluster_R_Analysis_Single.py: a cluster must
contain at least 16 of the 32 positive-edge or negative-edge target
oscillators. A cluster may satisfy both classes.

Positive- and negative-edge frequencies remain separate because averaging
them together would cancel their signs. Outputs include per-class statistics,
cluster and seed counts, 20-bin size/N histograms, the four most populated size
bins, and the first K meeting the seed quorum.

Usage: python Cluster_Omega_Analysis_Single.py <degree_exponent> <exponent> <m>
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

N_WORKERS = 32  # Adjust between 16 and 64 according to NFS performance.

CLASSES = ('edge_pos', 'edge_neg')
STAT_KEYS = ('mean', 'std', 'median', 'q5', 'q25', 'q75', 'q95')

# ==================== Size-histogram settings ====================
# Normalize cluster size by N and divide [0, 1] into 20 bins.
SIZE_HIST_N_BINS = 20
SIZE_HIST_RANGE = (0.0, 1.0)
SIZE_TOP_K = 4

SIZE_bin_edges = np.linspace(SIZE_HIST_RANGE[0], SIZE_HIST_RANGE[1],
                             SIZE_HIST_N_BINS + 1)
SIZE_bin_centers = (SIZE_bin_edges[:-1] + SIZE_bin_edges[1:]) / 2


def sample_powerlaw_positive(power_exp, N, seed, MIN=1.0, MAX=4.0):
    """Generate frequencies exactly as in Cluster_R_Analysis_Single.py."""
    np.random.seed(seed)
    x    = np.arange(1, N + 1)[::-1]
    mgp1 = -power_exp + 1
    km   = MIN ** mgp1
    kM   = MAX ** mgp1
    omega = (kM + (x / N) * (km - kM)) ** (1.0 / mgp1) - MIN
    np.random.shuffle(omega)
    return omega


def build_class_sets(natural_freq, n_target=32):
    """Build positive- and negative-edge target sets once per seed."""
    order_signed = np.argsort(natural_freq)
    return (
        set(order_signed[-n_target:].tolist()),   # edge +
        set(order_signed[:n_target].tolist()),    # edge -
    )


def cluster_overlaps(cluster_indices, edge_pos_osc, edge_neg_osc):
    """Return overlaps with the positive- and negative-edge target sets."""
    cluster_set = set(np.asarray(cluster_indices).ravel().tolist())
    return (
        len(edge_pos_osc & cluster_set),
        len(edge_neg_osc & cluster_set),
    )


def stat7(values):
    """Return seven summary statistics; return NaN values for an empty input."""
    if len(values) == 0:
        return {k: np.nan for k in STAT_KEYS}
    arr = np.asarray(values, dtype=float)
    return {
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr)),
        'median': float(np.median(arr)),
        'q5':     float(np.percentile(arr, 5)),
        'q25':    float(np.percentile(arr, 25)),
        'q75':    float(np.percentile(arr, 75)),
        'q95':    float(np.percentile(arr, 95)),
    }


def top_hist_bins(values, n_bins=SIZE_HIST_N_BINS, hist_range=SIZE_HIST_RANGE,
                  bin_centers=SIZE_bin_centers, top_k=SIZE_TOP_K):
    """Build a 20-bin histogram over [0, 1] and return its top_k bins.

    Returns counts, top indices, top counts, and top centers. Unused slots are
    filled with index=-1, count=0, and center=NaN.
    """
    top_idx = np.full(top_k, -1, dtype=int)
    top_counts = np.zeros(top_k, dtype=int)
    top_centers = np.full(top_k, np.nan, dtype=float)

    if len(values) == 0:
        return np.zeros(n_bins, dtype=int), top_idx, top_counts, top_centers

    counts, _ = np.histogram(np.asarray(values, dtype=float),
                             bins=n_bins, range=hist_range)
    order = np.argsort(counts, kind='stable')[::-1]
    for j in range(min(top_k, n_bins)):
        b = order[j]
        top_idx[j] = b
        top_counts[j] = counts[b]
        top_centers[j] = bin_centers[b]

    return counts, top_idx, top_counts, top_centers


# ==================== Worker: load and classify one (K, seed) pair ====================
def load_one_seed(K, seed, base_path, degree_exponent, exponent, m,
                  natural_freq, n_target, threshold):
    """Load clusters for one (K, seed) pair and classify them by edge."""
    melnikov_dir = (f"{base_path}/melnikov_domain/"
                    f"degree_exponent_{degree_exponent:.1f}/"
                    f"exponent_{exponent:.2f}/"
                    f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")

    out = {
        'status': 'ok', 'seed': seed,
        'omegas': {c: [] for c in CLASSES},
        'sizes':  {c: [] for c in CLASSES},
        'all_clusters': [],   # All records, retained for later reclassification.
    }

    try:
        num_clusters = int(np.load(f"{melnikov_dir}/num_clusters.npy"))
        if num_clusters == 0:
            return out

        cluster_mean_freqs = np.load(f"{melnikov_dir}/cluster_mean_natural_freqs.npy")
        cluster_sizes = np.load(f"{melnikov_dir}/cluster_sizes.npy")

        edge_pos_osc, edge_neg_osc = build_class_sets(natural_freq, n_target)

        for i in range(num_clusters):
            cluster_indices = np.load(f"{melnikov_dir}/cluster_{i}_indices.npy")
            ov_pos, ov_neg = cluster_overlaps(cluster_indices,
                                              edge_pos_osc, edge_neg_osc)

            omega = float(cluster_mean_freqs[i])
            s = int(cluster_sizes[i])

            out['all_clusters'].append({
                'seed': seed, 'omega': omega, 'size': s,
                'ov_edge_pos': ov_pos, 'ov_edge_neg': ov_neg,
            })

            flags = {
                'edge_pos': ov_pos >= threshold,
                'edge_neg': ov_neg >= threshold,
            }
            for c in CLASSES:
                if flags[c]:
                    out['omegas'][c].append(omega)
                    out['sizes'][c].append(s)

        return out

    except FileNotFoundError:
        out['status'] = 'missing'
        return out
    except Exception:
        out['status'] = 'missing'
        return out


def collect_cluster_data(degree_exponent, exponent, m, K_array,
                         base_path, max_seed=64, N=1024):
    """Collect classified cluster records at every K."""
    n_K = len(K_array)
    K_to_idx = {round(float(K), 2): i for i, K in enumerate(K_array)}

    # Match the single-peak R analysis: sample positive values, then mean-center.
    natural_freqs_by_seed = {}
    for seed in range(max_seed):
        omega_raw = sample_powerlaw_positive(degree_exponent, N, seed,
                                             MIN=1.0, MAX=4.0)
        natural_freqs_by_seed[seed] = omega_raw - np.mean(omega_raw)

    n_target = N // 32           # 32
    threshold = n_target // 2    # 16

    per_K_seed_results = [dict() for _ in range(n_K)]
    total_seeds_missing = 0

    tasks = [(round(float(K), 2), seed)
             for K in K_array
             for seed in range(max_seed)]
    n_total = len(tasks)

    print(f"[omega-single] Loading {n_total} (K, seed) tasks with "
          f"{N_WORKERS} threads...", flush=True)
    t0 = time.time()
    progress_every = max(1, n_total // 20)
    done = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_K = {
            ex.submit(load_one_seed, K, seed, base_path,
                      degree_exponent, exponent, m,
                      natural_freqs_by_seed[seed], n_target, threshold): K
            for (K, seed) in tasks
        }
        for fut in as_completed(future_to_K):
            K = future_to_K[fut]
            k_idx = K_to_idx[K]
            res = fut.result()
            if res['status'] == 'missing':
                total_seeds_missing += 1
            else:
                per_K_seed_results[k_idx][res['seed']] = res
            done += 1
            if done % progress_every == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{100*done/n_total:5.1f}%] {done}/{n_total}  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  "
                      f"({rate:.0f} tasks/s)", flush=True)

    records = {c: [] for c in CLASSES}
    all_records_full = []
    counts = {c: 0 for c in CLASSES}

    for k_idx, K in enumerate(K_array):
        rec = {c: {'K': float(K), 'omegas': [], 'sizes': [], 'seeds': set()}
               for c in CLASSES}
        rec_all = {'K': float(K), 'clusters': []}

        seed_res = per_K_seed_results[k_idx]
        for seed in sorted(seed_res.keys()):
            r = seed_res[seed]

            for c in CLASSES:
                ws, ss = r['omegas'][c], r['sizes'][c]
                if ws:
                    rec[c]['omegas'].extend(ws)
                    rec[c]['sizes'].extend(ss)
                    rec[c]['seeds'].add(seed)
                    counts[c] += len(ws)

            rec_all['clusters'].extend(r['all_clusters'])

        for c in CLASSES:
            records[c].append(rec[c])
        all_records_full.append(rec_all)

    return records, all_records_full, counts, total_seeds_missing


def summarize_class(rec_list, N):
    """Build per-class K-axis arrays for statistics and size/N histograms."""
    n_K = len(rec_list)
    out = {}
    for prefix in ('omega', 'size'):
        for k in STAT_KEYS:
            out[f'{prefix}_{k}'] = np.full(n_K, np.nan)
    out['n_clusters'] = np.zeros(n_K, dtype=int)
    out['n_seeds'] = np.zeros(n_K, dtype=int)
    out['size_hist_counts'] = np.zeros((n_K, SIZE_HIST_N_BINS), dtype=int)
    out['size_top_bin_idx'] = np.full((n_K, SIZE_TOP_K), -1, dtype=int)
    out['size_top_bin_counts'] = np.zeros((n_K, SIZE_TOP_K), dtype=int)
    out['size_top_bin_centers'] = np.full((n_K, SIZE_TOP_K), np.nan)

    for i, rec in enumerate(rec_list):
        w_stat = stat7(rec['omegas'])
        s_stat = stat7(rec['sizes'])
        for k in STAT_KEYS:
            out[f'omega_{k}'][i] = w_stat[k]
            out[f'size_{k}'][i] = s_stat[k]
        out['n_clusters'][i] = len(rec['omegas'])
        out['n_seeds'][i] = len(rec['seeds'])

        sizes_norm = np.asarray(rec['sizes'], dtype=float) / N
        c_all, t_idx, t_cnt, t_ctr = top_hist_bins(sizes_norm)
        out['size_hist_counts'][i] = c_all
        out['size_top_bin_idx'][i] = t_idx
        out['size_top_bin_counts'][i] = t_cnt
        out['size_top_bin_centers'][i] = t_ctr

    return out


def find_Kc(n_seeds_arr, K_array, quorum):
    """Return the first K at which the number of matching seeds reaches quorum."""
    idx = np.nonzero(n_seeds_arr >= quorum)[0]
    return float(K_array[idx[0]]) if len(idx) else None


if __name__ == "__main__":

    degree_exponent = float(sys.argv[1])
    exponent = float(sys.argv[2])
    m = float(sys.argv[3])

    K_array = np.round(np.arange(0.0, 20.0, 0.1), 2)
    max_seed = 64
    N = 1024
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(repo_root / "results"))).expanduser()
    base_path = str(data_root / "DF_single_peak")
    seed_quorum = max_seed // 2      # K_c requires at least 32 seeds.

    records, records_full, counts, n_missing = collect_cluster_data(
        degree_exponent, exponent, m, K_array, base_path, max_seed, N)

    save_dir = (f"{base_path}/cluster_omega_analysis/"
                f"degree_exponent_{degree_exponent:.1f}/"
                f"exponent_{exponent:.2f}/"
                f"m_{m:.1f}")
    os.makedirs(save_dir, exist_ok=True)

    save_data = {
        'K_array': K_array,
        'n_K': len(K_array),
        'classes': list(CLASSES),
    }

    Kc = {}
    for c in CLASSES:
        summ = summarize_class(records[c], N)
        for key, val in summ.items():
            save_data[f'{c}_{key}'] = val
        Kc[c] = find_Kc(summ['n_seeds'], K_array, seed_quorum)
        save_data[f'K_c_{c}'] = Kc[c]

    valid = [v for v in Kc.values() if v is not None]
    save_data['K_critical'] = min(valid) if valid else None
    save_data['seed_quorum'] = seed_quorum
    save_data['K_c_def'] = ('first K where >= seed_quorum seeds carry a cluster '
                            'of that class')

    save_data.update({
        # Size-histogram metadata.
        'size_hist_n_bins': SIZE_HIST_N_BINS,
        'size_hist_range': np.array(SIZE_HIST_RANGE),
        'size_bin_edges': SIZE_bin_edges,
        'size_bin_centers': SIZE_bin_centers,
        'size_top_k': SIZE_TOP_K,
        'size_normalization': 'size / N',

        'degree_exponent': degree_exponent,
        'exponent': exponent,
        'm': m,
        'max_seed': max_seed,
        'N': N,
        'n_target': N // 32,
        'threshold': (N // 32) // 2,
        'omega_setup': ('single_peak; sample_powerlaw_positive + zero-mean; '
                        'edge_pos = argsort(omega)[-32:], '
                        'edge_neg = argsort(omega)[:32] '
                        '(identical to Cluster_R_Analysis_Single.py)'),
        'aggregation': ('plain per-K aggregation of classified clusters; '
                        'no KDE peak finding'),
        'total_edge_pos_found': counts['edge_pos'],
        'total_edge_neg_found': counts['edge_neg'],
        'total_seeds_missing':  n_missing,
    })

    np.save(f"{save_dir}/omega_aggregation_results.npy", save_data)

    raw_data = {
        'records_edge_pos': records['edge_pos'],
        'records_edge_neg': records['edge_neg'],
        # All clusters: seed, omega, size, and two overlap measures.
        'all_records_full': records_full,
        'K_array': K_array,
        'n_target': N // 32,
        'threshold': (N // 32) // 2,
    }
    np.save(f"{save_dir}/omega_raw_records.npy", raw_data)

    print(f"\nDone! deg={degree_exponent:.1f}, exp={exponent:.2f}, m={m:.1f}")
    print(f"  K points: {len(K_array)}")
    print(f"  Edge+ clusters found: {counts['edge_pos']}")
    print(f"  Edge- clusters found: {counts['edge_neg']}")
    print(f"  Seeds missing: {n_missing}")
    print(f"  K_c[edge_pos] = {Kc['edge_pos']}, K_c[edge_neg] = {Kc['edge_neg']}, "
          f"K_critical = {save_data['K_critical']}")
