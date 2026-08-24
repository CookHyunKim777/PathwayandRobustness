"""Analyze R distributions of edge clusters in the single-peak case.

Classification requires at least 16 of 32 target oscillators: the largest
signed ω values for positive-edge clusters and the smallest signed ω values
for negative-edge clusters. All seeds, including trivial cases, are retained
regardless of convergence.

Statistics are stored separately at every K without K binning. Q25 and Q75 are
included with the median, Q5, and Q95. For each edge class, a 20-bin histogram
over R in [0, 1] records the four most populated bins. All (K, seed) tasks are
loaded and processed in parallel.

Usage: python Cluster_R_Analysis_Single.py <degree_exponent> <exponent> <m>
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

N_WORKERS = 32  # Adjust between 16 and 64 according to NFS performance.


def sample_powerlaw_positive(power_exp, N, seed, MIN=1.0, MAX=4.0):
    np.random.seed(seed)
    x    = np.arange(1, N + 1)[::-1]
    mgp1 = -power_exp + 1
    km   = MIN ** mgp1
    kM   = MAX ** mgp1
    omega = (kM + (x / N) * (km - kM)) ** (1.0 / mgp1) - MIN
    np.random.shuffle(omega)
    return omega


def find_edge_positive_clusters(cluster_list, natural_freq, n_target=32, threshold=16):
    edge_oscillators = set(np.argsort(natural_freq)[-n_target:])
    result = []
    for i, cluster in enumerate(cluster_list):
        cluster_set = set(cluster)
        overlap = len(edge_oscillators & cluster_set)
        if overlap >= threshold:
            result.append(i)
    return result


def find_edge_negative_clusters(cluster_list, natural_freq, n_target=32, threshold=16):
    edge_oscillators = set(np.argsort(natural_freq)[:n_target])
    result = []
    for i, cluster in enumerate(cluster_list):
        cluster_set = set(cluster)
        overlap = len(edge_oscillators & cluster_set)
        if overlap >= threshold:
            result.append(i)
    return result


# ==================== Quantile and histogram helpers ====================

# Divide R in [0, 1] into 20 histogram bins.
R_HIST_N_BINS = 20
R_HIST_RANGE = (0.0, 1.0)
R_TOP_K = 4  # Number of most populated bins to retain.

R_bin_edges = np.linspace(R_HIST_RANGE[0], R_HIST_RANGE[1], R_HIST_N_BINS + 1)
R_bin_centers = (R_bin_edges[:-1] + R_bin_edges[1:]) / 2


def quantile_stats(values):
    """Return median, Q5, Q25, Q75, and Q95; return NaN values if empty."""
    if len(values) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    arr = np.asarray(values, dtype=float)
    return (
        np.median(arr),
        np.percentile(arr, 5),
        np.percentile(arr, 25),
        np.percentile(arr, 75),
        np.percentile(arr, 95),
    )


def top_hist_bins(values, n_bins=R_HIST_N_BINS, hist_range=R_HIST_RANGE, top_k=R_TOP_K):
    """
    Bin values into a 20-bin histogram over [0, 1] and return the top_k bins.

    Returns:
      counts        : shape (n_bins,), all histogram counts
      top_idx       : shape (top_k,), indices in descending count order; -1 if unused
      top_counts    : shape (top_k,), corresponding counts; 0 if unused
      top_centers   : shape (top_k,), corresponding bin centers; NaN if unused
    """
    top_idx = np.full(top_k, -1, dtype=int)
    top_counts = np.zeros(top_k, dtype=int)
    top_centers = np.full(top_k, np.nan, dtype=float)

    if len(values) == 0:
        return np.zeros(n_bins, dtype=int), top_idx, top_counts, top_centers

    counts, _ = np.histogram(np.asarray(values, dtype=float),
                             bins=n_bins, range=hist_range)

    # Sort by descending count; ties favor the lower bin index.
    order = np.argsort(counts, kind='stable')[::-1]
    n_fill = min(top_k, n_bins)
    for j in range(n_fill):
        b = order[j]
        top_idx[j] = b
        top_counts[j] = counts[b]
        top_centers[j] = R_bin_centers[b]

    return counts, top_idx, top_counts, top_centers


# ==================== Worker: load and classify one (K, seed) pair ====================
def process_one_seed(K, seed, base_path, degree_exponent, exponent, m,
                     natural_freq, n_target, threshold):
    """
    Load NPY files for one (K, seed) pair and extract edge-cluster R values.
    Natural frequencies are precomputed per seed and mean-centered.

    Returned dictionary:
      status            : 'ok' | 'skipped_file'
      not_converged     : bool
      trivial           : bool
      system_R          : float | None
      edge_pos_Rs       : list[float]
      edge_neg_Rs       : list[float]
      n_edge_pos / n_edge_neg : int
    """
    conv_dir = (f"{base_path}/convergence_info/"
                f"degree_exponent_{degree_exponent:.1f}/"
                f"exponent_{exponent:.2f}/"
                f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")
    melnikov_dir = (f"{base_path}/melnikov_domain/"
                    f"degree_exponent_{degree_exponent:.1f}/"
                    f"exponent_{exponent:.2f}/"
                    f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")
    order_dir = (f"{base_path}/order_parameter/"
                 f"degree_exponent_{degree_exponent:.1f}/"
                 f"exponent_{exponent:.2f}/"
                 f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")

    out = {
        'status': 'ok',
        'not_converged': False,
        'trivial': False,
        'system_R': None,
        'edge_pos_Rs': [],
        'edge_neg_Rs': [],
        'n_edge_pos': 0,
        'n_edge_neg': 0,
    }

    try:
        try:
            converged = bool(np.load(f"{conv_dir}/converged.npy")[0])
            trivial = bool(np.load(f"{conv_dir}/trivial_state.npy")[0])
        except FileNotFoundError:
            converged = False
            trivial = False

        if not converged:
            out['not_converged'] = True
        if trivial:
            out['trivial'] = True

        mean_R = np.load(f"{order_dir}/mean_R.npy")[0]
        out['system_R'] = mean_R

        num_clusters = int(np.load(f"{melnikov_dir}/num_clusters.npy"))

        if num_clusters == 0:
            return out  # status='ok'; no clusters were detected.

        cluster_Rs = np.load(f"{melnikov_dir}/cluster_Rs.npy")

        cluster_list = []
        for i in range(num_clusters):
            cluster_indices = np.load(f"{melnikov_dir}/cluster_{i}_indices.npy")
            cluster_list.append(cluster_indices)

        edge_pos_indices = find_edge_positive_clusters(cluster_list, natural_freq, n_target, threshold)
        edge_neg_indices = find_edge_negative_clusters(cluster_list, natural_freq, n_target, threshold)

        for idx in edge_pos_indices:
            out['edge_pos_Rs'].append(cluster_Rs[idx])
        out['n_edge_pos'] = len(edge_pos_indices)

        for idx in edge_neg_indices:
            out['edge_neg_Rs'].append(cluster_Rs[idx])
        out['n_edge_neg'] = len(edge_neg_indices)

        return out

    except FileNotFoundError:
        out['status'] = 'skipped_file'
        return out
    except Exception:
        out['status'] = 'skipped_file'
        return out


if __name__ == "__main__":

    degree_exponent = float(sys.argv[1])
    exponent = float(sys.argv[2])
    m = float(sys.argv[3])

    print(f"Processing: deg={degree_exponent:.1f}, exp={exponent:.2f}, m={m:.1f}")

    K_array = np.round(np.arange(0.0, 20.0, 0.1), 2)
    n_K = len(K_array)
    max_seed = 64
    N = 1024
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(repo_root / "results"))).expanduser()
    base_path = str(data_root / "DF_single_peak")

    n_target = N // 32
    threshold = n_target // 2

    print(f"Cluster criteria: n_target={n_target}, threshold={threshold}")
    print(f"R histogram: {R_HIST_N_BINS} bins over {R_HIST_RANGE}, top-{R_TOP_K} bins saved")

    # Precompute mean-centered single-peak natural frequencies.
    natural_freqs_by_seed = {}
    for seed in range(max_seed):
        omega_raw = sample_powerlaw_positive(degree_exponent, N, seed, MIN=1.0, MAX=4.0)
        natural_freqs_by_seed[seed] = omega_raw - np.mean(omega_raw)

    # Map K to output indices.
    K_to_idx = {round(float(K), 2): i for i, K in enumerate(K_array)}

    # Initialize per-K collection lists.
    edge_pos_R_by_K = [[] for _ in range(n_K)]
    edge_neg_R_by_K = [[] for _ in range(n_K)]
    system_R_by_K = [[] for _ in range(n_K)]

    total_seeds_used = 0
    total_seeds_not_converged = 0
    total_seeds_trivial = 0
    total_seeds_skipped_file = 0
    total_edge_pos_found = 0
    total_edge_neg_found = 0

    # ---------- Process all (K, seed) tasks in one ThreadPool ----------
    tasks = [(round(float(K), 2), seed)
             for K in K_array
             for seed in range(max_seed)]
    n_total = len(tasks)

    print(f"Loading/processing {n_total} (K, seed) tasks with {N_WORKERS} threads...",
          flush=True)
    t0 = time.time()
    progress_every = max(1, n_total // 20)
    done = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_K = {
            ex.submit(process_one_seed, K, seed, base_path,
                      degree_exponent, exponent, m,
                      natural_freqs_by_seed[seed], n_target, threshold): K
            for (K, seed) in tasks
        }
        for fut in as_completed(future_to_K):
            K = future_to_K[fut]
            k_idx = K_to_idx[K]
            res = fut.result()

            if res['status'] == 'skipped_file':
                total_seeds_skipped_file += 1
                done += 1
                if done % progress_every == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"  [{100*done/n_total:5.1f}%] {done}/{n_total}  "
                          f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  "
                          f"({rate:.0f} tasks/s)", flush=True)
                continue

            # status 'ok'
            if res['not_converged']:
                total_seeds_not_converged += 1
            if res['trivial']:
                total_seeds_trivial += 1

            if res['system_R'] is not None:
                system_R_by_K[k_idx].append(res['system_R'])

            edge_pos_R_by_K[k_idx].extend(res['edge_pos_Rs'])
            edge_neg_R_by_K[k_idx].extend(res['edge_neg_Rs'])

            total_edge_pos_found += res['n_edge_pos']
            total_edge_neg_found += res['n_edge_neg']
            total_seeds_used += 1

            done += 1
            if done % progress_every == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{100*done/n_total:5.1f}%] {done}/{n_total}  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  "
                      f"({rate:.0f} tasks/s)", flush=True)

    # ==================== Per-K statistics without binning ====================

    edge_pos_R_median = np.full(n_K, np.nan)
    edge_pos_R_q5     = np.full(n_K, np.nan)
    edge_pos_R_q25    = np.full(n_K, np.nan)
    edge_pos_R_q75    = np.full(n_K, np.nan)
    edge_pos_R_q95    = np.full(n_K, np.nan)

    edge_neg_R_median = np.full(n_K, np.nan)
    edge_neg_R_q5     = np.full(n_K, np.nan)
    edge_neg_R_q25    = np.full(n_K, np.nan)
    edge_neg_R_q75    = np.full(n_K, np.nan)
    edge_neg_R_q95    = np.full(n_K, np.nan)

    system_R_mean   = np.full(n_K, np.nan)
    system_R_var    = np.full(n_K, np.nan)
    system_R_median = np.full(n_K, np.nan)
    system_R_q5     = np.full(n_K, np.nan)
    system_R_q25    = np.full(n_K, np.nan)
    system_R_q75    = np.full(n_K, np.nan)
    system_R_q95    = np.full(n_K, np.nan)

    edge_pos_n = np.array([len(x) for x in edge_pos_R_by_K])
    edge_neg_n = np.array([len(x) for x in edge_neg_R_by_K])
    system_n   = np.array([len(x) for x in system_R_by_K])

    # Positive/negative edge R histograms: all counts and the top_k bins.
    edge_pos_R_hist_counts = np.zeros((n_K, R_HIST_N_BINS), dtype=int)
    edge_pos_R_top_bin_idx     = np.full((n_K, R_TOP_K), -1, dtype=int)
    edge_pos_R_top_bin_counts  = np.zeros((n_K, R_TOP_K), dtype=int)
    edge_pos_R_top_bin_centers = np.full((n_K, R_TOP_K), np.nan)

    edge_neg_R_hist_counts = np.zeros((n_K, R_HIST_N_BINS), dtype=int)
    edge_neg_R_top_bin_idx     = np.full((n_K, R_TOP_K), -1, dtype=int)
    edge_neg_R_top_bin_counts  = np.zeros((n_K, R_TOP_K), dtype=int)
    edge_neg_R_top_bin_centers = np.full((n_K, R_TOP_K), np.nan)

    for k_idx in range(n_K):
        p_med, p_q5, p_q25, p_q75, p_q95 = quantile_stats(edge_pos_R_by_K[k_idx])
        edge_pos_R_median[k_idx] = p_med
        edge_pos_R_q5[k_idx]  = p_q5
        edge_pos_R_q25[k_idx] = p_q25
        edge_pos_R_q75[k_idx] = p_q75
        edge_pos_R_q95[k_idx] = p_q95

        n_med, n_q5, n_q25, n_q75, n_q95 = quantile_stats(edge_neg_R_by_K[k_idx])
        edge_neg_R_median[k_idx] = n_med
        edge_neg_R_q5[k_idx]  = n_q5
        edge_neg_R_q25[k_idx] = n_q25
        edge_neg_R_q75[k_idx] = n_q75
        edge_neg_R_q95[k_idx] = n_q95

        s_vals = system_R_by_K[k_idx]
        if len(s_vals) > 0:
            s_arr = np.asarray(s_vals, dtype=float)
            system_R_mean[k_idx] = np.mean(s_arr)
            system_R_var[k_idx]  = np.var(s_arr)
            s_med, s_q5, s_q25, s_q75, s_q95 = quantile_stats(s_vals)
            system_R_median[k_idx] = s_med
            system_R_q5[k_idx]  = s_q5
            system_R_q25[k_idx] = s_q25
            system_R_q75[k_idx] = s_q75
            system_R_q95[k_idx] = s_q95

        # Most populated positive-edge histogram bins.
        p_counts, p_top_idx, p_top_cnt, p_top_ctr = top_hist_bins(edge_pos_R_by_K[k_idx])
        edge_pos_R_hist_counts[k_idx]     = p_counts
        edge_pos_R_top_bin_idx[k_idx]     = p_top_idx
        edge_pos_R_top_bin_counts[k_idx]  = p_top_cnt
        edge_pos_R_top_bin_centers[k_idx] = p_top_ctr

        # Most populated negative-edge histogram bins.
        n_counts, n_top_idx, n_top_cnt, n_top_ctr = top_hist_bins(edge_neg_R_by_K[k_idx])
        edge_neg_R_hist_counts[k_idx]     = n_counts
        edge_neg_R_top_bin_idx[k_idx]     = n_top_idx
        edge_neg_R_top_bin_counts[k_idx]  = n_top_cnt
        edge_neg_R_top_bin_centers[k_idx] = n_top_ctr

    save_path = (f"{base_path}/cluster_R_analysis/"
                f"degree_exponent_{degree_exponent:.1f}/"
                f"exponent_{exponent:.2f}/"
                f"m_{m:.1f}")
    os.makedirs(save_path, exist_ok=True)

    results = {
        # --- K axis without binning ---
        'K_array': K_array,
        'n_K': n_K,

        # --- Positive-edge cluster R quantiles ---
        'edge_pos_R_median': edge_pos_R_median,
        'edge_pos_R_q5':  edge_pos_R_q5,
        'edge_pos_R_q25': edge_pos_R_q25,
        'edge_pos_R_q75': edge_pos_R_q75,
        'edge_pos_R_q95': edge_pos_R_q95,
        'edge_pos_n': edge_pos_n,

        # --- Negative-edge cluster R quantiles ---
        'edge_neg_R_median': edge_neg_R_median,
        'edge_neg_R_q5':  edge_neg_R_q5,
        'edge_neg_R_q25': edge_neg_R_q25,
        'edge_neg_R_q75': edge_neg_R_q75,
        'edge_neg_R_q95': edge_neg_R_q95,
        'edge_neg_n': edge_neg_n,

        # --- System-level R statistics ---
        'system_R_mean': system_R_mean,
        'system_R_var': system_R_var,
        'system_R_median': system_R_median,
        'system_R_q5':  system_R_q5,
        'system_R_q25': system_R_q25,
        'system_R_q75': system_R_q75,
        'system_R_q95': system_R_q95,
        'system_n': system_n,

        # --- 20-bin R histograms over [0, 1] and their four leading bins ---
        'R_hist_n_bins': R_HIST_N_BINS,
        'R_hist_range': np.array(R_HIST_RANGE),
        'R_bin_edges': R_bin_edges,
        'R_bin_centers': R_bin_centers,
        'R_top_k': R_TOP_K,

        'edge_pos_R_hist_counts': edge_pos_R_hist_counts,        # (n_K, 20)
        'edge_pos_R_top_bin_idx': edge_pos_R_top_bin_idx,        # (n_K, 4); unused slots are -1
        'edge_pos_R_top_bin_counts': edge_pos_R_top_bin_counts,  # (n_K, 4); unused slots are 0
        'edge_pos_R_top_bin_centers': edge_pos_R_top_bin_centers,# (n_K, 4); unused slots are NaN

        'edge_neg_R_hist_counts': edge_neg_R_hist_counts,        # (n_K, 20)
        'edge_neg_R_top_bin_idx': edge_neg_R_top_bin_idx,        # (n_K, 4)
        'edge_neg_R_top_bin_counts': edge_neg_R_top_bin_counts,  # (n_K, 4)
        'edge_neg_R_top_bin_centers': edge_neg_R_top_bin_centers,# (n_K, 4)

        # --- Metadata and diagnostics ---
        'degree_exponent': degree_exponent,
        'exponent': exponent,
        'm': m,
        'max_seed': max_seed,
        'n_target': n_target,
        'threshold': threshold,
        'total_seeds_used': total_seeds_used,
        'total_seeds_not_converged': total_seeds_not_converged,
        'total_seeds_trivial': total_seeds_trivial,
        'total_seeds_skipped_file': total_seeds_skipped_file,
        'total_edge_pos_found': total_edge_pos_found,
        'total_edge_neg_found': total_edge_neg_found,
    }

    np.save(f"{save_path}/cluster_R_results.npy", results)

    print(f"\nDone! Saved to {save_path}/cluster_R_results.npy")
    print(f"  Seeds used: {total_seeds_used}")
    print(f"  Seeds not converged: {total_seeds_not_converged}")
    print(f"  Seeds trivial: {total_seeds_trivial}")
    print(f"  Seeds skipped (file not found): {total_seeds_skipped_file}")
    print(f"  Edge+ clusters found: {total_edge_pos_found}")
    print(f"  Edge- clusters found: {total_edge_neg_found}")
    print(f"  System R samples: {np.sum(system_n)}")
