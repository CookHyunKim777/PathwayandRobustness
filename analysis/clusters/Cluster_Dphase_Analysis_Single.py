"""Aggregate cluster phase velocities for the single-peak case.

At every K, collect cluster dphase and size directly and summarize them by
mean, standard deviation, median, Q5, Q25, Q75, and Q95. Classification is
identical to Cluster_R_Analysis_Single.py: a cluster must contain at least 16
of the 32 positive-edge or negative-edge target oscillators. A cluster may
satisfy both classes, and all seeds are retained regardless of convergence.

Positive- and negative-edge phase velocities remain separate because their
opposite signs would cancel if averaged. Outputs include per-class statistics,
cluster and seed counts, and the first K meeting the seed quorum.

Usage: python Cluster_Dphase_Analysis_Single.py <degree_exponent> <exponent> <m>
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

N_WORKERS = 32  # Adjust between 16 and 64 according to NFS performance.

# Whether to collect the same not_converged/trivial diagnostics as the R analysis.
# If True, two additional NPY files are read for each (K, seed) pair.
LOAD_CONVERGENCE = True

CLASSES = ('edge_pos', 'edge_neg')
STAT_KEYS = ('mean', 'std', 'median', 'q5', 'q25', 'q75', 'q95')


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
    """Build positive- and negative-edge target sets once per seed.

    This matches find_edge_positive_clusters and find_edge_negative_clusters
    in the single-peak R analysis:
      edge+ : argsort(omega)[-n_target:]
      edge- : argsort(omega)[:n_target]
    """
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


# ==================== Worker: load and classify one (K, seed) pair ====================
def load_one_seed(K, seed, base_path, degree_exponent, exponent, m,
                  natural_freq, n_target, threshold):
    """Load and classify clusters and the system angular-velocity variance.

    Return the same diagnostic fields as process_one_seed in the R analysis.
    """
    conv_dir = (f"{base_path}/convergence_info/"
                f"degree_exponent_{degree_exponent:.1f}/"
                f"exponent_{exponent:.2f}/"
                f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")
    melnikov_dir = (f"{base_path}/melnikov_domain/"
                    f"degree_exponent_{degree_exponent:.1f}/"
                    f"exponent_{exponent:.2f}/"
                    f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")
    arnold_dir = (f"{base_path}/arnold_tongue/"
                  f"degree_exponent_{degree_exponent:.1f}/"
                  f"exponent_{exponent:.2f}/"
                  f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")

    out = {
        'status': 'ok', 'seed': seed,
        'not_converged': False,
        'trivial': False,
        'dphases': {c: [] for c in CLASSES},
        'sizes':   {c: [] for c in CLASSES},
        'all_clusters': [],   # All records, retained for later reclassification.
        'sys_var': None,
    }

    # Convergence diagnostics match the R analysis; count without filtering.
    if LOAD_CONVERGENCE:
        try:
            converged = bool(np.load(f"{conv_dir}/converged.npy")[0])
            trivial = bool(np.load(f"{conv_dir}/trivial_state.npy")[0])
        except FileNotFoundError:
            converged = False
            trivial = False
        except Exception:
            converged = False
            trivial = False
        if not converged:
            out['not_converged'] = True
        if trivial:
            out['trivial'] = True

    # Spatial variance of system angular velocity.
    try:
        mean_dphase_field = np.load(f"{arnold_dir}/mean_dphase.npy")
        out['sys_var'] = float(np.var(mean_dphase_field))
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Cluster dphase and size, classified by edge.
    try:
        num_clusters = int(np.load(f"{melnikov_dir}/num_clusters.npy"))
        if num_clusters == 0:
            return out

        cluster_mean_dphases = np.load(f"{melnikov_dir}/cluster_mean_dphases.npy")
        cluster_sizes = np.load(f"{melnikov_dir}/cluster_sizes.npy")

        edge_pos_osc, edge_neg_osc = build_class_sets(natural_freq, n_target)

        for i in range(num_clusters):
            cluster_indices = np.load(f"{melnikov_dir}/cluster_{i}_indices.npy")
            ov_pos, ov_neg = cluster_overlaps(cluster_indices,
                                              edge_pos_osc, edge_neg_osc)

            d = float(cluster_mean_dphases[i])
            s = int(cluster_sizes[i])

            out['all_clusters'].append({
                'seed': seed, 'dphase': d, 'size': s,
                'ov_edge_pos': ov_pos, 'ov_edge_neg': ov_neg,
            })

            flags = {
                'edge_pos': ov_pos >= threshold,
                'edge_neg': ov_neg >= threshold,
            }
            for c in CLASSES:
                if flags[c]:
                    out['dphases'][c].append(d)
                    out['sizes'][c].append(s)

        return out

    except FileNotFoundError:
        out['status'] = 'skipped_file'
        return out
    except Exception:
        out['status'] = 'skipped_file'
        return out


def collect_cluster_data(degree_exponent, exponent, m, K_array,
                         base_path, max_seed=64, N=1024):
    """Collect classified records and system angular-velocity variance by K."""
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

    tasks = [(round(float(K), 2), seed)
             for K in K_array
             for seed in range(max_seed)]
    n_total = len(tasks)

    print(f"[dphase-single] Loading {n_total} (K, seed) tasks with "
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
            per_K_seed_results[k_idx][res['seed']] = res
            done += 1
            if done % progress_every == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{100*done/n_total:5.1f}%] {done}/{n_total}  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  "
                      f"({rate:.0f} tasks/s)", flush=True)

    # Reassemble per-class records at every K.
    records = {c: [] for c in CLASSES}
    all_records_full = []
    system_dphase_var_by_K = []

    # Diagnostic counters use the same names as the single-peak R analysis.
    diag = {
        'total_seeds_used': 0,
        'total_seeds_not_converged': 0,
        'total_seeds_trivial': 0,
        'total_seeds_skipped_file': 0,
        'total_edge_pos_found': 0,
        'total_edge_neg_found': 0,
    }

    for k_idx, K in enumerate(K_array):
        rec = {c: {'K': float(K), 'dphases': [], 'sizes': [], 'seeds': set()}
               for c in CLASSES}
        rec_all = {'K': float(K), 'clusters': []}
        sys_var_list = []

        seed_res = per_K_seed_results[k_idx]
        for seed in sorted(seed_res.keys()):
            r = seed_res[seed]

            if r['status'] == 'skipped_file':
                diag['total_seeds_skipped_file'] += 1
                continue

            if r['not_converged']:
                diag['total_seeds_not_converged'] += 1
            if r['trivial']:
                diag['total_seeds_trivial'] += 1
            diag['total_seeds_used'] += 1

            if r['sys_var'] is not None:
                sys_var_list.append(r['sys_var'])

            for c in CLASSES:
                ds, ss = r['dphases'][c], r['sizes'][c]
                if ds:
                    rec[c]['dphases'].extend(ds)
                    rec[c]['sizes'].extend(ss)
                    rec[c]['seeds'].add(seed)
                    diag[f'total_{c}_found'] += len(ds)

            rec_all['clusters'].extend(r['all_clusters'])

        for c in CLASSES:
            records[c].append(rec[c])
        all_records_full.append(rec_all)
        system_dphase_var_by_K.append(sys_var_list)

    return records, all_records_full, system_dphase_var_by_K, diag


def summarize_class(rec_list):
    """Build per-class arrays along the K axis."""
    n_K = len(rec_list)
    out = {}
    for prefix in ('dphase', 'size'):
        for k in STAT_KEYS:
            out[f'{prefix}_{k}'] = np.full(n_K, np.nan)
    out['n_clusters'] = np.zeros(n_K, dtype=int)
    out['n_seeds'] = np.zeros(n_K, dtype=int)

    for i, rec in enumerate(rec_list):
        d_stat = stat7(rec['dphases'])
        s_stat = stat7(rec['sizes'])
        for k in STAT_KEYS:
            out[f'dphase_{k}'][i] = d_stat[k]
            out[f'size_{k}'][i] = s_stat[k]
        out['n_clusters'][i] = len(rec['dphases'])
        out['n_seeds'][i] = len(rec['seeds'])

    return out


def find_Kc(n_seeds_arr, K_array, quorum):
    """Return the first K at which the number of matching seeds reaches quorum."""
    idx = np.nonzero(n_seeds_arr >= quorum)[0]
    return float(K_array[idx[0]]) if len(idx) else None


def summarize_system_var(system_dphase_var_by_K):
    """Summarize the cross-seed system angular-velocity variance at every K."""
    n_K = len(system_dphase_var_by_K)
    out = {k: np.full(n_K, np.nan) for k in STAT_KEYS}
    n_arr = np.zeros(n_K, dtype=int)
    for k_idx, vals in enumerate(system_dphase_var_by_K):
        n_arr[k_idx] = len(vals)
        s = stat7(vals)
        for key in out:
            out[key][k_idx] = s[key]
    return out, n_arr


if __name__ == "__main__":

    degree_exponent = float(sys.argv[1])
    exponent = float(sys.argv[2])
    m = float(sys.argv[3])

    print(f"Processing: deg={degree_exponent:.1f}, exp={exponent:.2f}, m={m:.1f}")

    K_array = np.round(np.arange(0.0, 20.0, 0.1), 2)
    max_seed = 64
    N = 1024
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(repo_root / "results"))).expanduser()
    base_path = str(data_root / "DF_single_peak")
    seed_quorum = max_seed // 2      # K_c requires at least 32 seeds.

    n_target = N // 32
    threshold = n_target // 2
    print(f"Cluster criteria: n_target={n_target}, threshold={threshold}")

    records, records_full, system_dphase_var_by_K, diag = collect_cluster_data(
        degree_exponent, exponent, m, K_array, base_path, max_seed, N)

    sys_var_stat, sys_var_n = summarize_system_var(system_dphase_var_by_K)

    save_dir = (f"{base_path}/cluster_dphase_analysis/"
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
        summ = summarize_class(records[c])
        for key, val in summ.items():
            save_data[f'{c}_{key}'] = val
        Kc[c] = find_Kc(summ['n_seeds'], K_array, seed_quorum)
        save_data[f'K_c_{c}'] = Kc[c]

    valid = [v for v in Kc.values() if v is not None]
    save_data['K_critical'] = min(valid) if valid else None
    save_data['seed_quorum'] = seed_quorum
    save_data['K_c_def'] = ('first K where >= seed_quorum seeds carry a cluster '
                            'of that class')

    # Cross-seed system angular-velocity variance, shared by both classes.
    for k in STAT_KEYS:
        save_data[f'system_dphase_var_{k}'] = sys_var_stat[k]
    save_data['system_dphase_var_n'] = sys_var_n
    save_data['system_dphase_var_def'] = (
        'per-seed spatial variance of arnold_tongue/mean_dphase (N nodes), '
        'aggregated over seeds')

    save_data.update({
        'degree_exponent': degree_exponent,
        'exponent': exponent,
        'm': m,
        'max_seed': max_seed,
        'N': N,
        'n_target': n_target,
        'threshold': threshold,
        'omega_setup': ('single_peak; sample_powerlaw_positive + zero-mean; '
                        'edge_pos = argsort(omega)[-32:], '
                        'edge_neg = argsort(omega)[:32] '
                        '(identical to Cluster_R_Analysis_Single.py)'),
        'aggregation': ('plain per-K aggregation of classified clusters; '
                        'no KDE peak finding'),

        # --- Diagnostics directly comparable with the single-peak R analysis ---
        'total_seeds_used':          diag['total_seeds_used'],
        'total_seeds_not_converged': diag['total_seeds_not_converged'],
        'total_seeds_trivial':       diag['total_seeds_trivial'],
        'total_seeds_skipped_file':  diag['total_seeds_skipped_file'],
        'total_edge_pos_found':      diag['total_edge_pos_found'],
        'total_edge_neg_found':      diag['total_edge_neg_found'],
    })

    np.save(f"{save_dir}/dphase_aggregation_results.npy", save_data)

    raw_data = {
        'records_edge_pos': records['edge_pos'],
        'records_edge_neg': records['edge_neg'],
        # All clusters: seed, dphase, size, and two overlap measures. These
        # records allow reclassification without reloading the source files.
        'all_records_full': records_full,
        'system_dphase_var_by_K': system_dphase_var_by_K,
        'K_array': K_array,
        'n_target': n_target,
        'threshold': threshold,
    }
    np.save(f"{save_dir}/dphase_raw_records.npy", raw_data)

    print(f"\nDone! deg={degree_exponent:.1f}, exp={exponent:.2f}, m={m:.1f}")
    print(f"  K points: {len(K_array)}")
    print(f"  Seeds used: {diag['total_seeds_used']}")
    print(f"  Seeds not converged: {diag['total_seeds_not_converged']}")
    print(f"  Seeds trivial: {diag['total_seeds_trivial']}")
    print(f"  Seeds skipped (file not found): {diag['total_seeds_skipped_file']}")
    print(f"  Edge+ clusters found: {diag['total_edge_pos_found']}")
    print(f"  Edge- clusters found: {diag['total_edge_neg_found']}")
    print(f"  K_c[edge_pos] = {Kc['edge_pos']}, K_c[edge_neg] = {Kc['edge_neg']}, "
          f"K_critical = {save_data['K_critical']}")
