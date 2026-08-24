"""Aggregate cluster phase velocities for the double-peak case.

At every K, collect cluster dphase and size directly and summarize them by
mean, standard deviation, median, Q5, Q25, Q75, and Q95. Classification is
identical to Cluster_R_Analysis.py: a cluster must contain at least 16 of the
32 center, positive-edge, or negative-edge target oscillators. A cluster may
satisfy more than one class.

Positive- and negative-edge phase velocities remain separate because their
opposite signs would cancel if averaged. Outputs include per-class statistics,
cluster and seed counts, the first K meeting the seed quorum, and union counts
for clusters containing either edge.

Usage: python Cluster_Dphase_Analysis.py <degree_exponent> <exponent> <m>
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.regular_sampling import PowerLaw_distribution

N_WORKERS = 32  # Adjust between 16 and 64 according to NFS performance.

CLASSES = ('center', 'edge_pos', 'edge_neg')
STAT_KEYS = ('mean', 'std', 'median', 'q5', 'q25', 'q75', 'q95')


def build_class_sets(natural_freq, n_target=32):
    """Return center, positive-edge, and negative-edge target oscillator sets."""
    order_abs = np.argsort(np.abs(natural_freq))
    order_signed = np.argsort(natural_freq)
    return (
        set(order_abs[:n_target].tolist()),        # center
        set(order_signed[-n_target:].tolist()),    # edge +
        set(order_signed[:n_target].tolist()),     # edge -
    )


def cluster_overlaps(cluster_indices, center_osc, edge_pos_osc, edge_neg_osc):
    """Return overlaps with the center, positive-edge, and negative-edge sets."""
    cluster_set = set(np.asarray(cluster_indices).ravel().tolist())
    return (
        len(center_osc & cluster_set),
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
    """Load and classify clusters and the system angular-velocity variance."""
    melnikov_dir = (f"{base_path}/melnikov_domain/"
                    f"degree_exponent_{degree_exponent:.1f}/"
                    f"exponent_{exponent:.2f}/"
                    f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")
    arnold_dir = (f"{base_path}/arnold_tongue/"
                  f"degree_exponent_{degree_exponent:.1f}/"
                  f"exponent_{exponent:.2f}/"
                  f"m_{m:.1f}/K_{K:.2f}/seed_{seed}")

    out = {
        'seed': seed,
        'dphases': {c: [] for c in CLASSES},
        'sizes':   {c: [] for c in CLASSES},
        'has_edge_any': False,     # Positive or negative edge, as in the R analysis.
        'n_edge_any': 0,
        'all_clusters': [],        # All records, retained for later reclassification.
        'sys_var': None,
    }

    # Spatial variance of nodewise time-averaged angular velocities.
    try:
        mean_dphase_field = np.load(f"{arnold_dir}/mean_dphase.npy")
        out['sys_var'] = float(np.var(mean_dphase_field))
    except FileNotFoundError:
        pass
    except Exception:
        pass

    try:
        num_clusters = int(np.load(f"{melnikov_dir}/num_clusters.npy"))
        if num_clusters == 0:
            return out

        cluster_mean_dphases = np.load(f"{melnikov_dir}/cluster_mean_dphases.npy")
        cluster_sizes = np.load(f"{melnikov_dir}/cluster_sizes.npy")

        center_osc, edge_pos_osc, edge_neg_osc = build_class_sets(
            natural_freq, n_target)

        for i in range(num_clusters):
            cluster_indices = np.load(f"{melnikov_dir}/cluster_{i}_indices.npy")
            ov_ct, ov_pos, ov_neg = cluster_overlaps(
                cluster_indices, center_osc, edge_pos_osc, edge_neg_osc)

            d = float(cluster_mean_dphases[i])
            s = int(cluster_sizes[i])

            out['all_clusters'].append({
                'seed': seed, 'dphase': d, 'size': s,
                'ov_center': ov_ct, 'ov_edge_pos': ov_pos, 'ov_edge_neg': ov_neg,
            })

            flags = {
                'center':   ov_ct  >= threshold,
                'edge_pos': ov_pos >= threshold,
                'edge_neg': ov_neg >= threshold,
            }
            for c in CLASSES:
                if flags[c]:
                    out['dphases'][c].append(d)
                    out['sizes'][c].append(s)

            if flags['edge_pos'] or flags['edge_neg']:
                out['has_edge_any'] = True
                out['n_edge_any'] += 1

        return out

    except FileNotFoundError:
        return out
    except Exception:
        return out


def collect_cluster_data(degree_exponent, exponent, m, K_array,
                         base_path, max_seed=64, N=1024):
    """Collect classified records and system angular-velocity variance by K."""
    n_K = len(K_array)
    K_to_idx = {round(float(K), 2): i for i, K in enumerate(K_array)}

    # Reconstruct omega exactly as in main_ensemble.py.
    natural_freqs_by_seed = {}
    for seed in range(max_seed):
        natural_freqs_by_seed[seed] = PowerLaw_distribution(
            degree_exponent, N, True, seed, MIN=1.0, MAX=3.0 + 1.0)

    n_target = N // 32           # 32
    threshold = n_target // 2    # 16

    per_K_seed_results = [dict() for _ in range(n_K)]

    tasks = [(round(float(K), 2), seed)
             for K in K_array
             for seed in range(max_seed)]
    n_total = len(tasks)

    print(f"[dphase-double] Loading {n_total} (K, seed) tasks with "
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
    records = {c: [] for c in CLASSES}          # [{K, dphases, sizes, seeds}, ...]
    edge_any = []                               # [{K, n_clusters, seeds}, ...]
    all_records_full = []
    system_dphase_var_by_K = []
    counts = {c: 0 for c in CLASSES}

    for k_idx, K in enumerate(K_array):
        rec = {c: {'K': float(K), 'dphases': [], 'sizes': [], 'seeds': set()}
               for c in CLASSES}
        rec_edge_any = {'K': float(K), 'n_clusters': 0, 'seeds': set()}
        rec_all = {'K': float(K), 'clusters': []}
        sys_var_list = []

        seed_res = per_K_seed_results[k_idx]
        for seed in sorted(seed_res.keys()):
            r = seed_res[seed]
            if r['sys_var'] is not None:
                sys_var_list.append(r['sys_var'])

            for c in CLASSES:
                ds, ss = r['dphases'][c], r['sizes'][c]
                if ds:
                    rec[c]['dphases'].extend(ds)
                    rec[c]['sizes'].extend(ss)
                    rec[c]['seeds'].add(seed)
                    counts[c] += len(ds)

            if r['has_edge_any']:
                rec_edge_any['n_clusters'] += r['n_edge_any']
                rec_edge_any['seeds'].add(seed)

            rec_all['clusters'].extend(r['all_clusters'])

        for c in CLASSES:
            records[c].append(rec[c])
        edge_any.append(rec_edge_any)
        all_records_full.append(rec_all)
        system_dphase_var_by_K.append(sys_var_list)

    return records, edge_any, all_records_full, system_dphase_var_by_K, counts


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

    K_array = np.round(np.arange(0.0, 20.0, 0.1), 2)
    max_seed = 64
    N = 1024
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SYNC_PATHS_DATA_DIR", str(repo_root / "results"))).expanduser()
    base_path = str(data_root / "DF_double_peak")
    seed_quorum = max_seed // 2      # K_c requires at least 32 seeds.

    records, edge_any, records_full, system_dphase_var_by_K, counts = \
        collect_cluster_data(degree_exponent, exponent, m, K_array,
                             base_path, max_seed, N)

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

    # Union of positive and negative edges for direct comparison with edge_n.
    # Retain counts only because the opposite dphase signs cancel.
    save_data['edge_n_clusters'] = np.array(
        [r['n_clusters'] for r in edge_any], dtype=int)
    save_data['edge_n_seeds'] = np.array(
        [len(r['seeds']) for r in edge_any], dtype=int)
    save_data['edge_union_note'] = (
        'edge = edge_pos OR edge_neg, counts only. dphase stats are kept '
        'split by tail because +/- cancel when averaged together.')

    valid = [v for v in Kc.values() if v is not None]
    save_data['K_critical'] = min(valid) if valid else None
    save_data['seed_quorum'] = seed_quorum
    save_data['K_c_def'] = ('first K where >= seed_quorum seeds carry a cluster '
                            'of that class')

    # Cross-seed system angular-velocity variance, shared by all classes.
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
        'n_target': N // 32,
        'threshold': (N // 32) // 2,
        'omega_setup': ('double_peak; center = argsort(|omega|)[:32], '
                        'edge_pos = argsort(omega)[-32:], '
                        'edge_neg = argsort(omega)[:32] '
                        '(identical to Cluster_R_Analysis.py)'),
        'aggregation': ('plain per-K aggregation of classified clusters; '
                        'no KDE peak finding'),
        'total_center_found':   counts['center'],
        'total_edge_pos_found': counts['edge_pos'],
        'total_edge_neg_found': counts['edge_neg'],
    })

    np.save(f"{save_dir}/dphase_aggregation_results.npy", save_data)

    raw_data = {
        'records_center':   records['center'],
        'records_edge_pos': records['edge_pos'],
        'records_edge_neg': records['edge_neg'],
        # All clusters: seed, dphase, size, and three overlap measures. These
        # records allow reclassification without reloading the source files.
        'all_records_full': records_full,
        'system_dphase_var_by_K': system_dphase_var_by_K,
        'K_array': K_array,
        'n_target': N // 32,
        'threshold': (N // 32) // 2,
    }
    np.save(f"{save_dir}/dphase_raw_records.npy", raw_data)

    print(f"\nDone! deg={degree_exponent:.1f}, exp={exponent:.2f}, m={m:.1f}")
    print(f"  K points: {len(K_array)}")
    print(f"  Center clusters found: {counts['center']}")
    print(f"  Edge+ clusters found:  {counts['edge_pos']}")
    print(f"  Edge- clusters found:  {counts['edge_neg']}")
    print(f"  K_c[center]   = {Kc['center']}")
    print(f"  K_c[edge_pos] = {Kc['edge_pos']}")
    print(f"  K_c[edge_neg] = {Kc['edge_neg']}")
    print(f"  K_critical    = {save_data['K_critical']}")
