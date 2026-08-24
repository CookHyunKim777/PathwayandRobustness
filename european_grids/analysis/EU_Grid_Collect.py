"""Collect and aggregate Kuramoto simulation results for European grids.

The script computes system-level and cluster-level statistics, including
peak-restricted statistics based on the negative and positive maxima of
``h(P) = n * <d>^2``. A cluster is assigned to a peak when it contains at
least half of that peak bin's total coupling weight. If one cluster satisfies
the criterion for both peaks, it is also recorded in the ``both`` pool.

Peak-cluster statistics are retained at every simulated lambda value. The
peak-cluster phase velocity is summarized directly across seeds, whereas KDE
aggregation is retained for the much larger pool of all detected clusters.

Input tree::

    <data-root>/DF/<country>/<sim_type>/alpha_<alpha>/<lambda>/<seed>/
        <country>_forward_lmd<lambda>_seed<seed>.txt
    <data-root>/DF/<country>/alpha_<alpha>/<country>_node_rescale.txt
    <data-root>/DF/<country>/alpha_<alpha>/<country>_link_rescale.txt

Output tree::

    <data-root>/DF/<country>/<sim_type>_collect/alpha_<alpha>/

By default, ``<data-root>`` is the repository's ``results`` directory. Set
``SYNC_PATHS_DATA_DIR`` to use another location.

Usage::

    python -m european_grids.analysis.EU_Grid_Collect COUNTRY SIM_TYPE ALPHA
"""

import numpy as np
import os
import sys
import glob
import re
import json
from pathlib import Path
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


# ─── Argument handling ─────────────────────────────────────
SUPPORTED_COUNTRIES = ("DE", "ES", "FR", "UK")
SUPPORTED_SIM_TYPES = ("Annealed", "Quenched",
                       "Annealed_massless", "Quenched_massless")

if len(sys.argv) < 4:
    print("Usage: python EU_Grid_Collect.py <country> <sim_type> <alpha>")
    print(f"  country : {' | '.join(SUPPORTED_COUNTRIES)}")
    print(f"  sim_type: {' | '.join(SUPPORTED_SIM_TYPES)}")
    print(f"  alpha   : reparametrization knob label (e.g. 0.0, 3.0)")
    sys.exit(1)

country  = sys.argv[1].strip()
sim_type = sys.argv[2].strip()
alpha_arg = sys.argv[3].strip()

if country not in SUPPORTED_COUNTRIES:
    print(f"[ERROR] unknown country: {country}")
    sys.exit(1)
if sim_type not in SUPPORTED_SIM_TYPES:
    print(f"[ERROR] unknown sim_type: {sim_type}")
    sys.exit(1)

try:
    alpha_str = f"{float(alpha_arg):.1f}"
except ValueError:
    alpha_str = alpha_arg

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("SYNC_PATHS_DATA_DIR", str(REPO_ROOT / "results"))
).expanduser()
grid_dir = DATA_ROOT / "DF" / country

base_dir = grid_dir / sim_type / f"alpha_{alpha_str}"
save_dir = grid_dir / f"{sim_type}_collect" / f"alpha_{alpha_str}"
save_dir.mkdir(parents=True, exist_ok=True)

# Rescaled data used for peak calculations.
rescale_dir = grid_dir / f"alpha_{alpha_str}"
node_rescale_file = rescale_dir / f"{country}_node_rescale.txt"
link_rescale_file = rescale_dir / f"{country}_link_rescale.txt"

print(f"[EU grid: {country} / {sim_type} / alpha_{alpha_str}] base = {base_dir}")
print(f"[EU grid: {country} / {sim_type} / alpha_{alpha_str}] out  = {save_dir}")

if not base_dir.exists():
    print(f"[ERROR] base dir not found: {base_dir}")
    sys.exit(1)


# ─── Global settings ────────────────────────────────────────
# Omega bins used to identify peak seeds.
N_BINS_PEAK = 16

PEAK_NAMES = ('neg', 'pos', 'both')

# Top-N bins of the Fig. 2-style R histogram.
R_HIST_N_BINS = 20                                    # Divide R in [0, 1] into 20 bins.
R_BIN_EDGES   = np.linspace(0.0, 1.0, R_HIST_N_BINS + 1)
R_TOP_N       = 4                                     # Retain the top four bins.

# KDE parameters, used only for aggregating all-cluster dphase values.
KDE_BANDWIDTH     = 0.05
CLUSTER_MARGIN    = 0.2
MIN_PEAK_HEIGHT   = 0.01
MIN_CLUSTER_COUNT = 20
MIN_PEAK_COUNT    = 20
BIN_WIDTH_FACTOR  = 5
MERGE_OVERLAPPING = True
VALLEY_RATIO      = 0.7

# Peak-dphase summary method; KDE is not used for the peak pool.
# Select either 'median' or 'mean' as the representative peak_dphase value.
# The NPY output stores median, mean, and standard deviation for later selection.
PEAK_DPHASE_ESTIMATOR = 'median'


# ─── Peak-seed search: h(P)=n*<d>^2 ────────────────────────
def find_peaks_h(omega, d, n_bins=N_BINS_PEAK):
    """Select the largest h below zero and above zero as the two peaks.
    Returns (seed_neg, seed_pos, diag)."""
    edges = np.linspace(omega.min(), omega.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_arr = np.zeros(n_bins, dtype=int)
    s_mean = np.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (omega >= edges[i]) & (omega <= edges[i + 1])
        else:
            mask = (omega >= edges[i]) & (omega < edges[i + 1])
        if mask.sum() > 0:
            n_arr[i] = int(mask.sum())
            s_mean[i] = d[mask].mean()
    h_arr = n_arr * s_mean ** 2

    def side_peak(sign):
        side = (centers < 0) & (n_arr > 0) if sign < 0 else (centers > 0) & (n_arr > 0)
        if not np.any(side):
            return None, None
        h_side = np.where(side, h_arr, -np.inf)
        ip = int(np.argmax(h_side))
        return float(centers[ip]), ip

    seed_neg, ip_neg = side_peak(-1)
    seed_pos, ip_pos = side_peak(+1)
    diag = {'edges': edges, 'centers': centers, 'n': n_arr,
            'mean_s': s_mean, 'h': h_arr, 'ip_neg': ip_neg, 'ip_pos': ip_pos}
    return seed_neg, seed_pos, diag


def peak_bin_node_indices(omega, edges, ip):
    """Return node indices in peak bin ip using the original omega values."""
    if ip is None:
        return np.array([], dtype=int)
    lo, hi = edges[ip], edges[ip + 1]
    if ip == len(edges) - 2:
        m = (omega >= lo) & (omega <= hi)
    else:
        m = (omega >= lo) & (omega < hi)
    return np.where(m)[0]


# Prepare peak information; disable peak-restricted statistics if unavailable.
PEAK_ENABLED = node_rescale_file.exists() and link_rescale_file.exists()
peak_info = {}
d_rescale = None   # Per-node coupling constant d.
if PEAK_ENABLED:
    _node = np.loadtxt(node_rescale_file, ndmin=2)
    omega_rescale = _node[:, 2]
    d_rescale = np.loadtxt(link_rescale_file, ndmin=1).reshape(-1)
    if len(d_rescale) != len(omega_rescale):
        print(f"  [peak][WARN] link_rescale length ({len(d_rescale)}) != "
              f"node count ({len(omega_rescale)}). "
              f"d-weighted half criterion assumes per-node d aligned with node order.")
    seed_neg, seed_pos, pdiag = find_peaks_h(omega_rescale, d_rescale, N_BINS_PEAK)
    edges_peak = pdiag['edges']
    for name, seed_v, ip in [('neg', seed_neg, pdiag['ip_neg']),
                             ('pos', seed_pos, pdiag['ip_pos'])]:
        nodes = peak_bin_node_indices(omega_rescale, edges_peak, ip)
        if len(nodes) and len(d_rescale) >= (nodes.max() + 1):
            w_bin = float(np.sum(d_rescale[nodes]))
        else:
            w_bin = 0.0
        peak_info[name] = {
            'seed': seed_v, 'ip': ip,
            'bin_nodes': set(nodes.tolist()),
            'bin_nodes_arr': nodes,
            'n_bin': int(len(nodes)),
            'half': len(nodes) / 2.0,       # Count-based diagnostic threshold.
            'w_bin': w_bin,                 # Total d in the peak bin.
            'half_w': w_bin / 2.0,          # Half of the d-weighted total.
            'bin_lo': float(edges_peak[ip]) if ip is not None else np.nan,
            'bin_hi': float(edges_peak[ip + 1]) if ip is not None else np.nan,
        }
    print(f"  [peak] neg: seed={seed_neg}, bin_nodes={peak_info['neg']['n_bin']}, "
          f"W_bin={peak_info['neg']['w_bin']:.4f} (half_w={peak_info['neg']['half_w']:.4f}), "
          f"range=[{peak_info['neg']['bin_lo']:.4f},{peak_info['neg']['bin_hi']:.4f}]")
    print(f"  [peak] pos: seed={seed_pos}, bin_nodes={peak_info['pos']['n_bin']}, "
          f"W_bin={peak_info['pos']['w_bin']:.4f} (half_w={peak_info['pos']['half_w']:.4f}), "
          f"range=[{peak_info['pos']['bin_lo']:.4f},{peak_info['pos']['bin_hi']:.4f}]")
else:
    print(f"  [peak] rescale data not found ({node_rescale_file}); "
          f"peak-restricted stats DISABLED.")


def cluster_bin_weight(node_cluster, num_clusters, bin_nodes_set, d):
    """Return the d weight from bin_nodes contained in each cluster."""
    w = np.zeros(num_clusters, dtype=float)
    for i in bin_nodes_set:
        if i < len(node_cluster) and i < len(d):
            cid = node_cluster[i]
            if 0 <= cid < num_clusters:
                w[cid] += d[i]
    return w


def assign_peak_cluster(node_cluster, num_clusters, bin_nodes_set, half_w, d):
    """Return the cluster entraining at least half_w of a peak bin, or None."""
    if num_clusters == 0 or len(bin_nodes_set) == 0 or half_w <= 0:
        return None
    w = cluster_bin_weight(node_cluster, num_clusters, bin_nodes_set, d)
    best_cid = int(np.argmax(w))
    if w[best_cid] >= half_w and w[best_cid] > 0:
        return best_cid
    return None


def find_both_cluster(node_cluster, num_clusters,
                      bin_nodes_neg, half_w_neg,
                      bin_nodes_pos, half_w_pos, d):
    """Return a cluster entraining at least half of both peak bins, or None."""
    if num_clusters == 0 or len(bin_nodes_neg) == 0 or len(bin_nodes_pos) == 0:
        return None
    if half_w_neg <= 0 or half_w_pos <= 0:
        return None
    w_neg = cluster_bin_weight(node_cluster, num_clusters, bin_nodes_neg, d)
    w_pos = cluster_bin_weight(node_cluster, num_clusters, bin_nodes_pos, d)
    qualifies = (w_neg >= half_w_neg) & (w_pos >= half_w_pos) & \
                (w_neg > 0) & (w_pos > 0)
    if not np.any(qualifies):
        return None
    combined = np.where(qualifies, w_neg + w_pos, -1.0)
    return int(np.argmax(combined))


# ─── File parsing ──────────────────────────────────────────
def parse_result_file(fpath, has_rloc):
    with open(fpath, 'r') as f:
        lines = f.readlines()

    section = 0
    sec1_data, sec2_data, sec3_data = None, [], []
    num_clusters = 0
    sec1_has_F = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            if 'lambda' in line.lower() and ('R' in line or 'r' in line.lower()):
                section = 1
                header_tokens = line.lstrip('#').split()
                sec1_has_F = ('F' in header_tokens) or ('F_weighted' in header_tokens)
            elif 'Node_Index' in line:
                section = 2
            elif 'Num_Clusters' in line:
                m = re.search(r'Num_Clusters:\s*(\d+)', line)
                if m:
                    num_clusters = int(m.group(1))
            elif 'Cluster_idx' in line:
                section = 3
            continue

        tokens = line.split()
        try:
            if section == 1 and sec1_data is None:
                start = 0
                try:
                    float(tokens[0])
                except ValueError:
                    start = 1
                vals = [float(t) for t in tokens[start:]]
                idx = 0
                lam_v = vals[idx]; idx += 1
                R_v = vals[idx]; idx += 1
                Rw_v = vals[idx]; idx += 1
                if has_rloc:
                    Rloc_v = vals[idx]; idx += 1
                else:
                    Rloc_v = np.nan
                if sec1_has_F:
                    F_v = vals[idx]; idx += 1
                    Fw_v = vals[idx]; idx += 1
                else:
                    F_v = np.nan; Fw_v = np.nan
                Time_v = vals[idx] if idx < len(vals) else np.nan
                sec1_data = {'lambda': lam_v, 'R': R_v, 'R_weighted': Rw_v,
                             'R_local': Rloc_v, 'F': F_v, 'F_weighted': Fw_v,
                             'Time': Time_v}
            elif section == 2:
                sec2_data.append((int(tokens[0]), float(tokens[1]), int(tokens[2])))
            elif section == 3:
                base = (int(tokens[0]), int(tokens[1]),
                        float(tokens[2]), float(tokens[3]),
                        float(tokens[4]), float(tokens[5]))
                if len(tokens) >= 8:
                    F_c = float(tokens[6]); Fw_c = float(tokens[7])
                else:
                    F_c = np.nan; Fw_c = np.nan
                sec3_data.append(base + (F_c, Fw_c))
        except (ValueError, IndexError):
            continue

    if sec1_data is None:
        return None

    avg_omega_per_node = np.array([r[1] for r in sec2_data], dtype=float) if sec2_data else np.array([])
    node_cluster_id = np.array([r[2] for r in sec2_data], dtype=int) if sec2_data else np.array([], dtype=int)

    if sec3_data:
        cls_arr = np.array(sec3_data, dtype=float)
        cluster_size = cls_arr[:, 1].astype(int)
        cluster_mean_omega = cls_arr[:, 2]
        cluster_R_w = cls_arr[:, 3]
        cluster_R_unw = cls_arr[:, 4]
        cluster_dphase = cls_arr[:, 5]
        cluster_F = cls_arr[:, 6]
        cluster_F_w = cls_arr[:, 7]
    else:
        cluster_size = np.array([], dtype=int)
        cluster_mean_omega = np.array([])
        cluster_R_w = np.array([]); cluster_R_unw = np.array([])
        cluster_dphase = np.array([]); cluster_F = np.array([]); cluster_F_w = np.array([])

    return {
        'system': sec1_data,
        'avg_omega': avg_omega_per_node,
        'node_cluster': node_cluster_id,
        'num_clusters': num_clusters,
        'cluster_size': cluster_size,
        'cluster_mean_omega': cluster_mean_omega,
        'cluster_R_w': cluster_R_w,
        'cluster_R_unw': cluster_R_unw,
        'cluster_dphase': cluster_dphase,
        'cluster_F': cluster_F,
        'cluster_F_w': cluster_F_w,
    }


has_rloc = sim_type.startswith("Quenched")


# ─── Scan lambda directories ───────────────────────────────
lambda_list = []
for p in sorted(base_dir.iterdir()):
    if not p.is_dir():
        continue
    try:
        lam = float(p.name)
        lambda_list.append((lam, p))
    except ValueError:
        continue
lambda_list.sort(key=lambda x: x[0])
print(f"[EU grid: {country} / {sim_type} / alpha_{alpha_str}] found {len(lambda_list)} lambda values")


# ─── Statistical functions ────────────────────────────────
def stats_dict(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {k: np.nan for k in ['mean','std','median','q5','q25','q75','q95','min','max']} | {'count': 0}
    return {
        'count': int(len(arr)), 'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
        'median': float(np.median(arr)), 'q5': float(np.percentile(arr, 5)),
        'q25': float(np.percentile(arr, 25)), 'q75': float(np.percentile(arr, 75)),
        'q95': float(np.percentile(arr, 95)), 'min': float(np.min(arr)), 'max': float(np.max(arr)),
    }


def weighted_stats_dict(arr, weights):
    arr = np.asarray(arr, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    arr = arr[mask]; w = w[mask]
    if len(arr) == 0:
        return {k: np.nan for k in ['wmean','wstd','wmedian','wq5','wq25','wq75','wq95']}
    W = w.sum()
    wmean = float(np.sum(w * arr) / W)
    wvar = float(np.sum(w * (arr - wmean)**2) / W)
    wstd = float(np.sqrt(wvar))
    order = np.argsort(arr)
    a_sorted = arr[order]; w_sorted = w[order]
    cum = np.cumsum(w_sorted) / W
    def wquantile(q):
        idx = np.searchsorted(cum, q, side='left')
        idx = min(idx, len(a_sorted) - 1)
        return float(a_sorted[idx])
    return {'wmean': wmean, 'wstd': wstd, 'wmedian': wquantile(0.50),
            'wq5': wquantile(0.05), 'wq25': wquantile(0.25),
            'wq75': wquantile(0.75), 'wq95': wquantile(0.95)}


# ─── KDE-based dphase grouping ─────────────────────────────
# This is used only for all-cluster aggregation below, not for peak pools.
def cluster_by_kde(dphase_values, bandwidth=KDE_BANDWIDTH, x_grid_points=2000,
                   min_peak_height=MIN_PEAK_HEIGHT, cluster_margin=CLUSTER_MARGIN,
                   min_cluster_count=MIN_CLUSTER_COUNT, min_peak_count=MIN_PEAK_COUNT,
                   merge_overlapping=MERGE_OVERLAPPING, valley_ratio=VALLEY_RATIO):
    dphase_values = np.array(dphase_values)
    N_val = len(dphase_values)
    if N_val < 2:
        return []
    std_val = np.std(dphase_values, ddof=1)
    if not np.isfinite(std_val) or std_val == 0:
        return []
    kde = gaussian_kde(dphase_values)
    kde.set_bandwidth(bw_method=bandwidth / std_val)
    val_min, val_max = dphase_values.min(), dphase_values.max()
    x_grid = np.linspace(val_min - 0.5, val_max + 0.5, x_grid_points)
    kde_values = kde(x_grid)
    peaks_idx, _ = find_peaks(kde_values, height=min_peak_height)
    peak_positions = x_grid[peaks_idx]
    peak_heights = kde_values[peaks_idx]
    valid_mask = []
    for p in peak_positions:
        cnt = np.sum((dphase_values >= p - cluster_margin) & (dphase_values <= p + cluster_margin))
        valid_mask.append(cnt >= min_peak_count)
    valid_mask = np.array(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        return []
    valid_peaks = peak_positions[valid_mask]
    valid_heights = peak_heights[valid_mask]
    valid_grididx = peaks_idx[valid_mask]
    if merge_overlapping and len(valid_peaks) > 1:
        order = np.argsort(valid_peaks)
        sp_pos = valid_peaks[order]; sp_height = valid_heights[order]; sp_grididx = valid_grididx[order]
        n_p = len(sp_pos)
        parent = list(range(n_p))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj: parent[rj] = ri
        for i in range(n_p - 1):
            j = i + 1
            if sp_pos[j] - sp_pos[i] >= 2.0 * cluster_margin:
                continue
            g_i, g_j = sp_grididx[i], sp_grididx[j]
            valley = float(kde_values[g_i:g_j + 1].min())
            shallower = float(min(sp_height[i], sp_height[j]))
            if shallower <= 0.0:
                continue
            if valley >= valley_ratio * shallower:
                union(i, j)
        groups_of_peaks = {}
        for i in range(n_p):
            r = find(i)
            groups_of_peaks.setdefault(r, []).append(i)
        merged_peaks = []; merged_heights = []; merged_windows = []
        for r, members in groups_of_peaks.items():
            members = np.array(members)
            mem_pos = sp_pos[members]; mem_h = sp_height[members]
            i_top = members[np.argmax(mem_h)]
            merged_peaks.append(sp_pos[i_top]); merged_heights.append(sp_height[i_top])
            merged_windows.append((mem_pos.min() - cluster_margin, mem_pos.max() + cluster_margin))
        merged_peaks = np.array(merged_peaks); merged_heights = np.array(merged_heights)
    else:
        merged_peaks = valid_peaks; merged_heights = valid_heights
        merged_windows = [(p - cluster_margin, p + cluster_margin) for p in valid_peaks]
    groups = []
    assigned_mask = np.zeros(N_val, dtype=bool)
    indices = np.arange(N_val)
    sorted_order = np.argsort(merged_heights)[::-1]
    for idx in sorted_order:
        lo, hi = merged_windows[idx]
        in_group = ((dphase_values >= lo) & (dphase_values <= hi) & (~assigned_mask))
        count = int(np.sum(in_group))
        if count >= min_cluster_count:
            member_indices = indices[in_group]
            groups.append({'peak_dphase': float(merged_peaks[idx]),
                           'window': (float(lo), float(hi)),
                           'member_indices': member_indices})
            assigned_mask[in_group] = True
    return groups


# ─── Helper for cluster-level statistics rows ──────────────
def cluster_row(lam, pool_size, pool_omega, pool_R_w, pool_R_unw,
                pool_dphase, pool_F, pool_F_w, extra=None):
    pool_size_arr = np.asarray(pool_size, dtype=float)
    row = {'lambda': lam, 'n_clusters_pooled': len(pool_size)}
    if extra:
        row.update(extra)
    s = stats_dict(pool_size)
    for k, v in s.items():
        row[f"size_{k}"] = v
    for qty, arr in [('mean_omega', pool_omega), ('R_cls_w', pool_R_w),
                     ('R_cls_unw', pool_R_unw), ('dphase', pool_dphase),
                     ('F_cls', pool_F), ('F_cls_w', pool_F_w)]:
        s = stats_dict(arr)
        for k, v in s.items():
            row[f"{qty}_{k}"] = v
        ws = weighted_stats_dict(arr, pool_size_arr)
        for k, v in ws.items():
            row[f"{qty}_{k}"] = v
    return row


# ─── Top-N R-histogram bins used by Fig. 2 plotting ────────
def top_bins_of(values, n_top=R_TOP_N, edges=R_BIN_EDGES):
    """Return the n_top most populated histogram bins.

    Unused slots have idx=-1 and count=0.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    idx = np.full(n_top, -1, dtype=int)
    cnt = np.zeros(n_top, dtype=int)
    if len(v) == 0:
        return idx, cnt
    hist, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    order = np.argsort(hist)[::-1]          # Descending count order.
    k = 0
    for b in order:
        if hist[b] <= 0 or k >= n_top:
            break
        idx[k] = int(b); cnt[k] = int(hist[b]); k += 1
    return idx, cnt


# ─── Peak-pool dphase summary without KDE for Fig. 2 bubbles ───
def peak_dphase_summary(dphase_vals, size_vals,
                        estimator=PEAK_DPHASE_ESTIMATOR):
    """Summarize a peak pool at one lambda by one value across seeds.

    ``assign_peak_cluster`` has already selected one representative entraining
    cluster per seed. The pool therefore contains realizations of the same
    physical object and does not require further KDE subdivision.

    The previous KDE step returned an empty list when no density peak passed
    its thresholds, which could discard a populated lambda and create gaps in
    the bubble plot. This function always returns one group for a nonempty pool.

    The return schema matches the earlier kde_groups_of output:
      peak_dphase / window_lo / window_hi / dphase_median / dphase_mean /
      dphase_std / size_median / size_mean / count
    window_lo and window_hi contain the actual pool range rather than a KDE window.

    Duplicate negative/positive assignments are allowed when both peaks have
    merged into one cluster; that case is also recorded in the 'both' pool.
    """
    d = np.asarray(dphase_vals, dtype=float)
    s = np.asarray(size_vals, dtype=float)

    m = np.isfinite(d)
    d = d[m]
    s = s[m] if len(s) == len(m) else np.asarray([], dtype=float)

    if len(d) == 0:
        return []                      # No entraining cluster exists at this lambda.

    has_s = (len(s) == len(d))
    med = float(np.median(d))
    avg = float(np.mean(d))
    rep = med if estimator == 'median' else avg

    return [{
        'peak_dphase':   rep,                       # Representative dphase.
        'window_lo':     float(d.min()),            # Actual pool range, not a KDE window.
        'window_hi':     float(d.max()),
        'dphase_median': med,
        'dphase_mean':   avg,
        'dphase_std':    float(np.std(d)),          # Cross-seed spread for error bars.
        'size_median':   float(np.median(s)) if has_s else np.nan,
        'size_mean':     float(np.mean(s))   if has_s else np.nan,
        'count':         int(len(d)),               # Number of contributing seeds.
    }]


# ─── Main loop ─────────────────────────────────────────────
per_lambda_sys_rows = []
per_lambda_cls_rows = []

# Peak CSV rows are accumulated at each individual lambda.
peak_rows      = {name: [] for name in PEAK_NAMES}   # name -> [row, ...]
peak_lam_list  = {name: [] for name in PEAK_NAMES}   # name -> [lam, ...]
peak_top_idx   = {name: [] for name in PEAK_NAMES}   # name -> [(4,) idx, ...]
peak_top_cnt   = {name: [] for name in PEAK_NAMES}
peak_n_list    = {name: [] for name in PEAK_NAMES}   # name -> [n_clusters, ...]
peak_kde_list  = {name: [] for name in PEAK_NAMES}   # name -> [ {lambda, groups}, ... ]

n_files_total = 0; n_files_ok = 0; n_files_missing = 0
pool_dphase_by_lambda = {}; pool_size_by_lambda = {}

for lam, lam_dir in lambda_list:
    seed_dirs = sorted(
        [p for p in lam_dir.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else -1,
    )

    sys_R = []; sys_R_w = []; sys_R_loc = []; sys_F = []; sys_F_w = []; sys_time = []
    pool_size = []; pool_omega = []; pool_R_w = []; pool_R_unw = []
    pool_dphase = []; pool_F = []; pool_F_w = []
    largest_cluster_size = []; largest_cluster_R_w = []; largest_cluster_R_unw = []
    largest_cluster_dphase = []; largest_cluster_F = []; largest_cluster_F_w = []
    num_clusters_list = []

    # Entraining-cluster pools for each peak, combined across seeds at this lambda.
    peak_pool = {name: {'size': [], 'omega': [], 'R_w': [], 'R_unw': [],
                        'dphase': [], 'F': [], 'F_w': [],
                        'n_seed_hit': 0, 'n_seed_none': 0}
                 for name in PEAK_NAMES}

    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name)
        except ValueError:
            continue
        pattern = f"{country}_forward_lmd*_seed{seed}.txt"
        matches = list(seed_dir.glob(pattern))
        n_files_total += 1
        if not matches:
            n_files_missing += 1
            continue
        fpath = matches[0]
        data = parse_result_file(fpath, has_rloc)
        if data is None:
            n_files_missing += 1
            continue

        n_files_ok += 1
        sys_ = data['system']
        sys_R.append(sys_['R']); sys_R_w.append(sys_['R_weighted'])
        sys_R_loc.append(sys_['R_local']); sys_F.append(sys_['F'])
        sys_F_w.append(sys_['F_weighted']); sys_time.append(sys_['Time'])

        if len(data['cluster_size']) > 0:
            pool_size.extend(data['cluster_size'].tolist())
            pool_omega.extend(data['cluster_mean_omega'].tolist())
            pool_R_w.extend(data['cluster_R_w'].tolist())
            pool_R_unw.extend(data['cluster_R_unw'].tolist())
            pool_dphase.extend(data['cluster_dphase'].tolist())
            pool_F.extend(data['cluster_F'].tolist())
            pool_F_w.extend(data['cluster_F_w'].tolist())
            imax = 0
            largest_cluster_size.append(int(data['cluster_size'][imax]))
            largest_cluster_R_w.append(float(data['cluster_R_w'][imax]))
            largest_cluster_R_unw.append(float(data['cluster_R_unw'][imax]))
            largest_cluster_dphase.append(float(data['cluster_dphase'][imax]))
            largest_cluster_F.append(float(data['cluster_F'][imax]))
            largest_cluster_F_w.append(float(data['cluster_F_w'][imax]))

            # Select each peak's entraining cluster and test for a merged both case.
            if PEAK_ENABLED:
                def _push(name, cid):
                    if cid is None or cid >= len(data['cluster_size']):
                        peak_pool[name]['n_seed_none'] += 1
                        return
                    peak_pool[name]['n_seed_hit'] += 1
                    peak_pool[name]['size'].append(int(data['cluster_size'][cid]))
                    peak_pool[name]['omega'].append(float(data['cluster_mean_omega'][cid]))
                    peak_pool[name]['R_w'].append(float(data['cluster_R_w'][cid]))
                    peak_pool[name]['R_unw'].append(float(data['cluster_R_unw'][cid]))
                    peak_pool[name]['dphase'].append(float(data['cluster_dphase'][cid]))
                    peak_pool[name]['F'].append(float(data['cluster_F'][cid]))
                    peak_pool[name]['F_w'].append(float(data['cluster_F_w'][cid]))

                if peak_info['neg']['n_bin'] > 0:
                    cid_neg = assign_peak_cluster(
                        data['node_cluster'], data['num_clusters'],
                        peak_info['neg']['bin_nodes'], peak_info['neg']['half_w'],
                        d_rescale)
                    _push('neg', cid_neg)
                if peak_info['pos']['n_bin'] > 0:
                    cid_pos = assign_peak_cluster(
                        data['node_cluster'], data['num_clusters'],
                        peak_info['pos']['bin_nodes'], peak_info['pos']['half_w'],
                        d_rescale)
                    _push('pos', cid_pos)
                if peak_info['neg']['n_bin'] > 0 and peak_info['pos']['n_bin'] > 0:
                    cid_both = find_both_cluster(
                        data['node_cluster'], data['num_clusters'],
                        peak_info['neg']['bin_nodes'], peak_info['neg']['half_w'],
                        peak_info['pos']['bin_nodes'], peak_info['pos']['half_w'],
                        d_rescale)
                    _push('both', cid_both)

        num_clusters_list.append(data['num_clusters'])

    pool_dphase_by_lambda[lam] = np.asarray(pool_dphase, dtype=float)
    pool_size_by_lambda[lam] = np.asarray(pool_size, dtype=int)

    # System-level row.
    sys_R = np.asarray(sys_R, dtype=float); sys_R_w = np.asarray(sys_R_w, dtype=float)
    sys_R_loc = np.asarray(sys_R_loc, dtype=float); sys_F = np.asarray(sys_F, dtype=float)
    sys_F_w = np.asarray(sys_F_w, dtype=float)
    row_sys = {'lambda': lam, 'n_seeds': len(sys_R)}
    for qty, arr in [('R', sys_R), ('R_weighted', sys_R_w), ('R_local', sys_R_loc),
                     ('F', sys_F), ('F_weighted', sys_F_w)]:
        s = stats_dict(arr)
        for k, v in s.items():
            row_sys[f"{qty}_{k}"] = v
    row_sys['n_clusters_mean'] = float(np.mean(num_clusters_list)) if num_clusters_list else np.nan
    row_sys['n_clusters_median'] = float(np.median(num_clusters_list)) if num_clusters_list else np.nan
    lcs = stats_dict(largest_cluster_size); lcR = stats_dict(largest_cluster_R_w)
    lcR_unw = stats_dict(largest_cluster_R_unw); lcD = stats_dict(largest_cluster_dphase)
    lcF = stats_dict(largest_cluster_F); lcF_w = stats_dict(largest_cluster_F_w)
    row_sys['largest_cluster_size_mean'] = lcs['mean']; row_sys['largest_cluster_size_median'] = lcs['median']
    row_sys['largest_cluster_R_w_mean'] = lcR['mean']; row_sys['largest_cluster_R_w_median'] = lcR['median']
    row_sys['largest_cluster_R_unw_mean'] = lcR_unw['mean']; row_sys['largest_cluster_R_unw_median'] = lcR_unw['median']
    row_sys['largest_cluster_dphase_mean'] = lcD['mean']; row_sys['largest_cluster_dphase_median'] = lcD['median']
    row_sys['largest_cluster_F_mean'] = lcF['mean']; row_sys['largest_cluster_F_median'] = lcF['median']
    row_sys['largest_cluster_F_w_mean'] = lcF_w['mean']; row_sys['largest_cluster_F_w_median'] = lcF_w['median']
    per_lambda_sys_rows.append(row_sys)

    # All-cluster row.
    per_lambda_cls_rows.append(
        cluster_row(lam, pool_size, pool_omega, pool_R_w, pool_R_unw,
                    pool_dphase, pool_F, pool_F_w))

    # Peak row at this individual lambda, without lambda binning.
    if PEAK_ENABLED:
        for name in PEAK_NAMES:
            pp = peak_pool[name]

            # Four most populated R_w histogram bins.
            ti, tc = top_bins_of(pp['R_w'])
            n_cls = int(np.sum(np.isfinite(np.asarray(pp['R_w'], dtype=float))))

            peak_lam_list[name].append(float(lam))
            peak_top_idx[name].append(ti)
            peak_top_cnt[name].append(tc)
            peak_n_list[name].append(n_cls)

            # Summarize peak-pool dphase without KDE; a nonempty pool yields one group.
            peak_kde_list[name].append({
                'lambda':         float(lam),
                'total_clusters': int(len(pp['dphase'])),
                'groups':         peak_dphase_summary(pp['dphase'], pp['size']),
            })

            extra = {
                'n_seed_hit':  pp['n_seed_hit'],
                'n_seed_none': pp['n_seed_none'],
            }
            for j in range(R_TOP_N):
                extra[f'R_top_bin_idx_{j}']   = int(ti[j])
                extra[f'R_top_bin_count_{j}'] = int(tc[j])
                extra[f'R_top_bin_lo_{j}']    = float(R_BIN_EDGES[ti[j]]) if ti[j] >= 0 else np.nan
                extra[f'R_top_bin_hi_{j}']    = float(R_BIN_EDGES[ti[j] + 1]) if ti[j] >= 0 else np.nan
            extra['R_hist_n'] = n_cls

            if name == 'both':
                extra.update({
                    'peak_neg_seed':   peak_info['neg']['seed'],
                    'peak_neg_bin_lo': peak_info['neg']['bin_lo'],
                    'peak_neg_bin_hi': peak_info['neg']['bin_hi'],
                    'peak_neg_w_bin':  peak_info['neg']['w_bin'],
                    'peak_neg_half_w': peak_info['neg']['half_w'],
                    'peak_pos_seed':   peak_info['pos']['seed'],
                    'peak_pos_bin_lo': peak_info['pos']['bin_lo'],
                    'peak_pos_bin_hi': peak_info['pos']['bin_hi'],
                    'peak_pos_w_bin':  peak_info['pos']['w_bin'],
                    'peak_pos_half_w': peak_info['pos']['half_w'],
                })
            else:
                extra.update({
                    'peak_seed':       peak_info[name]['seed'],
                    'peak_bin_lo':     peak_info[name]['bin_lo'],
                    'peak_bin_hi':     peak_info[name]['bin_hi'],
                    'peak_bin_nnodes': peak_info[name]['n_bin'],
                    'peak_w_bin':      peak_info[name]['w_bin'],
                    'peak_half_w':     peak_info[name]['half_w'],
                })

            peak_rows[name].append(
                cluster_row(lam, pp['size'], pp['omega'], pp['R_w'], pp['R_unw'],
                            pp['dphase'], pp['F'], pp['F_w'], extra=extra))

    # Save raw arrays.
    raw_kwargs = dict(
        sys_R=sys_R, sys_R_w=sys_R_w, sys_R_loc=sys_R_loc, sys_F=sys_F, sys_F_w=sys_F_w,
        pool_size=np.asarray(pool_size, dtype=int),
        pool_omega=np.asarray(pool_omega, dtype=float),
        pool_R_w=np.asarray(pool_R_w, dtype=float),
        pool_R_unw=np.asarray(pool_R_unw, dtype=float),
        pool_dphase=np.asarray(pool_dphase, dtype=float),
        pool_F=np.asarray(pool_F, dtype=float),
        pool_F_w=np.asarray(pool_F_w, dtype=float),
        num_clusters_list=np.asarray(num_clusters_list, dtype=int),
        largest_cluster_size=np.asarray(largest_cluster_size, dtype=int),
        largest_cluster_R_w=np.asarray(largest_cluster_R_w, dtype=float),
        largest_cluster_R_unw=np.asarray(largest_cluster_R_unw, dtype=float),
        largest_cluster_dphase=np.asarray(largest_cluster_dphase, dtype=float),
        largest_cluster_F=np.asarray(largest_cluster_F, dtype=float),
        largest_cluster_F_w=np.asarray(largest_cluster_F_w, dtype=float),
    )
    if PEAK_ENABLED:
        for name in PEAK_NAMES:
            pp = peak_pool[name]
            raw_kwargs[f'peak_{name}_size']   = np.asarray(pp['size'], dtype=int)
            raw_kwargs[f'peak_{name}_omega']  = np.asarray(pp['omega'], dtype=float)
            raw_kwargs[f'peak_{name}_R_w']    = np.asarray(pp['R_w'], dtype=float)
            raw_kwargs[f'peak_{name}_R_unw']  = np.asarray(pp['R_unw'], dtype=float)
            raw_kwargs[f'peak_{name}_dphase'] = np.asarray(pp['dphase'], dtype=float)
            raw_kwargs[f'peak_{name}_F']      = np.asarray(pp['F'], dtype=float)
            raw_kwargs[f'peak_{name}_F_w']    = np.asarray(pp['F_w'], dtype=float)
    np.savez_compressed(save_dir / f"raw_lambda_{lam:.4f}.npz", **raw_kwargs)


# ─── Save system and all-cluster CSV files ─────────────────
df_sys = pd.DataFrame(per_lambda_sys_rows).sort_values('lambda').reset_index(drop=True)
df_cls = pd.DataFrame(per_lambda_cls_rows).sort_values('lambda').reset_index(drop=True)
df_sys.to_csv(save_dir / "per_lambda_stats.csv", index=False)
df_cls.to_csv(save_dir / "per_lambda_clusters.csv", index=False)


# ─── Per-peak CSV and top-four R-bin/dphase summaries at each lambda ───
if PEAK_ENABLED:
    print(f"[EU grid: {country} / {sim_type} / alpha_{alpha_str}] "
          f"peak stats: per-lambda (no binning), {len(lambda_list)} lambda values")

    peak_extra_save = {}
    for name in PEAK_NAMES:
        df_pk = pd.DataFrame(peak_rows[name]).sort_values('lambda').reset_index(drop=True)
        df_pk.to_csv(save_dir / f"per_lambda_clusters_peak_{name}.csv", index=False)

        lam_arr = np.asarray(peak_lam_list[name], dtype=float)
        order = np.argsort(lam_arr)
        peak_extra_save[name] = {
            'lambda_array':     lam_arr[order],
            'R_bin_edges':      R_BIN_EDGES.copy(),
            'R_top_bin_idx':    np.asarray(peak_top_idx[name], dtype=int)[order],   # (n_lambda, 4), -1 = empty
            'R_top_bin_counts': np.asarray(peak_top_cnt[name], dtype=int)[order],   # (n_lambda, 4)
            'n':                np.asarray(peak_n_list[name], dtype=int)[order],    # (n_lambda,)
            'lambda_results':   [peak_kde_list[name][i] for i in order],            # dphase summary groups
        }

        # An empty group should occur only when no entraining cluster exists.
        n_empty = sum(1 for r in peak_extra_save[name]['lambda_results']
                      if len(r['groups']) == 0)
        n_pool0 = sum(1 for r in peak_extra_save[name]['lambda_results']
                      if r['total_clusters'] == 0)
        print(f"  [peak dphase] {name:4s}: "
              f"{len(lam_arr) - n_empty}/{len(lam_arr)} lambdas with a group; "
              f"empty={n_empty} (of which pool==0: {n_pool0})")

    np.save(save_dir / "peak_cluster_hist_results.npy", {
        'peaks':          peak_extra_save,
        'binned':         False,
        'R_hist_n_bins':  R_HIST_N_BINS,
        'R_top_n':        R_TOP_N,
        'peak_dphase_method':    'pooled summary (no KDE)',
        'peak_dphase_estimator': PEAK_DPHASE_ESTIMATOR,
        'country': country, 'sim_type': sim_type, 'alpha': alpha_str,
    })


# ─── KDE-based dphase aggregation for all clusters ─────────
# KDE remains useful here because each lambda bin contains thousands of clusters.
print(f"\n[EU grid: {country} / {sim_type} / alpha_{alpha_str}] KDE dphase aggregation ...")

lambda_vals = np.array([lam for lam, _ in lambda_list])
if len(lambda_vals) > 1:
    raw_dlam = float(np.median(np.diff(np.sort(lambda_vals))))
    bin_width = max(raw_dlam * BIN_WIDTH_FACTOR, raw_dlam)
else:
    bin_width = 0.1; raw_dlam = 0.0
lam_min = float(lambda_vals.min()); lam_max = float(lambda_vals.max())
lam_bin_edges = np.arange(lam_min, lam_max + bin_width + 1e-9, bin_width)
lam_bin_centers = 0.5 * (lam_bin_edges[:-1] + lam_bin_edges[1:])
n_bins = len(lam_bin_centers)
lam_bin_idx = np.digitize(lambda_vals, lam_bin_edges) - 1
lam_bin_idx = np.clip(lam_bin_idx, 0, n_bins - 1)
print(f"  raw lambda spacing ≈ {raw_dlam:.4f}, bin_width = {bin_width:.4f}, n_bins = {n_bins}")

bin_dphases = [[] for _ in range(n_bins)]
bin_sizes = [[] for _ in range(n_bins)]
for i, lam in enumerate(lambda_vals):
    b = lam_bin_idx[i]
    bin_dphases[b].extend(pool_dphase_by_lambda[lam].tolist())
    bin_sizes[b].extend(pool_size_by_lambda[lam].tolist())

bin_results = []
for b in range(n_bins):
    dphases = np.asarray(bin_dphases[b], dtype=float)
    sizes = np.asarray(bin_sizes[b], dtype=float)
    result = {'lambda_center': float(lam_bin_centers[b]), 'K_center': float(lam_bin_centers[b]),
              'total_clusters': int(len(dphases)), 'groups': []}
    if len(dphases) < 5:
        bin_results.append(result); continue
    groups = cluster_by_kde(dphases)
    for g in groups:
        member_dphases = dphases[g['member_indices']]
        member_sizes = sizes[g['member_indices']]
        result['groups'].append({
            'peak_dphase': float(g['peak_dphase']),
            'window_lo': float(g['window'][0]), 'window_hi': float(g['window'][1]),
            'mean_dphase': float(np.mean(member_dphases)), 'mean_size': float(np.mean(member_sizes)),
            'median_size': float(np.median(member_sizes)), 'std_dphase': float(np.std(member_dphases)),
            'std_size': float(np.std(member_sizes)), 'count': int(len(member_dphases))})
    bin_results.append(result)

total_groups = sum([len(br['groups']) for br in bin_results])
total_in_groups = sum([g['count'] for br in bin_results for g in br['groups']])
print(f"  → {total_groups} dphase groups, {total_in_groups} clusters assigned")

agg_save = {'bin_results': bin_results, 'lambda_bin_centers': lam_bin_centers,
            'lambda_bin_edges': lam_bin_edges, 'K_bin_centers': lam_bin_centers,
            'K_bin_edges': lam_bin_edges, 'bin_width': float(bin_width),
            'raw_dlam': float(raw_dlam), 'kde_bandwidth': KDE_BANDWIDTH,
            'cluster_margin': CLUSTER_MARGIN, 'min_peak_height': MIN_PEAK_HEIGHT,
            'min_cluster_count': MIN_CLUSTER_COUNT, 'min_peak_count': MIN_PEAK_COUNT,
            'merge_overlapping': MERGE_OVERLAPPING, 'valley_ratio': VALLEY_RATIO,
            'country': country, 'sim_type': sim_type, 'alpha': alpha_str}
np.save(save_dir / "dphase_aggregation_results.npy", agg_save)


# ─── metadata ──────────────────────────────────────────────
peak_meta = None
if PEAK_ENABLED:
    peak_meta = {name: {'seed': peak_info[name]['seed'], 'bin_index': peak_info[name]['ip'],
                        'bin_lo': peak_info[name]['bin_lo'], 'bin_hi': peak_info[name]['bin_hi'],
                        'n_bin_nodes': peak_info[name]['n_bin'],
                        'w_bin': peak_info[name]['w_bin'], 'half_w': peak_info[name]['half_w']}
                 for name in ('neg', 'pos')}

metadata = {
    'country': country, 'sim_type': sim_type, 'alpha': alpha_str,
    'has_rloc': bool(has_rloc), 'n_lambda': len(lambda_list),
    'lambda_min': float(lambda_list[0][0]) if lambda_list else None,
    'lambda_max': float(lambda_list[-1][0]) if lambda_list else None,
    'n_files_total': int(n_files_total), 'n_files_ok': int(n_files_ok),
    'n_files_missing': int(n_files_missing),
    'peak_restricted': {
        'enabled': bool(PEAK_ENABLED),
        'criterion': 'd-weighted half (coupling-constant weight)',
        'lambda_binned': False,
        'method': 'find_peaks_h (h=n*<d>^2); neg/pos peak bin; cluster entraining '
                  '>= half of the peak-bin oscillators\' TOTAL d (coupling constant) sum '
                  '(node membership weighted by d). None if no such cluster. '
                  'both = one cluster entraining >= half of the d-sum of BOTH neg and pos '
                  'peak bins simultaneously. Stats are computed per individual lambda '
                  '(pooled over seeds only); NO lambda binning.',
        'n_bins_peak': N_BINS_PEAK,
        'peaks': peak_meta,
        'node_rescale': str(node_rescale_file),
        'link_rescale': str(link_rescale_file),
    },
    'peak_hist': {
        'enabled': bool(PEAK_ENABLED),
        'file': 'peak_cluster_hist_results.npy',
        'binned': False,
        'R_hist_n_bins': R_HIST_N_BINS,
        'R_top_n': R_TOP_N,
        'peak_dphase_method': 'pooled summary (no KDE)',
        'peak_dphase_estimator': PEAK_DPHASE_ESTIMATOR,
        'description':
            'Per individual lambda, the top-4 most-populated bins of the peak-cluster '
            'R_w histogram over R in [0,1] (R_top_bin_idx / R_top_bin_counts / '
            'R_bin_edges, empty slot = -1), plus n (cluster count) for '
            'relative-frequency thresholding. Also lambda_results: the peak-cluster '
            'dphase pooled over seeds and summarized into a SINGLE representative group '
            'per lambda (dphase_median / dphase_mean / dphase_std / size_median / count; '
            'window_lo,hi = actual pool min,max). NO KDE grouping is applied: the peak '
            'pool already holds exactly one entraining cluster per seed, so the seeds are '
            'realizations of the same object and there is nothing to split. The previous '
            'KDE step silently discarded whole lambdas whose density had no qualifying '
            'peak (e.g. 162 of 201 lambdas for DE massless) even though the pool was full, '
            'which punched holes in the bubble plot. Here groups is empty ONLY when the '
            'pool itself is empty, i.e. no cluster met the d-weighted half criterion at '
            'that lambda -- a real physical absence, not a thresholding artifact. '
            'lambda_array holds the actual simulated lambda values.',
    },
    'kde_aggregation': {
        'note': 'KDE is still used HERE (whole-cluster aggregation), where each lambda '
                'bin pools thousands of clusters and multimodal structure is genuine.',
        'bin_width': float(bin_width), 'n_bins': int(n_bins),
        'kde_bandwidth': KDE_BANDWIDTH, 'cluster_margin': CLUSTER_MARGIN,
        'min_peak_height': MIN_PEAK_HEIGHT, 'min_peak_count': MIN_PEAK_COUNT,
        'min_cluster_count': MIN_CLUSTER_COUNT, 'merge_overlapping': MERGE_OVERLAPPING,
        'valley_ratio': VALLEY_RATIO, 'total_groups': int(total_groups),
        'total_in_groups': int(total_in_groups),
    },
    'description': {
        'system_level': 'R, R_weighted, R_local, F, F_weighted per seed; stats per lambda.',
        'cluster_level': 'Clusters pooled across seeds within each lambda.',
        'peak_restricted_cluster_level':
            'Only the cluster entraining >= half of each peak bin\'s TOTAL d (coupling '
            'constant) sum, pooled across seeds. Files '
            'per_lambda_clusters_peak_{neg,pos,both}.csv, one row per SIMULATED lambda '
            '(no binning). neg/pos: representative cluster of each peak (duplicates '
            'allowed when merged -- a single cluster may entrain both peaks, in which '
            'case it appears in the neg pool, the pos pool, AND the both pool). '
            'both: a single cluster entraining >= half of the d-sum of BOTH peak bins. '
            'Columns mirror per_lambda_clusters.csv plus peak metadata (w_bin/half_w), '
            'n_seed_hit / n_seed_none, and R_top_bin_{idx,count,lo,hi}_0..3 / R_hist_n. '
            'NaN stats when no qualifying cluster.',
    },
    'source_base': str(base_dir), 'save_dir': str(save_dir),
}
with open(save_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"[EU grid: {country} / {sim_type} / alpha_{alpha_str}] Done")
print(f"{'='*60}")
print(f"  lambdas  : {len(lambda_list)}")
print(f"  files OK : {n_files_ok} / {n_files_total}  (missing: {n_files_missing})")
print(f"  Saved    : {save_dir}")
print(f"    - per_lambda_stats.csv")
print(f"    - per_lambda_clusters.csv")
if PEAK_ENABLED:
    print(f"    - per_lambda_clusters_peak_neg.csv   (per-lambda, neg peak entraining cluster)")
    print(f"    - per_lambda_clusters_peak_pos.csv   (per-lambda, pos peak entraining cluster)")
    print(f"    - per_lambda_clusters_peak_both.csv  (per-lambda, both peaks merged)")
    print(f"    - peak_cluster_hist_results.npy      (per-lambda top-4 R bins + dphase summary, no KDE)")
    print(f"    - raw_lambda_*.npz  (peak arrays included)")
else:
    print(f"    - raw_lambda_*.npz")
print(f"    - dphase_aggregation_results.npy")
print(f"    - metadata.json")
