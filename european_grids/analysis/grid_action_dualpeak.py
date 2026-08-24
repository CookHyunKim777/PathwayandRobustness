"""
grid_action_dualpeak.py
=======================
Ad-hoc potential (action) A(R) for European power grids, computed from a
NEGATIVE-side peak seed and a POSITIVE-side peak seed separately, for both
the massive (2nd-order, m=10) and massless (1st-order, m=0) variants.

INPUT (per country, per alpha), produced by Prepare_Grid_Knobs.py:
  {data_root}/{country}/alpha_{alpha}/{country}_node_rescale.txt
      columns: mass, gamma, omega   (omega == mean-centered P == used as ω)
  {data_root}/{country}/alpha_{alpha}/{country}_link_rescale.txt
      one column, N values: per-node coupling weight d_i  (<d>=1 normalized)

COUPLING
  Per-node effective coupling  K_i = K * d_i    (d_i used directly; <d>=1).
  Cluster weighting uses d_i / sum(d).

SEED (peak) DEFINITION  -- h(P) = n_bin * <d>_bin^2  (coupling-weighted)
  P range is uniformly binned (n_bins). Among bins whose center < 0 the
  argmax of h gives the NEGATIVE peak; among bins whose center > 0 the
  argmax of h gives the POSITIVE peak. Each peak's bin-center ω is the
  nucleation seed; the cluster grows two-sidedly from that seed.

VARIANTS
  massive  (m>0, 2nd order):  Melnikov-radius entrainment,
                              drift g = -1/2 * gamma^2 m K_i R /((dω)^2 m^2 + gamma^4)
  massless (m=0, 1st order):  standard Kuramoto locking |dω| < K_i R,
                              drift g = -(K_i R)/(2 dω) * (1 - sqrt(1-(K_i R/dω)^2))

MELNIKOV RADIUS (massive)  -- matched to the single/double-peak convention:
      r_mel = (4/pi) * sqrt(K_i R / m) + alpha / sqrt(K_i R * m^3)
  Note the SECOND term is ADDED (not subtracted) and uses a separate
  constant `alpha` (distinct from the damping `gamma` in g), with m^3 under
  the root.
"""

import numpy as np


# ---------------------------------------------------------------------------
# h(P) peak finder (coupling-weighted): h_bin = n_bin * <d>_bin^2
# ---------------------------------------------------------------------------
def find_peaks_h(omega, d, n_bins=16):
    """Return (omega_seed_neg, omega_seed_pos, diag) using h(P)=n*<d>^2.

    Negative peak: argmax of h among bins with center < 0.
    Positive peak: argmax of h among bins with center > 0.
    If a sign side has no populated bin, its seed is None.
    """
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
        if sign < 0:
            side = (centers < 0) & (n_arr > 0)
        else:
            side = (centers > 0) & (n_arr > 0)
        if not np.any(side):
            return None, None
        h_side = np.where(side, h_arr, -np.inf)
        ip = int(np.argmax(h_side))
        return float(centers[ip]), ip

    seed_neg, ip_neg = side_peak(-1)
    seed_pos, ip_pos = side_peak(+1)

    diag = {
        'edges': edges, 'centers': centers,
        'n': n_arr, 'mean_s': s_mean, 'h': h_arr,
        'ip_neg': ip_neg, 'ip_pos': ip_pos,
    }
    return seed_neg, seed_pos, diag


# ---------------------------------------------------------------------------
# f, g  --  MASSIVE (2nd order, m > 0)
# ---------------------------------------------------------------------------
def f_massive(Rp, omega, omega_c, K_i, d_per_total, cluster_mask):
    if Rp <= 0 or not np.any(cluster_mask):
        return 0.0
    val = np.sqrt(np.maximum(0.0, (K_i * Rp) ** 2 - (omega - omega_c) ** 2)) / (K_i * Rp)
    return np.sum(d_per_total[cluster_mask] * val[cluster_mask])


def g_massive(Rp, omega, omega_c, K_i, gamma, m, d_per_total, cluster_mask):
    if Rp <= 0:
        return 0.0
    outside = ~cluster_mask
    if not np.any(outside):
        return 0.0
    dw = omega - omega_c
    val = -0.5 * (gamma ** 2 * m * K_i * Rp) / (dw ** 2 * m ** 2 + gamma ** 4)
    return np.sum(d_per_total[outside] * val[outside])


def melnikov_radius_massive(Rp, K_i, alpha, m):
    """Entrainment radius matched to single/double-peak convention:
        r_mel = (4/pi)*sqrt(K_i R / m) + alpha / sqrt(K_i R * m^3)
    Second term is ADDED and uses constant `alpha` with m^3 under the root.
    """
    KR = np.maximum(K_i * Rp, 1e-12)
    term1 = (4.0 / np.pi) * np.sqrt(KR / m)
    term2 = alpha / np.sqrt(KR * m ** 3)
    return np.maximum(0.0, term1 + term2)


# ---------------------------------------------------------------------------
# f  --  MASSLESS (1st order, standard Kuramoto); NO drift term (g=0)
# ---------------------------------------------------------------------------
def f_massless(Rp, omega, omega_c, K_i, d_per_total, cluster_mask):
    if Rp <= 0 or not np.any(cluster_mask):
        return 0.0
    val = np.sqrt(np.maximum(0.0, (K_i * Rp) ** 2 - (omega - omega_c) ** 2)) / (K_i * Rp)
    return np.sum(d_per_total[cluster_mask] * val[cluster_mask])


# ---------------------------------------------------------------------------
# A(R) from a given seed, two-sided growth
# ---------------------------------------------------------------------------
def _cluster_mask_at(R_current, omega, omega_c, K_i, alpha, gamma, m, variant):
    """Recompute the entrainment mask at a given R and omega_c."""
    if variant == 'massive':
        mel = melnikov_radius_massive(R_current, K_i, alpha, m)
        return np.abs(omega - omega_c) < mel
    elif variant == 'massless':
        return np.abs(omega - omega_c) < (K_i * R_current)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def compute_action_from_seed(K, omega, d, alpha, gamma, m, omega_seed,
                             R_max=1.0, n_R=10000, variant='massive',
                             include_g=True):
    """Grow a cluster two-sidedly from omega_seed.

    Parameters
    ----------
    alpha : Melnikov second-term constant (massive variant only).
    gamma : damping constant used in the drift term g (massive variant only).
    include_g : if False, drop the incoherent drift term g in the massive
                variant (integrand becomes R - f only, like the massless
                case but with Melnikov-radius entrainment). Default True.

    Returns
    -------
    R_grid       : (n_R+1,) radius grid
    A_vals       : (n_R+1,) action A(R)
    omega_c_vals : (n_R+1,) cluster centre frequency at each R
                   (== theta_dot_c, the entrained cluster's common rotation
                   speed under the drift-free mean-field approximation)
    """
    N = len(omega)
    R_grid = np.linspace(0.0, R_max, n_R + 1)
    dR = R_max / n_R
    A_vals = np.zeros(n_R + 1)
    omega_c_vals = np.full(n_R + 1, np.nan)

    K_i = K * d
    d_per_total = d / np.sum(d)

    cluster_mask = np.zeros(N, dtype=bool)
    cum_area = 0.0

    for i in range(n_R + 1):
        R_current = R_grid[i]
        if R_current <= 0.0:
            A_vals[i] = 0.0
            omega_c_vals[i] = omega_seed
            continue

        omega_c = np.mean(omega[cluster_mask]) if np.any(cluster_mask) else omega_seed
        omega_c_vals[i] = omega_c

        if variant == 'massive':
            mel = melnikov_radius_massive(R_current, K_i, alpha, m)
            cluster_mask = np.abs(omega - omega_c) < mel
            f_val = f_massive(R_current, omega, omega_c, K_i, d_per_total, cluster_mask)
            if include_g:
                g_val = g_massive(R_current, omega, omega_c, K_i, gamma, m, d_per_total, cluster_mask)
            else:
                g_val = 0.0
            integrand = R_current - (f_val + g_val)
        elif variant == 'massless':
            # standard Kuramoto locking condition; NO drift term
            cluster_mask = np.abs(omega - omega_c) < (K_i * R_current)
            f_val = f_massless(R_current, omega, omega_c, K_i, d_per_total, cluster_mask)
            integrand = R_current - f_val
        else:
            raise ValueError(f"Unknown variant: {variant}")

        cum_area += integrand * dR
        A_vals[i] = K * cum_area

    return R_grid, A_vals, omega_c_vals


def mask_at_Rstar(R_star, omega_c_star, omega, d, K, alpha, gamma, m, variant):
    """Reconstruct the entrainment mask at R_star given omega_c_star."""
    K_i = K * d
    return _cluster_mask_at(R_star, omega, omega_c_star, K_i, alpha, gamma, m, variant)


def analyze_action(R_grid, A_vals):
    """Return R_star, A_min, R_barrier, A_barrier."""
    idx_min = int(np.argmin(A_vals))
    R_star = R_grid[idx_min]
    A_min = A_vals[idx_min]
    if idx_min > 0:
        rng = np.where(R_grid <= R_star)[0]
        idx_b = rng[int(np.argmax(A_vals[rng]))]
        R_barrier = R_grid[idx_b]
        A_barrier = A_vals[idx_b]
    else:
        R_barrier = 0.0
        A_barrier = 0.0
    return R_star, A_min, R_barrier, A_barrier