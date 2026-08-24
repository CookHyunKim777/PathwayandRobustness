from core.regular_sampling import *
from core.detect_cluster_GCC import *
import numpy as np


def sample_powerlaw_positive(power_exp, N, seed, MIN=1.0, MAX=4.0):
    """Generate single-peak ω values as in RK4; mean-centering is done by the caller."""
    np.random.seed(seed)
    x    = np.arange(1, N + 1)[::-1]
    mgp1 = -power_exp + 1
    km   = MIN ** mgp1
    kM   = MAX ** mgp1
    omega = (kM + (x / N) * (km - kM)) ** (1.0 / mgp1) - MIN
    np.random.shuffle(omega)
    return omega


def f_of_Rp_Gradually(Rp, omega, omega_c, K, alpha, gamma, m, N, exponent,
                      cluster_mask, omega_raw):
    """Coherent contribution; degree is based on pre-centered omega_raw."""
    if Rp <= 0 or not np.any(cluster_mask):
        return 0.0
    degree = np.power(np.abs(omega_raw) + 1.0, exponent)
    degree_bar = np.mean(degree)
    degree_per_total = degree / np.sum(degree)
    K_i = K * degree / degree_bar
    val = np.sqrt(np.maximum(0, (K_i * Rp)**2 - (omega - omega_c)**2)) / (K_i * Rp)
    return np.sum(degree_per_total[cluster_mask] * val[cluster_mask])


def g_of_Rp_Gradually(Rp, omega, omega_c, K, alpha, gamma, m, N, exponent,
                      cluster_mask, omega_raw):
    """Incoherent contribution; degree is based on pre-centered omega_raw."""
    if Rp <= 0:
        return 0.0
    degree = np.power(np.abs(omega_raw) + 1.0, exponent)
    degree_bar = np.mean(degree)
    degree_per_total = degree / np.sum(degree)
    K_i = K * degree / degree_bar
    outside_cluster = ~cluster_mask
    omega_shifted = omega - omega_c
    val = -0.5 * (gamma**2 * m * K_i * Rp) / (omega_shifted**2 * m**2 + gamma**4)
    return np.sum(degree_per_total[outside_cluster] * val[outside_cluster])


def A_of_Rp_array_discrete(K, omega, omega_raw, omega_bar,
                           alpha, gamma, m, N,
                           R_max, n_R, exponent):
    """
    Compute the action A(R) for the massive single-peak case.
    Degree is based on pre-centered omega_raw; dynamical omega is centered.
    Two nucleation scenarios are evaluated: from_max and from_min.

    Returns
    -------
    R_grid, A_vals_from_max, A_vals_from_min,
    omega_c_from_max, omega_c_from_min
    """
    R_grid = np.linspace(0, R_max, n_R + 1)
    dR = R_max / n_R

    degree = np.power(np.abs(omega_raw) + 1.0, exponent)
    degree_bar = np.mean(degree)
    K_i = K * degree / degree_bar

    omega_max = np.max(omega)
    omega_min = np.min(omega)

    cluster_mask_max = np.zeros(N, dtype=bool)
    omega_c_max = omega_max
    A_vals_from_max = np.zeros_like(R_grid)
    omega_c_vals_max = np.full_like(R_grid, np.nan)
    cum_area_max = 0.0

    cluster_mask_min = np.zeros(N, dtype=bool)
    omega_c_min = omega_min
    A_vals_from_min = np.zeros_like(R_grid)
    omega_c_vals_min = np.full_like(R_grid, np.nan)
    cum_area_min = 0.0

    for i in range(n_R + 1):
        R_current = R_grid[i]

        if R_current <= 0:
            A_vals_from_max[i] = 0.0
            A_vals_from_min[i] = 0.0
            omega_c_vals_max[i] = omega_max
            omega_c_vals_min[i] = omega_min
            continue

        # from_max
        if np.any(cluster_mask_max):
            omega_c_max = np.mean(omega[cluster_mask_max])
        else:
            omega_c_max = omega_max
        omega_c_vals_max[i] = omega_c_max

        term1_max = (4.0 / np.pi) * np.sqrt(K_i * R_current / m)
        term2_max = alpha / np.sqrt(K_i * R_current * m**3)
        melnikov_radius_max = np.maximum(0.0, term1_max + term2_max)
        cluster_mask_max = (np.abs(omega - omega_c_max) < melnikov_radius_max) & (omega <= omega_max)

        f_val_max = f_of_Rp_Gradually(R_current, omega, omega_c_max, K, alpha, gamma, m, N, exponent, cluster_mask_max, omega_raw)
        g_val_max = g_of_Rp_Gradually(R_current, omega, omega_c_max, K, alpha, gamma, m, N, exponent, cluster_mask_max, omega_raw)
        integrand_max = R_current - (f_val_max + g_val_max)
        cum_area_max += integrand_max * dR
        A_vals_from_max[i] = K * cum_area_max

        # from_min
        if np.any(cluster_mask_min):
            omega_c_min = np.mean(omega[cluster_mask_min])
        else:
            omega_c_min = omega_min
        omega_c_vals_min[i] = omega_c_min

        term1_min = (4.0 / np.pi) * np.sqrt(K_i * R_current / m)
        term2_min = alpha / np.sqrt(K_i * R_current * m**3)
        melnikov_radius_min = np.maximum(0.0, term1_min + term2_min)
        cluster_mask_min = (np.abs(omega - omega_c_min) < melnikov_radius_min) & (omega >= omega_min)

        f_val_min = f_of_Rp_Gradually(R_current, omega, omega_c_min, K, alpha, gamma, m, N, exponent, cluster_mask_min, omega_raw)
        g_val_min = g_of_Rp_Gradually(R_current, omega, omega_c_min, K, alpha, gamma, m, N, exponent, cluster_mask_min, omega_raw)
        integrand_min = R_current - (f_val_min + g_val_min)
        cum_area_min += integrand_min * dR
        A_vals_from_min[i] = K * cum_area_min

    return (R_grid, A_vals_from_max, A_vals_from_min,
            omega_c_vals_max, omega_c_vals_min)
