from core.regular_sampling import *
from core.detect_cluster_GCC import *
import numpy as np


def f_of_Rp_Gradually(Rp, omega, omega_c, K, alpha, gamma, m, N, exponent, cluster_mask):
    """Contribution from coherent oscillators inside the cluster."""
    if Rp <= 0 or not np.any(cluster_mask):
        return 0.0
    degree = np.power(np.abs(omega) + 1.0, exponent)
    degree_bar = np.mean(degree)
    degree_per_total = degree / np.sum(degree)
    K_i = K * degree / degree_bar
    val = np.sqrt(np.maximum(0, (K_i * Rp)**2 - (omega - omega_c)**2)) / (K_i * Rp)
    return np.sum(degree_per_total[cluster_mask] * val[cluster_mask])


def g_of_Rp_Gradually(Rp, omega, omega_c, K, alpha, gamma, m, N, exponent, cluster_mask):
    """Contribution from incoherent oscillators outside the cluster."""
    if Rp <= 0:
        return 0.0
    degree = np.power(np.abs(omega) + 1.0, exponent)
    degree_bar = np.mean(degree)
    degree_per_total = degree / np.sum(degree)
    K_i = K * degree / degree_bar
    outside_cluster = ~cluster_mask
    omega_shifted = omega - omega_c
    val = -0.5 * (gamma**2 * m * K_i * Rp) / (omega_shifted**2 * m**2 + gamma**4)
    return np.sum(degree_per_total[outside_cluster] * val[outside_cluster])


def A_of_Rp_array_discrete(K, omega_cluster, omega, omega_bar,
                           alpha, gamma, m, N,
                           R_max, n_R, exponent):
    """
    Compute the action A(R) for the massive double-peak case.
    Two nucleation scenarios are evaluated: from_max (edge) and from_center.

    Returns
    -------
    R_grid, A_vals_from_max, A_vals_from_center,
    omega_c_from_max, omega_c_from_center
        omega_c_* : cluster-center frequency at each R (equal to theta_dot_c).
    """
    R_grid = np.linspace(0, R_max, n_R + 1)
    dR = R_max / n_R

    degree = np.power(np.abs(omega) + 1.0, exponent)
    degree_bar = np.mean(degree)
    K_i = K * degree / degree_bar

    omega_max = np.max(omega)
    omega_min = np.min(omega)

    # Scenario 1: start at omega_max.
    cluster_mask_max = np.zeros(N, dtype=bool)
    omega_c_max = omega_max
    A_vals_from_max = np.zeros_like(R_grid)
    omega_c_vals_max = np.full_like(R_grid, np.nan)
    cum_area_max = 0.0

    # Scenario 2: start at omega = 0 (center).
    cluster_mask_center = np.zeros(N, dtype=bool)
    omega_c_center = 0.0
    A_vals_from_center = np.zeros_like(R_grid)
    omega_c_vals_center = np.full_like(R_grid, np.nan)
    cum_area_center = 0.0

    for i in range(n_R + 1):
        R_current = R_grid[i]

        if R_current <= 0:
            A_vals_from_max[i] = 0.0
            A_vals_from_center[i] = 0.0
            omega_c_vals_max[i] = omega_max
            omega_c_vals_center[i] = 0.0
            continue

        # Scenario 1: from_max.
        if np.any(cluster_mask_max):
            omega_c_max = np.mean(omega[cluster_mask_max])
        else:
            omega_c_max = omega_max
        omega_c_vals_max[i] = omega_c_max

        term1_max = (4.0 / np.pi) * np.sqrt(K_i * R_current / m)
        term2_max = alpha / np.sqrt(K_i * R_current * m**3)
        melnikov_radius_max = np.maximum(0.0, term1_max + term2_max)
        cluster_mask_max = (np.abs(omega - omega_c_max) < melnikov_radius_max) & (omega <= omega_max)

        f_val_max = f_of_Rp_Gradually(R_current, omega, omega_c_max, K, alpha, gamma, m, N, exponent, cluster_mask_max)
        g_val_max = g_of_Rp_Gradually(R_current, omega, omega_c_max, K, alpha, gamma, m, N, exponent, cluster_mask_max)
        integrand_max = R_current - (f_val_max + g_val_max)
        cum_area_max += integrand_max * dR
        A_vals_from_max[i] = K * cum_area_max

        # Scenario 2: from_center.
        if np.any(cluster_mask_center):
            omega_c_center = np.mean(omega[cluster_mask_center])
        else:
            omega_c_center = 0.0
        omega_c_vals_center[i] = omega_c_center

        term1_center = (4.0 / np.pi) * np.sqrt(K_i * R_current / m)
        term2_center = alpha / np.sqrt(K_i * R_current * m**3)
        melnikov_radius_center = np.maximum(0.0, term1_center + term2_center)
        cluster_mask_center = (np.abs(omega - omega_c_center) < melnikov_radius_center)

        f_val_center = f_of_Rp_Gradually(R_current, omega, omega_c_center, K, alpha, gamma, m, N, exponent, cluster_mask_center)
        g_val_center = g_of_Rp_Gradually(R_current, omega, omega_c_center, K, alpha, gamma, m, N, exponent, cluster_mask_center)
        integrand_center = R_current - (f_val_center + g_val_center)
        cum_area_center += integrand_center * dR
        A_vals_from_center[i] = K * cum_area_center

    return (R_grid, A_vals_from_max, A_vals_from_center,
            omega_c_vals_max, omega_c_vals_center)
