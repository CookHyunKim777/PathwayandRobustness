from core.regular_sampling import *
from core.detect_cluster_GCC import *
import numpy as np


def f_of_Rp_Gradually_massless(Rp, omega, omega_c, K, N, exponent, cluster_mask):
    """Contribution from coherent oscillators in the massless case."""
    if Rp <= 0 or not np.any(cluster_mask):
        return 0.0
    degree = np.power(np.abs(omega) + 1.0, exponent)
    degree_bar = np.mean(degree)
    degree_per_total = degree / np.sum(degree)
    K_i = K * degree / degree_bar
    val = np.sqrt(np.maximum(0, (K_i * Rp)**2 - (omega - omega_c)**2)) / (K_i * Rp)
    return np.sum(degree_per_total[cluster_mask] * val[cluster_mask])


def A_of_Rp_array_discrete_massless(K, omega_cluster, omega, omega_bar,
                                    alpha, gamma, N,
                                    R_max, n_R, exponent):
    """
    Compute the action A(R) for the massless double-peak case, with
    lock_radius = K_i * R. The two nucleation scenarios are from_max (edge)
    and from_center.

    Returns
    -------
    R_grid, A_vals_from_max, A_vals_from_center,
    omega_c_from_max, omega_c_from_center
    """
    R_grid = np.linspace(0, R_max, n_R + 1)
    dR = R_max / n_R

    degree = np.power(np.abs(omega) + 1.0, exponent)
    degree_bar = np.mean(degree)
    K_i = K * degree / degree_bar

    omega_max = np.max(omega)
    omega_min = np.min(omega)

    cluster_mask_max = np.zeros(N, dtype=bool)
    omega_c_max = omega_max
    A_vals_from_max = np.zeros_like(R_grid)
    omega_c_vals_max = np.full_like(R_grid, np.nan)
    cum_area_max = 0.0

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

        # from_max
        if np.any(cluster_mask_max):
            omega_c_max = np.mean(omega[cluster_mask_max])
        else:
            omega_c_max = omega_max
        omega_c_vals_max[i] = omega_c_max

        lock_radius_max = K_i * R_current
        cluster_mask_max = (np.abs(omega - omega_c_max) < lock_radius_max) & (omega <= omega_max)

        f_val_max = f_of_Rp_Gradually_massless(R_current, omega, omega_c_max, K, N, exponent, cluster_mask_max)
        integrand_max = R_current - f_val_max  # massless: no drift term
        cum_area_max += integrand_max * dR
        A_vals_from_max[i] = K * cum_area_max

        # from_center
        if np.any(cluster_mask_center):
            omega_c_center = np.mean(omega[cluster_mask_center])
        else:
            omega_c_center = 0.0
        omega_c_vals_center[i] = omega_c_center

        lock_radius_center = K_i * R_current
        cluster_mask_center = (np.abs(omega - omega_c_center) < lock_radius_center)

        f_val_center = f_of_Rp_Gradually_massless(R_current, omega, omega_c_center, K, N, exponent, cluster_mask_center)
        integrand_center = R_current - f_val_center  # massless: no drift term
        cum_area_center += integrand_center * dR
        A_vals_from_center[i] = K * cum_area_center

    return (R_grid, A_vals_from_max, A_vals_from_center,
            omega_c_vals_max, omega_c_vals_center)
