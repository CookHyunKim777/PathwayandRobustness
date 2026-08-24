"""Numerical integrators for the degree-weighted massless Kuramoto model.

d theta_i / dt = omega_i + K_i * [<d*sin(theta)> * cos(theta_i) - <d*cos(theta)> * sin(theta_i)]

where:
- K_i = K * degree_i / mean(degree)  (node-specific coupling)
- <d*sin(theta)> = sum_j (degree_j / total_degree) * sin(theta_j)
- <d*cos(theta)> = sum_j (degree_j / total_degree) * cos(theta_j)

This is equivalent to mean-field coupling weighted by degree.

The time-stepping functions are compiled with Numba.
"""

import numpy as np
import numpy.typing as npt
from numba import njit

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]
arr = arr32 | arr64


@njit(fastmath=True)
def get_velocity(
    weighted_adjacency: arr, degree_per_total: arr, omega: arr, phase: arr
) -> arr:
    """
    Calculate dtheta/dt for massless Kuramoto model with degree weighting
    
    weighted_adjacency: K * degree / mean(degree) for each node (N,)
    degree_per_total: degree / sum(degree) for each node (N,)
    omega: natural frequencies (N,)
    phase: current phases (N,)
    
    Returns: dtheta/dt (N,)
    """
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    
    # Degree-weighted mean field
    mean_sin = np.sum(degree_per_total * sin_phase)
    mean_cos = np.sum(degree_per_total * cos_phase)
    
    # omega_i + K_i * [<d*sin(theta)> * cos(theta_i) - <d*cos(theta)> * sin(theta_i)]
    velocity = omega + weighted_adjacency * (mean_sin * cos_phase - mean_cos * sin_phase)
    return velocity


@njit(fastmath=True)
def rk1_massless(
    weighted_adjacency: arr, degree_per_total: arr, omega: arr, phase: arr, dt: float
) -> arr:
    """Euler method for massless Kuramoto with degree weighting"""
    velocity = get_velocity(weighted_adjacency, degree_per_total, omega, phase)
    return phase + dt * velocity


@njit(fastmath=True)
def rk2_massless(
    weighted_adjacency: arr, degree_per_total: arr, omega: arr, phase: arr, dt: float
) -> arr:
    """RK2 (Heun's method) for massless Kuramoto with degree weighting"""
    k1 = get_velocity(weighted_adjacency, degree_per_total, omega, phase)
    k2 = get_velocity(weighted_adjacency, degree_per_total, omega, phase + dt * k1)
    return phase + dt * 0.5 * (k1 + k2)


@njit(fastmath=True)
def rk4_massless(
    weighted_adjacency: arr, degree_per_total: arr, omega: arr, phase: arr, dt: float
) -> arr:
    """
    RK4 for massless Kuramoto model with degree weighting
    
    weighted_adjacency: K * degree / mean(degree) for each node (N,)
    degree_per_total: degree / sum(degree) for each node (N,)
    omega: natural frequencies (N,)
    phase: current phases (N,)
    dt: time step
    
    Returns: next phase (N,)
    """
    k1 = get_velocity(weighted_adjacency, degree_per_total, omega, phase)
    k2 = get_velocity(weighted_adjacency, degree_per_total, omega, phase + 0.5 * dt * k1)
    k3 = get_velocity(weighted_adjacency, degree_per_total, omega, phase + 0.5 * dt * k2)
    k4 = get_velocity(weighted_adjacency, degree_per_total, omega, phase + dt * k3)
    
    return phase + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
