"""Numerical integrators for the degree-weighted swing equation.

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij * sin(theta_j-theta_i)

Calculate interaction term as follows:
sum_j K_ij * A_ij * sin(theta_j-theta_i)
= cos(theta_i) * [KA @ sin(theta)]_i - sin(theta_i) * [KA @ cos(theta)]_i

The time-stepping functions are compiled with Numba.
"""

from typing import overload

import numpy as np
import numpy.typing as npt
import scipy.sparse
from numba import njit

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]
arr = arr32 | arr64
sparse = scipy.sparse.csr_matrix | scipy.sparse.coo_matrix | scipy.sparse.csc_matrix


@njit(fastmath=True)
def get_acceleration(
    weighted_adjacency_matrix: arr, degree: arr, params: arr, phase: arr, dphase: arr
) -> arr:
    """Return acceleration for ``params = [power, damping, inertia]``."""

    # Interaction
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    mean_sin, mean_cos = np.sum(degree * sin_phase), np.sum(degree * cos_phase)
    sin_phase_adj = weighted_adjacency_matrix * sin_phase
    cos_phase_adj = weighted_adjacency_matrix * cos_phase

    force = (
        params[0]  # P
        - params[1] * dphase  # -gamma * velocity
        + mean_sin * cos_phase_adj  # interaction1
        - mean_cos * sin_phase_adj  # interaction2
    )
    return force / params[2]  # a = F / m


@overload
def rk1(
    weighted_adjacency_matrix: arr32,
    degree: arr32,
    params: arr32,
    phase: arr32,
    dphase: arr32,
    dt: arr32,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk1(
    weighted_adjacency_matrix: arr64,
    degree: arr64,
    params: arr64,
    phase: arr64,
    dphase: arr64,
    dt: arr64,
) -> tuple[arr64, arr64]:
    ...


@njit(fastmath=True)
def rk1(
    weighted_adjacency_matrix: arr, degree: arr, params: arr, phase: arr, dphase: arr, dt: arr
) -> tuple[arr, arr]:
    """Return next phase, next dphase"""
    acceleration = get_acceleration(weighted_adjacency_matrix, degree, params, phase, dphase)
    velocity = dphase
    return phase + dt * velocity, dphase + dt * acceleration


@overload
def rk2(
    weighted_adjacency_matrix: arr32,
    degree: arr32,
    params: arr32,
    phase: arr32,
    dphase: arr32,
    dt: arr32,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk2(
    weighted_adjacency_matrix: arr64,
    degree: arr64,
    params: arr64,
    phase: arr64,
    dphase: arr64,
    dt: arr64,
) -> tuple[arr64, arr64]:
    ...


@njit(fastmath=True)
def rk2(
    weighted_adjacency_matrix: arr, degree: arr, params: arr, phase: arr, dphase: arr, dt: arr
) -> tuple[arr, arr]:
    """Return next phase, next dphase"""
    acceleration1 = get_acceleration(weighted_adjacency_matrix, degree, params, phase, dphase)
    velocity1 = dphase

    temp_phase = phase + dt * velocity1
    temp_dphase = dphase + dt * acceleration1
    acceleration2 = get_acceleration(
        weighted_adjacency_matrix, degree, params, temp_phase, temp_dphase
    )
    velocity2 = temp_dphase

    velocity = np.array(0.5, dtype=dt.dtype) * (velocity1 + velocity2)
    acceleration = np.array(0.5, dtype=dt.dtype) * (acceleration1 + acceleration2)
    return phase + dt * velocity, dphase + dt * acceleration


@overload
def rk4(
    weighted_adjacency_matrix: arr32,
    degree: arr32,
    params: arr32,
    phase: arr32,
    dphase: arr32,
    dt: arr32,
) -> tuple[arr32, arr32]:
    ...


@overload
def rk4(
    weighted_adjacency_matrix: arr64,
    degree: arr64,
    params: arr64,
    phase: arr64,
    dphase: arr64,
    dt: arr64,
) -> tuple[arr64, arr64]:
    ...


@njit(fastmath=True)
def rk4(
    weighted_adjacency_matrix: arr, degree: arr, params: arr, phase: arr, dphase: arr, dt: arr
) -> tuple[arr, arr]:
    """
    Return next phase, next dphase

    weighted adjacency matrix: (N, N)
    params: (3, N) power, gamma, mass
    phase: (N, )
    dphase: (N, )
    """
    acceleration1 = get_acceleration(weighted_adjacency_matrix, degree, params, phase, dphase)
    velocity1 = dphase

    temp_phase = phase + np.array(0.5) * dt * velocity1
    temp_dphase = dphase + np.array(0.5) * dt * acceleration1
    acceleration2 = get_acceleration(
        weighted_adjacency_matrix, degree, params, temp_phase, temp_dphase
    )
    velocity2 = temp_dphase

    temp_phase = phase + np.array(0.5) * dt * velocity2
    temp_dphase = dphase + np.array(0.5) * dt * acceleration2
    acceleration3 = get_acceleration(
        weighted_adjacency_matrix, degree, params, temp_phase, temp_dphase
    )
    velocity3 = temp_dphase

    temp_phase = phase + dt * velocity3
    temp_dphase = dphase + dt * acceleration3
    acceleration4 = get_acceleration(
        weighted_adjacency_matrix, degree, params, temp_phase, temp_dphase
    )
    velocity4 = temp_dphase

    velocity = (
        velocity1
        + np.array(2.0) * velocity2
        + np.array(2.0) * velocity3
        + velocity4
    ) / np.array(6.0)
    acceleartion = (
        acceleration1
        + np.array(2.0) * acceleration2
        + np.array(2.0) * acceleration3
        + acceleration4
    ) / np.array(6.0)
    return phase + dt * velocity, dphase + dt * acceleartion
