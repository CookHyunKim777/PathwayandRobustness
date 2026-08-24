"""Sampling routines for initial phases and natural frequencies."""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy
import scipy.integrate as integrate
from scipy.linalg import solve
from scipy.optimize import brentq, fsolve, newton, root
from scipy.integrate import quad
import time

import networkx as nx
import numba as nb
import numpy as np

def Gaussian_distribution(sigma, N, RegularSampling, seed = None):
    """Sample a Gaussian distribution with standard deviation ``sigma``."""
    np.random.seed(seed)
    if RegularSampling:
        x = np.arange(1, N + 1)
        np.random.shuffle(x)
        return np.sqrt(2) * sigma * scipy.special.erfinv(-1 + (2 * x - 1) / N)
    else:
        return np.random.normal(0, sigma, N)

def Bimodal_distribution_density(x, delta, sigma):
    return 1/2/np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power((x-delta),2)/2/sigma**2) + 1/2/np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power((x+delta),2)/2/sigma**2)
def Bimodal_distribution_cumulative(y, delta, sigma):
    return quad(Bimodal_distribution_density, 0, y, args=(delta, sigma), epsabs=1.0e-16, epsrel=1.0e-16, limit=1000000)[0]
def solve_Bimodal_distribution_cumulative(w, z, delta, sigma):
    return z - Bimodal_distribution_cumulative(w, delta, sigma)

def Bimodal_distribution(delta, sigma, N, RegularSampling, seed = None):
    """Sample a symmetric bimodal Gaussian distribution."""
    np.random.seed(seed)

    if RegularSampling:
        v = np.zeros(N)     
        for i in np.arange(1, N + 1):
            z = -1/2 + (i - 1/2) / N
            epsilon = delta * np.sign(z)
            solution= fsolve(solve_Bimodal_distribution_cumulative, x0=epsilon, xtol=1e-16, args=(z, delta, sigma,))
            v[i-1] = solution
        np.random.shuffle(v)
        return v
    else:
        print("Bimodal Distribution needs Regular Sampling")
    
def Uniform_distribution(min_v, max_v, N, RegularSampling, seed = None):
    """Sample a uniform distribution on ``[min_v, max_v]``."""
    np.random.seed(seed)
    if RegularSampling:
        x = np.linspace(min_v, max_v, N)
        np.random.shuffle(x)
        return x
    else:
        return np.random.uniform(min_v, max_v, N)

def PowerLaw_distribution(
    power_exp,
    N,
    RegularSampling,
    seed,
    MIN=1,
    MAX=0,
    esl=1e-2
):
    """Sample the symmetric shifted power-law distribution used in the model.

    When ``RegularSampling`` is true, inverse-CDF quantiles are generated on
    the positive and negative branches and then randomly permuted.
    """
    np.random.seed(seed)
    if RegularSampling:
        x = np.arange(1, int(N / 2) + 1)[::-1]
        y = np.arange(1, N - int(N / 2) + 1)[::-1]
        if MAX <= MIN:  # No Upper bound Case
            # MAX = int(np.power(N, 1 / (power_exp - 1))
            dist = np.concatenate(((MIN * (N / 2 / x) ** (1 / (power_exp - 1)) - MIN), -(MIN * (N / 2 / y) ** (1 / (power_exp - 1)) - MIN)))
            np.random.shuffle(dist)
            return dist
        else:  # with Upper bound Case
            mgp1 = -power_exp + 1
            km, kM = MIN**mgp1, MAX**mgp1
            dist = np.concatenate(((kM + (x/N*2)*(km - kM))**(1/mgp1) - MIN, -((kM + (y/N*2)*(km - kM))**(1/mgp1) - MIN)))
            np.random.shuffle(dist)
            return dist
    else:
        if MAX < 1:
            try:
                power_exp += esl
                MAX = int(np.power(N, 1 / (power_exp - 1)))
            except:
                MAX = int(N / 2)

        power_exp -= 1  # Because of {power_exp-1}
        power_exp *= -1
        x = np.random.random(size=int(N / 2))
        y = np.random.random(size=N-int(N / 2))
        mg, Mg = MIN**power_exp, MAX**power_exp
        dist = np.concatenate(((mg + (Mg - mg) * x) ** (1.0 / power_exp) - MIN, -((mg + (Mg - mg) * y) ** (1.0 / power_exp) - MIN)))
        np.random.shuffle(dist)
        return dist


# def PowerLaw_distribution(
#     power_exp,
#     N,
#     RegularSampling,
#     seed,
#     MIN=1,
#     MAX=0,
#     esl=1e-2
# ):
#     """
#     note
#     ----
#     Sampling from Power-law distribution propto x^{-(power_exp-1)} for m<=x<=M
#     with 2 < power_exp < 3.
#     If Regular Sampling, p(x) = ((power_exp-1)/x_min) * (x/x_min)^(-power_exp).
#     """
#     np.random.seed(seed)
#     if RegularSampling:
#         x = np.arange(1, N + 1)[::-1]
#         np.random.shuffle(x)
#         if MAX <= MIN:  # No Upper bound Case
#             # MAX = int(np.power(N, 1 / (power_exp - 1)))
#             return MIN * (N / x) ** (1 / (power_exp - 1))
#         else:  # with Upper bound Case
#             mgp1 = -power_exp + 1
#             km, kM = MIN**mgp1, MAX**mgp1
#             return (kM + (x/N)*(km - kM))**(1/mgp1)
#     else:
#         if MAX < 1:
#             try:
#                 power_exp += esl
#                 MAX = int(np.power(N, 1 / (power_exp - 1)))
#             except:
#                 MAX = N

#         power_exp -= 1  # Because of {power_exp-1}
#         power_exp *= -1
#         x = np.random.random(size=N)
#         mg, Mg = MIN**power_exp, MAX**power_exp
#         return (mg + (Mg - mg) * x) ** (1.0 / power_exp)
