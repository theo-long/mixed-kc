from typing import TYPE_CHECKING, Collection

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import multivariate_normal

from kc.types import LikelihoodType, epsilon

if TYPE_CHECKING:
    from kc.real_values import GaussianVariable


def update_covariance(cov: NDArray, v: NDArray):
    """Update covariance matrix cov for Gaussian r.v. x based on observing v^T x = b."""
    return cov - (cov @ v @ v.T @ cov) / (v.T @ cov @ v)


def update_mean(mu: NDArray, cov: NDArray, v: NDArray, b: ArrayLike):
    """Update mean vector mu for Gaussian r.v. x based on observing Ax = b."""
    return mu + (b - v.T @ mu) * (cov @ v) / (v.T @ cov @ v)


def score_observation(mu: NDArray, cov: NDArray, v: NDArray, b: ArrayLike):
    """Compute the likelihood of observing v^T x = b for Gaussian r.v. x."""
    mu_obs = v.T @ mu
    cov_obs = v.T @ cov @ v
    # Handle singular covariance case
    if np.allclose(cov_obs, 0):
        return np.isclose(mu_obs, b) * 1.0

    # Multiply by epsilon since we are observing measure 0 event
    return multivariate_normal.pdf(b, mean=mu_obs, cov=cov_obs) * epsilon  # type: ignore


def create_observation_vector(
    vars: Collection["GaussianVariable"], val: float, gaussian_dim: int
):
    """Create a numpy array representing observation v^T x = val"""
    v = np.zeros((1, gaussian_dim + 1))
    for var in vars:
        v[var.var] = var.scale
    v[-1] = val
    return v
