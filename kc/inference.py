import dd.autoref as _bdd
import numpy as np
from numpy.typing import NDArray

from kc.base import PExpr
from kc.config import settings
from kc.model_count import model_count
from kc.real_values import GaussianSum, Zero
from kc.state import KCState, TruncationState


def get_gaussian_posterior(mu: NDArray, cov: NDArray, A: NDArray, b: NDArray):
    # 1. Calculate the Innovation Covariance (S)
    # S = A @ cov @ A.T
    S = A @ cov @ A.T

    # 2. Calculate the Kalman Gain (K)
    # K = cov @ A.T @ inv(S)
    # Note: np.linalg.solve is more numerically stable than np.linalg.inv
    K = cov @ A.T @ np.linalg.pinv(S)

    # 3. Update the Mean (mu_new)
    # mu_new = mu + K @ (b - A @ mu)
    mu_new = mu + K @ (b - A @ mu)

    # 4. Update the Covariance (cov_new)
    # cov_new = (I - K @ A) @ cov
    cov_new = (np.eye(cov.shape[0]) - K @ A) @ cov
    return mu_new, cov_new


def get_expr_distribution(mu, cov, v, b):
    y_mean = v.T @ mu + b
    y_cov = v.T @ cov @ v
    return (y_mean, y_cov)


def log_score_singular(mu, cov, A, b, tol=1e-12):
    # Calculate innovation mean and covariance
    y_mean = A @ mu
    S = A @ cov @ A.T
    residual = b - y_mean

    # Use SVD for rank, pseudo-determinant, and pseudo-inverse
    # S = U @ diag(eigenvals) @ Vh
    U, s, Vh = np.linalg.svd(S)

    # Identify non-zero eigenvalues based on a tolerance
    non_zero = s > tol
    s_nonzero = s[non_zero]
    k = np.sum(non_zero)  # The rank

    # 1. Check for consistency: Is the residual in the span of S?
    # Project residual onto the null space of S
    null_space_proj = residual - (U[:, non_zero] @ (U[:, non_zero].T @ residual))
    if np.any(np.abs(null_space_proj) > tol):
        return -np.inf  # Observation is impossible under the model

    # 2. Compute Squared Mahalanobis Distance using SVD components
    # (b-Am).T @ S+ @ (b-Am)
    # Equivalent to: sum( (U.T @ residual)**2 / s_nonzero )
    res_transformed = U[:, non_zero].T @ residual
    mahalanobis_sq = np.sum((res_transformed**2) / s_nonzero)

    # 3. Compute Log Pseudo-determinant
    log_pseudo_det = np.sum(np.log(s_nonzero))

    # 4. Final Log-Score
    log_score = -0.5 * (k * np.log(2 * np.pi) + log_pseudo_det + mahalanobis_sq)

    return log_score


def run_kc(expr: PExpr):
    trunc_state = TruncationState()
    expr.collect_real_truncation({}, trunc_state)
    state = KCState(trunc_state)
    val = expr.kc({}, state)
    if settings.debug:
        print(f"BDD vars: {state.bdd.vars}")
        if isinstance(val, _bdd.Function):
            print(f"Result expr: {val.to_expr()}")
        else:
            print(f"Result expr: {val}")
        print(f"Observes expr: {state.observes_all_hold.to_expr()}")

    posterior_update_mixture = model_count(
        state.bdd,
        state.observes_all_hold,
        state.gaussian_count,
        state.weights,
        state.priors,
    )
    normalizing_constant = 0.0
    mu, cov = (
        np.zeros((state.gaussian_count, 1)),
        np.eye(state.gaussian_count),
    )
    posterior_mixture: list[tuple[float, tuple[NDArray, NDArray]]] = []
    for likelihood, posterior_updates in posterior_update_mixture:
        A, b = posterior_updates[:, 1:], posterior_updates[:, :1]
        log_gaussian_score = log_score_singular(mu, cov, A, b)
        normalizing_constant += likelihood * np.exp(log_gaussian_score)
        mu_posterior, cov_posterior = get_gaussian_posterior(mu, cov, A, b)
        posterior_mixture.append((likelihood, (mu_posterior, cov_posterior)))

    if normalizing_constant == 0:
        return None, normalizing_constant

    # Inference for binary variable
    if isinstance(val, _bdd.Function):
        posterior_mixture_with_val = model_count(
            state.bdd,
            val & state.observes_all_hold,
            state.gaussian_count,
            state.weights,
            state.priors,
        )
        unnormalized_prob = 0.0
        mu, cov = (
            np.zeros((state.gaussian_count, 1)),
            np.eye(state.gaussian_count),
        )
        for likelihood, posterior_updates in posterior_mixture_with_val:
            A, b = posterior_updates[:, 1:], posterior_updates[:, :1]
            log_gaussian_score = log_score_singular(mu, cov, A, b)
            unnormalized_prob += likelihood * np.exp(log_gaussian_score)

        return (unnormalized_prob / normalizing_constant, normalizing_constant)
    elif isinstance(val, GaussianSum):
        normalized_posterior = []
        for weight, (mu, cov) in posterior_mixture:
            v = np.zeros((1, state.gaussian_count))
            b = 0.0
            for var in val.rvs:
                assert not isinstance(var, Zero)
                v[0, var.var - 1] = var.scale
                b += var.shift
            expr_mu, expr_cov = get_expr_distribution(mu, cov, v, b)
            normalized_posterior.append(
                (weight / normalizing_constant, (expr_mu, expr_cov))
            )
        return normalized_posterior, normalizing_constant
    else:
        raise TypeError(f"Cannot perform inference for value of type {type(val)}")
