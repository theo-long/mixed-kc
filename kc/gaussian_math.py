import numpy as np
from numpy.typing import NDArray


def get_gaussian_posterior(
    mu: NDArray,
    cov: NDArray,
    A: NDArray | None = None,
    b: NDArray | None = None,
    Q: NDArray | None = None,
    R: NDArray | None = None,
):
    if Q is not None and R is not None:
        import scipy.linalg

        # Optimization: Use QR decomposition of A.T = Q @ R
        # This implies A = R.T @ Q.T

        # 1. Compute Innovation
        # We want nu = R^{-T} b - Q.T mu
        # Solve R.T @ x = b for x
        # R is upper triangular, so R.T is lower triangular.
        try:
            # We assume R is square (k x k) and full rank for improved efficiency
            # If R is not square, we might need a different approach, but standard QR gives sq R
            # for economic update.
            # b should be shape (k,) or (k, 1)
            sol_b = scipy.linalg.solve_triangular(R, b, trans="T")
        except (scipy.linalg.LinAlgError, ValueError):
            # Fallback if R is singular? Or just proceed with pinv.
            # R should be full rank from incremental system.
            # But let's use lstsq or pinv if needed, though solve_triangular is faster.
            # For robustness sake in generic func, maybe pinv?
            # But let's stick to solve_triangular as primary path.
            sol_b = np.linalg.lstsq(R.T, b, rcond=None)[0]

        qt_mu = Q.T @ mu
        innovation = sol_b - qt_mu

        # 2. Compute Omega = Q.T @ cov @ Q
        # This is strictly smaller than full S if k < n
        # S = R.T @ Omega @ R
        qv = Q.T @ cov
        Omega = qv @ Q

        # 3. Compute Inverse of Omega
        # Omega is k x k.
        inv_Omega = np.linalg.pinv(Omega)

        # 4. Update Gain terms
        # K = cov @ Q @ inv_Omega @ R^{-T}
        # delta_mu = K @ (b - A mu)
        #          = cov @ Q @ inv_Omega @ (R^{-T} b - Q.T mu)
        #          = cov @ Q @ inv_Omega @ innovation_projected
        # Let Z = inv_Omega @ innovation
        Z = inv_Omega @ innovation

        # mu_update = cov @ Q @ Z
        # We calculate (cov @ Q) earlier as qv.T
        mu_update = qv.T @ Z
        mu_new = mu + mu_update

        # 5. Update Covariance
        # cov_new = cov - K @ S @ K.T
        #         = cov - (cov Q invOmega R^{-T}) (R.T Omega R) (R^{-1} invOmega Q.T cov)
        #         = cov - cov Q invOmega Q.T cov
        # Let H = cov @ Q
        # cov_new = cov - H @ inv_Omega @ H.T
        term = qv.T @ inv_Omega @ qv
        cov_new = cov - term

        return mu_new, cov_new

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
    y_mean = v @ mu + b
    y_cov = v @ cov @ v.T
    return (y_mean, y_cov)


def log_score_singular(mu, cov, A=None, b=None, Q=None, R=None, tol=1e-12):
    if Q is not None and R is not None:
        import scipy.linalg

        # Optimization using QR of A.T
        # A.T = Q @ R  => A = R.T @ Q.T
        # b - A@mu = b - R.T @ Q.T @ mu

        # 1. Project to "innovation space"
        # We want to evaluate (b - A mu).T @ S^{-1} @ (b - A mu)
        # S = R.T @ Omega @ R
        # S^{-1} = R^{-1} @ Omega^{-1} @ R^{-T}
        # term = (b - R.T Q.T mu).T @ R^{-1} Omega^{-1} R^{-T} @ (b - R.T Q.T mu)
        #      = (R^{-T} b - Q.T mu).T @ Omega^{-1} @ (R^{-T} b - Q.T mu)
        # Let v = R^{-T} b - Q.T mu

        try:
            # b should be shape (k,) or (k, 1)
            sol_b = scipy.linalg.solve_triangular(R, b, trans="T")
        except (scipy.linalg.LinAlgError, ValueError):
            # Fallback if R is singular or rectangular (m > k)
            # R is (k, m). R.T is (m, k).
            # b is (m, 1) or (m,).
            result = np.linalg.lstsq(R.T, b, rcond=None)
            sol_b = result[0]
            residuals = result[1]
            if residuals.size > 0 and residuals[0] > tol:
                return -float("inf")

        qt_mu = Q.T @ mu
        v = sol_b - qt_mu

        # 2. Compute Omega = Q.T @ cov @ Q
        Omega = Q.T @ cov @ Q

        # 3. SVD of Omega for pseudo-inverse and determinant
        U, s, Vh = np.linalg.svd(Omega)

        # 4. Filter eigenvalues
        non_zero = s > tol
        s_nonzero = s[non_zero]
        k_rank = np.sum(non_zero)  # Rank of Omega

        # 5. Check consistency
        # v must be in span of Omega
        # Project v onto null space of Omega
        U_nz = U[:, non_zero]
        v_transformed = U_nz.T @ v
        v_proj = U_nz @ v_transformed

        # Note: If R is full rank, any inconsistency comes from Omega rank deficiency
        # However, we also need to check consistency in the null space of A if m > n?
        # Actually, if we are in the Q-space projection, we are fine.
        # But wait, if R was rectangular (m > k), that handles the row-space.
        # Here we assume A.T = Q R, so Range(A.T) = Range(Q).
        # Null(A) is orthogonal to Range(A.T).
        # Actually this formula is consistent for the "active" constraints captured by Q, R.
        # If there were constraints NOT in Q, R, they'd be ignored here.
        # Assuming Q, R captures the relevant system.

        if np.linalg.norm(v - v_proj) > tol:
            return -float("inf")

        # 6. Mahalanobis
        # val = v.T @ Omega+ @ v
        mahalanobis_sq = np.sum((v_transformed.flatten() ** 2) / s_nonzero)

        # 7. Log Pseudo-det
        # log|S| = log|R.T Omega R| = log|Omega| + 2 log|R|
        log_pseudo_det_omega = np.sum(np.log(s_nonzero))
        # Sum of log of diagonal elements of R
        # R might be shaped (k, k)
        log_det_R = np.sum(np.log(np.abs(np.diag(R))))
        log_pseudo_det = log_pseudo_det_omega + 2 * log_det_R

        # 8. Final Log Score
        return -0.5 * (k_rank * np.log(2 * np.pi) + log_pseudo_det + mahalanobis_sq)

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
    k_rank = np.sum(non_zero)  # The rank

    # 1. Check for consistency: Is the residual in the span of S?
    # Project residual onto the null space of S
    null_space_proj = residual - (U[:, non_zero] @ (U[:, non_zero].T @ residual))
    if np.any(np.abs(null_space_proj) > tol):
        return -np.inf  # Observation is impossible under the model

    # 2. Compute Squared Mahalanobis Distance using SVD components
    # (b-Am).T @ S+ @ (b-Am)
    # Equivalent to: sum( (U.T @ residual)**2 / s_nonzero )
    res_transformed = U[:, non_zero].T @ residual
    mahalanobis_sq = np.sum((res_transformed.flatten() ** 2) / s_nonzero)

    # 3. Compute Log Pseudo-determinant
    log_pseudo_det = np.sum(np.log(s_nonzero))

    # 4. Final Log-Score
    log_score = -0.5 * (k_rank * np.log(2 * np.pi) + log_pseudo_det + mahalanobis_sq)

    return log_score
