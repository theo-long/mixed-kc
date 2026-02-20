import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())


from kc.gaussian_math import get_gaussian_posterior, log_score_singular


def verify():
    np.random.seed(42)
    n = 5  # State dimension
    m = 3  # Observation dimension

    mu = np.random.randn(n, 1)
    # Make random PSD covariance
    C = np.random.randn(n, n)
    cov = C @ C.T

    A = np.random.randn(m, n)
    b = np.random.randn(m, 1)

    # Perform QR decomposition
    Q, R = np.linalg.qr(A.T)

    # 1. Verify Posterior Update
    mu1, cov1 = get_gaussian_posterior(mu, cov, A, b)
    mu2, cov2 = get_gaussian_posterior(mu, cov, A, b, Q=Q, R=R)

    print("Posterior Diff Mean:", np.linalg.norm(mu1 - mu2))
    print("Posterior Diff Cov:", np.linalg.norm(cov1 - cov2))

    assert np.allclose(mu1, mu2)
    assert np.allclose(cov1, cov2)
    print("Posterior Verification Passed for m < n")

    # 2. Verify Log Score
    ls1 = log_score_singular(mu, cov, A, b)
    ls2 = log_score_singular(mu, cov, A, b, Q=Q, R=R)
    print("LogScore Diff:", abs(ls1 - ls2))
    assert np.isclose(ls1, ls2)
    print("LogScore Verification Passed for m < n")

    # Case m > n (Overconstrained)
    m_over = 6
    A_over = np.random.randn(m_over, n)
    b_over = np.random.randn(m_over, 1)
    Q_over, R_over = np.linalg.qr(A_over.T)

    mu1_o, cov1_o = get_gaussian_posterior(mu, cov, A_over, b_over)
    mu2_o, cov2_o = get_gaussian_posterior(mu, cov, A_over, b_over, Q=Q_over, R=R_over)

    ls1_o = log_score_singular(mu, cov, A_over, b_over)
    ls2_o = log_score_singular(mu, cov, A_over, b_over, Q=Q_over, R=R_over)
    print("LogScore Diff (Over):", abs(ls1_o - ls2_o))
    assert np.isclose(ls1_o, ls2_o)

    print("Posterior Diff Mean (Over):", np.linalg.norm(mu1_o - mu2_o))
    print("Posterior Diff Cov (Over):", np.linalg.norm(cov1_o - cov2_o))

    if np.allclose(mu1_o, mu2_o) and np.allclose(cov1_o, cov2_o):
        print("Posterior Verification Passed for m > n")


if __name__ == "__main__":
    verify()
