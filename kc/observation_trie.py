from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from kc.types import LikelihoodType


@dataclass
class LikelihoodNode:
    likelihood: LikelihoodType


@dataclass
class ObservationNode:
    """We maintain a QR decomposition of our observation matrix V = (A, b) which represents Ax = b"""

    Q: NDArray
    R: NDArray
    children: set["ObservationNode | LikelihoodNode"] = field(default_factory=set)

    def add_obs(self, observation_vector: NDArray):
        raise NotImplementedError()


class IncrementalSystem:
    def __init__(self, n_features, tol=1e-9):
        self.n = n_features
        self.tol = tol
        # Store QR of A.T (because we add rows to A, which are columns of A.T)
        self.Q = np.zeros((self.n, 0))
        self.R = np.zeros((0, 0))
        self.b = np.array([])

    def process_equation(self, v, c):
        v = np.array(v, dtype=float)

        # --- STEP 1: Fast Check (Projection) ---
        # Project v onto current basis Q. z = Q.T @ v
        if self.Q.shape[1] > 0:
            z = self.Q.T @ v
            v_projected = self.Q @ z
            v_res = v - v_projected
            norm_res = np.linalg.norm(v_res)
        else:
            z = np.array([])
            norm_res = np.linalg.norm(v)

        # --- STEP 2: Branch Logic ---

        # CASE A: Dependent (Redundant or Incompatible)
        if norm_res < self.tol:
            if self.R.shape[0] == 0:
                return "Incompatible" if abs(c) > self.tol else "Redundant"

            # Solve R * w = z to get weights
            w = scipy.linalg.solve_triangular(self.R, z, lower=False)
            c_predicted = w @ self.b

            if abs(c - c_predicted) > self.tol:
                return f"Incompatible (Contradiction: {c_predicted:.2f} != {c})"
            return "Redundant"

        # CASE B: Independent -> Use Scipy to Update
        else:
            # We are adding a row to A, which is a COLUMN to A.T
            # We insert vector 'v' at the end (index k=m) of the decomposition
            # qr_insert(Q, R, u, k, which='col')

            # Note: qr_insert handles the Gram-Schmidt / Householder logic internally
            self.Q, self.R = scipy.linalg.qr_insert(
                self.Q, self.R, v, self.Q.shape[1], which="col"
            )

            self.b = np.append(self.b, c)
            return "Accepted (New Independent Row)"
