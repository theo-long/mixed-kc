from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from kc.gaussian_math import get_gaussian_posterior, log_score_singular
from kc.types import LikelihoodType, WeightType, epsilon


class UpdateResult(Enum):
    REDUNDANT = 0
    UPDATE = 1
    INCOMPATIBLE = -1


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

        # CASE A: Dependent (Redundant or Incompatible)
        if norm_res < self.tol:
            if self.R.shape[0] == 0:
                return (
                    UpdateResult.INCOMPATIBLE
                    if abs(c) > self.tol
                    else UpdateResult.REDUNDANT
                )

            # Solve R * w = z to get weights
            w = scipy.linalg.solve_triangular(self.R, z, lower=False)
            c_predicted = w @ self.b

            if abs(c - c_predicted) > self.tol:
                return UpdateResult.INCOMPATIBLE
            return UpdateResult.REDUNDANT

        # CASE B: Independent -> Use Scipy to Update
        else:
            # We are adding a row to A, which is a COLUMN to A.T
            # We insert vector 'v' at the end (index k=m) of the decomposition
            # qr_insert(Q, R, u, k, which='col')

            # Note: qr_insert handles the Gram-Schmidt / Householder logic internally
            self.Q, self.R = scipy.linalg.qr_insert(  # type:ignore
                self.Q, self.R, v, self.Q.shape[1], which="col"
            )

            self.b = np.append(self.b, c)
            return UpdateResult.UPDATE

    def merge(self, other: "IncrementalSystem"):
        final_result = UpdateResult.REDUNDANT
        A = (other.Q @ other.R).T
        for v, c in zip(A, other.b):
            result = self.process_equation(v, c)
            if result == UpdateResult.INCOMPATIBLE:
                return result
            if result == UpdateResult.UPDATE:
                final_result = result
        return final_result


@dataclass
class LikelihoodNode:
    likelihood: LikelihoodType

    def __mul__(self, other: LikelihoodType):
        self.likelihood *= other  # type: ignore
        return self

    def _recursive_compute_posterior(
        self, posterior_mixture: list[tuple[float, NDArray, NDArray]]
    ):
        new_posterior_mixture = []
        for i in range(len(posterior_mixture)):
            likelihood, mu, cov = posterior_mixture[i]
            new_posterior_mixture.append((likelihood * self.likelihood, mu, cov))  # type: ignore

        return new_posterior_mixture


@dataclass
class ObservationNode:
    """We maintain a QR decomposition of our observation matrix V = (A, b) which represents Ax = b"""

    observations: IncrementalSystem
    children: list["ObservationNode | LikelihoodNode"] = field(default_factory=list)

    def add_obs(self, observation_vector: NDArray):
        raise NotImplementedError()

    def __add__(self, other: "ObservationNode") -> "ObservationNode":
        if not isinstance(other, ObservationNode):
            raise TypeError

        return ObservationNode(
            IncrementalSystem(self.observations.n), children=[self, other]
        )

    def _collect_qr_update_recursive(self, qr: IncrementalSystem):
        """Recursively check if some sequence of observations is still valid."""
        result = qr.merge(self.observations)
        if result == UpdateResult.INCOMPATIBLE:
            return None

        new_children = []
        for i in range(len(self.children)):
            child = self.children[i]
            if isinstance(child, ObservationNode):
                child = child._collect_qr_update_recursive(deepcopy(qr))
            if child:
                new_children.append(child)

        self.children = new_children
        # If there is a single observation node child, collapse them together
        if len(self.children) == 1 and not isinstance(self.children[0], LikelihoodNode):
            merged_node = self.children[0]
            merged_node.observations.merge(self.observations)
            return merged_node

        if self.children:
            return self

        return None

    def __mul__(self, other: WeightType) -> "ObservationNode | LikelihoodNode | None":
        if isinstance(other, LikelihoodType):
            new_children = []
            for child in self.children:
                child *= other
                if child:
                    new_children.append(child)
            self.children = new_children
            return self

        result = self.observations.process_equation(other[1:], other[0])
        if result == UpdateResult.INCOMPATIBLE:
            return None

        new_children = []
        for i in range(len(self.children)):
            child = self.children[i]
            if isinstance(child, ObservationNode):
                child = child._collect_qr_update_recursive(deepcopy(self.observations))
            if child:
                new_children.append(child)

        self.children = new_children
        # If there is a single observation node child, collapse them together
        if len(self.children) == 1 and not isinstance(self.children[0], LikelihoodNode):
            merged_node = self.children[0]
            merged_node.observations.merge(self.observations)
            return merged_node

        if self.children:
            return self
        return None

    def __rmul__(self, other: WeightType) -> "ObservationNode | LikelihoodNode | None":
        return self.__mul__(other)

    def _recursive_compute_posterior(
        self, posterior_mixture: list[tuple[float, NDArray, NDArray]]
    ):
        # If no observations in this node, no need to update
        if self.observations.R.size:
            for i in range(len(posterior_mixture)):
                likelihood, mu, cov = posterior_mixture[i]
                log_score = log_score_singular(
                    mu,
                    cov,
                    Q=self.observations.Q,
                    R=self.observations.R,
                    b=self.observations.b,
                )
                likelihood *= np.exp(log_score) * (
                    epsilon ** self.observations.Q.shape[1]
                )
                mu_new, cov_new = get_gaussian_posterior(
                    mu,
                    cov,
                    Q=self.observations.Q,
                    R=self.observations.R,
                    b=self.observations.b,
                )
                posterior_mixture[i] = (likelihood, mu_new, cov_new)

        new_posterior_mixture = []
        for child in self.children:
            new_posterior_mixture.extend(
                child._recursive_compute_posterior(posterior_mixture)
            )
        return new_posterior_mixture

    def compute_posterior(self):
        cov = np.eye(self.observations.n)
        mu = np.zeros((self.observations.n, 1))
        posterior_mixture = [(1.0, mu, cov)]
        return self._recursive_compute_posterior(posterior_mixture)
