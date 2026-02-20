from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from kc.config import settings
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
        # B1: Empty QR
        elif self.Q.shape[1] == 0:
            self.Q, self.R = scipy.linalg.qr(v.reshape(-1, 1), mode="economic")
            self.b = np.array([c])
            return UpdateResult.UPDATE
        # B2: merge into existing QR
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

    @property
    def A(self):
        return (self.Q @ self.R).T


@dataclass(frozen=True)
class LikelihoodNode:
    likelihood: LikelihoodType

    def __mul__(self, other: WeightType):
        if isinstance(other, LikelihoodType):
            return LikelihoodNode(self.likelihood * other)
        elif self.likelihood == 0:
            return LikelihoodNode(0.0)
        else:
            node = ObservationNode(
                IncrementalSystem(other.shape[0] - 1),
                children=[LikelihoodNode(self.likelihood)],
            )
            node.observations.process_equation(other[1:], other[0])
            return node

    # TODO - refactor so that Obs and Likelihood Nodes share the same __add__ method
    def __add__(
        self, other: "ObservationNode | LikelihoodNode"
    ) -> "ObservationNode | LikelihoodNode":
        if isinstance(other, LikelihoodNode):
            return LikelihoodNode(self.likelihood + other.likelihood)
        elif self.likelihood == 0:
            return other
        else:
            children = []
            likelihood_value = self.likelihood
            if other.num_obs == 0:
                for child in other.children:
                    if isinstance(child, LikelihoodNode):
                        likelihood_value += child.likelihood
                    else:
                        children.append(child)
            else:
                children = [other]
            children.append(LikelihoodNode(self.likelihood))
            if len(children) == 1:
                return children[0]
            return ObservationNode(
                IncrementalSystem(other.observations.n), children=children
            )

    def _recursive_compute_posterior(
        self, posterior_mixture: list[tuple[float, NDArray, NDArray]]
    ):
        new_posterior_mixture = []
        for i in range(len(posterior_mixture)):
            likelihood, mu, cov = posterior_mixture[i]
            new_posterior_mixture.append((likelihood * self.likelihood, mu, cov))  # type: ignore

        return new_posterior_mixture

    def __str__(self):
        return f"LikelihoodNode(val={self.likelihood})"

    def compute_posterior(self):
        return [(self.likelihood, np.zeros((0, 0)), np.eye(0))]

    @property
    def num_obs_to_root(self):
        return 0


@dataclass
class ObservationNode:
    """We maintain a QR decomposition of our observation matrix V = (A, b) which represents Ax = b"""

    observations: IncrementalSystem
    children: list["ObservationNode | LikelihoodNode"] = field(default_factory=list)

    @property
    def num_obs(self):
        return len(self.observations.b)

    @property
    def num_obs_to_root(self):
        self.num_obs + max(map(lambda x: x.num_obs_to_root, self.children))

    def __add__(
        self, other: "ObservationNode | LikelihoodNode"
    ) -> "ObservationNode | LikelihoodNode":
        if isinstance(other, ObservationNode):
            children = []
            # TODO - might you be able to merge the two obs nodes if they share observations?
            for node in [self, other]:
                if node.num_obs == 0:
                    children.extend(node.children)
                else:
                    children.append(node)
            return ObservationNode(
                IncrementalSystem(self.observations.n), children=children
            )
        elif other.likelihood == 0:
            return self
        else:
            return ObservationNode(
                IncrementalSystem(self.observations.n),
                children=[self, LikelihoodNode(other.likelihood)],
            )

    def _collect_qr_update_recursive(self, qr: IncrementalSystem):
        """Recursively check if some sequence of observations is still valid."""
        result = qr.merge(self.observations)
        if result == UpdateResult.INCOMPATIBLE:
            return LikelihoodNode(0.0)

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

        return LikelihoodNode(0.0)

    def __mul__(self, other: WeightType) -> "ObservationNode | LikelihoodNode":
        if isinstance(other, LikelihoodType):
            if other == 0:
                return LikelihoodNode(0.0)
            new_children = []
            for child in self.children:
                child *= other
                if child:
                    new_children.append(deepcopy(child))
            self.children = new_children
            return self

        result = self.observations.process_equation(other[1:], other[0])
        if result == UpdateResult.INCOMPATIBLE:
            return LikelihoodNode(0.0)

        new_children = []
        for i in range(len(self.children)):
            child = self.children[i]
            if isinstance(child, ObservationNode):
                child = child._collect_qr_update_recursive(deepcopy(self.observations))
            if isinstance(child, LikelihoodNode) and child.likelihood == 0:
                continue
            new_children.append(deepcopy(child))

        self.children = new_children
        # If there is a single observation node child, collapse them together
        if len(self.children) == 1 and not isinstance(self.children[0], LikelihoodNode):
            merged_node = self.children[0]
            merged_node.observations.merge(self.observations)
            return merged_node

        if self.children:
            return self
        return LikelihoodNode(0.0)

    def __rmul__(self, other: WeightType) -> "ObservationNode | LikelihoodNode":
        return self.__mul__(other)

    def _apply_observations(self, posterior_mixture):
        if self.observations.R.size:
            new_posterior_mixture = []
            for likelihood, mu, cov in posterior_mixture:
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
                if settings.debug:
                    print(
                        "Updated posterior",
                        (mu, cov),
                        "to\n",
                        (mu_new, cov_new),
                        f"with log score {log_score: .4f}",
                        "and epsilon factor",
                        self.observations.Q.shape[1],
                    )
                new_posterior_mixture.append((likelihood, mu_new, cov_new))
            return new_posterior_mixture
        else:
            if settings.debug:
                print("No observations in this node, skipping update")
            return posterior_mixture

    def _recursive_compute_posterior(
        self, posterior_mixture: list[tuple[float, NDArray, NDArray]]
    ):
        posterior_mixture = self._apply_observations(posterior_mixture)
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

    def _tree_str(self, prefix="", is_last=True):
        res = ""
        connector = "└── " if is_last else "├── "

        node_str = f"ObservationNode(n_eqs={self.num_obs})"
        res += f"{prefix}{connector}{node_str}\n"

        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child_is_last = i == len(self.children) - 1
            if hasattr(child, "_tree_str"):
                res += child._tree_str(new_prefix, child_is_last)
            else:  # LikelihoodNode
                child_connector = "└── " if child_is_last else "├── "
                res += f"{new_prefix}{child_connector}{child}\n"
        return res

    def __str__(self):
        return self._tree_str(prefix="", is_last=True).replace("└── ", "", 1).strip()
