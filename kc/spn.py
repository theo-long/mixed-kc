"""Sum-Product Network representation for continuous latents."""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from scipy.special import betaln
import numpy as np

from kc.gaussian_math import log_score_singular
from kc.types import GradedLikelihoodType, LikelihoodType


@dataclass
class ObservationWeights:
    likelihood: LikelihoodType
    gaussian_obs_coefficients: list[dict[int, float]] = field(default_factory=list)
    gaussian_obs_values: list[float] = field(default_factory=list)
    beta_counts: dict[int, tuple[int, int]] = field(default_factory=dict)
    truncated_gaussian_obs: int = 0

    @property
    def scope(self) -> set[int]:
        scope = set()
        for obs_vector in self.gaussian_obs_coefficients:
            scope |= obs_vector.keys()
        scope |= self.beta_counts.keys()
        return scope

    def __str__(self):
        return f"Obs(likelihood={self.likelihood}, beta_counts={self.beta_counts}, n_gaussian_obs={len(self.gaussian_obs_coefficients)}, trunc_obs={self.truncated_gaussian_obs}, scope={self.scope})"

    def __mul__(self, other: "ObservationWeights") -> "ObservationWeights":
        beta_counts = dict(self.beta_counts)
        for var, (other_true_count, other_false_count) in other.beta_counts.items():
            true_count, false_count = beta_counts.get(var, (0, 0))
            beta_counts[var] = (
                true_count + other_true_count,
                false_count + other_false_count,
            )
        return ObservationWeights(
            self.likelihood * other.likelihood,  # type: ignore
            self.gaussian_obs_coefficients + other.gaussian_obs_coefficients,
            self.gaussian_obs_values + other.gaussian_obs_values,
            beta_counts,
            self.truncated_gaussian_obs + other.truncated_gaussian_obs,
        )

    def __add__(self, other: "ObservationWeights"):
        if len(self.scope) + len(other.scope) == 0:
            # Prefer the one with fewer obs *if* it has likelihood > 0
            if (
                self.truncated_gaussian_obs < other.truncated_gaussian_obs
                and self.likelihood
            ):
                return self
            elif (
                other.truncated_gaussian_obs < self.truncated_gaussian_obs
                and other.likelihood
            ):
                return other
            else:
                return ObservationWeights(
                    self.likelihood + other.likelihood,  # type: ignore
                    truncated_gaussian_obs=self.truncated_gaussian_obs,
                )
        else:
            # TODO: technically we could if everything had *the same* set of observations (or equivalent set)
            raise ValueError("Cannot add weights with observations")


def _build_gaussian_observation_matrix(
    dim: int,
    scope_map: dict[int, int],
    gaussian_obs_coefficients: list[dict[int, float]],
    gaussian_obs_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((len(gaussian_obs_coefficients), dim))
    b = np.array(gaussian_obs_values)

    for i, vector in enumerate(gaussian_obs_coefficients):
        for gaussian_var, val in vector.items():
            j = scope_map[gaussian_var]
            A[i, j] = val

    return A, b


def _get_gaussian_observation_likelihood_update(observation: ObservationWeights):
    assert observation.gaussian_obs_values
    assert observation.gaussian_obs_coefficients
    scope_map = {s: i for i, s in enumerate(observation.scope)}
    dim = len(observation.scope)
    A, b = _build_gaussian_observation_matrix(
        dim,
        scope_map,
        observation.gaussian_obs_coefficients,
        observation.gaussian_obs_values,
    )
    log_score = log_score_singular(np.zeros((dim, 1)), np.eye(dim), A, b)
    if log_score is None:
        return None, 0
    return log_score, np.linalg.matrix_rank(A)


def _get_beta_observation_likelihood_update(
    observation: ObservationWeights, beta_priors: dict[int, tuple[float, float]]
) -> float:
    log_likelihood = 0
    for var, (s, f) in observation.beta_counts.items():
        alpha, beta = beta_priors[var]

        # Log of the Beta ratio (Probability of this specific sequence of flips)
        log_likelihood += betaln(s + alpha, f + beta) - betaln(alpha, beta)

    return log_likelihood


def _get_observation_likelihood(
    observation: ObservationWeights, beta_priors: dict[int, tuple[float, float]]
) -> GradedLikelihoodType | None:
    if observation.likelihood == 0:
        return None
    log_likelihood = np.log(observation.likelihood).item()  # type: ignore
    n_obs = observation.truncated_gaussian_obs
    if observation.gaussian_obs_coefficients:
        log_score, new_obs = _get_gaussian_observation_likelihood_update(observation)
        if log_score is None:
            return None
        log_likelihood += log_score
        n_obs += new_obs

    if observation.beta_counts:
        log_score = _get_beta_observation_likelihood_update(observation, beta_priors)
        log_likelihood += log_score

    return GradedLikelihoodType(log_likelihood, n_obs)


class Node(ABC):
    ADD_OPS: dict[
        tuple[type["Node"], type["Node"]],
        Callable,
    ] = {}
    MUL_OPS: dict[
        tuple[type["Node"], type["Node"]],
        Callable,
    ] = {}

    @property
    @abstractmethod
    def scope(self) -> set[int]: ...

    @abstractmethod
    def compute_log_likelihood(
        self, beta_priors: dict[int, tuple[float, float]]
    ) -> GradedLikelihoodType | None: ...

    @abstractmethod
    def _tree_str(self, prefix="", is_last=True) -> str: ...

    def __str__(self):
        return self._tree_str(prefix="", is_last=True).replace("└── ", "", 1).strip()

    def __add__(self, other) -> "Sum | WeightNode":
        registry_key = (type(self), type(other))
        func = self.ADD_OPS.get(registry_key)

        if func:
            return func(self, other)

        return NotImplemented

    def __radd__(self, other) -> "Sum | WeightNode":
        return self.__add__(other)

    def __mul__(self, other) -> "Product | WeightNode":
        registry_key = (type(self), type(other))
        func = self.MUL_OPS.get(registry_key)

        if func:
            return func(self, other)

        return NotImplemented

    def __rmul__(self, other) -> "Product | WeightNode":
        return self.__mul__(other)


class WeightNode(Node):
    def __init__(self, weight: ObservationWeights) -> None:
        self.weight = weight

    @property
    def scope(self):
        return self.weight.scope

    def compute_log_likelihood(
        self, beta_priors: dict[int, tuple[float, float]]
    ) -> GradedLikelihoodType | None:
        return _get_observation_likelihood(self.weight, beta_priors)

    def _tree_str(self, prefix="", is_last=True):
        res = ""
        connector = "└── " if is_last else "├── "
        node_str = f"Weight({self.weight})"
        res += f"{prefix}{connector}{node_str}\n"
        return res


class Product(Node):
    def __init__(self, *children: Node) -> None:
        # TODO: all merging logic should happen here so it is not duplicated in the add/mul methods
        # Merging logic:
        # - Weight nodes should be multiplied together i.e. only a single weight node should be present in the children
        self.children: list[Node] = list(children)

    @property
    def scope(self):
        return set.union(*[c.scope for c in self.children])

    def is_valid(self):
        for a, b in itertools.combinations(self.children, 2):
            if a.scope & b.scope:
                return False
        return True

    def compute_log_likelihood(self, beta_priors: dict[int, tuple[float, float]]):
        log_likelihood = GradedLikelihoodType(0.0, 0)
        for child in self.children:
            increment = child.compute_log_likelihood(beta_priors)
            if increment is None:
                # If anything is None i.e. p=0 in product, whole product is p=0
                return None
            log_likelihood = log_likelihood * increment
        return log_likelihood

    def _tree_str(self, prefix="", is_last=True):
        res = ""
        connector = "└── " if is_last else "├── "

        node_str = f"Product({self.scope})"
        res += f"{prefix}{connector}{node_str}\n"

        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child_is_last = i == len(self.children) - 1
            res += child._tree_str(new_prefix, child_is_last)
        return res


class Sum(Node):
    def __init__(self, *children: Node) -> None:
        # TODO: all merging logic should happen here so it is not duplicated in the add/mul methods
        self.children: list[Node] = list(children)

    @property
    def scope(self):
        if not self.children:
            return set()
        return set.union(*[c.scope for c in self.children])

    def compute_log_likelihood(self, beta_priors: dict[int, tuple[float, float]]):  # type: ignore
        log_likelihood = None
        for child in self.children:
            log_likelihood_update = child.compute_log_likelihood(beta_priors)
            # If update has p=0, we can just ignore it in sum
            if log_likelihood_update is not None:
                log_likelihood = log_likelihood + log_likelihood_update
        return log_likelihood

    def _tree_str(self, prefix="", is_last=True):
        res = ""
        connector = "└── " if is_last else "├── "

        node_str = f"Sum({self.scope})"
        res += f"{prefix}{connector}{node_str}\n"

        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child_is_last = i == len(self.children) - 1
            res += child._tree_str(new_prefix, child_is_last)
        return res


def _add_sum_sum(a: Sum, b: Sum):
    return Sum(*a.children, *b.children)


def _add_sum_product(a: Sum, b: Product):
    return Sum(*a.children, b)


def _add_sum_weight(a: Sum, b: WeightNode):
    new_children = []
    add_b = True
    for child in a.children:
        # TODO: we can also sum weights when the observations are *equivalent*
        if isinstance(child, WeightNode) and len(child.scope.union(b.scope)) == 0:
            new_children.append(child + b)
            add_b = False
        else:
            new_children.append(child)

    if add_b:
        new_children.append(b)
    return Sum(*new_children)


def _add_product_product(a: Product, b: Product):
    return Sum(a, b)


def _add_product_weight(a: Product, b: WeightNode):
    return Sum(a, b)


def _add_weight_weight(a: WeightNode, b: WeightNode):
    # Can only sum weights when there are no latent observations
    # TODO: we can also sum weights when the observations are *equivalent*
    if len(a.scope.union(b.scope)) == 0:
        return WeightNode(a.weight + b.weight)
    else:
        return Sum(a, b)


def _get_max_disjoint_matching(
    left_children: list[Node], right_children: list[Node]
) -> tuple[list[Node], list[Node]]:
    raise NotImplementedError()


def _recursive_mul_sum_sum(
    new_children: list[Node], left_children: list[Node], right_children: list[Node]
):
    raise NotImplementedError()


def _mul_sum_sum(a: Sum, b: Sum):
    if a.scope.isdisjoint(b.scope):
        return Product(*a.children, *b.children)
    else:
        # What we really want here is to find the largest possible subsets of a and b that are disjoint
        # that way instead of distributing the whole sum, we can just compute (a + b)*(c + b)
        # The algorithm for this is iterative bipartite matching
        # First find largest disjoint subsets a', b', add the product a' * b' to children
        # Then find the largest disjoint subsets of a', b_rem and a_rem, b' recursively until we are 'done' with a', b'
        # Then repeat for a_rem, b_rem
        raise NotImplementedError()


def _mul_sum_product(a: Sum, b: Product):
    if a.scope.isdisjoint(b.scope):
        return Product(*a.children, b)
    else:
        disjoint_children = []
        overlapping_children = []
        for child in a.children:
            if child.scope.isdisjoint(b.scope):
                disjoint_children.append(child)
            else:
                overlapping_children.append(child * b)
        return Sum(Product(b, Sum(*disjoint_children)), *overlapping_children)


def _mul_product_product(a: Product, b: Product):
    if a.scope.isdisjoint(b.scope):
        return Product(*a.children, *b.children)
    else:
        raise NotImplementedError


def _mul_sum_weight(a: Sum, b: WeightNode):
    if b.weight.likelihood == 0:
        return WeightNode(ObservationWeights(0.0))

    if a.scope.isdisjoint(b.scope):
        return Product(a, b)
    else:
        disjoint_children = []
        overlapping_children = []
        for child in a.children:
            if child.scope.isdisjoint(b.scope):
                disjoint_children.append(child)
            else:
                overlapping_children.append(child * b)
        return Sum(Product(b, Sum(*disjoint_children)), *overlapping_children)


def _mul_product_weight(a: Product, b: WeightNode):
    if b.weight.likelihood == 0:
        return WeightNode(ObservationWeights(0.0))
    if a.scope.isdisjoint(b.scope):
        return Product(a, b)
    new_children = []
    for child in a.children:
        if a.scope.isdisjoint(b.scope):
            new_children.append(child)
        else:
            new_children.append(child * b)
    return Product(*new_children)


def _mul_weight_weight(a: WeightNode, b: WeightNode):
    if b.weight.likelihood == 0:
        return WeightNode(ObservationWeights(0.0))
    return WeightNode(a.weight * b.weight)


Node.ADD_OPS = {
    (Sum, Sum): _add_sum_sum,
    (Sum, Product): _add_sum_product,
    (Sum, WeightNode): _add_sum_weight,
    (Product, Product): _add_product_product,
    (Product, WeightNode): _add_product_weight,
    (WeightNode, WeightNode): _add_weight_weight,
}

# Although we might be able to define reasonable mul ops for other combinations
# Only Weight * Node shows up in WMC
# Should think about what the other cases might represent
Node.MUL_OPS = {
    (Sum, WeightNode): _mul_sum_weight,
    (Product, WeightNode): _mul_product_weight,
    (WeightNode, WeightNode): _mul_weight_weight,
}
