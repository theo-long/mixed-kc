"""Sum-Product Network representation for continuous latents."""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import logsumexp

from kc.gaussian_math import get_gaussian_posterior, log_score_singular
from kc.types import LikelihoodType, GradedLikelihoodType


@dataclass
class ObservationWeights:
    likelihood: LikelihoodType
    gaussian_obs_A: np.ndarray | None = None
    gaussian_obs_b: np.ndarray | None = None

    @property
    def scope(self) -> set[int]:
        if self.gaussian_obs_A is None:
            return set()
        return set(self.gaussian_obs_A.nonzero()[1].astype(int))

    def __mul__(self, other: "ObservationWeights") -> "ObservationWeights":
        if self.gaussian_obs_A and other.gaussian_obs_A:
            assert self.gaussian_obs_b is not None and other.gaussian_obs_b is not None
            gaussian_obs_A = np.stack(
                [self.gaussian_obs_A, other.gaussian_obs_A], axis=-1
            )
            gaussian_obs_b = np.concatenate([self.gaussian_obs_b, other.gaussian_obs_b])
        elif self.gaussian_obs_A:
            gaussian_obs_A, gaussian_obs_b = self.gaussian_obs_A, self.gaussian_obs_b
        else:
            gaussian_obs_A, gaussian_obs_b = other.gaussian_obs_A, other.gaussian_obs_b

        return ObservationWeights(
            self.likelihood * other.likelihood,  # type: ignore
            gaussian_obs_A,
            gaussian_obs_b,
        )

    def __add__(self, other: "ObservationWeights"):
        if len(self.scope) + len(other.scope) == 0:
            return ObservationWeights(self.likelihood + other.likelihood)  # type: ignore
        else:
            raise ValueError("Cannot add weights with observations")


@dataclass(frozen=True)
class LatentState:
    cov: np.ndarray
    mu: np.ndarray
    log_likelihood: LikelihoodType

    @classmethod
    def initial_state(cls, n: int) -> "LatentState":
        return cls(cov=np.eye(n), mu=np.zeros((n, 1)), log_likelihood=1.0)


def _update_latent_with_observation(
    observation: ObservationWeights, state: LatentState
) -> LatentState | None:
    if observation.likelihood == 0:
        return None

    log_likelihood = state.log_likelihood + np.log(observation.likelihood).item()  # type: ignore
    if observation.gaussian_obs_A is not None:
        assert observation.gaussian_obs_b is not None
        log_likelihood += log_score_singular(
            state.mu, state.cov, observation.gaussian_obs_A, observation.gaussian_obs_b
        )
        new_mu, new_cov = get_gaussian_posterior(
            state.mu, state.cov, observation.gaussian_obs_A, observation.gaussian_obs_b
        )
    else:
        new_mu, new_cov = state.mu, state.cov

    return LatentState(new_cov, new_mu, log_likelihood)


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
        self, state: LatentState
    ) -> GradedLikelihoodType | None: ...

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

    def compute_log_likelihood(self, state: LatentState) -> GradedLikelihoodType | None:
        latent = _update_latent_with_observation(self.weight, state)
        if latent is None:
            return None

        rank: int = np.linalg.matrix_rank(latent.cov).item()
        return GradedLikelihoodType(latent.log_likelihood, latent.cov.shape[0] - rank)


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

    def compute_log_likelihood(self, state: LatentState):
        log_likelihood = None
        for child in self.children:
            increment = child.compute_log_likelihood(state)
            if log_likelihood is None:
                log_likelihood = None
            elif increment is None:
                continue
            else:
                log_likelihood += increment
        return log_likelihood


class Sum(Node):
    def __init__(self, *children: Node) -> None:
        # TODO: all merging logic should happen here so it is not duplicated in the add/mul methods
        self.children: list[Node] = list(children)

    @property
    def scope(self):
        return set.union(*[c.scope for c in self.children])

    def compute_log_likelihood(self, state: LatentState):  # type: ignore
        log_likelihood = GradedLikelihoodType(1.0, 0)
        nonzero = False
        for child in self.children:
            log_likelihood_update = child.compute_log_likelihood(state)
            if log_likelihood_update:
                nonzero = True
                log_likelihood += log_likelihood_update
        return log_likelihood if nonzero else None


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
