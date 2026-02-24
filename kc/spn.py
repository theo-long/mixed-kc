"""Sum-Product Network representation for continuous latents."""

from typing import Callable
import itertools
from abc import ABC
from typing import NamedTuple

import numpy as np

from kc.types import LikelihoodType


class ObservationWeights(NamedTuple):
    gaussian_obs: np.ndarray | None
    likelihood: LikelihoodType

    @property
    def scope(self):
        if self.gaussian_obs is None:
            return set()
        return set(self.gaussian_obs[:-1].flatnonzero())

    def __mul__(self, other: "ObservationWeights") -> "ObservationWeights":
        if self.gaussian_obs and other.gaussian_obs:
            gaussian_obs = np.stack([self.gaussian_obs, other.gaussian_obs], axis=-1)
        elif self.gaussian_obs:
            gaussian_obs = self.gaussian_obs
        else:
            gaussian_obs = other.gaussian_obs

        return ObservationWeights(
            gaussian_obs,
            self.likelihood * other.likelihood,
        )


class Node(ABC):
    ADD_OPS: dict[
        tuple[type["Node"], type["Node"]],
        Callable[["Node", "Node"], "Sum | WeightNode"],
    ] = {}
    MUL_OPS: dict[
        tuple[type["Node"], type["Node"]],
        Callable[["Node", "Node"], "Product | WeightNode"],
    ] = {}

    def __init__(self) -> None:
        self.scope: set[int]

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


class Product(Node):
    def __init__(self, *children) -> None:
        self.children: list[Node] = list(children)
        self.scope = set.union(*[c.scope for c in children])

    def is_valid(self):
        for a, b in itertools.combinations(self.children, 2):
            if a.scope & b.scope:
                return False
        return True


class Sum(Node):
    def __init__(self, *children: Node) -> None:
        self.children: list[Node] = list(children)
        self.scope = set.union(*[c.scope for c in children])


def _add_sum_sum(a: Sum, b: Sum):
    # TODO we might be able to merge children
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


def _mul_sum_sum(a: Sum, b: Sum):
    if a.scope.isdisjoint(b.scope):
        return Product(*a.children, *b.children)
    else:
        raise NotImplementedError


def _mul_sum_product(a: Sum, b: Product):
    if a.scope.isdisjoint(b.scope):
        return Product(*a.children, b)
    else:
        raise NotImplementedError


def _mul_sum_weight(a: Sum, b: WeightNode):
    return Product(a, b)


def _mul_product_product(a: Product, b: Product):
    if a.scope.isdisjoint(b.scope):
        return Product(*a.children, *b.children)
    else:
        raise NotImplementedError


def _mul_product_weight(a: Product, b: WeightNode):
    if a.scope.isdisjoint(b.scope):
        return Product(a, b)
    else:
        raise NotImplementedError


def _mul_weight_weight(a: WeightNode, b: WeightNode):
    return WeightNode(a.weight * b.weight)


Node.ADD_OPS = {
    (Sum, Sum): _add_sum_sum,
    (Sum, Product): _add_sum_product,
    (Sum, WeightNode): _add_sum_weight,
    (Product, Product): _add_product_product,
    (Product, WeightNode): _add_product_weight,
    (WeightNode, WeightNode): _add_weight_weight,
}

Node.MUL_OPS = {
    (Sum, Sum): _mul_sum_sum,
    (Sum, Product): _mul_sum_product,
    (Sum, WeightNode): _mul_sum_weight,
    (Product, Product): _mul_product_product,
    (Product, WeightNode): _mul_product_weight,
    (WeightNode, WeightNode): _mul_weight_weight,
}
