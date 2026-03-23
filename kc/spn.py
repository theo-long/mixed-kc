"""Sum-Product Network representation for posterior."""

import itertools
from abc import ABC, abstractmethod
from math import prod
from typing import Callable

from kc.observation_weights import FullPosterior, GradedLikelihood, ObservationWeights


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
    def get_log_likelihood(
        self, beta_priors: dict[int, tuple[float, float]]
    ) -> GradedLikelihood | None: ...

    @abstractmethod
    def get_posterior(
        self, var_selection: list[int], **kwargs
    ) -> list[FullPosterior]: ...

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

    def __str__(self):
        return f"Weight({self.weight})"

    @property
    def scope(self):
        return self.weight.scope

    def get_log_likelihood(
        self, beta_priors: dict[int, tuple[float, float]]
    ) -> GradedLikelihood | None:
        return self.weight.get_log_likelihood(beta_priors=beta_priors)

    def get_posterior(self, var_selection: list[int], **kwargs) -> list[FullPosterior]:
        child_vars = [v for v in var_selection if v in self.scope]
        if not child_vars:
            likelihood = self.weight.get_log_likelihood(**kwargs)
            return [FullPosterior(likelihood)] if likelihood else []
        posterior = self.weight.get_posterior(child_vars, **kwargs)
        return [posterior] if posterior else []

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

    def get_log_likelihood(self, beta_priors: dict[int, tuple[float, float]]):
        log_likelihood = GradedLikelihood(0.0, 0)
        for child in self.children:
            increment = child.get_log_likelihood(beta_priors)
            if increment is None:
                # If anything is None i.e. p=0 in product, whole product is p=0
                return None
            log_likelihood = log_likelihood * increment
        return log_likelihood

    def get_posterior(self, var_selection: list[int], **kwargs) -> list[FullPosterior]:
        posterior_combinations = itertools.product(
            *(
                c.get_posterior([v for v in var_selection if v in c.scope], **kwargs)
                for c in self.children
            )
        )
        product_posterior: list[FullPosterior] = []
        for posterior_combination in posterior_combinations:
            if not posterior_combination:
                product_posterior.append(FullPosterior(GradedLikelihood(0.0, 0)))
                continue
            combined_posterior = prod(
                posterior_combination, start=FullPosterior(GradedLikelihood(0.0, 0))
            )
            product_posterior.append(combined_posterior)
        return product_posterior

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

    def get_log_likelihood(self, beta_priors: dict[int, tuple[float, float]]):  # type: ignore
        log_likelihood = None
        for child in self.children:
            log_likelihood_update = child.get_log_likelihood(beta_priors)
            # If update has p=0, we can just ignore it in sum
            if log_likelihood_update is not None:
                log_likelihood = log_likelihood + log_likelihood_update
        return log_likelihood

    def get_posterior(self, var_selection: list[int], **kwargs) -> list[FullPosterior]:
        return sum(
            [c.get_posterior(var_selection, **kwargs) for c in self.children], []
        )

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
    if b.weight.likelihood == 0.0:
        return a
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
    elif a.weight.likelihood == 0.0:
        return b
    elif b.weight.likelihood == 0.0:
        return a
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

        new_children = list(overlapping_children)
        if disjoint_children:
            if len(disjoint_children) == 1:
                new_children.append(Product(b, disjoint_children[0]))
            else:
                new_children.append(Product(b, Sum(*disjoint_children)))
        return Sum(*new_children)


def _mul_product_weight(a: Product, b: WeightNode):
    if b.weight.likelihood == 0:
        return WeightNode(ObservationWeights(0.0))
    if a.scope.isdisjoint(b.scope):
        return Product(a, b)

    disjoint_children = []
    overlapping_children = []
    for child in a.children:
        if child.scope.isdisjoint(b.scope):
            disjoint_children.append(child)
        else:
            overlapping_children.append(child)

    if len(overlapping_children) == 1:
        merged_overlap = overlapping_children[0] * b
    else:
        merged_overlap = Product(*overlapping_children) * b

    if disjoint_children:
        return Product(*disjoint_children, merged_overlap)
    else:
        return merged_overlap


def _mul_weight_weight(a: WeightNode, b: WeightNode):
    if b.weight.likelihood == 0 or a.weight.likelihood == 0:
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
