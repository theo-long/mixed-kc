import bisect
import itertools
import operator
from collections import defaultdict
from functools import reduce

import dd.autoref as _bdd
import sympy

from kc.config import settings
from kc.real_values import (
    DistributionWithDensity,
    DistributionWithMoments,
    RealVariable,
    Union,
)
from kc.types import InequalityLiteral, WeightType, epsilon, inequality_flip_mapping


class RandomVariableCounter:
    def __init__(self) -> None:
        self.rv_counter = 0
        self.rvs: dict[int, DistributionWithDensity] = {}

    def next_variable(self, rv: DistributionWithDensity):
        self.rv_counter += 1
        self.rvs[self.rv_counter] = rv
        return self.rv_counter


class TruncationState(RandomVariableCounter):
    def __init__(self):
        self.truncations: dict[int, set[float]] = defaultdict(set)
        super().__init__()

    def add_truncation(self, var: int, scale: float, shift: float, value: float):
        self.truncations[var].add((value - shift) / scale)

    def get_truncations(self, var: int) -> set[float]:
        return self.truncations.get(var, set())


class KCState(RandomVariableCounter):
    def __init__(self, truncation_state: TruncationState):
        self.bdd = _bdd.BDD()
        self.flips = 0
        self.flip_params = 0
        self.weights: dict[int, tuple[WeightType, int | float | WeightType]] = {}
        self.priors: dict[sympy.Symbol, DistributionWithMoments] = {}
        self._observes_all_hold = self.bdd.true
        self.truncations = truncation_state.truncations
        self.bdd_equality_nodes: dict[int, set[str]] = defaultdict(set)
        self._gaussian_observes_all_hold = None
        super().__init__()

    def _get_gaussian_pair_eq_node_name(self, var: int, other: int):
        # Need to sort so that we don't have separate nodes g1=g2 and g2=g1
        first, second = min(var, other), max(var, other)
        return f"_g{first}=g{second}"

    def get_gaussian_variable_pair_equality_expression(self, var: int, other: int):
        assert not settings.single_observe_eps, (
            "This function should only be called when `single_observe_eps=False`"
        )
        node = self._get_gaussian_pair_eq_node_name(var, other)
        self.bdd.declare(node)
        self.set_weight(node, epsilon, 1.0)
        return self.bdd.var(node)

    def get_gaussian_union_equality_expression(
        self,
        symbolic_value: Union,
        val: float,
    ):
        unguarded_clauses = []
        equality_nodes = []
        for v in symbolic_value.values:
            # get the observe clause for this value
            clause, equality_node = self.get_gaussian_variable_equality_expression(
                v.var, v.scale, v.shift, val
            )
            unguarded_clauses.append(clause)
            equality_nodes.append(equality_node)

        guarded_clause = self.bdd.false
        for i in range(len(symbolic_value.values)):
            clause = unguarded_clauses[i]
            # In the case where single_observe_eps, this is handled automatically by the eps logic
            # since the double equality setting will have weight eps^2, single equality weight eps
            if not settings.single_observe_eps:
                # For equality, we need to ensure that only one of the equality clauses is true
                # This is to ensure that the density is only counted once
                for j in range(len(symbolic_value.values)):
                    if i != j:
                        clause = clause & (
                            ~equality_nodes[j]
                            | self.get_gaussian_variable_pair_equality_expression(
                                symbolic_value.values[i].var,
                                symbolic_value.values[j].var,
                            )
                        )
            # Add formula guarding this value to the clause
            clause = clause & symbolic_value.formulae[i]
            guarded_clause = guarded_clause | clause

        return guarded_clause

    def get_gaussian_union_inequality_expression(
        self,
        symbolic_value: Union,
        inequality: InequalityLiteral,
        val: float,
    ):
        unguarded_clauses = []
        for v in symbolic_value.values:
            # get the observe clause for this value
            clause = self.get_gaussian_variable_inequality_expression(
                v.var, v.scale, v.shift, inequality, val
            )
            unguarded_clauses.append(clause)

        return reduce(
            operator.or_,
            (
                formula & clause
                for formula, clause in zip(symbolic_value.formulae, unguarded_clauses)
            ),
            self.bdd.false,
        )

    def _get_eq_node_name(self, var: int, val: float, lower: float, upper: float):
        return f"_g{var}={val}|{lower}<g{var}<={upper}"

    def _get_interval_node_name(self, var: int, lower: float, upper: float):
        # Avoid issues with -0.0
        lower += 0.0
        upper += 0.0
        return f"_g{var}<={upper}|>{lower}"

    def _create_eq_node(
        self, var: int, val: float, lower: float, upper: float, scale: float
    ):
        equality_node_name = self._get_eq_node_name(var, val, lower, upper)
        self.bdd.declare(equality_node_name)
        # Compute weight for equality node
        # It is the density at val divided by the normalization constant for the interval
        self.rvs[var].pdf(val)
        weight = self.rvs[var].pdf(val) / (
            self.rvs[var].cdf(upper) - self.rvs[var].cdf(lower)
        )
        if settings.single_observe_eps:
            weight = weight * epsilon
        if settings.transform_measures:
            weight /= scale
        self.set_weight(equality_node_name, weight, 1.0)
        self.bdd_equality_nodes[var].add(equality_node_name)
        return self.bdd.var(equality_node_name)

    def add_bdd_nodes_for_gaussian_variable(self, var: int):
        # Get Gaussian parameters
        rv = self.rvs[var]
        sorted_thresholds = sorted(self.truncations.get(var, set()))
        sorted_thresholds.insert(0, float("-inf"))
        sorted_thresholds.append(float("inf"))

        # Create BDD nodes for intervals (half open intervals: a < x <= b)
        for i in range(len(sorted_thresholds) - 1):
            lower = sorted_thresholds[i]
            upper = sorted_thresholds[i + 1]

            # Create BDD node for this interval
            interval_node_name = self._get_interval_node_name(var, lower, upper)
            self.bdd.declare(interval_node_name)

            if lower == float("-inf"):
                remaining_probability_mass = 1
            else:
                remaining_probability_mass = 1 - rv.cdf(lower)

            flip_prob = (rv.cdf(upper) - rv.cdf(lower)) / remaining_probability_mass

            # Set weights: true weight = probability of being in interval
            self.set_weight(interval_node_name, flip_prob, 1.0 - flip_prob)

        return sorted_thresholds

    def get_gaussian_variable_inequality_expression(
        self,
        var: int,
        scale: float,
        shift: float,
        inequality: InequalityLiteral,
        val: float,
    ):
        val = (val - shift) / scale
        if scale < 0:
            inequality = inequality_flip_mapping[inequality]
        sorted_thresholds = (
            [float("-inf")]
            + sorted(self.truncations.get(var, set()))
            + [
                float("inf"),
            ]
        )
        split_index = sorted_thresholds.index(val)
        if inequality in ["<=", "<"]:
            # Some node (x <= upper | x > lower) where upper <= val must be true
            clause = self.bdd.false
            lower, upper = 0.0, 0.0
            for i in range(0, split_index):
                lower, upper = sorted_thresholds[i], sorted_thresholds[i + 1]
                clause = clause | self.bdd.var(
                    self._get_interval_node_name(var, lower, upper)
                )
            if inequality == "<":
                eq_node_name = self._get_eq_node_name(var, val, lower, upper)
                if eq_node_name not in self.bdd_equality_nodes[var]:
                    eq_node = self._create_eq_node(var, val, lower, upper, scale)
                else:
                    eq_node = self.bdd.var(eq_node_name)
                clause = clause & (~eq_node)
        elif inequality in [">", ">="]:
            # Every node representing (x <= t | x > s) for t <= val must be false
            clause = self.bdd.true
            lower, upper = 0.0, 0.0
            for i in range(1, split_index + 1):
                lower = sorted_thresholds[i - 1]
                upper = sorted_thresholds[i]
                clause = clause & ~self.bdd.var(
                    self._get_interval_node_name(var, lower, upper)
                )
            if inequality == ">=":
                eq_node_name = self._get_eq_node_name(var, val, lower, upper)
                if eq_node_name not in self.bdd_equality_nodes[var]:
                    eq_node = self._create_eq_node(var, val, lower, upper, scale)
                else:
                    eq_node = self.bdd.var(eq_node_name)
                clause = clause | eq_node
        else:
            raise ValueError(f"Unexpected inequality: {inequality}")

        return clause

    def get_gaussian_variable_equality_expression(
        self, var: int, scale: float, shift: float, val: float
    ):
        val = (val - shift) / scale
        sorted_thresholds = (
            [float("-inf")]
            + sorted(self.truncations.get(var, set()))
            + [
                float("inf"),
            ]
        )
        bisect_index = bisect.bisect_left(sorted_thresholds, val)
        lower, upper = (
            sorted_thresholds[bisect_index - 1],
            sorted_thresholds[bisect_index],
        )

        inequality_clause = self.bdd.true
        if lower != float("-inf"):
            inequality_clause = self.get_gaussian_variable_inequality_expression(
                var, 1.0, 0.0, ">", lower
            )
        if upper != float("inf"):
            inequality_clause = (
                inequality_clause
                & self.get_gaussian_variable_inequality_expression(
                    var, 1.0, 0.0, "<=", upper
                )
            )

        equality_node = self._create_eq_node(var, val, lower, upper, scale)
        equality_clause = inequality_clause & equality_node
        return equality_clause, equality_node

    @property
    def mutually_compatible_equalities(self):
        mutually_compatible_equalities = self.bdd.true
        for node_names in self.bdd_equality_nodes.values():
            for node, other_node in itertools.combinations(node_names, 2):
                # Ensure that only one equality node for this variable can be true at a time
                mutually_compatible_equalities = mutually_compatible_equalities & ~(
                    self.bdd.var(node) & self.bdd.var(other_node)
                )
        return mutually_compatible_equalities

    @property
    def observes_all_hold(self):
        return self._observes_all_hold & self.mutually_compatible_equalities

    def next_flip(self):
        self.flips += 1
        return self.flips

    def next_flip_param(self):
        self.flip_params += 1
        return self.flip_params

    def set_weight(
        self,
        var,
        pos_weight: WeightType,
        neg_weight: WeightType,
    ):
        self.weights[var] = (pos_weight, neg_weight)


def get_gaussian_var_name(gaussian: int):
    return f"_g{gaussian}"


def get_truncated_flip_name(gaussian: int, upper: float, lower: float):
    return f"_g{gaussian}<{upper}|>{lower}"
