import itertools
from collections import defaultdict

import dd.autoref as _bdd

from kc.real_values import (
    DistributionWithDensity,
    Gaussian,
    GaussianVariable,
)
from kc.spn import ObservationWeights


class RandomVariableCounter:
    def __init__(self) -> None:
        self.rv_counter = -1  # Start at -1 so first return is 0
        self.rvs: dict[int, DistributionWithDensity] = {}

    def next_variable(self, rv: DistributionWithDensity):
        self.rv_counter += 1
        self.rvs[self.rv_counter] = rv
        return self.rv_counter

    def variable_count(self, variable_type: type):
        count = 0
        for rv in self.rvs.values():
            if isinstance(rv, variable_type):
                count += 1
        return count


class TruncationCounter:
    def __init__(self):
        self.truncations: dict[int, set[float]] = defaultdict(set)
        super().__init__()

    def add_truncation(self, var: int, scale: float, shift: float, value: float):
        self.truncations[var].add((value - shift) / scale)

    def get_truncations(self, var: int) -> set[float]:
        return self.truncations.get(var, set())


class PreprocessState:
    def __init__(self) -> None:
        self.truncation_counter = TruncationCounter()
        self.rv_counter = RandomVariableCounter()


class KCState(RandomVariableCounter):
    def __init__(self, preprocess_state: PreprocessState):
        self.bdd = _bdd.BDD()
        self.flips = 0
        self.flip_params = 0
        self.weights: dict[
            int,
            tuple[
                ObservationWeights,
                ObservationWeights,
            ],
        ] = {}
        self.beta_priors: dict[int, tuple[float, float]] = {}
        self._observes_all_hold = self.bdd.true
        self.truncations = preprocess_state.truncation_counter.truncations
        self.bdd_equality_nodes: dict[int, set[str]] = defaultdict(set)
        self.gaussian_count = preprocess_state.rv_counter.variable_count(Gaussian)
        super().__init__()

    def _get_symbolic_observe_eq_node_name(
        self, rvs: list[GaussianVariable], val: float
    ):
        rvs = sorted(rvs, key=lambda x: x.var)
        vars_str = ",".join(f"{v.scale}*g{v.var}" for v in rvs)
        return f"_{{{vars_str}}}={val}"

    def _get_eq_node_name(self, var: int, val: float, lower: float, upper: float):
        return f"_g{var}={val}|{lower}<g{var}<={upper}"

    def _get_interval_node_name(self, var: int, lower: float, upper: float):
        # Avoid issues with -0.0
        lower += 0.0
        upper += 0.0
        return f"_g{var}<={upper}|>{lower}"

    def add_bdd_nodes_for_truncatable_variable(self, var: int):
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

            self.set_weight(
                interval_node_name,
                ObservationWeights(flip_prob),
                ObservationWeights(1.0 - flip_prob),
            )

        return sorted_thresholds

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
        pos_weight: ObservationWeights,
        neg_weight: ObservationWeights,
    ):
        self.weights[var] = (pos_weight, neg_weight)


def get_gaussian_var_name(gaussian: int):
    return f"_g{gaussian}"


def get_truncated_flip_name(gaussian: int, upper: float, lower: float):
    return f"_g{gaussian}<{upper}|>{lower}"
