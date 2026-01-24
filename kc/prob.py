import bisect
import itertools
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Any

import dd.autoref as _bdd
import sympy
from scipy.stats import norm

from kc.config import settings
from kc.model_count import model_count
from kc.types import InequalityLiteral, WeightType, epsilon, inequality_flip_mapping


class GaussianVariableCounter:
    def __init__(self) -> None:
        self.gaussians = 0
        self.gaussian_params: dict[int, tuple[float, float]] = {}

    def next_gaussian(self):
        self.gaussians += 1
        return self.gaussians

    def set_gaussian_params(self, var, mu, sigma):
        self.gaussian_params[var] = (mu, sigma)


class TruncationState(GaussianVariableCounter):
    def __init__(self):
        self.truncations: dict[int, set[float]] = defaultdict(set)
        super().__init__()

    def add_truncation(self, var: int, scale: float, shift: float, value: float):
        self.truncations[var].add((value - shift) / scale)

    def get_truncations(self, var: int) -> set[float]:
        return self.truncations.get(var, set())


def get_gaussian_var_name(gaussian: int):
    return f"_g{gaussian}"


def get_truncated_flip_name(gaussian: int, upper: float, lower: float):
    return f"_g{gaussian}<{upper}|>{lower}"


class KCState(GaussianVariableCounter):
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
        symbolic_value: "GaussianUnion",
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
        symbolic_value: "GaussianUnion",
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
        weight = gaussian_pdf(*self.gaussian_params[var], val) / (
            gaussian_cdf(*self.gaussian_params[var], upper)
            - gaussian_cdf(*self.gaussian_params[var], lower)
        )
        if settings.single_observe_eps:
            weight *= epsilon
        if settings.transform_measures:
            weight /= scale
        self.set_weight(equality_node_name, weight, 1.0)
        self.bdd_equality_nodes[var].add(equality_node_name)
        return self.bdd.var(equality_node_name)

    def add_bdd_nodes_for_gaussian_variable(self, var: int):
        # Get Gaussian parameters
        mean, std = self.gaussian_params[var]
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
                remaining_probability_mass = 1 - gaussian_cdf(mean, std, lower)

            flip_prob = (
                gaussian_cdf(mean, std, upper) - gaussian_cdf(mean, std, lower)
            ) / remaining_probability_mass

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


class RealValue(ABC):
    pass


@dataclass
class GaussianVariable(RealValue):
    var: int
    scale: float = 1.0
    shift: float = 0.0

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


@dataclass
class GaussianUnion(RealValue):
    formulae: list[Any]
    values: list[GaussianVariable]

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


def gaussian_pdf(mean, std, val):
    return norm.pdf(val, loc=mean, scale=std).item()


def gaussian_cdf(mean, std, val):
    return norm.cdf(val, loc=mean, scale=std).item()


def merge_real_values_ignore_cond(t, f):
    """When performing truncation we don't need to worry about cond"""
    # Extract formulae and values from t (then branch)
    if isinstance(t, GaussianVariable):
        t_values = [t]
    elif isinstance(t, GaussianUnion):
        t_values = t.values
    else:
        raise TypeError(f"Unexpected type for t: {type(t)}")

    # Extract formulae and values from f (else branch)
    if isinstance(f, GaussianVariable):
        f_values = [f]
    elif isinstance(f, GaussianUnion):
        f_values = f.values
    else:
        raise TypeError(f"Unexpected type for f: {type(f)}")

    gaussian_vars = {
        (gaussian.var, gaussian.scale, gaussian.shift): gaussian
        for gaussian in f_values + t_values
    }

    return GaussianUnion(formulae=[], values=list(gaussian_vars.values()))


def merge_real_values(cond, t, f):
    """
    Merge two RealValues (t and f) based on condition cond.
    Returns a GaussianUnion with BDDs and deduplicated GaussianVariables.
    Each formula[i] guards the corresponding values[i].
    When the same value appears with different guards, the guards are ORed together.
    """
    # Extract formulae and values from t (then branch)
    if isinstance(t, GaussianVariable):
        t_formulae = [cond]
        t_values = [t]
    elif isinstance(t, GaussianUnion):
        # AND each formula with cond, ensuring formulae and values are aligned
        t_formulae = [cond & formula for formula in t.formulae]
        t_values = t.values
    else:
        raise TypeError(f"Unexpected type for t: {type(t)}")

    # Extract formulae and values from f (else branch)
    if isinstance(f, GaussianVariable):
        f_formulae = [~cond]
        f_values = [f]
    elif isinstance(f, GaussianUnion):
        # AND each formula with ~cond, ensuring formulae and values are aligned
        f_formulae = [~cond & formula for formula in f.formulae]
        f_values = f.values
    else:
        raise TypeError(f"Unexpected type for f: {type(f)}")

    # Build a map from var -> list of guards (formulae) for that variable
    # This allows us to OR together guards for the same variable
    var_and_transform_to_guards = {}
    var_and_transform_to_gaussian = {}

    # Process t then f branch: formulae and values are aligned
    for formula, gv in itertools.chain(
        zip(t_formulae, t_values), zip(f_formulae, f_values)
    ):
        var_and_transform = gv.var, gv.scale, gv.shift
        if var_and_transform not in var_and_transform_to_guards:
            var_and_transform_to_guards[var_and_transform] = []
            var_and_transform_to_gaussian[var_and_transform] = gv
        var_and_transform_to_guards[var_and_transform].append(formula)

    # Build aligned lists: OR together guards for each unique variable
    all_formulae = []
    all_values = []
    for var_and_transform, guards in var_and_transform_to_guards.items():
        # OR all guards together for this variable
        if len(guards) == 1:
            combined_guard = guards[0]
        else:
            # OR all guards together
            combined_guard = guards[0]
            for guard in guards[1:]:
                combined_guard = combined_guard | guard
        all_formulae.append(combined_guard)
        all_values.append(var_and_transform_to_gaussian[var_and_transform])

    return GaussianUnion(formulae=all_formulae, values=all_values)


class PExpr(ABC):
    @abstractmethod
    def kc(self, env: dict[str, "PExpr"], state: "KCState") -> Any:
        """Compile this probabilistic expression into the KCState and return the corresponding BDD."""
        raise NotImplementedError()

    @abstractmethod
    def collect_real_truncation(
        self, env: dict[str, "PExpr"], state: "TruncationState"
    ) -> Any:
        """Collect all the observed inequalities and the Gaussian variables they apply to."""
        raise NotImplementedError()


class AExpr(PExpr):
    pass


@dataclass
class Const(AExpr):
    val: bool

    def kc(self, env, state):
        if self.val:
            return state.bdd.true
        else:
            return state.bdd.false

    def collect_real_truncation(self, env, state):
        return


@dataclass
class Gaussian(AExpr):
    mean: float
    std: float

    def kc(self, env, state):
        var = state.next_gaussian()
        state.set_gaussian_params(var, self.mean, self.std)
        state.add_bdd_nodes_for_gaussian_variable(var)
        return GaussianVariable(var)

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: TruncationState
    ) -> Any:
        var = state.next_gaussian()
        state.set_gaussian_params(var, self.mean, self.std)
        return GaussianVariable(var)


@dataclass
class Affine(PExpr):
    """Corresponds to the expression body * scale + shift"""

    body: PExpr
    scale: float = 1.0
    shift: float = 0.0

    def _apply_to_gaussian_var(self, var: GaussianVariable) -> GaussianVariable:
        new_scale = var.scale * self.scale
        new_shift = var.shift * self.scale + self.shift
        return GaussianVariable(var.var, new_scale, new_shift)

    def _apply_to_gaussian_union(self, union: GaussianUnion) -> GaussianUnion:
        new_values = [self._apply_to_gaussian_var(var) for var in union.values]
        return GaussianUnion(union.formulae, new_values)

    def kc(self, env, state):
        body = self.body.kc(env, state)
        if isinstance(body, GaussianVariable):
            return self._apply_to_gaussian_var(body)
        elif isinstance(body, GaussianUnion):
            return self._apply_to_gaussian_union(body)
        else:
            raise TypeError("body should evaluate to a RealValue")

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: TruncationState
    ) -> Any:
        body = self.body.collect_real_truncation(env, state)
        if isinstance(body, GaussianVariable):
            return self._apply_to_gaussian_var(body)
        elif isinstance(body, GaussianUnion):
            return self._apply_to_gaussian_union(body)
        else:
            raise TypeError("body should evaluate to a RealValue")


@dataclass
class Var(AExpr):
    var: str

    def kc(self, env, state):
        return env[self.var]

    def collect_real_truncation(self, env, state: "TruncationState"):
        substitued_value = env[self.var]
        if substitued_value is not None:
            substitued_value = substitued_value.collect_real_truncation(env, state)
        return substitued_value


class DistributionWithMoments(AExpr):
    @abstractmethod
    def moment(self, n: int):
        raise NotImplementedError

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: TruncationState
    ) -> Any:
        return

    def kc(self, env, state):
        flip_param_id = state.next_flip_param()
        symbol = sympy.symbols(f"p{flip_param_id}")
        state.priors[symbol] = self
        return symbol


@dataclass
class BetaPrior(DistributionWithMoments):
    alpha: float
    beta: float

    def moment(self, n):
        if n == 0:
            return 1
        return (
            self.moment(n - 1) * (self.alpha + n - 1) / (self.alpha + self.beta + n - 1)
        )


@dataclass
class Flip(PExpr):
    prob: float | AExpr

    def kc(self, env, state):
        flip_id = state.next_flip()
        state.bdd.declare(f"flip_{flip_id}")
        if isinstance(self.prob, (float, int)):
            state.set_weight(f"flip_{flip_id}", self.prob, 1.0 - self.prob)
        else:
            prob_val = self.prob.kc(env, state)
            state.set_weight(f"flip_{flip_id}", prob_val, 1.0 - prob_val)
        return state.bdd.var(f"flip_{flip_id}")

    def collect_real_truncation(self, env, state: "TruncationState"):
        return


@dataclass
class IfThenElse(PExpr):
    cond: AExpr
    then_expr: PExpr
    else_expr: PExpr

    def kc(self, env, state):
        condition_bdd = self.cond.kc(env, state)
        then_result = self.then_expr.kc(env, state)
        else_result = self.else_expr.kc(env, state)

        if isinstance(then_result, RealValue):
            return merge_real_values(condition_bdd, then_result, else_result)
        else:
            return (condition_bdd & then_result) | (~condition_bdd & else_result)

    def collect_real_truncation(self, env, state: "TruncationState"):
        self.cond.collect_real_truncation(env, state)
        then_result = self.then_expr.collect_real_truncation(env, state)
        else_result = self.else_expr.collect_real_truncation(env, state)
        if isinstance(then_result, RealValue):
            return merge_real_values_ignore_cond(then_result, else_result)
        else:
            return


def extend_env(env: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any]:
    new_env = env.copy()
    new_env.update(extension)
    return new_env


@dataclass
class Let(PExpr):
    var: str
    binding: PExpr
    body: PExpr

    def kc(self, env, state):
        new_env = extend_env(env, {self.var: self.binding.kc(env, state)})
        return self.body.kc(new_env, state)

    def collect_real_truncation(self, env, state: "TruncationState"):
        new_env = extend_env(
            env, {self.var: self.binding.collect_real_truncation(env, state)}
        )
        return self.body.collect_real_truncation(new_env, state)


class Rejection(Exception):
    pass


@dataclass
class Observe(PExpr):
    cond: PExpr

    def kc(self, env, state: KCState):
        state._observes_all_hold = state._observes_all_hold & self.cond.kc(env, state)
        return state.bdd.true

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: TruncationState
    ) -> Any:
        return self.cond.collect_real_truncation(env, state)


@dataclass
class ObserveReal(PExpr):
    symbolic_value: PExpr
    val: float

    def kc(self, env, state: KCState):
        # Modify the self.observes_all_hold formula in some way...
        # We want something like score(density(symbolic_value, val)),
        #  where this density depends on which GaussianVariable symbolic_value
        symbolic_value = self.symbolic_value.kc(env, state)
        if isinstance(symbolic_value, GaussianVariable):
            clause, _ = state.get_gaussian_variable_equality_expression(
                symbolic_value.var, symbolic_value.scale, symbolic_value.shift, self.val
            )
        elif isinstance(symbolic_value, GaussianUnion):
            clause = state.get_gaussian_union_equality_expression(
                symbolic_value, self.val
            )
        else:
            raise ValueError(f"Unexpected type: {type(symbolic_value)}")

        state._observes_all_hold = state._observes_all_hold & clause
        return state.bdd.true

    def collect_real_truncation(self, env, state):
        return


@dataclass
class Inequality(PExpr):
    symbolic_value: PExpr
    inequality: InequalityLiteral
    val: float

    def kc(self, env, state: KCState):
        symbolic_value = self.symbolic_value.kc(env, state)
        if isinstance(symbolic_value, GaussianVariable):
            clause = state.get_gaussian_variable_inequality_expression(
                symbolic_value.var,
                symbolic_value.scale,
                symbolic_value.shift,
                self.inequality,
                self.val,
            )
        elif isinstance(symbolic_value, GaussianUnion):
            clause = state.get_gaussian_union_inequality_expression(
                symbolic_value, self.inequality, self.val
            )
        else:
            raise TypeError(f"Unexpected type: {type(symbolic_value)}")
        return clause

    def collect_real_truncation(self, env, state):
        symbolic_value = self.symbolic_value.collect_real_truncation(env, state)

        if isinstance(symbolic_value, GaussianVariable):
            state.add_truncation(
                symbolic_value.var, symbolic_value.scale, symbolic_value.shift, self.val
            )
        elif isinstance(symbolic_value, GaussianUnion):
            for gv in symbolic_value.values:
                state.add_truncation(gv.var, gv.scale, gv.shift, self.val)
        else:
            raise TypeError(
                f"Unexpected type for symbolic_value: {type(self.symbolic_value)}"
            )

        return


def run_kc(expr: PExpr):
    state = TruncationState()
    expr.collect_real_truncation({}, state)
    state = KCState(state)
    bdd = expr.kc({}, state)
    if settings.debug:
        print(f"BDD vars: {state.bdd.vars}")
        print(f"Result expr: {bdd.to_expr()}")
        print(f"Observes expr: {state.observes_all_hold.to_expr()}")
        print(f"Result & Observes expr {(bdd & state.observes_all_hold).to_expr()}")
    unnormalized_count = model_count(
        state.bdd, bdd & state.observes_all_hold, state.weights, state.priors
    )
    normalizing_constant = model_count(
        state.bdd, state.observes_all_hold, state.weights, state.priors
    )
    if normalizing_constant == 0:
        return None, normalizing_constant
    return (unnormalized_count / normalizing_constant, normalizing_constant)
