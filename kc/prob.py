import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Literal

import dd.autoref as _bdd
from scipy.stats import norm

from kc.model_count import model_count


class GaussianVariableCounter:
    def __init__(self) -> None:
        self.gaussians = 0
        self.gaussian_params: dict[int, tuple[float, float, float, float]] = {}

    def next_gaussian(self):
        self.gaussians += 1
        return self.gaussians

    def set_gaussian_params(
        self, var, mu, sigma, lower=float("-inf"), upper=float("inf")
    ):
        self.gaussian_params[var] = (mu, sigma, lower, upper)


class TruncationState(GaussianVariableCounter):
    def __init__(self):
        self.truncations: dict[int, set[float]] = defaultdict(set)
        super().__init__()

    def add_truncation(self, var: int, value: float):
        self.truncations[var].add(value)

    def get_truncations(self, var: int) -> set[float]:
        return self.truncations.get(var, set())


def get_gaussian_var_name(gaussian: int):
    return f"_g{gaussian}"


def get_truncated_flip_name(gaussian: int, upper: float, lower: float):
    return f"_g{gaussian}<{upper}|>{lower}"


def create_truncated_gaussian_vars(
    expr: "PExpr",
    state: "TruncationState",
):
    """Pulls all gaussians out to top-level variables and splits them into truncated gaussians."""
    return _create_truncated_gaussian_vars_helper(
        expr.replace_gaussians({}, GaussianVariableCounter()),
        state.gaussian_params.copy(),
        {k: v.copy() for k, v in state.truncations.items()},
    )


def _create_truncated_gaussian_vars_helper(
    expr: "PExpr",
    gaussians_to_params: dict[int, tuple[float, float, float, float]],
    gaussians_to_truncations: dict[int, set[float]],
):
    """Recursive helper"""
    if len(gaussians_to_params) == 0:
        return expr

    gaussian, (mean, std, _, _) = gaussians_to_params.popitem()
    truncations = gaussians_to_truncations.pop(gaussian, set())

    def build_nested_if_then_else(lower: float, truncations: list[float]):
        if truncations:
            upper = truncations.pop(0)
        else:
            upper = float("inf")

        truncated_gaussian = TruncatedGaussian(mean, std, lower, upper)

        if upper == float("inf"):
            return truncated_gaussian

        if lower == float("-inf"):
            remaining_probability_mass = 1
        else:
            remaining_probability_mass = 1 - gaussian_cdf(mean, std, lower)

        flip_prob = (
            gaussian_cdf(mean, std, upper) - gaussian_cdf(mean, std, lower)
        ) / remaining_probability_mass

        flip_name = get_truncated_flip_name(gaussian, upper, lower)

        return Let(
            flip_name,
            Flip(flip_prob),
            IfThenElse(
                Var(flip_name),
                truncated_gaussian,
                build_nested_if_then_else(upper, truncations),
            ),
        )

    return Let(
        get_gaussian_var_name(gaussian),
        build_nested_if_then_else(float("-inf"), sorted(truncations)),
        _create_truncated_gaussian_vars_helper(
            expr, gaussians_to_params, gaussians_to_truncations
        ),
    )


class KCState(GaussianVariableCounter):
    def __init__(self):
        self.bdd = _bdd.BDD()
        self.flips = 0
        self.weights = {}
        self._observes_all_hold = self.bdd.true

        self._gaussian_observe_stack: list[
            tuple[GaussianUnion | GaussianVariable, Literal["<", ">", "="], float],
        ] = []
        self._gaussian_observes_all_hold = None
        super().__init__()

    def _get_eq_conjuction(
        self, gv: "GaussianVariable", thresholds: list[float], val: float
    ):
        equality_clause = self.bdd.true
        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]
            if low == val:
                equality_clause = equality_clause & self.bdd.var(
                    self._get_eq_node_name(gv.var, low)
                )
            elif low != float("-inf"):
                equality_clause = equality_clause & ~self.bdd.var(
                    self._get_eq_node_name(gv.var, low)
                )
        return equality_clause

    def _get_lt_conjuction(
        self, gv: "GaussianVariable", thresholds: list[float], val: float
    ):
        interval_allowed_clause = self.bdd.false
        interval_disallowed_clause = self.bdd.true
        equality_disallowed_clause = self.bdd.true
        equality_allowed_clause = self.bdd.false
        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]

            # Cannot observe equality for values greater than val
            if (high >= val) and (high != float("inf")):
                equality_disallowed_clause = equality_disallowed_clause & ~self.bdd.var(
                    self._get_eq_node_name(gv.var, high)
                )

            if high < val:
                equality_allowed_clause = equality_allowed_clause | self.bdd.var(
                    self._get_eq_node_name(gv.var, high)
                )

            # Can observe interval for intervals less than or equal to val
            if high <= val:
                interval_allowed_clause = interval_allowed_clause | self.bdd.var(
                    self._get_interval_node_name(gv.var, low, high)
                )

            # Cannot observe interval for intervals greater than val
            if low >= val:
                interval_disallowed_clause = interval_disallowed_clause & ~self.bdd.var(
                    self._get_interval_node_name(gv.var, low, high)
                )

        # XOR between interval and equality clauses
        allowed_clause = (interval_allowed_clause & ~equality_allowed_clause) | (
            ~interval_allowed_clause & equality_allowed_clause
        )
        return allowed_clause & interval_disallowed_clause & equality_disallowed_clause

    def _get_gt_conjuction(
        self, gv: "GaussianVariable", thresholds: list[float], val: float
    ):
        interval_allowed_clause = self.bdd.false
        interval_disallowed_clause = self.bdd.true
        equality_disallowed_clause = self.bdd.true
        equality_allowed_clause = self.bdd.false
        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]

            # Cannot observe equality for values less than val
            if (low <= val) and (low != float("-inf")):
                equality_disallowed_clause = equality_disallowed_clause & ~self.bdd.var(
                    self._get_eq_node_name(gv.var, low)
                )

            if low > val:
                equality_allowed_clause = equality_allowed_clause | self.bdd.var(
                    self._get_eq_node_name(gv.var, low)
                )

            # Can observe interval for intervals greater than or equal to val
            if low >= val:
                interval_allowed_clause = interval_allowed_clause | self.bdd.var(
                    self._get_interval_node_name(gv.var, low, high)
                )

            if high <= val:
                interval_disallowed_clause = interval_disallowed_clause & ~self.bdd.var(
                    self._get_interval_node_name(gv.var, low, high)
                )

        # XOR between interval and equality clauses
        allowed_clause = (interval_allowed_clause & ~equality_allowed_clause) | (
            ~interval_allowed_clause & equality_allowed_clause
        )

        return allowed_clause & interval_disallowed_clause & equality_disallowed_clause

    def _get_gaussian_variable_observe_clause(
        self,
        symbolic_value: "GaussianVariable",
        inequality: Literal["<", ">", "="],
        val: float,
        thresholds: list[float],
    ):
        if inequality == "=":
            return self._get_eq_conjuction(symbolic_value, thresholds, val)
        elif inequality == "<":
            return self._get_lt_conjuction(symbolic_value, thresholds, val)
        elif inequality == ">":
            return self._get_gt_conjuction(symbolic_value, thresholds, val)
        else:
            raise ValueError(f"Unexpected inequality: {inequality}")

    def _get_gaussian_union_observe_clause(
        self,
        symbolic_value: "GaussianUnion",
        inequality: Literal["<", ">", "="],
        val: float,
        thresholds: dict[int, list[float]],
    ):
        unguarded_clauses = []
        for v in symbolic_value.values:
            # get the observe clause for this value
            clause = self._get_gaussian_variable_observe_clause(
                v, inequality, val, thresholds[v.var]
            )
            unguarded_clauses.append(clause)

        guarded_clauses = []
        for i in range(len(symbolic_value.values)):
            # this clause is true and all the other clauses in the union are false
            clause = unguarded_clauses[i]
            if inequality == "=":
                # For equality, we need to ensure that only one of the equality clauses is true
                # This is to ensure that the density is only counted once
                clause = clause & reduce(
                    operator.and_,
                    [
                        ~self.bdd.var(
                            self._get_eq_node_name(symbolic_value.values[j].var, val)
                        )
                        for j in range(len(symbolic_value.values))
                        if j != i
                    ],
                )
            # Add formula guarding this value to the clause
            clause = clause & symbolic_value.formulae[i]
            guarded_clauses.append(clause)

        # We OR together all the guarded clauses
        clause = reduce(operator.or_, guarded_clauses)
        return clause

    def _get_eq_node_name(self, var: int, threshold: float):
        return f"gaussian_{var}_eq_{threshold}"

    def _get_interval_node_name(self, var: int, a: float, b: float):
        return f"gaussian_{var}_interval_{a}_{b}"

    def _add_bdd_nodes_for_gaussian_variable(
        self, var: int, threshold_list: set[float]
    ):
        # Get Gaussian parameters
        mean, std, _, _ = self.gaussian_params[var]

        # Sort thresholds and remove duplicates
        sorted_thresholds = sorted(list(threshold_list))
        sorted_thresholds.insert(0, float("-inf"))
        sorted_thresholds.append(float("inf"))

        # Create BDD nodes for intervals (open intervals: a < x < b)
        for i in range(len(sorted_thresholds) - 1):
            a = sorted_thresholds[i]
            b = sorted_thresholds[i + 1]

            # Create BDD node for this interval
            interval_node_name = self._get_interval_node_name(var, a, b)
            self.bdd.declare(interval_node_name)

            # Calculate weight using CDF
            if b == float("inf"):
                prob_interval = 1.0 - gaussian_cdf(mean, std, a)
            elif a == float("-inf"):
                prob_interval = gaussian_cdf(mean, std, b)
            else:
                prob_interval = gaussian_cdf(mean, std, b) - gaussian_cdf(mean, std, a)

            # Set weights: true weight = probability of being in interval
            self.set_weight(interval_node_name, prob_interval, 1.0)

        # Create BDD nodes for equality at each threshold value
        for threshold_val in sorted_thresholds:
            if threshold_val in (float("-inf"), float("inf")):
                continue
            equality_node_name = self._get_eq_node_name(var, threshold_val)
            self.bdd.declare(equality_node_name)

            # Calculate PDF at threshold value
            pdf_val = gaussian_pdf(mean, std, threshold_val)

            # Set weights: true weight = PDF value, false weight = 1.0
            self.set_weight(equality_node_name, pdf_val, 1.0)

        return sorted_thresholds

    def _compile_gaussian_observes_all_hold_clause(self):
        clause = self.bdd.true
        if not self._gaussian_observe_stack:
            return clause

        # Collect all the values appear in observe equality/inequality statements
        thresholds: dict[int, set[float]] = defaultdict(set)
        for symbolic_value, _, val in self._gaussian_observe_stack:
            if isinstance(symbolic_value, GaussianVariable):
                thresholds[symbolic_value.var].add(val)
            elif isinstance(symbolic_value, GaussianUnion):
                for v in symbolic_value.values:
                    thresholds[v.var].add(val)
            else:
                raise ValueError(f"Unexpected type: {type(symbolic_value)}")

        sorted_thresholds: dict[int, list[float]] = {}
        for gv, threshold_set in thresholds.items():
            sorted_thresholds[gv] = self._add_bdd_nodes_for_gaussian_variable(
                gv, threshold_set
            )

        for symbolic_value, inequality, val in self._gaussian_observe_stack:
            if isinstance(symbolic_value, GaussianVariable):
                clause = clause & self._get_gaussian_variable_observe_clause(
                    symbolic_value,
                    inequality,
                    val,
                    sorted_thresholds[symbolic_value.var],
                )
            elif isinstance(symbolic_value, GaussianUnion):
                clause = clause & self._get_gaussian_union_observe_clause(
                    symbolic_value, inequality, val, sorted_thresholds
                )
            else:
                raise ValueError(f"Unexpected type: {type(symbolic_value)}")

        return clause

    @property
    def observes_all_hold(self):
        if self._gaussian_observes_all_hold is None:
            self._gaussian_observes_all_hold = (
                self._compile_gaussian_observes_all_hold_clause()
            )
        return self._observes_all_hold & self._gaussian_observes_all_hold

    def next_flip(self):
        self.flips += 1
        return self.flips

    def set_weight(self, var, pos_weight, neg_weight):
        self.weights[var] = (pos_weight, neg_weight)


class RealValue(ABC):
    pass


@dataclass
class GaussianVariable(RealValue):
    var: int

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

    gaussian_vars = {gaussian.var: gaussian for gaussian in f_values + t_values}

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
    var_to_guards = {}
    var_to_gaussian = {}

    # Process t branch: formulae and values are aligned
    for formula, gv in zip(t_formulae, t_values):
        var = gv.var
        if var not in var_to_guards:
            var_to_guards[var] = []
            var_to_gaussian[var] = gv
        var_to_guards[var].append(formula)

    # Process f branch: formulae and values are aligned
    for formula, gv in zip(f_formulae, f_values):
        var = gv.var
        if var not in var_to_guards:
            var_to_guards[var] = []
            var_to_gaussian[var] = gv
        var_to_guards[var].append(formula)

    # Build aligned lists: OR together guards for each unique variable
    all_formulae = []
    all_values = []
    for var, guards in var_to_guards.items():
        # OR all guards together for this variable
        if len(guards) == 1:
            combined_guard = guards[0]
        else:
            # OR all guards together
            combined_guard = guards[0]
            for guard in guards[1:]:
                combined_guard = combined_guard | guard
        all_formulae.append(combined_guard)
        all_values.append(var_to_gaussian[var])

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

    def replace_gaussians(
        self, env: dict[str, "PExpr"], state: "GaussianVariableCounter"
    ) -> Any:
        """Apply expression transformation which truncates real variables according to their observed inequalities."""
        # By default, do nothing (e.g. for expressions that do not involve real variables)
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

    def replace_gaussians(self, env, state):
        return self


@dataclass
class TruncatedGaussian(AExpr):
    mean: float
    std: float
    lower: float
    upper: float

    def kc(self, env, state):
        var = state.next_gaussian()
        state.set_gaussian_params(var, self.mean, self.std, self.lower, self.upper)
        return GaussianVariable(var)

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        return self

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: TruncationState
    ) -> Any:
        raise ValueError(
            "TruncatedGaussian expressions should not appear in original expr"
        )


@dataclass
class Gaussian(TruncatedGaussian):
    mean: float
    std: float
    lower: float = field(default=float("-inf"), init=False)
    upper: float = field(default=float("inf"), init=False)

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: TruncationState
    ) -> Any:
        var = state.next_gaussian()
        state.set_gaussian_params(var, self.mean, self.std)
        return GaussianVariable(var)

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        var = state.next_gaussian()
        return Var(get_gaussian_var_name(var))


@dataclass
class GatedGaussian(AExpr):
    mean: float
    std: float
    truncations: list[float]

    def kc(self, env, state):
        raise NotImplementedError()


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

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        return self


@dataclass
class Flip(PExpr):
    prob: float

    def kc(self, env, state):
        flip_id = state.next_flip()
        state.bdd.declare(f"flip_{flip_id}")
        state.set_weight(f"flip_{flip_id}", self.prob, 1.0 - self.prob)
        return state.bdd.var(f"flip_{flip_id}")

    def collect_real_truncation(self, env, state: "TruncationState"):
        return

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        return self


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

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        new_cond = self.cond.replace_gaussians(env, state)
        new_then_expr = self.then_expr.replace_gaussians(env, state)
        new_else_expr = self.else_expr.replace_gaussians(env, state)
        return IfThenElse(new_cond, new_then_expr, new_else_expr)


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

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        new_binding = self.binding.replace_gaussians(env, state)
        new_body = self.body.replace_gaussians(env, state)
        return Let(self.var, new_binding, new_body)


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

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        new_cond = self.cond.replace_gaussians(env, state)
        return Observe(new_cond)


# Things to think about:
#   When multiple observes talk about potentially the same GaussianVariable
#     ObserveReal(3.0, Var("x"))
#     ObserveReal(6.0, (Mul(2.0, Var("x"))))
#     where Var("x") refers to a GaussianVariable(i).
#       or where Var("x") is a union
#     This is related to the question in Pun that we saw of density of [x, 2x] at [3, 6] when x ~ N(0, 1)?
# How to handle the non-independence of ObserveRealInequality?


@dataclass
class ObserveReal(PExpr):
    symbolic_value: PExpr
    inequality: Literal["<", ">", "="]
    val: float

    def kc(self, env, state: KCState):
        # Modify the self.observes_all_hold formula in some way...
        # We want something like score(density(symbolic_value, val)),
        #  where this density depends on which GaussianVariable symbolic_value
        symbolic_value = self.symbolic_value.kc(env, state)
        state._gaussian_observe_stack.append(
            (symbolic_value, self.inequality, self.val)
        )
        return state.bdd.true

    def collect_real_truncation(self, env, state):
        if self.inequality == "=":
            return

        symbolic_value = self.symbolic_value.collect_real_truncation(env, state)

        if isinstance(symbolic_value, GaussianVariable):
            state.add_truncation(symbolic_value.var, self.val)
        elif isinstance(symbolic_value, GaussianUnion):
            for gv in symbolic_value.values:
                state.add_truncation(gv.var, self.val)
        else:
            raise TypeError(
                f"Unexpected type for symbolic_value: {type(self.symbolic_value)}"
            )

        return

    def replace_gaussians(
        self, env: dict[str, PExpr], state: GaussianVariableCounter
    ) -> Any:
        new_symbolic_value = self.symbolic_value.replace_gaussians(env, state)
        return ObserveReal(new_symbolic_value, self.inequality, self.val)


def run_kc(expr: PExpr):
    state = TruncationState()
    expr.collect_real_truncation({}, state)
    expr = create_truncated_gaussian_vars(expr, state)
    state = KCState()
    bdd = expr.kc({}, state)
    unnormalized_count = model_count(
        state.bdd, bdd & state.observes_all_hold, state.weights
    )
    normalizing_constant = model_count(
        state.bdd, state.observes_all_hold, state.weights
    )
    if normalizing_constant == 0:
        return None, normalizing_constant
    return (unnormalized_count / normalizing_constant, normalizing_constant)
