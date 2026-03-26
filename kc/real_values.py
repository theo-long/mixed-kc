import bisect
import itertools
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Self, Sequence

import dd.autoref as _bdd
from scipy.stats import beta, norm

from kc.base import AExpr, PExpr
from kc.config import settings
from kc.observation_weights import (
    GaussianWeight,
    ObservationWeights,
    TruncatedGaussianWeight,
)
from kc.types import InequalityLiteral, inequality_flip_mapping

if TYPE_CHECKING:
    from kc.state import KCState


class DistributionWithDensity:
    @abstractmethod
    def pdf(self, val: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def cdf(self, val: float) -> float:
        raise NotImplementedError


class DistributionWithMoments:
    @abstractmethod
    def moment(self, n: int):
        raise NotImplementedError


@dataclass
class Beta(AExpr, DistributionWithMoments, DistributionWithDensity):
    alpha: float
    beta: float

    def moment(self, n):
        if n == 0:
            return 1
        return (
            self.moment(n - 1) * (self.alpha + n - 1) / (self.alpha + self.beta + n - 1)
        )

    def cdf(self, val):
        return beta.cdf(val, a=self.alpha, b=self.beta).item()

    def pdf(self, val):
        return beta.pdf(val, a=self.alpha, b=self.beta).item()

    def preprocess(self, env: dict[str, PExpr], state) -> Any:
        return

    def kc(self, env, state):
        var = state.next_variable(self)
        state.beta_priors[var] = self.alpha, self.beta
        return BetaVariable(var)


@dataclass
class Gaussian(AExpr, DistributionWithDensity):
    mean: float
    std: float

    def kc(self, env, state):
        var = state.next_variable(self)
        # TODO - should we have this return a GaussianSum with a single variable?
        # that way we don't need to handle wrapping/unwrapping GaussianVariables elsewhere
        return GaussianSum(
            frozenset([GaussianVariable(var, scale=self.std, shift=self.mean)])
        )

    def preprocess(self, env: dict[str, PExpr], state) -> Any:
        var = state.rv_counter.next_variable(self)
        return GaussianSum(
            frozenset([GaussianVariable(var, scale=self.std, shift=self.mean)])
        )

    def pdf(self, val):
        return norm.pdf(val, loc=0, scale=1).item()

    def cdf(self, val):
        return norm.cdf(val, loc=0, scale=1).item()


@dataclass
class TruncatableGaussian(AExpr, DistributionWithDensity):
    mean: float
    std: float

    def kc(self, env, state):
        var = state.next_variable(self)
        state.add_bdd_nodes_for_truncatable_variable(var)
        return TruncatableGaussianVariable(var, scale=self.std, shift=self.mean)

    def preprocess(self, env: dict[str, PExpr], state) -> Any:
        var = state.rv_counter.next_variable(self)
        return TruncatableGaussianVariable(var, scale=self.std, shift=self.mean)

    def pdf(self, val):
        return norm.pdf(val, loc=0, scale=1).item()

    def cdf(self, val):
        return norm.cdf(val, loc=0, scale=1).item()


class RealValue(ABC):
    @abstractmethod
    def get_observe_expr(self, val: float, state: "KCState") -> _bdd.Function: ...


class RealVariable(RealValue):
    pass


class AffineTransformable(ABC):
    @abstractmethod
    def apply_affine(self, scale: float, shift: float) -> Self | "RealConstant":
        raise NotImplementedError()


@dataclass(eq=True, frozen=True)
class RealConstant(RealVariable, AffineTransformable, AExpr):
    value: float

    def get_observe_expr(self, val: float, state: "KCState"):
        if self.value == val:
            return state.bdd.true
        else:
            return state.bdd.false

    def apply_affine(self, scale: float, shift: float):
        return RealConstant(self.value * scale + shift)

    def preprocess(self, env, state):
        return self

    def kc(self, env, state):
        return self


@dataclass(eq=True, frozen=True)
class GaussianVariable(AffineTransformable):
    var: int
    scale: float = 1.0
    shift: float = 0.0

    def preprocess(self, env, state):
        return self

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return RealConstant(0.0)
        new_scale = self.scale * scale
        new_shift = self.shift * scale + shift
        return GaussianVariable(self.var, new_scale, new_shift)

    def __add__(self, other: "GaussianVariable") -> "GaussianVariable | RealConstant":
        if not isinstance(other, GaussianVariable):
            raise TypeError("Can only add GaussianVariable to GaussianVariable")
        if self.var == other.var:
            new_scale = self.scale + other.scale
            new_shift = self.shift + other.shift
            if new_scale == 0:
                return RealConstant(0.0)
            return GaussianVariable(self.var, new_scale, new_shift)
        else:
            raise ValueError("Cannot add GaussianVariables with different vars")


class Truncatable(ABC):
    @abstractmethod
    def get_inequality_expr(
        self, val: float, state: "KCState", inequality: InequalityLiteral
    ) -> _bdd.Function: ...


@dataclass(eq=True, frozen=True)
class TruncatableGaussianVariable(GaussianVariable, RealVariable, Truncatable):
    def get_inequality_expr(
        self, val: float, state: "KCState", inequality: InequalityLiteral
    ):
        val = (val - self.shift) / self.scale
        if self.scale < 0:
            inequality = inequality_flip_mapping[inequality]
        sorted_thresholds = (
            [float("-inf")]
            + sorted(state.truncations.get(self.var, set()))
            + [
                float("inf"),
            ]
        )
        split_index = sorted_thresholds.index(val)
        if inequality in ["<=", "<"]:
            # Some node (x <= upper | x > lower) where upper <= val must be true
            clause = state.bdd.false
            lower, upper = 0.0, 0.0
            for i in range(0, split_index):
                lower, upper = sorted_thresholds[i], sorted_thresholds[i + 1]
                clause = clause | state.bdd.var(
                    state._get_interval_node_name(self.var, lower, upper)
                )
            if inequality == "<":
                eq_node_name = state._get_eq_node_name(self.var, val, lower, upper)
                if eq_node_name not in state.bdd_equality_nodes[self.var]:
                    eq_node = self._create_eq_node(val, state, lower, upper)
                else:
                    eq_node = state.bdd.var(eq_node_name)
                clause = clause & (~eq_node)
        elif inequality in [">", ">="]:
            # Every node representing (x <= t | x > s) for t <= val must be false
            clause = state.bdd.true
            lower, upper = 0.0, 0.0
            for i in range(1, split_index + 1):
                lower = sorted_thresholds[i - 1]
                upper = sorted_thresholds[i]
                clause = clause & ~state.bdd.var(
                    state._get_interval_node_name(self.var, lower, upper)
                )
            if inequality == ">=":
                eq_node_name = state._get_eq_node_name(self.var, val, lower, upper)
                if eq_node_name not in state.bdd_equality_nodes[self.var]:
                    eq_node = self._create_eq_node(val, state, lower, upper)
                else:
                    eq_node = state.bdd.var(eq_node_name)
                clause = clause | eq_node
        else:
            raise ValueError(f"Unexpected inequality: {inequality}")

        return clause

    def _create_eq_node(self, val: float, state: "KCState", lower: float, upper: float):
        equality_node_name = state._get_eq_node_name(self.var, val, lower, upper)
        state.bdd.declare(equality_node_name)
        # Compute weight for equality node
        # It is the density at val divided by the normalization constant for the interval
        state.rvs[self.var].pdf(val)
        weight = state.rvs[self.var].pdf(val) / (
            state.rvs[self.var].cdf(upper) - state.rvs[self.var].cdf(lower)
        )
        if settings.transform_measures:
            weight /= self.scale
        state.set_weight(
            equality_node_name,
            ObservationWeights(
                weight, truncated_gaussian_obs=TruncatedGaussianWeight(1)
            ),
            1.0,
        )
        state.bdd_equality_nodes[self.var].add(equality_node_name)
        return state.bdd.var(equality_node_name)

    def get_observe_expr(self, val: float, state: "KCState"):
        val = (val - self.shift) / self.scale
        sorted_thresholds = (
            [float("-inf")]
            + sorted(state.truncations.get(self.var, set()))
            + [
                float("inf"),
            ]
        )
        bisect_index = bisect.bisect_left(sorted_thresholds, val)
        lower, upper = (
            sorted_thresholds[bisect_index - 1],
            sorted_thresholds[bisect_index],
        )

        inequality_clause = state.bdd.true
        if lower != float("-inf"):
            inequality_clause = self.get_inequality_expr(
                lower * self.scale + self.shift, state, ">"
            )
        if upper != float("inf"):
            inequality_clause = inequality_clause & self.get_inequality_expr(
                upper * self.scale + self.shift, state, ">="
            )

        equality_node = self._create_eq_node(val, state, lower, upper)
        equality_clause = inequality_clause & equality_node
        return equality_clause

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return RealConstant(0.0)
        new_scale = self.scale * scale
        new_shift = self.shift * scale + shift
        return TruncatableGaussianVariable(self.var, new_scale, new_shift)

    def __add__(self, other) -> GaussianVariable:
        raise TypeError("Cannot add TruncatableGaussianVariables")


@dataclass(eq=True, frozen=True)
class BetaVariable(RealVariable):
    var: int

    def get_observe_expr(self, val, state) -> _bdd.Function:
        raise NotImplementedError

    def preprocess(self, env, state):
        return self


@dataclass(eq=True, frozen=True)
class Union[T: RealVariable](RealValue, AffineTransformable, Truncatable):
    formulae: tuple[Any, ...]
    values: tuple[T, ...]

    def get_inequality_expr(
        self, val: float, state: "KCState", inequality: InequalityLiteral
    ) -> _bdd.Function:
        union_clause = state.bdd.false
        for f, v in zip(self.formulae, self.values):
            if not isinstance(v, Truncatable):
                raise ValueError("All elements of union must be Truncatable")
            # get the observe clause for this value
            clause = v.get_inequality_expr(val, state, inequality)
            guarded_clause = f & clause
            union_clause = union_clause | guarded_clause
        return union_clause

    def get_observe_expr(self, val: float, state: "KCState"):
        union_clause = state.bdd.false
        for f, v in zip(self.formulae, self.values):
            # get the observe clause for this value
            clause = v.get_observe_expr(val, state)
            guarded_clause = f & clause
            union_clause = union_clause | guarded_clause
        return union_clause

    def preprocess(self, env, state):
        return self

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return RealConstant(0.0)
        assert all(isinstance(var, AffineTransformable) for var in self.values), (
            "All values must be AffineTransformable"
        )
        new_values = [var.apply_affine(scale, shift) for var in self.values]  # type: ignore
        return Union(self.formulae, tuple(new_values))

    @property
    def truncatable(self):
        return all(getattr(v, "truncatable") for v in self.values)


@dataclass(frozen=True, eq=True)
class GaussianSum(RealVariable, AffineTransformable):
    """Sum of n Gaussian variables after evaluation"""

    rvs: frozenset[GaussianVariable | RealConstant]

    def get_observe_expr(self, val: float, state: "KCState"):
        # Move all shift terms into the value
        new_vars: list[GaussianVariable] = []
        for v in self.rvs:
            if isinstance(v, RealConstant):
                val -= v.value
                continue
            val -= v.shift
            new_vars.append(
                GaussianVariable(
                    var=v.var,
                    scale=v.scale,
                    shift=0.0,
                )
            )

        # If sum is empty/contains only RealConstant, then val must be 0
        if len(new_vars) == 0:
            return state.bdd.true if val == 0. else state.bdd.false

        node_name = state._get_symbolic_observe_eq_node_name(new_vars, val)
        state.bdd.declare(node_name)
        state.set_weight(
            node_name,
            GaussianWeight([{v.var: v.scale for v in new_vars}], [val]),
            1.0,
        )
        return state.bdd.var(node_name)

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return RealConstant(0.0)
        new_rvs = []
        for rv in self.rvs:
            if isinstance(rv, (GaussianVariable, Union)):
                new_rvs.append(rv.apply_affine(scale, shift))
            else:
                raise TypeError("rv must be GaussianVariable or Union")
        return GaussianSum(frozenset(new_rvs))

    def preprocess(self, env, state):
        return self

    def __add__(self, other: "GaussianSum") -> "GaussianSum":
        if not isinstance(other, GaussianSum):
            raise TypeError("Can only add GaussianSum to GaussianSum")

        # Combine GaussianVariables which have same underlying var
        # Combine Unions which are identical by counting occurrences
        new_vars: dict[int, GaussianVariable] = {}
        new_unions: dict[Union, int] = defaultdict(int)
        constant_val = 0.0
        for rv in itertools.chain(self.rvs, other.rvs):
            if isinstance(rv, GaussianVariable):
                if rv.var in new_vars:
                    existing_var = new_vars[rv.var]
                    combined_var = existing_var + rv
                    if isinstance(combined_var, RealConstant):
                        assert combined_var.value == 0.0
                        # Sum is 0 so can remove this var
                        new_vars.pop(rv.var)
                    else:
                        new_vars[rv.var] = combined_var
                else:
                    new_vars[rv.var] = rv
            elif isinstance(rv, Union):
                new_unions[rv] += 1
            elif isinstance(rv, RealConstant):
                constant_val += rv.value
            else:
                raise TypeError("rv must be GaussianVariable, Union, or RealConstant")
        combined_unions = []
        for union, count in new_unions.items():
            for _ in range(count):
                combined_unions.append(union.apply_affine(count, 0.0))
        new_rvs: list[GaussianVariable | RealConstant] = []
        new_rvs.extend(list(new_vars.values()) + combined_unions)
        if constant_val != 0.0 or not new_rvs:
            new_rvs.append(RealConstant(constant_val))
        return GaussianSum(frozenset(new_rvs))


@dataclass
class Sum(PExpr):
    """Sum of 2 Gaussian variables expression"""

    left: PExpr
    right: PExpr

    def kc(self, env, state):
        left = self.left.kc(env, state)
        right = self.right.kc(env, state)

        # We make everything a Union to simplify the logic
        # We effectively 'invert' Unions so that sum of Unions becomes Union of Sums
        if not isinstance(left, Union):
            left = Union((state.bdd.true,), (left,))
        if not isinstance(right, Union):
            right = Union((state.bdd.true,), (right,))

        sum_to_formula = defaultdict(list)
        for (lhs_formula, lhs_value), (rhs_formula, rhs_value) in itertools.product(
            zip(left.formulae, left.values), zip(right.formulae, right.values)
        ):
            if not isinstance(lhs_value, GaussianSum):
                lhs_value = GaussianSum(frozenset([lhs_value]))
            if not isinstance(rhs_value, GaussianSum):
                rhs_value = GaussianSum(frozenset([rhs_value]))

            assert isinstance(lhs_value, GaussianSum) and isinstance(
                rhs_value, GaussianSum
            ), "Elements of Union must GaussianSum"
            sum_value = lhs_value + rhs_value
            sum_to_formula[sum_value].append(lhs_formula & rhs_formula)

        sums, formulae = [], []
        for sum_value, formulas in sum_to_formula.items():
            combined_formula = reduce(operator.or_, formulas)
            sums.append(sum_value)
            formulae.append(combined_formula)

        # Check for the case where we have only one sum value
        if len(sums) == 1:
            return sums[0]

        return Union(tuple(formulae), tuple(sums))

    def preprocess(self, env, state):
        left = self.left.preprocess(env, state)
        right = self.right.preprocess(env, state)

        # We make everything a Union to simplify the logic
        # We effectively 'invert' Unions so that sum of Unions becomes Union of Sums
        if not isinstance(left, Union):
            left = Union((None,), (left,))
        if not isinstance(right, Union):
            right = Union((None,), (right,))

        # For truncation we don't need to track formulae, just the sums
        sums = set()
        for lhs_value, rhs_value in itertools.product(left.values, right.values):
            if not isinstance(lhs_value, GaussianSum):
                lhs_value = GaussianSum(frozenset([lhs_value]))
            if not isinstance(rhs_value, GaussianSum):
                rhs_value = GaussianSum(frozenset([rhs_value]))

            assert isinstance(lhs_value, GaussianSum) and isinstance(
                rhs_value, GaussianSum
            ), "Elements of Union must GaussianSum"
            sum_value = lhs_value + rhs_value
            sums.add(sum_value)

        # Check for the case where we have only one sum value
        if len(sums) == 1:
            return sums.pop()

        return Union(formulae=(None,), values=tuple(sums))


@dataclass
class Affine(PExpr):
    """Corresponds to the expression body * scale + shift"""

    body: PExpr
    scale: float = 1.0
    shift: float = 0.0

    def kc(self, env, state):
        body = self.body.kc(env, state)
        if isinstance(body, AffineTransformable):
            return body.apply_affine(self.scale, self.shift)
        else:
            raise TypeError("body should evaluate to a Gaussian Variable or Union")

    def preprocess(self, env: dict[str, PExpr], state) -> Any:
        body = self.body.preprocess(env, state)
        if isinstance(body, AffineTransformable):
            return body.apply_affine(self.scale, self.shift)
        else:
            raise TypeError("body should evaluate to a Gaussian Variable or Union")


def merge_real_values_ignore_cond(t, f):
    """When performing truncation we don't need to worry about cond"""
    # Extract formulae and values from t (then branch)
    if isinstance(t, RealVariable):
        t_values = [t]
    elif isinstance(t, Union):
        t_values = list(t.values)
    else:
        raise TypeError(f"Unexpected type for t: {type(t)}")

    # Extract formulae and values from f (else branch)
    if isinstance(f, RealVariable):
        f_values = [f]
    elif isinstance(f, Union):
        f_values = list(f.values)
    else:
        raise TypeError(f"Unexpected type for f: {type(f)}")

    return Union(formulae=(None,), values=tuple(set(f_values + t_values)))  # type: ignore


def merge_guarded_unions(
    guarded_values: Sequence[tuple[_bdd._Ref, Union[RealVariable] | RealVariable]],
) -> Union[RealVariable]:
    """
    Merge two RealValues (t and f) based on condition cond where t and f have been reduced i.e. have no GaussianSum terms..
    """
    # Extract formulae and values from t (then branch)
    var_to_guards = defaultdict(list)
    for formula, value in guarded_values:
        if isinstance(value, Union):
            for union_guard, union_sub_value in zip(value.formulae, value.values):
                var_to_guards[union_sub_value].append(formula & union_guard)
        else:
            var_to_guards[value].append(formula)

    all_formulae = []
    all_values = []
    for var, guards in var_to_guards.items():
        # OR all guards together for this variable
        combined_guard = reduce(operator.or_, guards)
        all_formulae.append(combined_guard)
        all_values.append(var)

    return Union(formulae=tuple(all_formulae), values=tuple(all_values))
