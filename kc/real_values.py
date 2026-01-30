import itertools
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Self, TypeVar

import sympy
from scipy.stats import beta, norm

from kc.base import AExpr, PExpr
from kc.config import settings

if TYPE_CHECKING:
    from kc.state import TruncationState


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

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: "TruncationState"
    ) -> Any:
        return

    def kc(self, env, state):
        flip_param_id = state.next_flip_param()
        symbol = sympy.symbols(f"p{flip_param_id}")
        state.priors[symbol] = self
        return symbol


@dataclass
class Gaussian(AExpr, DistributionWithDensity):
    mean: float
    std: float

    def kc(self, env, state):
        var = state.next_variable(self)
        state.add_bdd_nodes_for_gaussian_variable(var)
        return GaussianVariable(var)

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: "TruncationState"
    ) -> Any:
        var = state.next_variable(self)
        return GaussianVariable(var)

    def pdf(self, val):
        return norm.pdf(val, loc=self.mean, scale=self.std).item()

    def cdf(self, val):
        return norm.cdf(val, loc=self.mean, scale=self.std).item()


class RealValue(ABC):
    pass


class RealVariable(RealValue):
    pass


class AffineTransformable(ABC):
    @abstractmethod
    def apply_affine(self, scale: float, shift: float) -> Self | "Zero":
        raise NotImplementedError()


class Zero(RealVariable, AffineTransformable):
    def apply_affine(self, scale: float, shift: float):
        return self


@dataclass(eq=True, frozen=True)
class GaussianVariable(RealVariable, AffineTransformable):
    var: int
    scale: float = 1.0
    shift: float = 0.0

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return Zero()
        new_scale = self.scale * scale
        new_shift = self.shift * scale + shift
        return GaussianVariable(self.var, new_scale, new_shift)

    def __add__(self, other: "GaussianVariable") -> "GaussianVariable":
        if not isinstance(other, GaussianVariable):
            raise TypeError("Can only add GaussianVariable to GaussianVariable")
        if self.var == other.var:
            new_scale = self.scale + other.scale
            new_shift = self.shift + other.shift
            return GaussianVariable(self.var, new_scale, new_shift)
        else:
            raise ValueError("Cannot add GaussianVariables with different vars")


@dataclass(eq=True, frozen=True)
class BetaVariable(RealVariable):
    var: int

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


T = TypeVar("T", bound=RealVariable)


@dataclass(eq=True, frozen=True)
class Union[T](RealValue):
    formulae: tuple[Any]
    values: tuple[T]

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return Zero()
        assert all(isinstance(var, AffineTransformable) for var in self.values), (
            "All values must be AffineTransformable"
        )
        new_values = [var.apply_affine(scale, shift) for var in self.values]  # type: ignore
        return Union(self.formulae, tuple(new_values))


@dataclass(frozen=True, eq=True)
class GaussianSum(RealVariable, AffineTransformable):
    """Sum of n Gaussian variables after evaluation"""

    rvs: frozenset[GaussianVariable | Zero]

    def apply_affine(self, scale: float, shift: float):
        if scale == 0.0:
            return Zero()
        new_rvs = []
        for rv in self.rvs:
            if isinstance(rv, (GaussianVariable, Union)):
                new_rvs.append(rv.apply_affine(scale, shift))
            else:
                raise TypeError("rv must be GaussianVariable or Union")
        return GaussianSum(frozenset(new_rvs))

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self

    def __add__(self, other: "GaussianSum") -> "GaussianSum":
        if not isinstance(other, GaussianSum):
            raise TypeError("Can only add GaussianSum to GaussianSum")

        # Combine GaussianVariables which have same underlying var
        # Combine Unions which are identical by counting occurrences
        new_vars: dict[int, GaussianVariable] = {}
        new_unions: dict[Union, int] = defaultdict(int)
        for rv in itertools.chain(self.rvs, other.rvs):
            if isinstance(rv, GaussianVariable):
                if rv.var in new_vars:
                    existing_var = new_vars[rv.var]
                    combined_var = existing_var + rv
                    new_vars[rv.var] = combined_var
                else:
                    new_vars[rv.var] = rv
            elif isinstance(rv, Union):
                new_unions[rv] += 1
            else:
                raise TypeError("rv must be GaussianVariable or Union")
        combined_unions = []
        for union, count in new_unions.items():
            for _ in range(count):
                combined_unions.append(union.apply_affine(count, 0.0))
        new_rvs = list(new_vars.values()) + combined_unions
        return GaussianSum(frozenset(new_rvs))


@dataclass
class Sum(PExpr):
    """Sum of 2 Gaussian variables expression"""

    left: PExpr
    right: PExpr

    def kc(self, env, state):
        left = self.left.kc(env, state)
        right = self.right.kc(env, state)

        if not settings.union_of_sums:
            # Reduce to case of sum of sums
            if not isinstance(left, GaussianSum):
                left = GaussianSum(frozenset([left]))
            if not isinstance(right, GaussianSum):
                right = GaussianSum(frozenset([right]))
            return left + right

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

    def collect_real_truncation(self, env, state):
        left = self.left.collect_real_truncation(env, state)
        right = self.right.collect_real_truncation(env, state)

        if not settings.union_of_sums:
            # Reduce to case of sum of sums
            if not isinstance(left, GaussianSum):
                left = GaussianSum(frozenset([left]))
            if not isinstance(right, GaussianSum):
                right = GaussianSum(frozenset([right]))
            return left + right

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

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: "TruncationState"
    ) -> Any:
        body = self.body.collect_real_truncation(env, state)
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


def merge_real_values(cond, t, f):
    """
    Merge two RealValues (t and f) based on condition cond.
    If t or f is a GaussianSum, we 'invert' the Union to create a Union of Sums if this setting is enabled.
    """
    # If neither t nor f is a GaussianSum, we can use the simpler 'reduced' merging logic
    if settings.union_of_sums or not (
        isinstance(t, GaussianSum) or isinstance(f, GaussianSum)
    ):
        return merge_real_values_reduced(cond, t, f)

    if isinstance(t, GaussianSum):
        t_vals = t.rvs
    else:
        t_vals = [t]
    if isinstance(f, GaussianSum):
        f_vals = f.rvs
    else:
        f_vals = [f]

    # Now we can merge each corresponding term in the Gaussian sums
    sum_terms: dict[Union | GaussianVariable, int] = defaultdict(int)
    for t_val, f_val in itertools.zip_longest(t_vals, f_vals):
        t_val = t_val if t_val is not None else Zero()
        f_val = f_val if f_val is not None else Zero()
        merged_term = merge_real_values_reduced(cond, t_val, f_val)
        if not isinstance(merged_term, Union):
            raise TypeError(
                "Merged term should be a Union when merging GaussianSum terms"
            )
        sum_terms[merged_term] += 1

    merged_sum_terms = []
    for term, count in sum_terms.items():
        if count == 1:
            merged_sum_terms.append(term)
        else:
            merged_sum_terms.append(term.apply_affine(count, 0.0))

    return GaussianSum(frozenset(merged_sum_terms))


def merge_real_values_reduced(cond, t, f):
    """
    Merge two RealValues (t and f) based on condition cond where t and f have been reduced i.e. have no GaussianSum terms..
    """
    # Extract formulae and values from t (then branch)
    if isinstance(t, RealVariable):
        t_formulae = [cond]
        t_values = [t]
    elif isinstance(t, Union):
        # AND each formula with cond, ensuring formulae and values are aligned
        t_formulae = [cond & formula for formula in t.formulae]
        t_values = t.values
    else:
        raise TypeError(f"Unexpected type for t: {type(t)}")

    # Extract formulae and values from f (else branch)
    if isinstance(f, RealVariable):
        f_formulae = [~cond]
        f_values = [f]
    elif isinstance(f, Union):
        # AND each formula with ~cond, ensuring formulae and values are aligned
        f_formulae = [~cond & formula for formula in f.formulae]
        f_values = f.values
    else:
        raise TypeError(f"Unexpected type for f: {type(f)}")

    # Build a map from var -> list of guards (formulae) for that variable
    # This allows us to OR together guards for the same variable
    var_to_guards = defaultdict(list)

    # Process t then f branch: formulae and values are aligned
    for formula, rv in itertools.chain(
        zip(t_formulae, t_values), zip(f_formulae, f_values)
    ):
        var_to_guards[rv].append(formula)

    # Build aligned lists: OR together guards for each unique variable
    all_formulae = []
    all_values = []
    for var, guards in var_to_guards.items():
        # OR all guards together for this variable
        combined_guard = reduce(operator.or_, guards)
        all_formulae.append(combined_guard)
        all_values.append(var)

    return Union(formulae=tuple(all_formulae), values=tuple(all_values))
