import itertools
import operator
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Self, TypeVar

import sympy
from scipy.stats import beta, norm
from multiset import FrozenMultiset

from kc.base import AExpr, PExpr

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
    def apply_affine(self, scale: float, shift: float) -> Self:
        raise NotImplementedError()


@dataclass(eq=True, frozen=True)
class GaussianVariable(RealVariable, AffineTransformable):
    var: int
    scale: float = 1.0
    shift: float = 0.0

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self

    def apply_affine(self, scale: float, shift: float) -> "GaussianVariable":
        new_scale = self.scale * scale
        new_shift = self.shift * scale + shift
        return GaussianVariable(self.var, new_scale, new_shift)


@dataclass(eq=True, frozen=True)
class BetaVariable(RealVariable):
    var: int

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


T = TypeVar("T", bound=RealVariable)


@dataclass(eq=True, frozen=True)
class Union[T](RealValue):
    formulae: list[Any]
    values: list[T]

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self

    def apply_affine(self, scale: float, shift: float) -> "Union":
        assert all(isinstance(var, AffineTransformable) for var in self.values), (
            "All values must be AffineTransformable"
        )
        new_values = [var.apply_affine(scale, shift) for var in self.values]  # type: ignore
        return Union(self.formulae, new_values)


@dataclass(frozen=True, eq=True)
class GaussianSum(RealVariable, AffineTransformable):
    """Sum of n Gaussian variables after evaluation"""

    rvs: FrozenMultiset[GaussianVariable]

    def apply_affine(self, scale: float, shift: float) -> "GaussianSum":
        new_rvs = []
        for rv in self.rvs:
            if isinstance(rv, GaussianVariable):
                new_rvs.append(rv.apply_affine(scale, shift))
            else:
                raise TypeError("rv must be GaussianVariable or Union")
        return GaussianSum(FrozenMultiset(new_rvs))

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self

    def __add__(self, other: "GaussianSum") -> "GaussianSum":
        if not isinstance(other, GaussianSum):
            raise TypeError("Can only add GaussianSum to GaussianSum")
        new_rvs = self.rvs + other.rvs
        return GaussianSum(new_rvs)


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
            left = Union([state.bdd.true], [left])
        if not isinstance(right, Union):
            right = Union([state.bdd.true], [right])

        sum_to_formula = defaultdict(list)
        for (lhs_formula, lhs_value), (rhs_formula, rhs_value) in itertools.product(
            zip(left.formulae, left.values), zip(right.formulae, right.values)
        ):
            if not isinstance(lhs_value, GaussianSum):
                lhs_value = GaussianSum(FrozenMultiset([lhs_value]))
            if not isinstance(rhs_value, GaussianSum):
                rhs_value = GaussianSum(FrozenMultiset([rhs_value]))
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

        return Union(formulae, sums)

    def collect_real_truncation(self, env, state):
        left = self.left.collect_real_truncation(env, state)
        right = self.right.collect_real_truncation(env, state)

        # We make everything a Union to simplify the logic
        # We effectively 'invert' Unions so that sum of Unions becomes Union of Sums
        if not isinstance(left, Union):
            left = Union([], [left])
        if not isinstance(right, Union):
            right = Union([], [right])

        # For truncation we don't need to track formulae, just the sums
        sums = set()
        for lhs_value, rhs_value in itertools.product(left.values, right.values):
            if not isinstance(lhs_value, GaussianSum):
                lhs_value = GaussianSum(FrozenMultiset([lhs_value]))
            if not isinstance(rhs_value, GaussianSum):
                rhs_value = GaussianSum(FrozenMultiset([rhs_value]))
            sum_value = lhs_value + rhs_value
            sums.add(sum_value)

        # Check for the case where we have only one sum value
        if len(sums) == 1:
            return sums.pop()

        return Union(formulae=[], values=list(sums))


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
        t_values = t.values
    else:
        raise TypeError(f"Unexpected type for t: {type(t)}")

    # Extract formulae and values from f (else branch)
    if isinstance(f, RealVariable):
        f_values = [f]
    elif isinstance(f, Union):
        f_values = f.values
    else:
        raise TypeError(f"Unexpected type for f: {type(f)}")

    return Union(formulae=[], values=list(set(f_values + t_values)))


def merge_real_values(cond, t, f):
    """
    Merge two RealValues (t and f) based on condition cond.
    Returns a GaussianUnion with BDDs and deduplicated GaussianVariables.
    Each formula[i] guards the corresponding values[i].
    When the same value appears with different guards, the guards are ORed together.
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
    if isinstance(f, GaussianVariable):
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

    return Union(formulae=all_formulae, values=all_values)
