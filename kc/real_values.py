import itertools
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar

import sympy
from scipy.stats import beta, norm

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


@dataclass(eq=True, frozen=True)
class GaussianVariable(RealVariable):
    var: int
    scale: float = 1.0
    shift: float = 0.0

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


@dataclass(eq=True, frozen=True)
class BetaVariable(RealVariable):
    var: int

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


T = TypeVar("T", bound=RealVariable)


@dataclass
class Union[T](RealValue):
    formulae: list[Any]
    values: list[T]

    def collect_real_truncation(self, env, state: "TruncationState"):
        return self


@dataclass
class Sum(PExpr):
    """Sum of two Gaussian variables"""

    left: PExpr
    right: PExpr

    def kc(self, env, state):
        pass

    def collect_real_truncation(self, env, state):
        raise NotImplementedError()


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

    def _apply_to_gaussian_union(
        self, union: Union[GaussianVariable]
    ) -> Union[GaussianVariable]:
        new_values = [self._apply_to_gaussian_var(var) for var in union.values]
        return Union(union.formulae, new_values)

    def kc(self, env, state):
        body = self.body.kc(env, state)
        if isinstance(body, GaussianVariable):
            return self._apply_to_gaussian_var(body)
        elif isinstance(body, Union):
            assert all(isinstance(rv, GaussianVariable) for rv in body.values), (
                "Must be gaussian union"
            )
            return self._apply_to_gaussian_union(body)
        else:
            raise TypeError("body should evaluate to a Gaussian Variable or Union")

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: "TruncationState"
    ) -> Any:
        body = self.body.collect_real_truncation(env, state)
        if isinstance(body, GaussianVariable):
            return self._apply_to_gaussian_var(body)
        elif isinstance(body, Union):
            assert all(isinstance(rv, GaussianVariable) for rv in body.values), (
                "Must be gaussian union"
            )
            return self._apply_to_gaussian_union(body)
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
