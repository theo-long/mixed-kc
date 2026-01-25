import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy
from scipy.stats import beta, norm

from kc.base import AExpr, PExpr

if TYPE_CHECKING:
    from kc.state import TruncationState


class DistributionWithCDF:
    @abstractmethod
    def cdf(self, val: float) -> float:
        raise NotImplementedError


class DistributionWithMoments(AExpr):
    @abstractmethod
    def moment(self, n: int):
        raise NotImplementedError

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
class BetaPrior(DistributionWithMoments, DistributionWithCDF):
    alpha: float
    beta: float

    def moment(self, n):
        if n == 0:
            return 1
        return (
            self.moment(n - 1) * (self.alpha + n - 1) / (self.alpha + self.beta + n - 1)
        )

    def cdf(self, val):
        return beta.cdf(val).item()


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
        self, env: dict[str, PExpr], state: "TruncationState"
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
        self, env: dict[str, PExpr], state: "TruncationState"
    ) -> Any:
        body = self.body.collect_real_truncation(env, state)
        if isinstance(body, GaussianVariable):
            return self._apply_to_gaussian_var(body)
        elif isinstance(body, GaussianUnion):
            return self._apply_to_gaussian_union(body)
        else:
            raise TypeError("body should evaluate to a RealValue")


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
