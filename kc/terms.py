from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kc.base import AExpr, PExpr
from kc.real_values import (
    GaussianUnion,
    GaussianVariable,
    RealValue,
    merge_real_values,
    merge_real_values_ignore_cond,
)
from kc.types import InequalityLiteral

if TYPE_CHECKING:
    from kc.state import KCState, TruncationState


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
class Var(AExpr):
    var: str

    def kc(self, env, state):
        return env[self.var]

    def collect_real_truncation(self, env, state: "TruncationState"):
        substitued_value = env[self.var]
        if substitued_value is not None:
            substitued_value = substitued_value.collect_real_truncation(env, state)
        return substitued_value


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

    def kc(self, env, state: "KCState"):
        state._observes_all_hold = state._observes_all_hold & self.cond.kc(env, state)
        return state.bdd.true

    def collect_real_truncation(
        self, env: dict[str, PExpr], state: "TruncationState"
    ) -> Any:
        return self.cond.collect_real_truncation(env, state)


@dataclass
class ObserveReal(PExpr):
    symbolic_value: PExpr
    val: float

    def kc(self, env, state: "KCState"):
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

    def kc(self, env, state: "KCState"):
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
