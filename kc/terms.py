from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kc.base import AExpr, PExpr
from kc.real_values import (
    Union,
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


def construct_flip_tree_rec(
    values: Sequence[PExpr],
    probs: Sequence[float],
    flip_name: str,
) -> PExpr:
    if len(values) == 1:
        return values[0]
    else:
        mid = len(values) // 2
        left_probs = probs[:mid]
        right_probs = probs[mid:]

        left_total = sum(left_probs)
        right_total = sum(right_probs)

        left_normalized_probs = [p / left_total for p in left_probs]
        right_normalized_probs = [p / right_total for p in right_probs]

        left_subtree = construct_flip_tree_rec(
            values[:mid], left_normalized_probs, f"{flip_name}_L"
        )
        right_subtree = construct_flip_tree_rec(
            values[mid:], right_normalized_probs, f"{flip_name}_R"
        )

        return Let(
            flip_name,
            Flip(left_total),
            IfThenElse(
                Var(flip_name),
                left_subtree,
                right_subtree,
            ),
        )


class Categorical(PExpr):
    cls_counter = 0

    def __new__(cls, values: Sequence[PExpr], probs: Sequence[float]) -> PExpr:
        if len(values) != len(probs):
            raise ValueError("Values and probabilities must have the same length")
        normalized_probs = []
        total = sum(probs)
        for p in probs:
            normalized_probs.append(p / total)

        name = f"{cls.__name__}_{cls.cls_counter}"
        cls.cls_counter += 1
        return construct_flip_tree_rec(values, normalized_probs, name)


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
        elif isinstance(symbolic_value, Union):
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
        elif isinstance(symbolic_value, Union):
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
        elif isinstance(symbolic_value, Union):
            for gv in symbolic_value.values:
                state.add_truncation(gv.var, gv.scale, gv.shift, self.val)
        else:
            raise TypeError(
                f"Unexpected type for symbolic_value: {type(self.symbolic_value)}"
            )

        return
