import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from kc.base import AExpr, PExpr
from kc.observation_weights import BetaWeight
from kc.real_values import (
    BetaVariable,
    RealValue,
    Truncatable,
    TruncatableGaussianVariable,
    Union,
    merge_guarded_unions,
    merge_real_values_ignore_cond,
)
from kc.types import InequalityLiteral

if TYPE_CHECKING:
    from kc.state import KCState


class EnumType:
    def __init__(self, name: str, values: Sequence[str]):
        self.name = name
        self.values = values
        self.n_bits = math.ceil(math.log2(len(values))) if len(values) > 1 else 1
        self._members = {val: EnumValue(self, i) for i, val in enumerate(values)}

    def __getattr__(self, name: str) -> "EnumValue":
        if name in self._members:
            return self._members[name]
        raise AttributeError(f"'{self.name}' has no attribute '{name}'")


@dataclass
class EnumResult:
    enum_type: EnumType
    bits: tuple

    def preprocess(self, env, state):
        return self


@dataclass
class EnumValue(AExpr):
    enum_type: EnumType
    index: int

    def kc(self, env, state):
        bits = []
        for i in range(self.enum_type.n_bits):
            bit_val = bool((self.index >> i) & 1)
            bits.append(state.bdd.true if bit_val else state.bdd.false)
        return EnumResult(self.enum_type, tuple(bits))

    def preprocess(self, env, state):
        bits = tuple(None for _ in range(self.enum_type.n_bits))
        return EnumResult(self.enum_type, bits)


@dataclass
class Equality(PExpr):
    left: PExpr
    right: PExpr

    def kc(self, env, state):
        l_res = self.left.kc(env, state)
        r_res = self.right.kc(env, state)

        if isinstance(l_res, EnumResult) and isinstance(r_res, EnumResult):
            if l_res.enum_type != r_res.enum_type:
                raise TypeError(
                    f"Cannot compare different Enum types: {l_res.enum_type.name} and {r_res.enum_type.name}"
                )

            eq_bdd = state.bdd.true
            for l_bit, r_bit in zip(l_res.bits, r_res.bits):
                bits_eq = (l_bit & r_bit) | (~l_bit & ~r_bit)
                eq_bdd = eq_bdd & bits_eq
            return eq_bdd
        elif hasattr(l_res, "to_expr") and hasattr(r_res, "to_expr"):
            return (l_res & r_res) | (~l_res & ~r_res)  # type: ignore
        else:
            raise TypeError(
                f"Equality only supported between Enums or Booleans, got {type(l_res)} and {type(r_res)}"
            )

    def preprocess(self, env, state):
        self.left.preprocess(env, state)
        self.right.preprocess(env, state)
        return


@dataclass
class Const(AExpr):
    val: bool

    def kc(self, env, state):
        if self.val:
            return state.bdd.true
        else:
            return state.bdd.false

    def preprocess(self, env, state):
        return


@dataclass
class Var(AExpr):
    var: str

    def kc(self, env, state):
        return env[self.var]

    def preprocess(self, env, state):
        return env[self.var]


@dataclass
class Flip(PExpr):
    prob: float | AExpr

    def kc(self, env, state):
        flip_id = state.next_flip()
        state.bdd.declare(f"flip_{flip_id}")
        if isinstance(self.prob, (float, int)):
            pos, neg = (
                self.prob,
                1.0 - self.prob,
            )
        else:
            prob_val = self.prob.kc(env, state)
            if not isinstance(prob_val, BetaVariable):
                raise ValueError("Can only set Flip prob to float or Beta expression")
            pos, neg = (
                BetaWeight({prob_val.var: (1, 0)}),
                BetaWeight({prob_val.var: (0, 1)}),
            )
        state.set_weight(
            f"flip_{flip_id}",
            pos,
            neg,
        )
        return state.bdd.var(f"flip_{flip_id}")

    def preprocess(self, env, state):
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
            return merge_guarded_unions(
                [(condition_bdd, then_result), (~condition_bdd, else_result)]  # type: ignore
            )
        elif isinstance(then_result, EnumResult):
            assert (
                isinstance(else_result, EnumResult)
                and then_result.enum_type == else_result.enum_type
            )
            new_bits = tuple(
                (condition_bdd & t) | (~condition_bdd & e)
                for t, e in zip(then_result.bits, else_result.bits)
            )
            return EnumResult(then_result.enum_type, new_bits)
        else:
            return (condition_bdd & then_result) | (~condition_bdd & else_result)

    def preprocess(self, env, state):
        self.cond.preprocess(env, state)
        then_result = self.then_expr.preprocess(env, state)
        else_result = self.else_expr.preprocess(env, state)
        if isinstance(then_result, RealValue):
            return merge_real_values_ignore_cond(then_result, else_result)
        elif isinstance(then_result, EnumResult):
            assert (
                isinstance(else_result, EnumResult)
                and then_result.enum_type == else_result.enum_type
            )
            return EnumResult(then_result.enum_type, then_result.bits)
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

    def preprocess(self, env, state):
        new_env = extend_env(env, {self.var: self.binding.preprocess(env, state)})
        return self.body.preprocess(new_env, state)


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

    def preprocess(self, env: dict[str, PExpr], state) -> Any:
        return self.cond.preprocess(env, state)


@dataclass
class ObserveReal(PExpr):
    symbolic_value: PExpr
    val: float

    def kc(self, env, state: "KCState"):
        symbolic_value = self.symbolic_value.kc(env, state)
        if not isinstance(symbolic_value, RealValue):
            raise ValueError(
                "Can only ObserveReal an expression that evaluates to a RealValue"
            )
        state._observes_all_hold &= symbolic_value.get_observe_expr(self.val, state)
        return state.bdd.true

    def preprocess(self, env, state):
        return


@dataclass
class Inequality(PExpr):
    symbolic_value: PExpr
    inequality: InequalityLiteral
    val: float

    def kc(self, env, state: "KCState"):
        symbolic_value = self.symbolic_value.kc(env, state)
        if isinstance(symbolic_value, Truncatable):
            clause = symbolic_value.get_inequality_expr(
                self.val,
                state,
                self.inequality,
            )
        else:
            raise TypeError(f"Unexpected type: {type(symbolic_value)}")
        return clause

    def preprocess(self, env, state):
        symbolic_value = self.symbolic_value.preprocess(env, state)
        if isinstance(symbolic_value, TruncatableGaussianVariable):
            state.truncation_counter.add_truncation(
                symbolic_value.var, symbolic_value.scale, symbolic_value.shift, self.val
            )
        elif isinstance(symbolic_value, Union):
            if not all(
                isinstance(v, TruncatableGaussianVariable)
                for v in symbolic_value.values
            ):
                raise ValueError("Can only truncate TruncatableGaussian")
            for gv in symbolic_value.values:
                state.truncation_counter.add_truncation(
                    gv.var, gv.scale, gv.shift, self.val
                )
        else:
            raise TypeError(
                f"Unexpected type for symbolic_value: {type(symbolic_value)}"
            )

        return
