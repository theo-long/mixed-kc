from dataclasses import dataclass
from typing import Self

import dd.autoref as _bdd

from kc.base import PExpr, SupportsEqualityComparison
from kc.observation_weights import DirichletPartitionWeight
from kc.partition import Equal, NotEqual, PartitionEnumerator
from kc.state import KCState, PreprocessState


@dataclass
class DirichletProcessDraw(SupportsEqualityComparison):
    draw_number: int
    process: "DirichletProcessVariable"

    def equals(self, other: Self, state: KCState) -> _bdd._Ref:
        if self.process.var != other.process.var:
            raise ValueError(
                f"Cannot compare draws from different Dirichlet Processes, got {self.process.var}!={other.process.var}"
            )
        if self.draw_number == other.draw_number:
            return state.bdd.true
        sorted_draw_numbers = (
            min(self.draw_number, other.draw_number),
            max(self.draw_number, other.draw_number),
        )
        var_name = (
            f"DP{self.process.var}{sorted_draw_numbers[0]}={sorted_draw_numbers[1]}"
        )
        state.bdd.declare(var_name)

        true_constraint, false_constraint = (
            Equal(self.draw_number, other.draw_number),
            NotEqual(self.draw_number, other.draw_number),
        )
        true_weight, false_weight = (
            DirichletPartitionWeight(
                {self.process.var: PartitionEnumerator(true_constraint)}
            ),
            DirichletPartitionWeight(
                {self.process.var: PartitionEnumerator(false_constraint)}
            ),
        )
        state.set_weight(var_name, true_weight, false_weight)

        return state.bdd.var(var_name)


@dataclass
class Draw(PExpr):
    process: PExpr

    def kc(self, env, state):
        process = self.process.kc(env, state)
        if not isinstance(process, DirichletProcessVariable):
            raise ValueError("Can only Draw from a DirichletProcess")
        return process.draw(env, state)

    def preprocess(self, env: dict[str, PExpr], state: PreprocessState):
        process = self.process.preprocess(env, state)
        if not isinstance(process, DirichletProcessVariable):
            raise ValueError("Can only Draw from a DirichletProcess")
        return process.draw(env, state)


class DirichletProcessVariable:
    def __init__(self, var: int, alpha: float):
        self.var = var
        self.alpha = alpha
        self._counter: int = 0

    def draw(self, env, state):
        draw_number = self._counter
        self._counter += 1
        return DirichletProcessDraw(draw_number, self)


@dataclass
class DirichletProcess(PExpr):
    alpha: float

    def preprocess(self, env: dict[str, PExpr], state):
        var = state.rv_counter.next_dp()
        return DirichletProcessVariable(var, self.alpha)

    def kc(self, env, state):
        var = state.next_dp()
        state.dp_priors[var] = self.alpha
        return DirichletProcessVariable(var, self.alpha)
