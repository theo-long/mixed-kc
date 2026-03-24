from dataclasses import dataclass

from kc.base import PExpr
from kc.observation_weights import DirichletProcessWeight
from kc.real_values import Gaussian, GaussianSum, Union
from kc.state import KCState, PreprocessState


@dataclass
class Draw(PExpr):
    process: PExpr

    def kc(self, env, state):
        process = self.process.kc(env, state)
        if not isinstance(process, DirichletProcessVariable):
            raise ValueError("Can only Draw from a DirichletProcess")
        return process.draw(env, state)

    def preprocess(self, env: dict[str, PExpr], state: PreprocessState):
        return


class DirichletProcessVariable:
    def __init__(self, var: int, alpha: float, base: Gaussian):
        self.var = var
        self.alpha = alpha
        self.base = base
        self._draws: list[GaussianSum] = []
        self._n = 0

    def draw(self, env: dict[str, PExpr], state: KCState):
        # TODO - this is super inefficient because we represent every possible cluster partitioning
        # Is there anything better we can do?

        # We draw a new value from the base distribution
        new_value = self.base.kc(env, state)
        self._draws.append(new_value)

        # If no draws yet, must sit at new table
        if self._n == 0:
            self._n += 1
            return new_value

        # We iterate through all the existing customers
        draw_number = self._n
        prev_draw = draw_number
        table_exprs = []
        table_expr = state.bdd.true
        while prev_draw > 0:
            # Variable representing customer draw_number sitting at same table as prev_draw
            var = f"DP_{self.var}_draw_{draw_number}_table_{prev_draw}"
            state.bdd.declare(var)

            # If sits at same table as customer prev_draw, update table counts
            # Otherwise, continue to next customer
            table_exprs.append(table_expr & state.bdd.var(var))

            # In next iteration we did *not* sit at the current table, so add that to the table_expr
            table_expr = table_expr & ~state.bdd.var(var)

            # If we do not choose draw 1, then we *must* choose draw 0
            if prev_draw == 1:
                false_weight = DirichletProcessWeight({self.var: {draw_number: 0}})
                # add the guard expr corresponding to picking table 0
                table_exprs.append(table_expr)
            else:
                false_weight = 1.0

            state.set_weight(
                var,
                DirichletProcessWeight({self.var: {draw_number: prev_draw}}),
                false_weight,
            )

            prev_draw -= 1

        self._n += 1

        assert len(table_exprs) == len(self._draws)
        return Union(tuple(table_exprs), tuple(self._draws))


@dataclass
class DirichletProcess(PExpr):
    alpha: float
    base: Gaussian

    def preprocess(self, env: dict[str, PExpr], state):
        return

    def kc(self, env, state):
        var = state.next_dp()
        state.dp_priors[var] = self.alpha
        return DirichletProcessVariable(var, self.alpha, self.base)
