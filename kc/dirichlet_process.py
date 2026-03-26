from dataclasses import dataclass

import dd.autoref as _bdd

from kc.base import PExpr
from kc.observation_weights import DirichletProcessWeight
from kc.real_values import (
    Gaussian,
    GaussianSum,
    GaussianVariable,
    RealVariable,
    Union,
    merge_guarded_unions,
)
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
        process = self.process.preprocess(env, state)
        if not isinstance(process, DirichletProcessVariable):
            raise ValueError("Can only Draw from a DirichletProcess")
        var = state.rv_counter.next_variable(process.base.__class__)  # type: ignore
        return GaussianSum(
            frozenset(
                [GaussianVariable(var, scale=process.base.std, shift=process.base.mean)]
            )
        )


class DirichletProcessVariable:
    def __init__(self, var: int, alpha: float, base: Gaussian):
        self.var = var
        self.alpha = alpha
        self.base = base
        self._assignment_exprs: list[Union[GaussianSum]] = []
        self._n = 0

    def draw(self, env: dict[str, PExpr], state: KCState):
        # TODO - this is super inefficient because we represent every possible cluster partitioning
        # Is there anything better we can do?

        # We draw a new value from the base distribution
        new_value = self.base.kc(env, state)

        # If no draws yet, must sit at new table
        if self._n == 0:
            self._n += 1
            self._assignment_exprs.append(Union((state.bdd.true,), (new_value,)))
            return new_value

        # We iterate through all the existing customers
        draw_number = self._n
        prev_draw = draw_number
        guarded_table_assignments: list[
            tuple[_bdd._Ref, Union[RealVariable] | RealVariable]
        ] = []
        table_expr = state.bdd.true
        while prev_draw > 0:
            # Variable representing customer draw_number sitting at same table as prev_draw
            var = f"DP_{self.var}_draw_{draw_number}_table_{prev_draw}"
            state.bdd.declare(var)

            # If sits at same table as customer prev_draw, update table counts
            # Otherwise, continue to next customer
            if prev_draw == self._n:
                assignment = new_value
            else:
                assignment = self._assignment_exprs[prev_draw]
            guarded_table_assignments.append(
                (table_expr & state.bdd.var(var), assignment)
            )

            # In next iteration we did *not* sit at the current table, so add that to the table_expr
            table_expr = table_expr & ~state.bdd.var(var)

            # If we do not choose draw 1, then we *must* choose draw 0
            if prev_draw == 1:
                false_weight = DirichletProcessWeight({self.var: {draw_number: 0}})
                # add the guard expr corresponding to picking table 0
                guarded_table_assignments.append(
                    (table_expr, self._assignment_exprs[0])
                )
            else:
                # PROBLEM: false weight = 1.0 doesn't work
                # Sometimes in a BDD we might select the false branch of some DP expr
                # e.g. we have ~DP_1_draw_2_table_2
                # If we do not have any additional conditions on the assignments, 
                # then bdd vars after this can be *removed* from the BDD (because all paths go true)
                # But this means the contributions from the cluster assignment of draw 2
                # which must sit at table 1 or 2 (since we know ~table_2) *doesn't get counted*
                # and leads to wrong probabilities
                false_weight = 1.0

            state.set_weight(
                var,
                DirichletProcessWeight({self.var: {draw_number: prev_draw}}),
                false_weight,
            )

            prev_draw -= 1

        self._n += 1

        # Expr representing table assignment of current draw
        assignment_expr = merge_guarded_unions(guarded_table_assignments)
        self._assignment_exprs.append(assignment_expr)  # type: ignore

        assert len(guarded_table_assignments) == len(self._assignment_exprs)
        return assignment_expr


@dataclass
class DirichletProcess(PExpr):
    alpha: float
    base: Gaussian

    def preprocess(self, env: dict[str, PExpr], state):
        var = state.rv_counter.next_dp()
        return DirichletProcessVariable(var, self.alpha, self.base)

    def kc(self, env, state):
        var = state.next_dp()
        state.dp_priors[var] = self.alpha
        return DirichletProcessVariable(var, self.alpha, self.base)
