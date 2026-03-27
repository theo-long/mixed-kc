from dataclasses import dataclass

import dd.autoref as _bdd

from kc.base import PExpr
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
        self._table_number_exprs: dict[int, _bdd._Ref] = {}
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
            self._table_number_exprs[1] = state.bdd.true
            return new_value

        # We iterate through all the existing customers
        draw_number = self._n
        # Note that in our representation, table_number = k represents 'the same table as customer k'
        table_number = draw_number
        guarded_table_assignments: list[
            tuple[_bdd._Ref, Union[RealVariable] | RealVariable]
        ] = []
        table_expr = state.bdd.true
        while table_number > 0:
            # Variable representing customer draw_number sitting at same table as table_number
            var = f"DP_{self.var}_draw_{draw_number}_table_{table_number}"
            state.bdd.declare(var)

            # If sits at same table as customer table_number, update table counts
            # Otherwise, continue to next customer
            if table_number == self._n:
                assignment = new_value
                assignment_prob = self.alpha / (self.alpha + self._n)
            else:
                assignment = self._assignment_exprs[table_number]
                assignment_prob = 1 / (table_number)

            state.set_weight(
                var,
                assignment_prob,
                1 - assignment_prob,
            )

            guarded_table_assignments.append(
                (table_expr & state.bdd.var(var), assignment)
            )

            # In next iteration we did *not* sit at the current table, so add that to the table_expr
            table_expr = table_expr & ~state.bdd.var(var)

            # If we do not choose draw 1, then we *must* choose draw 0
            if table_number == 1:
                # add the guard expr corresponding to picking table 0
                guarded_table_assignments.append(
                    (table_expr, self._assignment_exprs[0])
                )
                # '# tables == 1' = 'prev # tables == 1 & not new_table'
                self._table_number_exprs[table_number] = self._table_number_exprs[
                    table_number
                ] & (
                    ~state.bdd.var(
                        f"DP_{self.var}_draw_{draw_number}_table_{draw_number}"
                    )
                )
            else:
                # Expr representing '# tables is table_number' is just 'prev. # tables is table_number - 1 & new_table'
                self._table_number_exprs[table_number] = self._table_number_exprs[
                    table_number - 1
                ] & state.bdd.var(
                    f"DP_{self.var}_draw_{draw_number}_table_{draw_number}"
                )

            table_number -= 1

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
