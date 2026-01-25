from kc.base import PExpr
from kc.config import settings
from kc.model_count import model_count
from kc.state import KCState, TruncationState


def run_kc(expr: PExpr):
    state = TruncationState()
    expr.collect_real_truncation({}, state)
    state = KCState(state)
    bdd = expr.kc({}, state)
    if settings.debug:
        print(f"BDD vars: {state.bdd.vars}")
        print(f"Result expr: {bdd.to_expr()}")
        print(f"Observes expr: {state.observes_all_hold.to_expr()}")
        print(f"Result & Observes expr {(bdd & state.observes_all_hold).to_expr()}")
    unnormalized_count = model_count(
        state.bdd, bdd & state.observes_all_hold, state.weights, state.priors
    )
    normalizing_constant = model_count(
        state.bdd, state.observes_all_hold, state.weights, state.priors
    )
    if normalizing_constant == 0:
        return None, normalizing_constant
    return (unnormalized_count / normalizing_constant, normalizing_constant)
