import dd.autoref as _bdd
import numpy as np

from kc.base import PExpr
from kc.config import settings
from kc.gaussian_math import get_expr_distribution
from kc.model_count import model_count
from kc.real_values import GaussianSum, GaussianVariable, Zero
from kc.state import KCState, PreprocessState
from kc.types import get_float_value


def run_kc(expr: PExpr):
    preprocess_state = PreprocessState()
    expr.preprocess({}, preprocess_state)
    state = KCState(preprocess_state)
    val = expr.kc({}, state)
    if settings.debug:
        print(f"BDD vars: {state.bdd.vars}")
        if isinstance(val, _bdd.Function):
            print(f"Result expr: {val.to_expr()}")
        else:
            print(f"Result expr: {val}")
        print(f"Observes expr: {state.observes_all_hold.to_expr()}")

    posterior_mixture = model_count(
        state.bdd,
        state.observes_all_hold,
        state.gaussian_count,
        state.weights,
    )
    normalizing_constant = get_float_value(
        sum(map(lambda x: x[0], posterior_mixture)), state.priors
    )

    if settings.debug:
        print("Normalizing constant pre-simplification:", normalizing_constant)
    normalizing_constant = get_float_value(normalizing_constant, {})

    if normalizing_constant == 0:
        return None, normalizing_constant

    if isinstance(val, GaussianVariable):
        val = GaussianSum(frozenset({val}))

    # Inference for binary variable
    if isinstance(val, _bdd.Function):
        posterior_mixture_with_val = model_count(
            state.bdd,
            val & state.observes_all_hold,
            state.gaussian_count,
            state.weights,
        )
        unnormalized_prob = sum(map(lambda x: x[0], posterior_mixture_with_val))

        if settings.debug:
            print("Unnormalized prob pre-simplification:", unnormalized_prob)
        unnormalized_prob = get_float_value(unnormalized_prob, state.priors)

        return (unnormalized_prob / normalizing_constant, normalizing_constant)
    elif isinstance(val, GaussianSum):
        normalized_posterior = []
        for weight, (mu, cov) in posterior_mixture:
            v = np.zeros((1, state.gaussian_count))
            b = 0.0
            for var in val.rvs:
                assert not isinstance(var, Zero)
                v[0, var.var - 1] = var.scale
                b += var.shift
            expr_mu, expr_cov = get_expr_distribution(mu, cov, v, b)
            normalized_posterior.append(
                (weight / normalizing_constant, (expr_mu, expr_cov))
            )
        return normalized_posterior, normalizing_constant
    else:
        raise TypeError(f"Cannot perform inference for value of type {type(val)}")
