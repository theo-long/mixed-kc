import dd.autoref as _bdd
import numpy as np

from kc.base import PExpr
from kc.config import settings
from kc.gaussian_math import get_expr_distribution
from kc.model_count import model_count
from kc.real_values import GaussianSum, GaussianVariable, RealConstant
from kc.spn import LatentState, Node
from kc.state import KCState, PreprocessState
from kc.terms import EnumResult
from kc.types import get_degree, get_float_value


def compute_spn_likelihood(spn: Node, state: KCState) -> float:
    latent = LatentState.initial_state(state.gaussian_count)
    graded_log_likelihood = spn.compute_log_likelihood(latent)
    from IPython import embed; embed()
    if graded_log_likelihood is None:
        return 0.0

    if settings.debug:
        print("Log likelihood pre-simplification:", graded_log_likelihood)
    likelihood = get_float_value(
        np.exp(graded_log_likelihood.log_likelihood),  # type: ignore
        state.priors,
    )
    likelihood = get_float_value(likelihood, {})
    return likelihood


def preprocess(expr: PExpr):
    preprocess_state = PreprocessState()
    expr.preprocess({}, preprocess_state)
    return preprocess_state


def kc(expr: PExpr, preprocess_state: PreprocessState):
    state = KCState(preprocess_state)
    val = expr.kc({}, state)
    if settings.debug:
        print(f"BDD vars: {state.bdd.vars}")
        if isinstance(val, _bdd.Function):
            print(f"Result expr: {val.to_expr()}")
        else:
            print(f"Result expr: {val}")
        print(f"Observes expr: {state.observes_all_hold.to_expr()}")

    return val, state


def get_normalizing_constant(state: KCState):
    spn = model_count(
        state.bdd,
        state.observes_all_hold,
        state.weights,
    )
    return compute_spn_likelihood(spn, state)


def binary_inference(val, state: KCState, normalizing_constant: float):
    spn = model_count(
        state.bdd,
        val & state.observes_all_hold,
        state.weights,
    )
    return compute_spn_likelihood(spn, state)


def gaussian_inference(val, state, normalizing_constant, posterior_mixture):
    normalized_posterior = []
    min_degree = float("inf")
    for weight, mu, cov in posterior_mixture:
        degree = get_degree(weight)
        if degree > min_degree:
            continue
        elif degree < min_degree and weight > 0:
            # New min degree, clear out old weights
            normalized_posterior.clear()
            min_degree = degree
        v = np.zeros((1, state.gaussian_count))
        b = 0.0
        for var in val.rvs:
            assert not isinstance(var, RealConstant)
            v[0, var.var - 1] = var.scale
            b += var.shift
        expr_mu, expr_cov = get_expr_distribution(mu, cov, v, b)
        normalized_posterior.append(
            (weight / normalizing_constant, (expr_mu, expr_cov))
        )
    return normalized_posterior, normalizing_constant


def enum_inference(val: EnumResult, state: KCState, normalizing_constant: float):
    probs: dict[str, float] = {}
    for i, enum_str in enumerate(val.enum_type.values):
        constraint = state.bdd.true
        for bit_i in range(val.enum_type.n_bits):
            bit_val = bool((i >> bit_i) & 1)
            expected_bdd = state.bdd.true if bit_val else state.bdd.false
            bits_eq = (val.bits[bit_i] & expected_bdd) | (
                ~val.bits[bit_i] & ~expected_bdd
            )
            constraint = constraint & bits_eq

        spn = model_count(
            state.bdd,
            constraint & state.observes_all_hold,
            state.weights,
        )
        unnormalized_prob = compute_spn_likelihood(spn, state)
        probs[enum_str] = unnormalized_prob / normalizing_constant
    return probs


def run_kc(expr: PExpr):
    preprocess_state = preprocess(expr)
    val, state = kc(expr, preprocess_state)
    normalizing_constant = get_normalizing_constant(state)

    if normalizing_constant == 0:
        return None, normalizing_constant

    if isinstance(val, GaussianVariable):
        val = GaussianSum(frozenset({val}))

    # Inference for binary variable
    if isinstance(val, _bdd.Function):
        return binary_inference(val, state, normalizing_constant), normalizing_constant
    elif isinstance(val, EnumResult):
        return enum_inference(val, state, normalizing_constant), normalizing_constant
    else:
        raise TypeError(f"Cannot perform inference for value of type {type(val)}")
