from kc.observation_weights import FullPosterior, GradedLikelihood
import dd.autoref as _bdd
import numpy as np

from kc.base import PExpr
from kc.config import settings
from kc.model_count import model_count
from kc.real_values import GaussianSum, GaussianVariable, BetaVariable
from kc.spn import Node
from kc.state import KCState, PreprocessState
from kc.terms import EnumResult


def compute_spn_likelihood(spn: Node, state: KCState) -> float:
    graded_log_likelihood = spn.get_log_likelihood(state.beta_priors)
    if graded_log_likelihood is None:
        return 0.0
    return np.exp(graded_log_likelihood.log_likelihood)


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
    return compute_spn_likelihood(spn, state) / normalizing_constant


def normalize_posterior(posterior: list[FullPosterior], normalizing_constant: float):
    normalized_posterior = []
    n_obs = min(c.likelihood.n_obs for c in posterior)
    for component in posterior:
        if component.likelihood.n_obs > n_obs:
            continue
        component.likelihood = GradedLikelihood(
            component.likelihood.log_likelihood - np.log(normalizing_constant), n_obs
        )
        normalized_posterior.append(component)
    return normalized_posterior


def gaussian_inference(val: GaussianSum, state: KCState, normalizing_constant: float):
    spn = model_count(state.bdd, state.observes_all_hold, state.weights)
    posterior = spn.get_posterior(
        [v.var for v in val.rvs if isinstance(v, GaussianVariable)],
        beta_priors=state.beta_priors,
    )
    return normalize_posterior(posterior, normalizing_constant)


def beta_inference(val: BetaVariable, state: KCState, normalizing_constant: float):
    spn = model_count(state.bdd, state.observes_all_hold, state.weights)
    posterior = spn.get_posterior(
        [val.var],
        beta_priors=state.beta_priors,
    )
    return normalize_posterior(posterior, normalizing_constant)


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
    elif isinstance(val, GaussianSum):
        return gaussian_inference(
            val, state, normalizing_constant
        ), normalizing_constant
    elif isinstance(val, BetaVariable):
        return beta_inference(val, state, normalizing_constant), normalizing_constant
    else:
        raise TypeError(f"Cannot perform inference for value of type {type(val)}")
