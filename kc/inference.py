import dd.autoref as _bdd
import numpy as np
import scipy.linalg

from kc.base import PExpr
from kc.config import settings
from kc.gaussian_math import get_expr_distribution
from kc.model_count import model_count
from kc.observation_weights import FullPosterior, GaussianPosterior, GradedLikelihood
from kc.real_values import BetaVariable, GaussianSum, GaussianVariable, Union
from kc.spn import Node
from kc.state import KCState, PreprocessState
from kc.terms import EnumResult


def compute_spn_likelihood(spn: Node, state: KCState) -> float:
    graded_log_likelihood = spn.get_log_likelihood(state.beta_priors, state.dp_priors)
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
    normalized_posterior: list[FullPosterior] = []
    n_obs = min(c.likelihood.n_obs for c in posterior)
    for component in posterior:
        if component.likelihood.n_obs > n_obs:
            continue
        component.likelihood = GradedLikelihood(
            component.likelihood.log_likelihood - np.log(normalizing_constant), n_obs
        )
        normalized_posterior.append(component)
    return normalized_posterior


def _fill_in_missing_gaussian_vars(rvs: list[int], posterior: FullPosterior):
    """Add in gaussian vars left implicit that have no correlation with others."""
    missing_vars = [i for i in rvs if i not in posterior.gaussian.scope]
    mu = np.concat([np.zeros((len(missing_vars), 1)), posterior.gaussian.mu])
    cov = scipy.linalg.block_diag(np.eye(len(missing_vars)), posterior.gaussian.cov)
    desired_order = [rvs.index(var) for var in missing_vars + posterior.gaussian.scope]
    mu = mu[desired_order, :]
    cov = cov[np.ix_(desired_order, desired_order)]
    return mu, cov


def get_gaussian_sum_posterior(
    val: GaussianSum, spn: Node, state: KCState
) -> list[FullPosterior]:
    rvs: list[int] = []
    scales: list[float] = []
    shifts: list[float] = []
    for rv in val.rvs:
        if isinstance(rv, GaussianVariable):
            rvs.append(rv.var)
            scales.append(rv.scale)
            shifts.append(rv.shift)
        else:
            shifts.append(rv.value)
    v = np.array(scales)[None, :]
    b = sum(shifts)
    joint_posterior = spn.get_posterior(
        rvs,
        beta_priors=state.beta_priors,
        dp_priors=state.dp_priors,
    )
    posterior: list[FullPosterior] = []
    for component in joint_posterior:
        if set(component.gaussian.scope) != set(rvs):
            mu, cov = _fill_in_missing_gaussian_vars(rvs, component)
        else:
            mu, cov = component.gaussian.mu, component.gaussian.cov
            assert rvs == component.gaussian.scope, (
                "Scope and rvs should be in the same order"
            )

        scale, shift = get_expr_distribution(mu, cov, v, b)
        posterior.append(
            FullPosterior(
                likelihood=component.likelihood,
                gaussian=GaussianPosterior(rvs, scale, shift),
            )
        )
    return posterior


def gaussian_inference(val: GaussianSum, state: KCState, normalizing_constant: float):
    spn = model_count(state.bdd, state.observes_all_hold, state.weights)
    posterior = get_gaussian_sum_posterior(val, spn, state)
    return normalize_posterior(posterior, normalizing_constant)


def beta_inference(val: BetaVariable, state: KCState, normalizing_constant: float):
    spn = model_count(state.bdd, state.observes_all_hold, state.weights)
    posterior = spn.get_posterior(
        [val.var],
        beta_priors=state.beta_priors,
        dp_priors=state.dp_priors,
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


def union_inference(val: Union, state: KCState, normalizing_constant: float):
    posterior = []
    for formula, value in zip(val.formulae, val.values):
        spn = model_count(
            state.bdd,
            formula & state.observes_all_hold,
            state.weights,
        )
        if isinstance(value, GaussianSum):
            guarded_posterior = get_gaussian_sum_posterior(value, spn, state)
        elif isinstance(value, (BetaVariable, GaussianVariable)):
            guarded_posterior = spn.get_posterior(
                [value.var],
                beta_priors=state.beta_priors,
                dp_priors=state.dp_priors,
            )
        else:
            raise TypeError(
                f"Cannot perform inference on value in union of type {type(value)}"
            )
        for component in guarded_posterior:
            new_likelihood = GradedLikelihood(
                component.likelihood.log_likelihood - np.log(normalizing_constant),
                component.likelihood.n_obs,
            )
            component.likelihood = new_likelihood
        posterior.extend(guarded_posterior)
    return posterior


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
    elif isinstance(val, Union):
        return union_inference(val, state, normalizing_constant), normalizing_constant
    else:
        raise TypeError(f"Cannot perform inference for value of type {type(val)}")


def get_spn(expr: PExpr):
    preprocess_state = preprocess(expr)
    val, state = kc(expr, preprocess_state)
    return model_count(state.bdd, state.observes_all_hold, state.weights)
