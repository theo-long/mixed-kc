from collections import defaultdict
from functools import reduce
import math
import operator
import random
from abc import ABC
from dataclasses import dataclass
import itertools as it
from scipy.stats import norm

import dd.autoref as _bdd

from kc.model_count import model_count


class KCState:
    def __init__(self):
        self.bdd = _bdd.BDD()
        self.flips = 0
        self.gaussians = 0
        self.weights = {}
        self.gaussian_params = {}
        self.observes_all_hold = self.bdd.true

        # We track observes associated with each GaussianVariable
        # This allows us to check if the observes are mutually compatible
        self.gaussian_observes = defaultdict(set)

    @property
    def gaussian_observes_mutually_compatible(self):
        # This is a big AND over all the mutually incompatible pairs of observes
        # future cases to handle:
        # - what if we have two different GaussianVariables that are somehow related?
        # - what if we have non-equality observes e.g. x < 0.5?
        clause = self.bdd.true
        for gaussian_variables in self.gaussian_observes.values():
            for observe_pair in it.combinations(gaussian_variables, 2):
                # Currently, we only support gaussian equality observes
                # therefore any two observes that are not equal are mutually incompatible
                clause = clause & ~(observe_pair[0] & observe_pair[1])
        return clause

    def next_flip(self):
        self.flips += 1
        return self.flips

    def next_gaussian(self):
        self.gaussians += 1
        return self.gaussians

    def set_gaussian_params(self, var, mu, sigma):
        self.gaussian_params[var] = (mu, sigma)

    def set_weight(self, var, pos_weight, neg_weight):
        self.weights[var] = (pos_weight, neg_weight)


class RealValue(ABC):
    pass


@dataclass
class GaussianVariable(RealValue):
    var: int


@dataclass
class GaussianUnion(RealValue):
    formulae: list[any]
    values: list[GaussianVariable]


def gaussian_density(mean, std, val):
    return norm.pdf(val, loc=mean, scale=std)


def merge_real_values(cond, t, f):
    """
    Merge two RealValues (t and f) based on condition cond.
    Returns a GaussianUnion with BDDs and deduplicated GaussianVariables.
    Each formula[i] guards the corresponding values[i].
    When the same value appears with different guards, the guards are ORed together.
    """
    # Extract formulae and values from t (then branch)
    if isinstance(t, GaussianVariable):
        t_formulae = [cond]
        t_values = [t]
    elif isinstance(t, GaussianUnion):
        # AND each formula with cond, ensuring formulae and values are aligned
        t_formulae = [cond & formula for formula in t.formulae]
        t_values = t.values
    else:
        raise TypeError(f"Unexpected type for t: {type(t)}")

    # Extract formulae and values from f (else branch)
    if isinstance(f, GaussianVariable):
        f_formulae = [~cond]
        f_values = [f]
    elif isinstance(f, GaussianUnion):
        # AND each formula with ~cond, ensuring formulae and values are aligned
        f_formulae = [~cond & formula for formula in f.formulae]
        f_values = f.values
    else:
        raise TypeError(f"Unexpected type for f: {type(f)}")

    # Build a map from var -> list of guards (formulae) for that variable
    # This allows us to OR together guards for the same variable
    var_to_guards = {}
    var_to_gaussian = {}

    # Process t branch: formulae and values are aligned
    for formula, gv in zip(t_formulae, t_values):
        var = gv.var
        if var not in var_to_guards:
            var_to_guards[var] = []
            var_to_gaussian[var] = gv
        var_to_guards[var].append(formula)

    # Process f branch: formulae and values are aligned
    for formula, gv in zip(f_formulae, f_values):
        var = gv.var
        if var not in var_to_guards:
            var_to_guards[var] = []
            var_to_gaussian[var] = gv
        var_to_guards[var].append(formula)

    # Build aligned lists: OR together guards for each unique variable
    all_formulae = []
    all_values = []
    for var, guards in var_to_guards.items():
        # OR all guards together for this variable
        if len(guards) == 1:
            combined_guard = guards[0]
        else:
            # OR all guards together
            combined_guard = guards[0]
            for guard in guards[1:]:
                combined_guard = combined_guard | guard
        all_formulae.append(combined_guard)
        all_values.append(var_to_gaussian[var])

    return GaussianUnion(formulae=all_formulae, values=all_values)


class PExpr(ABC):
    pass


class AExpr(PExpr):
    pass


@dataclass
class Const(AExpr):
    val: bool

    def sample(self, env):
        return self.val

    def pmf(self, env, v):
        if self.val == v:
            return 1.0
        else:
            return 0.0

    def kc(self, env, state):
        if self.val:
            return state.bdd.true
        else:
            return state.bdd.false


@dataclass
class Gaussian(AExpr):
    mean: float
    std: float

    def kc(self, env, state):
        var = state.next_gaussian()
        state.set_gaussian_params(var, self.mean, self.std)
        return GaussianVariable(var)


@dataclass
class Var(AExpr):
    var: str

    def sample(self, env):
        return env[self.var]

    def pmf(self, env, v):
        if env[self.var] == v:
            return 1.0
        else:
            return 0.0

    def kc(self, env, state):
        return env[self.var]


@dataclass
class Flip(PExpr):
    prob: float

    def sample(self, env):
        return random.random() < self.prob

    def pmf(self, env, v):
        if v:
            return self.prob
        else:
            return 1 - self.prob

    def kc(self, env, state):
        flip_id = state.next_flip()
        state.bdd.declare(f"flip_{flip_id}")
        state.set_weight(f"flip_{flip_id}", self.prob, 1.0 - self.prob)
        return state.bdd.var(f"flip_{flip_id}")


@dataclass
class IfThenElse(PExpr):
    cond: AExpr
    then_expr: PExpr
    else_expr: PExpr

    def sample(self, env):
        if self.cond.sample(env):
            return self.then_expr.sample(env)
        else:
            return self.else_expr.sample(env)

    def pmf(self, env, v):
        # Because `cond` is an `AExpr`, we know that it
        # is either deterministically true or deterministically false.
        if self.cond.sample(env):
            return self.then_expr.pmf(env, v)
        else:
            return self.else_expr.pmf(env, v)

        # prob_true = self.cond.pmf(env, True)
        # prob_false = self.cond.pmf(env, False)
        # return prob_true * self.then_expr.pmf(env, v) + prob_false * self.else_expr.pmf(
        #     env, v
        # )

    def kc(self, env, state):
        condition_bdd = self.cond.kc(env, state)
        then_result = self.then_expr.kc(env, state)
        else_result = self.else_expr.kc(env, state)

        if isinstance(then_result, RealValue):
            return merge_real_values(condition_bdd, then_result, else_result)
        else:
            return (condition_bdd & then_result) | (~condition_bdd & else_result)


def extend_env(env, extension):
    new_env = env.copy()
    new_env.update(extension)
    return new_env


@dataclass
class Let(PExpr):
    var: str
    binding: PExpr
    body: PExpr

    def sample(self, env):
        new_env = extend_env(env, {self.var: self.binding.sample(env)})
        return self.body.sample(new_env)

    def pmf(self, env, v):
        prob_that_var_is_true = self.binding.pmf(env, True)
        prob_that_var_is_false = self.binding.pmf(env, False)
        # without `observe`, could do: 1 - prob_that_var_is_true

        answer_with_var_true = self.body.pmf(extend_env(env, {self.var: True}), v)
        answer_with_var_false = self.body.pmf(extend_env(env, {self.var: False}), v)

        return (
            prob_that_var_is_true * answer_with_var_true
            + prob_that_var_is_false * answer_with_var_false
        )

    def kc(self, env, state):
        new_env = extend_env(env, {self.var: self.binding.kc(env, state)})
        return self.body.kc(new_env, state)


class Rejection(Exception):
    pass


@dataclass
class Observe(PExpr):
    cond: PExpr

    def sample(self, env):
        if self.cond.sample(env):
            return True
        else:
            raise Rejection()

    def pmf(self, env, v):
        if v:
            return self.cond.pmf(env, True)
        else:
            return 0.0

    def kc(self, env, state):
        state.observes_all_hold = state.observes_all_hold & self.cond.kc(env, state)
        return state.bdd.true


@dataclass
class ObserveReal(PExpr):
    val: float
    symbolic_value: PExpr

    def kc(self, env, state: KCState):
        # Modify the self.observes_all_hold formula in some way...
        # We want something like score(density(symbolic_value, val)),
        #  where this density depends on which GaussianVariable symbolic_value
        symbolic_value = self.symbolic_value.kc(env, state)
        if isinstance(symbolic_value, GaussianVariable):
            score_node_name = f"{symbolic_value.var}={self.val}"
            state.bdd.declare(score_node_name)
            score_node = state.bdd.var(score_node_name)
            state.gaussian_observes[symbolic_value.var].add(score_node)
            mean, std = state.gaussian_params[symbolic_value.var]
            density = gaussian_density(mean, std, self.val)
            state.set_weight(score_node_name, density, 1.0)
            state.observes_all_hold = state.observes_all_hold & score_node
            return

        # Otherwise, we have a union
        #   Create a score node for each possibility in the union.
        #    observes_all_hold will be extended with a big "or", each clause of which
        #    "ands" together:
        #      - the fact that this score node is true
        #      - the formula guarding this value
        #      - the fact that all the other score nodes are false
        #    the weight of each score node will be the corresponding Gaussian density.
        score_nodes = []
        for v in symbolic_value.values:
            #  create a score node for each possibility in the union.
            score_node_name = f"{v.var}={self.val}"
            state.bdd.declare(score_node_name)
            score_node = state.bdd.var(score_node_name)
            state.gaussian_observes[v.var].add(score_node)
            score_nodes.append(score_node)
            mean, std = state.gaussian_params[v.var]
            density = gaussian_density(mean, std, self.val)
            state.set_weight(score_node_name, density, 1.0)

        clauses = []
        for i in range(len(score_nodes)):
            # this score node is true and all the other score nodes are false
            clause = score_nodes[i] & reduce(
                operator.and_, [~x for x in score_nodes[:i] + score_nodes[i + 1 :]]
            )
            # Add formula guarding this value to the clause
            clause = clause & symbolic_value.formulae[i]
            clauses.append(clause)

        # We OR together all the clauses and AND it with observes_all_hold
        clause = reduce(operator.or_, clauses)
        state.observes_all_hold = state.observes_all_hold & clause

        # Things to think about:
        #   When multiple observes talk about potentially the same GaussianVariable
        #     ObserveReal(3.0, Var("x"))
        #     ObserveReal(6.0, (Mul(2.0, Var("x"))))
        #     where Var("x") refers to a GaussianVariable(i).
        #       or where Var("x") is a union
        #     This is related to the question in Pun that we saw of density of [x, 2x] at [3, 6] when x ~ N(0, 1)?

        return state.bdd.true


def rejection_sample(expr):
    num_rej = 0
    val = None
    while True:
        try:
            val = expr.sample({})
            break
        except Rejection:
            num_rej += 1
            pass
    return (val, num_rej)


def posterior_pmf(expr, v):
    pmf_true = expr.pmf({}, True)
    pmf_false = expr.pmf({}, False)
    if v:
        return pmf_true / (pmf_true + pmf_false)
    else:
        return pmf_false / (pmf_true + pmf_false)


def run_kc(expr):
    state = KCState()
    bdd = expr.kc({}, state)
    unnormalized_count = model_count(
        state.bdd, bdd & state.observes_all_hold, state.weights
    )
    normalizing_constant = model_count(
        state.bdd, state.observes_all_hold, state.weights
    )
    return (unnormalized_count / normalizing_constant, normalizing_constant)
