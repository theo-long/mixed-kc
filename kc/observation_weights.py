from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Self

import numpy as np
import scipy.linalg
from scipy.special import betaln

from kc.gaussian_math import get_gaussian_posterior, log_score_singular


def _build_gaussian_observation_matrix(
    scope_map: dict[int, int],
    gaussian_obs_coefficients: list[dict[int, float]],
    gaussian_obs_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((len(gaussian_obs_coefficients), len(scope_map)))
    b = np.array(gaussian_obs_values)

    for i, vector in enumerate(gaussian_obs_coefficients):
        for gaussian_var, val in vector.items():
            j = scope_map[gaussian_var]
            A[i, j] = val

    return A, b


@dataclass(frozen=True)
class GradedLikelihood:
    log_likelihood: float
    n_obs: int

    def __add__(self, other: "GradedLikelihood | None"):
        if other is None or self.n_obs < other.n_obs:
            return self
        elif other.n_obs < self.n_obs:
            return other
        else:
            log_likelihood = np.logaddexp(
                self.log_likelihood,
                other.log_likelihood,
            ).item()
            return GradedLikelihood(log_likelihood, self.n_obs)

    def __radd__(self, other: "GradedLikelihood | None"):
        return self.__add__(other)

    def __mul__(self, other: "GradedLikelihood"):
        log_likelihood = self.log_likelihood + other.log_likelihood
        return GradedLikelihood(log_likelihood, self.n_obs + other.n_obs)

    def __rmul__(self, other: "GradedLikelihood"):
        return other.__mul__(self)


class WeightType(ABC):
    @property
    def scope(self) -> set[int]:
        return set()

    @abstractmethod
    def __mul__(self: Self, other) -> Self: ...

    @abstractmethod
    def get_log_likelihood(self, **kwargs) -> tuple[float | None, int]: ...

    @abstractmethod
    def get_posterior(self, var_selection: list[int], **kwargs) -> "Posterior": ...


@dataclass
class Posterior(ABC):
    scope: list[int] = field(default_factory=list)


@dataclass
class GaussianPosterior(Posterior):
    mu: np.typing.NDArray = field(default_factory=lambda: np.zeros((0, 1)))
    cov: np.typing.NDArray = field(default_factory=lambda: np.zeros((0, 0)))

    def __mul__(self, other: "GaussianPosterior"):
        assert not (set(self.scope) & set(other.scope)), (
            "Cannot combine posteriors with overlapping scopes"
        )
        mu = np.concatenate((self.mu, other.mu), axis=0)
        if self.cov.size == 0:
            cov = other.cov
        elif other.cov.size == 0:
            cov = self.cov
        else:
            cov = scipy.linalg.block_diag(self.cov, other.cov)
        return GaussianPosterior(self.scope + other.scope, mu, cov)


@dataclass
class GaussianWeight(WeightType):
    gaussian_obs_coefficients: list[dict[int, float]] = field(default_factory=list)
    gaussian_obs_values: list[float] = field(default_factory=list)

    @property
    def scope(self) -> set[int]:
        scope = set()
        for obs_vector in self.gaussian_obs_coefficients:
            scope |= obs_vector.keys()
        return scope

    def __str__(self) -> str:
        return f"n_gaussian_obs={len(self.gaussian_obs_coefficients)}"

    def __mul__(self, other: "GaussianWeight"):
        return GaussianWeight(
            self.gaussian_obs_coefficients + other.gaussian_obs_coefficients,
            self.gaussian_obs_values + other.gaussian_obs_values,
        )

    def get_log_likelihood(self, **kwargs):
        if len(self.gaussian_obs_values) + len(self.gaussian_obs_coefficients) == 0:
            return (0.0, 0)
        scope_map = {s: i for i, s in enumerate(self.scope)}
        A, b = _build_gaussian_observation_matrix(
            scope_map,
            self.gaussian_obs_coefficients,
            self.gaussian_obs_values,
        )
        log_score = log_score_singular(
            np.zeros((len(scope_map), 1)), np.eye(len(scope_map)), A, b
        )
        if log_score is None:
            return None, 0
        return log_score, np.linalg.matrix_rank(A)

    def get_posterior(self, var_selection: list[int], **kwargs) -> GaussianPosterior:
        assert var_selection, "Cannot query empty set of vars"
        assert all(var in self.scope for var in var_selection), (
            "Can only query in-scope vars"
        )

        scope_map = {s: i for i, s in enumerate(self.scope)}
        A, b = _build_gaussian_observation_matrix(
            scope_map,
            self.gaussian_obs_coefficients,
            self.gaussian_obs_values,
        )
        mu, cov = get_gaussian_posterior(
            np.zeros((len(scope_map), 1)), np.eye(len(scope_map)), A, b
        )
        indices = []
        for var in var_selection:
            indices.append(scope_map[var])
        return GaussianPosterior(
            var_selection,
            mu[indices],
            cov[indices, indices],
        )


@dataclass
class BetaPosterior(Posterior):
    alphas: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)

    def __mul__(self, other: "BetaPosterior"):
        assert not (set(self.scope) & set(other.scope)), (
            "Cannot combine posteriors with overlapping scopes"
        )
        return BetaPosterior(
            self.scope + other.scope,
            self.alphas + other.alphas,
            self.betas + other.betas,
        )


@dataclass
class BetaWeight(WeightType):
    beta_counts: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def scope(self) -> set[int]:
        return set(self.beta_counts)

    def __str__(self) -> str:
        return f"n_beta_obs={len(self.beta_counts)}"

    def __mul__(self, other: "BetaWeight"):
        beta_counts = dict(self.beta_counts)
        for var, (other_true_count, other_false_count) in other.beta_counts.items():
            true_count, false_count = beta_counts.get(var, (0, 0))
            beta_counts[var] = (
                true_count + other_true_count,
                false_count + other_false_count,
            )
        return BetaWeight(beta_counts)

    def get_log_likelihood(self, **kwargs):
        if not self.beta_counts:
            return 0.0, 0
        log_likelihood = 0
        beta_priors: dict[int, tuple[float, float]] = kwargs["beta_priors"]
        for var, (s, f) in self.beta_counts.items():
            alpha, beta = beta_priors[var]

            # Log of the Beta ratio (Probability of this specific sequence of flips)
            log_likelihood += betaln(s + alpha, f + beta) - betaln(alpha, beta)

        return log_likelihood, 0

    def get_posterior(self, var_selection: list[int], **kwargs) -> BetaPosterior:
        assert var_selection, "Cannot query empty set of vars"
        assert all(var in self.scope for var in var_selection), (
            "Can only query in-scope vars"
        )
        priors: dict[int, tuple[float, float]] = kwargs["beta_priors"]
        alphas: list[float] = []
        betas: list[float] = []
        for var in var_selection:
            alpha, beta = priors[var]
            succ, fail = self.beta_counts[var]
            alphas.append(alpha + succ)
            betas.append(beta + fail)

        return BetaPosterior(var_selection, alphas, betas)


# Note - conflicting TruncatedGaussian obs are handled inside the KCState class by directly constraining the BDD
# This means we don't need to keep track of interactions and can simply directly compute likelihoods
@dataclass
class TruncatedGaussianWeight(WeightType):
    n_obs: int = 0

    @property
    def scope(self) -> set[int]:
        return set()

    def __mul__(self, other: "TruncatedGaussianWeight"):
        return TruncatedGaussianWeight(self.n_obs + other.n_obs)

    def __add__(self, other: "TruncatedGaussianWeight"):
        return TruncatedGaussianWeight(self.n_obs + other.n_obs)

    def get_log_likelihood(self, **kwargs):
        return 0.0, self.n_obs

    def get_posterior(self, var_selection: list[int], **kwargs) -> Posterior:
        raise NotImplementedError(
            "No posterior inference for Truncated Gaussian variables"
        )


@dataclass
class FullPosterior:
    likelihood: GradedLikelihood
    gaussian: GaussianPosterior = field(default_factory=GaussianPosterior)
    beta: BetaPosterior = field(default_factory=BetaPosterior)

    @property
    def scope(self):
        return self.gaussian.scope + self.beta.scope

    def __mul__(self, other: "FullPosterior"):
        assert not (set(self.scope) & set(other.scope)), (
            "Cannot combine posteriors with overlapping scopes"
        )
        return FullPosterior(
            self.likelihood * other.likelihood,
            self.gaussian * other.gaussian,
            self.beta * other.beta,
        )


@dataclass
class ObservationWeights:
    likelihood: float = 1.0
    gaussian_obs: GaussianWeight = field(default_factory=GaussianWeight)
    beta_obs: BetaWeight = field(default_factory=BetaWeight)
    truncated_gaussian_obs: TruncatedGaussianWeight = field(
        default_factory=TruncatedGaussianWeight
    )

    @property
    def scope(self) -> set[int]:
        scope = set()
        for dataclass_field in fields(self):
            weight = getattr(self, dataclass_field.name)
            if isinstance(weight, WeightType):
                scope |= weight.scope
        return scope

    @classmethod
    def from_weight(cls, weight: int | float | WeightType) -> "ObservationWeights":
        if isinstance(weight, int | float):
            return cls(likelihood=weight)
        elif isinstance(weight, GaussianWeight):
            return cls(gaussian_obs=weight)
        elif isinstance(weight, BetaWeight):
            return cls(beta_obs=weight)
        elif isinstance(weight, TruncatedGaussianWeight):
            return cls(truncated_gaussian_obs=weight)
        else:
            raise TypeError(f"Unrecognized weight type {type(weight)}")

    def __str__(self):
        rep_str = f"Obs(scope={self.scope}, likelihood={self.likelihood}"
        for dataclass_field in fields(self):
            weight = getattr(self, dataclass_field.name)
            if isinstance(weight, WeightType):
                rep_str += ", "
                rep_str += str(weight)
        return rep_str + ")"

    def __mul__(self, other: "ObservationWeights") -> "ObservationWeights":
        return ObservationWeights(
            self.likelihood * other.likelihood,
            self.gaussian_obs * other.gaussian_obs,
            self.beta_obs * other.beta_obs,
            self.truncated_gaussian_obs * other.truncated_gaussian_obs,
        )

    def __add__(self, other: "ObservationWeights"):
        if len(self.scope) + len(other.scope) == 0:
            # No gaussian or beta obs
            return ObservationWeights(
                self.likelihood + other.likelihood,
                truncated_gaussian_obs=self.truncated_gaussian_obs
                + other.truncated_gaussian_obs,
            )
        else:
            # TODO: technically we could if everything had *the same* set of observations (or equivalent set)
            raise ValueError("Cannot add weights with observations")

    def get_log_likelihood(self, **kwargs) -> GradedLikelihood | None:
        if self.likelihood == 0:
            return None

        log_likelihood, n_obs = np.log(self.likelihood).item(), 0
        for dataclass_field in fields(self):
            weight = getattr(self, dataclass_field.name)
            if isinstance(weight, WeightType):
                log_likelihood_update, obs_update = weight.get_log_likelihood(**kwargs)
                if log_likelihood_update is None:
                    return None
                log_likelihood += log_likelihood_update
                n_obs += obs_update

        return GradedLikelihood(log_likelihood, n_obs)

    def get_posterior(self, var_selection: list[int], **kwargs) -> FullPosterior | None:
        likelihood = self.get_log_likelihood(**kwargs)
        if likelihood is None:
            return likelihood

        var_selection_set = set(var_selection)

        gaussian_vars = self.gaussian_obs.scope & var_selection_set
        if gaussian_vars:
            gaussian_posterior = self.gaussian_obs.get_posterior(
                list(gaussian_vars), **kwargs
            )
        else:
            gaussian_posterior = GaussianPosterior()

        beta_vars = self.beta_obs.scope & var_selection_set
        if beta_vars:
            beta_posterior = self.beta_obs.get_posterior(list(beta_vars), **kwargs)
        else:
            beta_posterior = BetaPosterior()

        # Verify that we have queried all vars
        assert var_selection_set == gaussian_vars | beta_vars, (
            "Must get posterior for all vars"
        )

        return FullPosterior(likelihood, gaussian_posterior, beta_posterior)
