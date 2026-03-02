from dataclasses import dataclass, field

import numpy as np
from scipy.special import betaln

from kc.gaussian_math import log_score_singular


def _build_gaussian_observation_matrix(
    dim: int,
    scope_map: dict[int, int],
    gaussian_obs_coefficients: list[dict[int, float]],
    gaussian_obs_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((len(gaussian_obs_coefficients), dim))
    b = np.array(gaussian_obs_values)

    for i, vector in enumerate(gaussian_obs_coefficients):
        for gaussian_var, val in vector.items():
            j = scope_map[gaussian_var]
            A[i, j] = val

    return A, b


def _get_gaussian_observation_likelihood_update(observation: "ObservationWeights"):
    assert observation.gaussian_obs_values
    assert observation.gaussian_obs_coefficients
    scope_map = {s: i for i, s in enumerate(observation.scope)}
    dim = len(observation.scope)
    A, b = _build_gaussian_observation_matrix(
        dim,
        scope_map,
        observation.gaussian_obs_coefficients,
        observation.gaussian_obs_values,
    )
    log_score = log_score_singular(np.zeros((dim, 1)), np.eye(dim), A, b)
    if log_score is None:
        return None, 0
    return log_score, np.linalg.matrix_rank(A)


def _get_beta_observation_likelihood_update(
    observation: "ObservationWeights", beta_priors: dict[int, tuple[float, float]]
) -> float:
    log_likelihood = 0
    for var, (s, f) in observation.beta_counts.items():
        alpha, beta = beta_priors[var]

        # Log of the Beta ratio (Probability of this specific sequence of flips)
        log_likelihood += betaln(s + alpha, f + beta) - betaln(alpha, beta)

    return log_likelihood


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
                self.log_likelihood,  # type: ignore
                other.log_likelihood,  # type: ignore
            ).item()
            return GradedLikelihood(log_likelihood, self.n_obs)

    def __radd__(self, other: "GradedLikelihood | None"):
        return self.__add__(other)

    def __mul__(self, other: "GradedLikelihood"):
        log_likelihood = self.log_likelihood + other.log_likelihood  # type: ignore
        return GradedLikelihood(log_likelihood, self.n_obs + other.n_obs)

    def __rmul__(self, other: "GradedLikelihood"):
        return other.__mul__(self)


@dataclass
class ObservationWeights:
    likelihood: float
    gaussian_obs_coefficients: list[dict[int, float]] = field(default_factory=list)
    gaussian_obs_values: list[float] = field(default_factory=list)
    beta_counts: dict[int, tuple[int, int]] = field(default_factory=dict)
    truncated_gaussian_obs: int = 0

    @property
    def scope(self) -> set[int]:
        scope = set()
        for obs_vector in self.gaussian_obs_coefficients:
            scope |= obs_vector.keys()
        scope |= self.beta_counts.keys()
        return scope

    def __str__(self):
        return f"Obs(likelihood={self.likelihood}, beta_counts={self.beta_counts}, n_gaussian_obs={len(self.gaussian_obs_coefficients)}, trunc_obs={self.truncated_gaussian_obs}, scope={self.scope})"

    def __mul__(self, other: "ObservationWeights") -> "ObservationWeights":
        beta_counts = dict(self.beta_counts)
        for var, (other_true_count, other_false_count) in other.beta_counts.items():
            true_count, false_count = beta_counts.get(var, (0, 0))
            beta_counts[var] = (
                true_count + other_true_count,
                false_count + other_false_count,
            )
        return ObservationWeights(
            self.likelihood * other.likelihood,  # type: ignore
            self.gaussian_obs_coefficients + other.gaussian_obs_coefficients,
            self.gaussian_obs_values + other.gaussian_obs_values,
            beta_counts,
            self.truncated_gaussian_obs + other.truncated_gaussian_obs,
        )

    def __add__(self, other: "ObservationWeights"):
        if len(self.scope) + len(other.scope) == 0:
            # Prefer the one with fewer obs *if* it has likelihood > 0
            if (
                self.truncated_gaussian_obs < other.truncated_gaussian_obs
                and self.likelihood
            ):
                return self
            elif (
                other.truncated_gaussian_obs < self.truncated_gaussian_obs
                and other.likelihood
            ):
                return other
            else:
                return ObservationWeights(
                    self.likelihood + other.likelihood,  # type: ignore
                    truncated_gaussian_obs=self.truncated_gaussian_obs,
                )
        else:
            # TODO: technically we could if everything had *the same* set of observations (or equivalent set)
            raise ValueError("Cannot add weights with observations")

    def _get_observation_likelihood(
        self, beta_priors: dict[int, tuple[float, float]]
    ) -> GradedLikelihood | None:
        if self.likelihood == 0:
            return None
        log_likelihood = np.log(self.likelihood).item()  # type: ignore
        n_obs = self.truncated_gaussian_obs
        if self.gaussian_obs_coefficients:
            log_score, new_obs = _get_gaussian_observation_likelihood_update(self)
            if log_score is None:
                return None
            log_likelihood += log_score
            n_obs += new_obs

        if self.beta_counts:
            log_score = _get_beta_observation_likelihood_update(self, beta_priors)
            log_likelihood += log_score

        return GradedLikelihood(log_likelihood, n_obs)
