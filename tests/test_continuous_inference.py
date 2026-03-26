from turtle import pos

import numpy as np

from kc import (
    Affine,
    Beta,
    Const,
    Flip,
    Gaussian,
    IfThenElse,
    Let,
    Observe,
    ObserveReal,
    Sum,
    Var,
    run_kc,
)


def test_basic_beta_posterior():
    # Model: b ~ Beta(2, 2)
    # f1 ~ Flip(b)
    # f2 ~ Flip(b)
    # Observe f1 and f2 are True
    # Posterior should be Beta(2+2, 2+0) => Beta(4, 2)
    p = Let(
        "b",
        Beta(2, 2),
        Let(
            "f1",
            Flip(Var("b")),
            Let(
                "f2",
                Flip(Var("b")),
                Let(
                    "obs",
                    Observe(IfThenElse(Var("f1"), Var("f2"), Const(False))),
                    Var("b"),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)
    assert isinstance(posterior, list)
    assert posterior[0].beta.alphas[0] == 4
    assert posterior[0].beta.betas[0] == 2


def test_beta_posterior():
    p = Let(
        "b1",
        Beta(1, 5),
        Let(
            "b2",
            Beta(5, 1),
            Let(
                "f1",
                Flip(Var("b1")),
                Let(
                    "f2",
                    Flip(Var("b2")),
                    Let(
                        "x",
                        IfThenElse(
                            Var("f1"),
                            IfThenElse(
                                Var("f2"),
                                Flip(Var("b1")),
                                Flip(Var("b2")),
                            ),
                            IfThenElse(
                                Var("f2"),
                                Flip(Var("b1")),
                                Const(True),
                            ),
                        ),
                        Let("_", Observe(Var("x")), Var("b1")),
                    ),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)
    assert posterior


def test_simple_gaussian_posterior():
    p = Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let("_", ObserveReal(Sum(Var("g1"), Var("g2")), 1.0), Var("g1")),
        ),
    )
    posterior, Z = run_kc(p)
    assert isinstance(posterior, list)
    assert np.allclose(posterior[0].gaussian.mu, 0.5)
    assert np.allclose(posterior[0].gaussian.cov, 0.5)


def test_mixture_gaussian_posterior():
    p = Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "flip",
                Flip(0.75),
                Let(
                    "x",
                    IfThenElse(
                        Var("flip"),
                        Sum(Var("g1"), Var("g2")),
                        Affine(Var("g1"), 2.0),
                    ),
                    Let("_", ObserveReal(Var("x"), 1.0), Var("g1")),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)

    assert isinstance(posterior, list)
    assert len(posterior) == 2
    assert np.allclose(sum(np.exp(c.likelihood.log_likelihood) for c in posterior), 1.0)

    # Slightly more likely to observe g1 + g2 = 1 than 2 * g1 = 1, so mixture weight slightly higher than 0.75
    assert np.allclose(posterior[0].gaussian.mu, 0.5)
    assert np.allclose(posterior[0].gaussian.cov, 0.5)
    assert np.allclose(
        posterior[0].likelihood.log_likelihood, np.log(0.7892126302729954)
    )

    assert np.allclose(posterior[1].gaussian.mu, 0.5)
    assert np.allclose(posterior[1].gaussian.cov, 0.0)
    assert np.allclose(
        posterior[1].likelihood.log_likelihood, np.log(0.21078736972700463)
    )


def test_mixture_gaussian_sum_posterior():
    # Same as test above, except now we get posterior of *sum* of gaussian variables
    p = Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "flip",
                Flip(0.75),
                Let(
                    "x",
                    IfThenElse(
                        Var("flip"),
                        Sum(Var("g1"), Var("g2")),
                        Affine(Var("g1"), 2.0),
                    ),
                    Let("_", ObserveReal(Var("x"), 1.0), Sum(Var("g1"), Var("g2"))),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)

    assert isinstance(posterior, list)
    assert len(posterior) == 2
    assert np.allclose(sum(np.exp(c.likelihood.log_likelihood) for c in posterior), 1.0)

    # In the first branch, we observe g1 + g2 = 1, so posterior has mean 1, 0 covariance
    assert np.allclose(posterior[0].gaussian.mu, 1.0)
    assert np.allclose(posterior[0].gaussian.cov, 0.0)
    assert np.allclose(
        posterior[0].likelihood.log_likelihood, np.log(0.7892126302729954)
    )

    # In the second branch, we observe g1 = 0.5
    # so posterior has mean 0.5 and covariance 1 since it is g2 + 0.5, g2 ~ N(0, 1)
    assert np.allclose(posterior[1].gaussian.mu, 0.5)
    assert np.allclose(posterior[1].gaussian.cov, 1.0)
    assert np.allclose(
        posterior[1].likelihood.log_likelihood, np.log(0.21078736972700463)
    )


def test_mixture_gaussian_difference_posterior():
    # Same as test above, except now we get posterior of *difference* of gaussian variables
    p = Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "flip",
                Flip(0.75),
                Let(
                    "x",
                    IfThenElse(
                        Var("flip"),
                        Sum(Var("g1"), Var("g2")),
                        Affine(Var("g1"), 2.0),
                    ),
                    Let(
                        "_",
                        ObserveReal(Var("x"), 1.0),
                        Sum(Var("g1"), Affine(Var("g2"), -1)),
                    ),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)

    assert isinstance(posterior, list)
    assert len(posterior) == 2
    assert np.allclose(sum(np.exp(c.likelihood.log_likelihood) for c in posterior), 1.0)

    # In the first branch, we observe g1 + g2 = 1, so g1 - g2 = g1 + g2 - 2 * g2 = 1 - 2 * g2
    # In this case, g2 has posterior N(0.5, 0.5), so g1 - g2 ~ N(0, 2.0) (2.0 = variance = std ** 2, so *2 => var * 4)
    assert np.allclose(posterior[0].gaussian.mu, 0.0)
    assert np.allclose(posterior[0].gaussian.cov, 2.0)
    assert np.allclose(
        posterior[0].likelihood.log_likelihood, np.log(0.7892126302729954)
    )

    # In the second branch, we observe g1 = 0.5
    # so posterior has mean 0.5 and covariance 1 since it is 0.5 - g2, g2 ~ N(0, 1)
    assert np.allclose(posterior[1].gaussian.mu, 0.5)
    assert np.allclose(posterior[1].gaussian.cov, 1.0)
    assert np.allclose(
        posterior[1].likelihood.log_likelihood, np.log(0.21078736972700463)
    )


def test_union_inference():
    p = Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "flip",
                Flip(0.75),
                Let(
                    "x",
                    IfThenElse(
                        Var("flip"),
                        Sum(Var("g1"), Var("g2")),
                        Affine(Var("g1"), 2.0),
                    ),
                    Let(
                        "_",
                        ObserveReal(Var("x"), 1.0),
                        IfThenElse(Flip(0.5), Var("g1"), Var("g2")),  # type: ignore
                    ),
                ),
            ),
        ),
    )
    posterior, Z = run_kc(p)

    assert isinstance(posterior, list)
    # Two union components (g1, g2), and each one has a mixture posterior
    assert len(posterior) == 4
    assert np.allclose(sum(np.exp(c.likelihood.log_likelihood) for c in posterior), 1.0)

    component_weights = [
        np.log(0.5 * 0.7892126302729954),
        np.log(0.5 * 0.21078736972700463),
        np.log(0.5 * 0.7892126302729954),
        np.log(0.5 * 0.21078736972700463),
    ]
    assert np.allclose(
        [c.likelihood.log_likelihood for c in posterior], component_weights
    )

    component_means = [
        0.5,
        0.5,
        0.5,
        0.0,
    ]
    assert np.allclose(
        np.concat([c.gaussian.mu for c in posterior]).squeeze(), component_means
    )

    component_covs = [
        0.5,
        0.0,
        0.5,
        1.0,
    ]
    assert np.allclose(
        np.concat([c.gaussian.cov for c in posterior]).squeeze(), component_covs
    )
