import pytest
from scipy.stats import norm

from kc import (
    Affine,
    Beta,
    Const,
    Flip,
    Gaussian,
    IfThenElse,
    Inequality,
    Let,
    Observe,
    ObserveReal,
    TruncatableGaussian,
    Var,
    run_kc,
    settings,
)
from kc.real_values import Sum
from kc.terms import Categorical, EnumType, Equality


def gaussian_pdf(mean, std, val):
    return norm.pdf(val, loc=mean, scale=std).item()


def gaussian_cdf(mean, std, val):
    return norm.cdf(val, loc=mean, scale=std).item()


# Flip a coin, choose between two different Gaussians, observe the result
p1 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(
            Var("b"),
            Gaussian(0, 1),
            Gaussian(0, 2),
        ),
        Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
    ),
)
# Expected probability of b is the ratio of the density of x=1.0 under N(0, 1) to the sum of the densities of x=1.0 under N(0, 1) and N(0, 2)
expected_p1 = gaussian_pdf(0.0, 1.0, 1.0) / (
    gaussian_pdf(0.0, 1.0, 1.0) + gaussian_pdf(0.0, 2.0, 1.0)
)

# Flip a coin, choose between two different Gaussians, observe the result twice
# We should see that the probability of b is the same as above
p2 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 2)),
        Let(
            "_",
            ObserveReal(Var("x"), 1.0),
            Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
        ),
    ),
)
expected_p2 = expected_p1

# Flip a coin, choose between two different Gaussians, observe the result twice but with *different* values
# We should get an error
p3 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 2)),
        Let(
            "_",
            ObserveReal(Var("x"), 1.0),
            Let("_", ObserveReal(Var("x"), 0.1), Var("b")),
        ),
    ),
)
expected_p3 = None

# Observe a Gaussian, then flip a coin, then choose between existing or new Gaussian
# then observe the *same* value again
p4 = Let(
    "x",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("x"), 0.5),
        Let(
            "b",
            Flip(0.5),
            Let(
                "y",
                IfThenElse(Var("b"), Var("x"), Gaussian(0, 1)),
                Let("_", ObserveReal(Var("y"), 0.5), Var("b")),
            ),
        ),
    ),
)
# We should see that the probability of b is 1.0 because once we've observed the value of x as 1.0,
# we know that also observing the other gaussian having the same value is measure 0.
# This works because for ObserveReal, for each variable in a GaussianUnion, the clause states that *this* score node is true and all the other score nodes are false.
expected_p4 = 1.0

# Observe a Gaussian union, then observe another Gaussian union that contains the original union
b1_and_b2 = IfThenElse(Var("b1"), Var("b2"), Var("b1"))
p5 = Let(
    "b1",
    Flip(0.5),
    Let(
        "b2",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(Var("b1"), Gaussian(0, 1), Gaussian(0, 2)),
            Let(
                "_",
                ObserveReal(Var("x"), 1.0),
                Let(
                    "y",
                    IfThenElse(Var("b2"), Var("x"), Gaussian(0, 10)),
                    Let("_", ObserveReal(Var("y"), 1.0), b1_and_b2),
                ),
            ),
        ),
    ),
)
# Should be the same as p1 because the first observe is same as in p1, and the second observe has the same value as the first observe so it is measure 0.
expected_p5 = expected_p1

# Same as p5, but now the second observe has a different value than the first observe so it is not measure 0.
p6 = Let(
    "b1",
    Flip(0.5),
    Let(
        "b2",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(Var("b1"), Gaussian(0, 1), Gaussian(0, 2)),
            Let(
                "_",
                ObserveReal(Var("x"), 1.0),
                Let(
                    "y",
                    IfThenElse(Var("b2"), Var("x"), Gaussian(0, 10)),
                    Let("_", ObserveReal(Var("y"), 2.0), b1_and_b2),
                ),
            ),
        ),
    ),
)
# Impossible to observe b1 and b2, because the second observe implies that y != x so b2 must be false.
expected_p6 = 0.0

# Observe that a Gaussian variable is between two values
# (Note that the expression has to evaluate to a boolean, we use Flip(0.5) as a dummy boolean)
p7 = Let(
    "x",
    TruncatableGaussian(0, 1),
    Let(
        "_",
        Observe(Inequality(Var("x"), "<=", 1.0)),
        Let("_", Observe(Inequality(Var("x"), ">", 0.0)), Flip(0.5)),
    ),
)
expected_p7 = 0.5
expected_Z_p7 = gaussian_cdf(0.0, 1.0, 1.0) - gaussian_cdf(0.0, 1.0, 0.0)


# Observe that a Gaussian union is > 0. and < 1.0
p8 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), TruncatableGaussian(-2, 1), TruncatableGaussian(-2, 10)),
        Let(
            "_",
            Observe(Inequality(Var("x"), "<=", -1.0)),  # Observe that x <= -1.0
            Let(
                "_", Observe(Inequality(Var("x"), ">", -2.0)), Var("b")
            ),  # Observe that x > -2.0
        ),
    ),
)
# We just shifted everything down by -2, can shift back up
expected_p8 = (gaussian_cdf(0.0, 1.0, 1.0) - gaussian_cdf(0.0, 1.0, 0.0)) / (
    gaussian_cdf(0.0, 1.0, 1.0)
    - gaussian_cdf(0.0, 1.0, 0.0)
    + gaussian_cdf(0.0, 10.0, 1.0)
    - gaussian_cdf(0.0, 10.0, 0.0)
)

# Observe that a Gaussian union is <= 0. and > 1.0
p9 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(Var("b"), TruncatableGaussian(0, 1), TruncatableGaussian(0, 10)),
        Let(
            "_",
            Observe(Inequality(Var("x"), ">", 1.0)),  # Observe that x <= 1.0
            Let(
                "_", Observe(Inequality(Var("x"), "<=", 0.0)), Var("b")
            ),  # Observe that x <= 0.0
        ),
    ),
)
expected_p9 = None  # Impossible observation

# Same as p1, except we also observe an inequality that agrees with the equality
p10 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(
            Var("b"),
            TruncatableGaussian(0, 1),
            TruncatableGaussian(0, 2),
        ),
        Let(
            "_",
            ObserveReal(Var("x"), 1.0),
            Let("_", Observe(Inequality(Var("x"), ">", 0.42)), Var("b")),
        ),
    ),
)
expected_p10 = expected_p1  # Should be the same as p1 since the inequality is redundant given the equality

# Observe that both Gaussians in a Gaussian union are equal to 1.0, then observe the union again
# This is to test that observing equalities for all components of a Gaussian union works correctly
p11 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 2),
            Let(
                "x",
                IfThenElse(
                    Var("b"),
                    Var("g1"),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("x"), 1.0),
                    Let(
                        "_",
                        ObserveReal(Var("g1"), 1.0),
                        Let("_", ObserveReal(Var("g2"), 1.0), Var("b")),
                    ),
                ),
            ),
        ),
    ),
)
# Since we observe both g1 and g2 to be 1.0, observing x=1.0 is redundant - observing b is 50/50
expected_p11 = 0.5

# We have IID Gaussians in a nested if-then-else
# We observe that the first is > 0, that the combination of the first two is > 0, and that the whole thing is > 1.0
flip_1_and_flip_2 = IfThenElse(
    Var("flip_1"),
    Var("flip_2"),
    Const(False),
)
p12 = Let(
    "flip_1",
    Flip(0.5),
    Let(
        "flip_2",
        Flip(0.5),
        Let(
            "g1",
            TruncatableGaussian(0, 1),
            Let(
                "_",
                Observe(Inequality(Var("g1"), ">", 0.0)),
                Let(
                    "g1 or g2",
                    IfThenElse(
                        Var("flip_1"),
                        Var("g1"),
                        TruncatableGaussian(0, 1),
                    ),
                    Let(
                        "_",
                        Observe(Inequality(Var("g1 or g2"), ">", 0.0)),
                        Let(
                            "g1 or g2 or g3",
                            IfThenElse(
                                Var("flip_2"),
                                Var("g1 or g2"),
                                TruncatableGaussian(0, 1),
                            ),
                            Let(
                                "_",
                                Observe(Inequality(Var("g1 or g2 or g3"), ">", 1.0)),
                                flip_1_and_flip_2,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)
# We know g1 or g2 is > 0, whereas g3 is 50/50 positive/negative
# Therefore 2/3 probability of flip_2
# Similarly for g2 vs g2
# So expected prob is (2/3) ** 2
expected_p12 = (2 / 3) ** 2

# Observe that one Gaussian in a Gaussian union is equal to 1.0, then observe the union is equal to 2.0
p13 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "x",
                IfThenElse(
                    Var("b"),
                    Var("g1"),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("x"), 2.0),
                    Let(
                        "_",
                        ObserveReal(Var("g2"), 1.0),
                        Var("b"),
                    ),
                ),
            ),
        ),
    ),
)
expected_p13 = 1.0

# Same as above but we also observe that the other Gaussian is equal to 2.0, which shouldn't change anything
p14 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        Gaussian(0, 1),
        Let(
            "g2",
            Gaussian(0, 1),
            Let(
                "x",
                IfThenElse(
                    Var("b"),
                    Var("g1"),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("x"), 2.0),
                    Let(
                        "_",
                        ObserveReal(Var("g2"), 1.0),
                        Let("_", ObserveReal(Var("g1"), 2.0), Var("b")),
                    ),
                ),
            ),
        ),
    ),
)
expected_p14 = expected_p13


# We have 4 gaussians in a nested if-then-else
# We observe that pairs (1,2) and (2,3) and (3,4) and (4,1) are all equal to 1.0
def quad_gaussian(y_obs, output):
    return Let(
        "b0",
        Flip(0.5),
        Let(
            "b1",
            Flip(0.5),
            Let(
                "b2",
                Flip(0.5),
                Let(
                    "b3",
                    Flip(0.5),
                    Let(
                        "b4",
                        Flip(0.5),
                        Let(
                            "g1",
                            Gaussian(0, 1),
                            Let(
                                "g2",
                                Gaussian(0, 1),
                                Let(
                                    "g3",
                                    Gaussian(0, 1),
                                    Let(
                                        "g4",
                                        Gaussian(0, 1),
                                        Let(
                                            "x1",
                                            IfThenElse(
                                                Var("b1"),
                                                Var("g1"),
                                                Var("g2"),
                                            ),
                                            Let(
                                                "x2",
                                                IfThenElse(
                                                    Var("b2"),
                                                    Var("g2"),
                                                    Var("g3"),
                                                ),
                                                Let(
                                                    "x3",
                                                    IfThenElse(
                                                        Var("b3"),
                                                        Var("g3"),
                                                        Var("g4"),
                                                    ),
                                                    Let(
                                                        "x4",
                                                        IfThenElse(
                                                            Var("b4"),
                                                            Var("g4"),
                                                            Var("g1"),
                                                        ),
                                                        Let(
                                                            "y",
                                                            IfThenElse(
                                                                Var("b0"),
                                                                Var("x1"),
                                                                Var("x3"),
                                                            ),
                                                            Let(
                                                                "_",
                                                                ObserveReal(
                                                                    Var("x1"), 1.0
                                                                ),
                                                                Let(
                                                                    "_",
                                                                    ObserveReal(
                                                                        Var("x2"), 1.0
                                                                    ),
                                                                    Let(
                                                                        "_",
                                                                        ObserveReal(
                                                                            Var("x3"),
                                                                            1.0,
                                                                        ),
                                                                        Let(
                                                                            "_",
                                                                            ObserveReal(
                                                                                Var(
                                                                                    "x4"
                                                                                ),
                                                                                1.0,
                                                                            ),
                                                                            Let(
                                                                                "_",
                                                                                ObserveReal(
                                                                                    Var(
                                                                                        "y"
                                                                                    ),
                                                                                    y_obs,
                                                                                ),
                                                                                output,
                                                                            ),
                                                                        ),
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


# Observe that y = 1.0, x1 = (g1 or g2) = 1.0, x2 = (g2 or g3) = 1.0, x3 = (g3 or g4) = 1.0, x4 = (g4 or g1) = 1.0
# In this case, we must have g1 = 1, g3 = 1, g2 != 1, g4 != 1 or g2 = 1, g4 = 1, g1 != 1, g3 != 1
# i.e. either only (g1, g3) are 1.0 or only (g2, g4) are 1.0
# since any of the other gaussians also being 1.0 would have measure 0.
# The case (g1, g3) are 1.0 happens when b1 = True, b3 = True
# The case (g2, g4) are 1.0 happens when b1 = False, b3 = False
b1_and_b3 = IfThenElse(
    Var("b1"),
    Var("b3"),
    Const(False),
)
p15 = quad_gaussian(1.0, b1_and_b3)
expected_p15 = 0.5

# b2 and b4 can be anything in both cases
p16 = quad_gaussian(1.0, Var("b2"))
p17 = quad_gaussian(1.0, Var("b4"))
expected_p16 = 0.5
expected_p17 = 0.5

# Adding priors on the flip probabilities
boolean_expr = Let(
    "result",
    IfThenElse(
        Var("b1"),
        Var("b2"),
        Var("b3"),
    ),
    Let(
        "_",
        Observe(Var("result")),
        Var("b1"),
    ),
)
p18 = Let(
    "b1",
    Flip(Beta(1, 1)),
    Let(
        "b2",
        Flip(Beta(1, 1)),
        Let(
            "b3",
            Flip(Beta(1, 1)),
            boolean_expr,
        ),
    ),
)
expected_p18 = 0.5


# Adding *shared* priors on the flip probabilities
p19 = Let(
    "p",
    Beta(1, 1),
    Let(
        "b1",
        Flip(Var("p")),
        Let(
            "b2",
            Flip(Var("p")),
            Let(
                "b3",
                Flip(Var("p")),
                boolean_expr,
            ),
        ),
    ),
)
expected_p19 = 2 / 3

# Same as p1, but now we multiply the second gaussian by 2
p20 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(
            Var("b"),
            Gaussian(0, 1),
            Affine(Gaussian(0, 1), 2),
        ),
        Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
    ),
)
# Should we score this under measure of N(0, 1) or N(0, 2)?
# flag changes it
if settings.transform_measures:
    expected_p20 = expected_p1
else:
    expected_p20 = gaussian_pdf(0.0, 1.0, 1.0) / (
        gaussian_pdf(0.0, 1.0, 1.0) + gaussian_pdf(0.0, 1.0, 0.5)
    )

# We observe the *same gaussian* in both branches, but in the second it is multiplied by 2
p21 = Let(
    "g",
    Gaussian(0, 1),
    Let(
        "b",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(
                Var("b"),
                Var("g"),
                Affine(Var("g"), 2),
            ),
            Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
        ),
    ),
)
# Either g is 0.5 or it is 1.0
expected_p21 = gaussian_pdf(0, 1, 1) / (gaussian_pdf(0, 1, 0.5) + gaussian_pdf(0, 1, 1))

# Same as above, but now we observe that the pair is > 1.
p21 = Let(
    "g",
    TruncatableGaussian(0, 1),
    Let(
        "b",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(
                Var("b"),
                Var("g"),
                Affine(Var("g"), 2),
            ),
            Let("_", Observe(Inequality(Var("x"), ">", 1.0)), Var("b")),
        ),
    ),
)
# Either g is between 0.5 and 1, or between 1 and inf
# In the first case, flip is False, in the second it is 50/50
p21_normalizing_const = 0.5 * (gaussian_cdf(0, 1, 1) - gaussian_cdf(0, 1, 0.5)) + (
    1 - gaussian_cdf(0, 1, 1)
)
expected_p21 = (0.5 * (1 - gaussian_cdf(0, 1, 1))) / p21_normalizing_const

# Observe both a gaussian variable and its transformation in an incompatible way
p22 = Let(
    "g",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("g"), 1),
        Let("_", ObserveReal(Affine(Var("g"), 2, 1), 1), Const(True)),
    ),
)
expected_p22 = None

# Observe both a gaussian variable and its transformation in a compatible way
p23 = Let(
    "g",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("g"), 1),
        Let("_", ObserveReal(Affine(Var("g"), 2, 1), 3), Const(True)),
    ),
)
expected_p23 = 1.0

# Observe both a gaussian variable and its transformation in a compatible way
p24 = Let(
    "g",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("g"), 1),
        Let("_", ObserveReal(Affine(Var("g"), 2, 1), 3), Const(True)),
    ),
)
expected_p24 = 1.0

# Observe both a gaussian variable and itself multiplied by -1 being equal to the same value
p25 = Let(
    "g",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("g"), 1),
        Let(
            "_",
            ObserveReal(Affine(Var("g"), -1), 3),
            Const(True),
        ),
    ),
)
expected_p25 = None

# Observe both a gaussian variable and itself multiplied by -1 being equal to the same value
p26 = Let(
    "g",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveReal(Var("g"), 1),
        Let(
            "_",
            ObserveReal(Affine(Var("g"), -1), 3),
            Const(True),
        ),
    ),
)
expected_p26 = None

# If a gaussian is <= -1, multiply it by -1
# What is prob it is > 0?
p27 = Let(
    "g",
    TruncatableGaussian(0, 1),
    Let(
        "g_transformed",
        IfThenElse(
            Inequality(Var("g"), "<=", -1),
            Affine(Var("g"), -1),
            Var("g"),
        ),
        Inequality(Var("g_transformed"), ">", 0),
    ),
)
expected_p27 = 1 - (gaussian_cdf(0, 1, 0) - gaussian_cdf(0, 1, -1))

# Same as p20, but now we observe both the result and result *2
p28 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        IfThenElse(
            Var("b"),
            Gaussian(0, 1),
            Affine(Gaussian(0, 1), 2),
        ),
        Let(
            "_",
            ObserveReal(Affine(Var("x"), 2), 2.0),
            Let("_", ObserveReal(Var("x"), 1.0), Var("b")),
        ),
    ),
)
# Should we score this under measure of N(0, 1) or N(0, 2)?
# flag changes it
if settings.transform_measures:
    expected_p28 = expected_p1
else:
    expected_p28 = gaussian_pdf(0.0, 1.0, 1.0) / (
        gaussian_pdf(0.0, 1.0, 1.0) + gaussian_pdf(0.0, 1.0, 0.5)
    )


# Branch to two different gaussians, but if one is > 0, then branch back to the other
p29 = Let(
    "b",
    Flip(0.5),
    Let(
        "g1",
        TruncatableGaussian(0, 1),
        Let(
            "g2",
            TruncatableGaussian(0, 1),
            Let(
                "result",
                IfThenElse(
                    Var("b"),
                    IfThenElse(Inequality(Var("g1"), "<=", 0), Var("g2"), Var("g1")),
                    Var("g2"),
                ),
                Let(
                    "_",
                    ObserveReal(Var("result"), 1),
                    Var("b"),
                ),
            ),
        ),
    ),
)
expected_p29 = 0.75 / (0.75 + 0.5)

p30 = Let(
    "b",
    Flip(0.5),
    Let(
        "x",
        Gaussian(0, 1),
        Let(
            "y",
            Gaussian(1, 2),
            Let(
                "z",
                Gaussian(-1, 1),
                Let(
                    "result",
                    IfThenElse(
                        Var("b"),
                        Sum(Var("x"), Var("y")),
                        Sum(Var("x"), Var("z")),
                    ),
                    Let("_", ObserveReal(Var("result"), 0.0), Var("b")),
                ),
            ),
        ),
    ),
)
expected_p30 = 0.42356749843592556

# Observe a Categorical over Enums
Color = EnumType("Color", ["R", "G", "B"])
p_enum = Let(
    "color",
    Categorical([Color.R, Color.G, Color.B], [0.1, 0.4, 0.5]),
    Let(
        "b",
        Flip(0.5),
        # If flip is true, it stays the same, else it's R
        Let(
            "final_color",
            IfThenElse(Var("b"), Var("color"), Color.R),
            Var("final_color"),
        ),
    ),
)
# If b is true (0.5), we get [0.1, 0.4, 0.5]
# If b is false (0.5), we get [1.0, 0.0, 0.0]
# Total prob: R = 0.5*0.1 + 0.5*1.0 = 0.55
# G = 0.5*0.4 = 0.20
# B = 0.5*0.5 = 0.25
expected_p_enum = {"R": 0.55, "G": 0.20, "B": 0.25}

p_enum_eq = Let(
    "color",
    Categorical([Color.R, Color.G, Color.B], [0.1, 0.4, 0.5]),
    Let("_", Observe(Equality(Var("color"), Color.G)), Var("color")),
)
expected_p_enum_eq = {"R": 0.0, "G": 1.0, "B": 0.0}

p_enum_with_gaussians = Let(
    "color",
    Categorical([Color.R, Color.G, Color.B], [0.5, 0.25, 0.25]),
    Let(
        "b",
        Flip(0.5),
        Let(
            "x",
            IfThenElse(
                IfThenElse(
                    Var("b"),
                    Equality(Var("color"), Color.G),
                    Equality(Var("color"), Color.R),
                ),
                Gaussian(0, 1),
                Gaussian(0, 2),
            ),
            Let(
                "_",
                ObserveReal(Var("x"), 0.0),
                Var("b"),
            ),
        ),
    ),
)
# We are twice as likely to observe 0 if the IfThen condition is true
# If b is true, color is G, IfThen condition prob is 0.25, prob of observing 0 is 0.5 * 0.25 * 1 + 0.5 * 0.75 * 0.5 = 0.3125
# If b is false, color is R, IfThen condition prob is 0.5, prob of observing 0 is 0.5 * 0.5 * 1 + 0.5 * 0.5 * 0.5 = 0.375
expected_p_enum_with_gaussians = 0.3125 / (0.3125 + 0.375)

# impossible enum
p_enum_impossible = Let(
    "color",
    Categorical([Color.R, Color.G, Color.B], [0.5, 0.25, 0.25]),
    IfThenElse(
        Equality(Var("color"), Color.R),
        Observe(Equality(Var("color"), Color.G)),
        Observe(Equality(Var("color"), Color.R)),
    ),
)
expected_p_enum_impossible = None


@pytest.mark.parametrize(
    "name, program, expected_prob, expected_z",
    [
        ("p1", p1, expected_p1, None),
        ("p2", p2, expected_p2, None),
        ("p3", p3, expected_p3, None),
        ("p4", p4, expected_p4, None),
        ("p5", p5, expected_p5, None),
        ("p6", p6, expected_p6, None),
        ("p7", p7, expected_p7, expected_Z_p7),
        ("p8", p8, expected_p8, None),
        ("p9", p9, expected_p9, None),
        ("p10", p10, expected_p10, None),
        ("p11", p11, expected_p11, None),
        ("p12", p12, expected_p12, None),
        ("p13", p13, expected_p13, None),
        ("p14", p14, expected_p14, None),
        ("p15", p15, expected_p15, None),
        ("p16", p16, expected_p16, None),
        ("p17", p17, expected_p17, None),
        ("p18", p18, expected_p18, None),
        ("p19", p19, expected_p19, None),
        ("p20", p20, expected_p20, None),
        ("p21", p21, expected_p21, None),
        ("p22", p22, expected_p22, None),
        ("p23", p23, expected_p23, None),
        ("p24", p24, expected_p24, None),
        ("p25", p25, expected_p25, None),
        ("p26", p26, expected_p26, None),
        ("p27", p27, expected_p27, None),
        ("p28", p28, expected_p28, None),
        ("p29", p29, expected_p29, None),
        ("p30", p30, expected_p30, None),
        ("p_enum", p_enum, expected_p_enum, None),
        ("p_enum_eq", p_enum_eq, expected_p_enum_eq, None),
        (
            "p_enum_with_gaussians",
            p_enum_with_gaussians,
            expected_p_enum_with_gaussians,
            None,
        ),
        (
            "p_enum_impossible",
            p_enum_impossible,
            expected_p_enum_impossible,
            None,
        ),
    ],
)
def test_kc_programs(name, program, expected_prob, expected_z):
    print(f"--- {name} ---")
    prob, normalizing_constant = run_kc(program)

    # Check if probability matches expected
    if prob is not None and expected_prob is not None:
        if isinstance(prob, dict) and isinstance(expected_prob, dict):
            for k, v in expected_prob.items():
                assert abs(prob.get(k, 0.0) - v) <= 1e-8, (
                    f"Probability mismatch for {name} at key {k}: {prob.get(k)} != {v}"
                )
        else:
            assert abs(prob - expected_prob) <= 1e-8, (
                f"Probability mismatch for {name}: {prob} != {expected_prob}"
            )
    elif prob is None and expected_prob is None:
        pass  # Both None is fine
    else:
        assert False, f"Probability mismatch for {name}: {prob} != {expected_prob}"

    # Check normalizing constant if provided
    if expected_z is not None:
        assert abs(normalizing_constant - expected_z) <= 1e-8, (
            f"Normalizing constant mismatch for {name}: {normalizing_constant} != {expected_z}"
        )
