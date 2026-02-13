from kc import dsl
from kc import terms
import pytest


def test_dsl_p1_mixture():
    # p1 = Let("b", Flip(0.5), Let("x", IfThenElse(Var("b"), Gaussian(0, 1), Gaussian(0, 2)), Let("_", ObserveReal(Var("x"), 1.0), Var("b"))))
    with dsl.Model() as m:
        b = dsl.flip(0.5, name="b")
        x = dsl.if_else(b, dsl.gaussian(0, 1), dsl.gaussian(0, 2))
        dsl.observe(x == 1.0)

    # Check graph structure indirectly by compiling
    # The output variable is 'b'
    ir = m.compile(b)
    # expected_p1 logic is in test_main, we just ensure IR compiles
    assert ir is not None


def test_dsl_p4_nested_observation():
    # p4 = Let("x", Gaussian(0, 1), Let("_", ObserveReal(Var("x"), 0.5), Let("b", Flip(0.5), Let("y", IfThenElse(Var("b"), Var("x"), Gaussian(0, 1)), Let("_", ObserveReal(Var("y"), 0.5), Var("b"))))))
    with dsl.Model() as m:
        x = dsl.gaussian(0, 1, name="x")
        dsl.observe(x == 0.5)

        b = dsl.flip(0.5, name="b")
        y = dsl.if_else(b, x, dsl.gaussian(0, 1))
        dsl.observe(y == 0.5)

    ir = m.compile(b)
    assert ir is not None


def test_dsl_p12_nested_logic():
    # Complex nested if-else and observations
    with dsl.Model() as m:
        flip1 = dsl.flip(0.5, name="flip1")
        flip2 = dsl.flip(0.5, name="flip2")

        g1 = dsl.gaussian(0, 1, name="g1")
        dsl.observe(g1 > 0.0)
        # Note: dsl.observe(g1 > 0) -> Inequality

        # g1 or g2
        # In the original test: "g1 or g2" is a variable name for the result of IfThenElse
        # If flip1 then g1 else new gaussian
        g1_or_g2 = dsl.if_else(flip1, g1, dsl.gaussian(0, 1))
        dsl.observe(g1_or_g2 > 0.0)

        g1_or_g2_or_g3 = dsl.if_else(flip2, g1_or_g2, dsl.gaussian(0, 1))
        dsl.observe(g1_or_g2_or_g3 > 1.0)

        # Result is logic: if flip1 then (if flip2 then False else False) ?
        # Original: flip_1_and_flip_2 = IfThenElse(Var("flip_1"), Var("flip_2"), Const(False))
        # dsl logic:
        res = dsl.if_else(flip1, flip2, dsl.ensure_bool(False))

    ir = m.compile(res)
    assert ir is not None


def test_dsl_quad_gaussian():
    # 4 gaussians, 5 flips structure
    # Case 1: y_obs = 1.0, query b1 & b3

    with dsl.Model() as m:
        b0 = dsl.flip(0.5, name="b0")
        b1 = dsl.flip(0.5, name="b1")
        b2 = dsl.flip(0.5, name="b2")
        b3 = dsl.flip(0.5, name="b3")
        b4 = dsl.flip(0.5, name="b4")

        g1 = dsl.gaussian(0, 1, name="g1")
        g2 = dsl.gaussian(0, 1, name="g2")
        g3 = dsl.gaussian(0, 1, name="g3")
        g4 = dsl.gaussian(0, 1, name="g4")

        x1 = dsl.if_else(b1, g1, g2)
        x2 = dsl.if_else(b2, g2, g3)
        x3 = dsl.if_else(b3, g3, g4)
        x4 = dsl.if_else(b4, g4, g1)

        y = dsl.if_else(b0, x1, x3)

        dsl.observe(x1 == 1.0)
        dsl.observe(x2 == 1.0)
        dsl.observe(x3 == 1.0)
        dsl.observe(x4 == 1.0)
        dsl.observe(y == 1.0)  # y_obs

        target = b1 & b3

    ir = m.compile(target)
    assert ir is not None
