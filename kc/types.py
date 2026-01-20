import sympy

epsilon = sympy.Symbol("epsilon")
WeightType = int | float | sympy.Expr


def get_float_value(v: WeightType) -> float:
    if isinstance(v, (int, float)):
        return float(v)

    weight_poly = v.as_poly(epsilon)
    assert weight_poly is not None, "Should get a polynomial"

    return float(weight_poly.coeffs()[-1])
