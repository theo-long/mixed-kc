from kc.prob import (
    Let,
    Var,
    Const,
    Flip,
    IfThenElse,
    Observe,
    GaussianVariable,
)


expr = Let(
    "x",
    Var("x"),
    IfThenElse(
        Flip(0.5),
        Const(True),
        Const(False),
    ),
)

def main():
    print("Hello from mixed-kc!")


if __name__ == "__main__":
    main()
