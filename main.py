from kc.prob import (
    Let,
    Var,
    Const,
    Flip,
    IfThenElse,
    Observe,
    GaussianVariable,
)

# Flip a coin, choose between two different Gaussians, observe the result
expr = Let(
    "x",
    Var("x"),
    IfThenElse(
        Flip(0.5),
        Const(True),
        Const(False),
    ),
)

# Flip a coin, choose between two different Gaussians, observe the result twice

# Observe a Gaussian, then flip a coin, then choose between existing or new Gaussian
# then observe the *same* value again



def main():
    print("Hello from mixed-kc!")


if __name__ == "__main__":
    main()
