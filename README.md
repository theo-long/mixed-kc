# Instructions

This project uses `uv`. To run the test programs, first install `uv` then run `uv run main.py`.

# Implementation details

## Implementation of Gaussian Variable equality observes

## Implementation of Gaussian inequality observes

To handle inequalities which reference Gaussian variables (or composed IfThenElse statements with Gaussian variables in the body), we must update our KC in a few ways:
1. We must add nodes representing the events (some_gaussian >= some_value), with the weights given by evaluating the gaussian CDF.
2. We must update the weights of the score nodes used for Gaussian equality observes to handle the fact that (some_gaussian >= 0) is *not independent* of (some_gaussian == 1.)

The way we do this is sing a program transformation that transforms Gaussian variables into unions of *truncated* Gaussian variables.

### Single Gaussian Variable Case
Let's first consider the case where we have a single Gaussian variable, with two inequality observes, and a single equality observe:
```
Let(
    "Z",
    Gaussian(0, 1),
    Let(
        "_",
        ObserveRealInequality(Var("Z"), >=, 0),
        Let(
            "_",
            ObserveRealInequality(Var("Z"), <, 1)
            Let(
                "_",
                ObserveReal(Var("Z"), 0.5)
                Const(True),
            )
        )
    )
)
```
In this case the observes effectively truncate the domain of Z into 3 disjoint intervals: $(-\infty, 0), [0, 1), [1, \infty)$.

We can express this as a set of nested `IfThenElse` statements with `TruncatedGaussianVariable`s, which have densities that are only supported on each of these 3 intervals:
```
Z_truncated_representation = 
Let(
    "Z < 0",
    Flip(0.5),
    Let(
        "Z < 1 | Z >= 0",
        Flip(0.6827),
        IfThenElse(
            Var("Z < 0"),
            TruncatedGaussian(0, 1, -float("inf"), 0),
            IfThenElse(
                Var("Z < 1 | Z >= 0"),
                TruncatedGaussian(0, 1, 0, 1),
                TruncatedGaussian(0, 1, 1, float("inf"))
            )
        )
    )
)
```
A `TruncatedGaussianVariable(mean, std, low, high)` represent a random variable with a density given by truncating the usual Gaussian density at `low` and `high` and normalizing, i.e. it is equal to $0$ outside of the interval `[low, high)`, and is proportional to the Gaussian density inside that interval.

Now our full program becomes
```
Let(
    "Z",
    Z_truncated_representation,
    Let(
        "_",
        Observe(
            IfThenElse(
                Var("Z < 0"),
                Const(False),
                Const(True),
            )
        ),
        Let(
            "_",
            Observe(
                IfThenElse(
                    Var("Z < 0"),
                    Const(True),
                    Var("Z < 1 | Z >= 0"),
                )
            )
            Let(
                "_",
                Observe(
                    IfThenElse(
                        Var("Z < 0"),
                        Const(True),
                        Var("Z < 1 | Z >= 0"),
                    )
                ),
                Let(
                    "_",
                    "ObserveReal(Var("Z_truncated_0_to_1"), 0.5)
                    Const(True),
                ),
            )
        )
    )
)
```
where the variable `Var("Z_truncated_0_to_1")` references the `TruncatedGaussian(0, 1, 0, 1)` value in `Z_truncated_representation` but we omit this outer `Let` statement for brevity.

We see that the two inequality observes have been transformed into observing boolean variables:
```
ObserveRealInequality(Var("Z"), >=, 0)
~~~~~
Observe(
    IfThenElse(
        Var("Z < 0"),
        Const(True),
        Var("Z < 1 | Z >= 0"),
    )
)

ObserveRealInequality(Var("Z"), <, 1)
~~~~~
Observe(
    IfThenElse(
        Var("Z < 0"),
        Const(True),
        Var("Z < 1 | Z >= 0"),
    )
)
```

The equality observe has been transformed into *both* a boolean observation (corresponding the to interval in which that value lies) and an equality observation (of the relevant `TruncatedGaussianVariable`)
```
ObserveReal(Var("Z"), 0.5)
~~~~
Observe(
    IfThenElse(
        Var("Z < 0"),
        Const(True),
        Var("Z < 1 | Z >= 0"),
    )
)
ObserveReal(Var("Z_truncated_0_to_1"), 0.5)
```
Note that this is not strictly necessary, since `ObserveReal(Var("Z_truncated_0_to_1"), 0.5)` has density $0$ under all the other truncated gaussians in the union, but it is more efficient since it allows us to avoid evaluating these branches we know have density `0`.

### Multiple Gaussian Variables Case
In the case where we have a union of Gaussian variables, we must handle the fact that observes might be referencing Gaussian variables appearing in any of the branches of an `IfThenElse` statement:
```
Let(
    "Z",
    IfThenElse(
        Flip(0.5),
        Gaussian(0, 1), # G1
        Gaussian(0, 2), # G2
    ),
    ...
)
```
In order to handle this, before the KC step, we perform a first pass over the program that identifies all the observe statements which could possibly interact with each Gaussian variable. In this case, any statement that observes `Var("Z")` (or a nested `IfThenElse` referencing `Var("Z")`) will potentially impact both `G1` and `G2`. These interactions are stored as a dictionary with Gaussian variable keys and values being a sorted list of all inequality values interacting with that variable.

This data can then be used to truncate all the Gaussian variables as above, where we now truncate the variable according to the list of inequalities that impact it.

# Misc. Notes

## TODO
- add some documentation and flags for turning features on/off
- add ability to reference inequalities as Boolean variables, and then inequality observes are just handled as usual boolean observes

## Questions
- Are we actually using the BDD library anywhere for WMC, or just to construct BDD?

## Ideas
- Can we allow flip params to be symbolic values and perform inference on them? Point is that we can use KC to compile a formula for P(outcome | flip_thetas) which is some rational function in flip_thetas (since it is P(outcome) / P(observes all hold), both of which are polynomials in flip_thetas).
- Adding in continuous latents that are conjugate

## Things to look at
- SPPL sum product networks