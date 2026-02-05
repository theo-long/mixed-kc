# Instructions

This project uses `uv`. To run the test programs, first install `uv` then run `uv run main.py`.

# Implementation details

## Implementation of Gaussian Variable equality observes

There are two cases that we need to handle:
1) Directly observing a single Gaussian Variable
2) Observing some 'union' of Gaussian Variables (i.e. a nested `IfThenElse` with Gaussian branches)

For case 1, we create a new node in the BDD representing the event `g_i = val` where `g_i` is the Gaussian variable and `val` is the observed value, and we add this node to our `observes_all_hold` clause. The true weight of this node is the likelihood i.e. the value of the density of `g_i` at `val`, which ensures that the weight of the `observes_all_hold` clause represents the marginal likelihood of the observed data. The false weight is `1.0` for reasons specified in the paragraph below.

For case 2, we have a union of Gaussian variables, represented by a set of gaussian variables each paired with a boolean formula which represents the flip assignments under which the union (a series of nested `IfThenElse`s) evaluates to that variable. Similarly to case 1, we create a BDD node `g_i = val` with the same true/false weights (likelihood, 1.0). However, we __AND__ each BDD node with a guard clause which is the associated boolean formula. We also __AND__ `g_i = val` with clauses of the form `~(g_j = val)` for `j != i` since in the case where the Gaussian union evalutes to `g_i` and we observe it we *do not* want to score this outcome using the weight for `g_j = val`. This is also why we set the false weight to `1.0` for these nodes, since we only want to score using the value of `g_i`. Note that this leads to some intricacies in handling multiple overlapping observes, discussed in the following section.

### Handling multiple interacting equality observes

There are cases where we might observe multiple Gaussian variable equality statements, either of the same Gaussian variable, or of Gaussian Unions which reference the same Gaussian variables.

The first case is easier - we add a 'mutual compatibility' condition which ensures that no two BDD nodes of the form `g1 = 1.0` and `g1 = 2.0` are both true, i.e. we cannot have any pairwise incompatible equalities both being true.

The second case is more complicated. Consider the situation where we observe a first Gaussian equal to 1.0, then observe a union of that Gaussian and another being equal to 1.0 as well:
```
Let(
    "g1",
    Gaussian(0, 1),
    Let(
        "g1 or g2",
        IfThenElse(
            Flip(0.5),
            Var("g1"),
            Gaussian(0, 1),
        ),
        Let(
            "_",
            ObserveReal(Var("g1"), 1.0),
            Let(
                "_",
                ObserveReal(Var("g1 or g2"), 1.0),
                ...
            )
        )
    )   
)
```
In this case there are two possible 'explanations': `g1` and `g2` are *both* equal to 1.0 and the `Flip(0.5)` chooses the second branch of the `IfThenElse`, or only `g1` is equal to 1.0 and the first branch is chosen. Under the density $P(\cdot | g_1 = 1.0)$, the first event has measure `0`, while the second event has measure $1$, therefore with probability $1.0$ we are in the second situation. 

To ensure that this is correctly handled in the WMC, when observing a Gaussian union, each score node of the form `g_i = val` is also __AND__ ed with the __NOT__ of the other nodes `~(g_j = val)` in that union, which effectively says 'if we observe some union is `= val`, then *exactly one* of the variables in this union is equal to  `val`.

This introduces two wrinkles: first, we do not want the statement `~(g_j = val)` to change the weight of the weighted models in the WMC, since the score node `g_i = val` already captures the appropriate weight. This is done by assigning a weight of `1.0` to the false branch of each score node.

The second is that this excludes explanations that may have measure 0 but are still forced by other observe statements. If our previous example also had a third `ObserveReal(Var("g2"), 1.0)` statement, the fact that observing equalities of unions forces *exactly one* of the branches to be equal to a particular value would mean there are no satisfying assignments for our BDD!

The solution is to add a second type of equality node representing events of the form `g_i = g_j`, so when observing a Gaussian Union each individual branch condition looks like:
```
(g_i = val) & (~(g_j = val) | (g_i = g_j)) # i != j
```

What weights should the $(g_i = g_j)$ node have? It is a measure 0 event under the conditional density $P(\cdot | g_i = \text{val})$, so we could assign it true/false weights of $(0., 1.0)$, but this would not change the situation since now all models with this node set to true will have weight 0. Instead, we want to give this some 'infinitesimal' weight $\eps$ which is strictly smaller than any real-valued weight, but is not 0. Put differently, we should strictly prefer explanations of our observe statements which do not require `g_i = g_j` (since these have non-zero measure under $P(\cdot | g_i = \text{val})$). However, if we are 'forced' to accept the conclusion that `g_i = g_j`, either because we observe this directly, or have a large enough number of overlapping observes that some statement `g_i = g_j` is true, then we should instead score outcomes under the conditional measure $P(\cdot | g_i = \text{val}, g_j = \text{val})$.

It turns out that the structure of these 'infinitesimal weights' is exactly like polynomials in the variable $\eps$, where we reinterpret all of our usual real-valued weights $w$ as constant-valued polynomials in $R[\eps]$, and we assign the 'true' weight $\eps$ to the event `g_i = g_j` and the false weight $1.0$ (for the same reason as before where we don't want to multiply extra terms into the weight when it is false). This means that the final weight will be some polynomial $f$ of the form $a + b\eps + c\eps^2 ...$, where the coefficient of the $\eps^k$ term represent the unnormalized density under the conditional measure where $k$ different statements of the form `g_i = g_j` are true. We then take the first non-zero coefficient of $f$ as our actual weight, which captures the fact that we strictly prefer evaluating under conditional densities where fewer statements `g_i = g_j` are true. 

For an example of a program where we will end up having a weight with higher powers of $\eps$, consider $n$ Gaussian variables, where we observe that every pair (in the form of an `IfThenElse`) is equal to $1.0$. This implies that at least $n - 1$ of them must be equal to $1.0$ (otherwise some pair would not be equal to $1.0$), however we strictly prefer this to the case where all $n$ are equal to $1.0$. In this case resulting polynomial weights will have both $\epsilon^{n-1}$ and $\epsilon^{n}$ terms, with all lower-order terms equal to 0, and the $\epsilon^{n-1}$ coefficient will be the final weight.

## Implementation of Gaussian inequality observes

To handle inequalities which reference Gaussian variables (or composed IfThenElse statements with Gaussian variables in the body), we must update our KC in a few ways:
1. We must add nodes representing the events (some_gaussian >= some_value), with the weights given by evaluating the gaussian CDF.
2. We must update the weights of the score nodes used for Gaussian equality observes to handle the fact that (some_gaussian >= 0) is *not independent* of (some_gaussian == 1.)

The way we do this is using a program transformation that transforms Gaussian variables into unions of *truncated* Gaussian variables.

Note that in practice our implementation does not actually rewrite the `PExpr` into another `PExpr` as outlined below. Instead this 'program transformation' is implemented within the logic of the KC step, but the algorithm is conceptually the same as what is described below.

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
- Can we do ObserveReal on the "p" variables?
- Can we do inference over the "p" variables vs. just marginalizing them out?
- add ability to have latent Gaussians that can be parameters of other distributions
- can we pass PExprs with Betas in the leaves to be the priors for p variables?
    - A PExpr with `dist_type` leaves is a mixture over *truncated* `dist_type` (in this case `dist_type` is beta)
    - moments of mixture is just weighted sum of component moments
        - note this assumes independence which we should have in our case? Need to think about if observes change this
    - therefore we just need to be able to calculate moments for truncated `dist_type`
        - this is possible for beta dist because $x p_{\alpha, \beta}(x) = p_{\alpha + 1, \beta}(x)$ we have recursive expr for moments
        - different trick for truncated normal dist
        - can we express what makes these kinds of recursive moment computations possible for various `dist_type` families? Or just hardcode that we know a formula for certain families?
        - need a different type of inference at the end since we are not doing inference on boolean vars (as in KC), we are inferring the *moments* so we need a different kind of interpreter something like eval_moment:
            - first eval the PExpr with continuous leaves into a `Union` with guards + dists
            - for each guard expression, use KC to evaluate probability that it is true
                - there should be some optimizations here due to the fact that normalizing const is the same for everything here, many of these guards will have shared subexprs - feels like there should be some natural way to rewrite KC such that it accumulates probabilities in the leaves instead of just eval p(true) for a single bool expr
            - for each leaf calculate whatever moment we need
            - then combine moments as weighted sum
- IfThenElse optimization where if we know cond is actually false/true we can skip a branch
    - is this handled by laziness somehow?
- Can we add mutually incompatible *affine* observes?
    - For d-dimensional gaussian latent, mutually incompatible group is d + 1 observes that are linearly independent
    - naive/brute force: enumerate all d + 1 subsets and check for lin. indep.
    - alternative: some kind of data structure that captures lin. indep. relations to efficiently construct

## Ideas
- Adding in continuous latents that are conjugate
- Can we allow for inequalties between two different Gaussian variables e.g. g1 <= g2? Or are we only able to do g1 <= val?

## Things to look at
- SPPL sum product networks
    - compare inequality implementation - is theirs somehow stronger/weaker/different?