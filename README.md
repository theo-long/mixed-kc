# Instructions

A small probabilistic programming language for performing exact inference over programs with both discrete and continuous variables using knowledge compilation.

This project uses `uv`. To run the test programs, first install `uv` then run `uv run pytest`.

The easiest way to write programs is using the DSL in `kc.dsl`. See the `examples` folder for how to write programs and perform inference on them. You can also directly write programs in the IR which is defined in `kc.terms` and `kc.real_values`.

Note: this is a work in progress implementation! Not everything is well documented and things may break.

# Misc. Notes

## TODO
- Can we do ObserveReal on the "p" variables?
- Can we do inference over the "p" variables vs. just marginalizing them out?
- Gaussian observe linalg optimizations
    - Efficiently check 3 cases:
        1. new lin. indep. observe --> add it
        2. new lin. dep. observe that is compatible --> continue
        3. new mutually incompatible observe --> prune this branch
- Shared Gaussian observes trie data structure
    - Rather than a list of all universes, have some way to capture gaussian observes being shared between mixture components
    - Option 1. trie data structure, store observe 'prefix' as we go
    - Option 2. have some way to represent 'bags' of observes that are shared between branches. Different pairs of branches may overlap in different ways that are not representable in a trie (i.e. one pair shares a, b, other shares b, c, another shares a, c)
- Add a basic parser
- Write some representative programs

## Gaussian obs data structure thoughts
- One issue with Trie structure is you want to do a compatibility/redundancy check for each new obs vector you add. One way to do this is with QR decomposition, but in a Trie you don't have a single matrix - you have rows stored at each level.
- Maybe something much more sparse, where you just have a packed array of all observation vectors, and different paths are just indexed by a trie (or just operate on paths in parallel).
- Try and come up with some 'minimal set basis' instead of a trie-like structure.
- Rather than having likelihood nodes all the way at the leaves, maybe we can push them up as much as possible in order to avoid having to do an O(n) pass to update all the likelihoods.

## Ideas
- Can we involve types within the inference and distribution representation?
    - The different 'leaves' that appear in the posterior mixture could have different types, and we could also observe those types.

## Things to look at
- SPPL sum product networks
    - compare inequality implementation - is theirs somehow stronger/weaker/different?

## Beta-Bernoulli
- We represent beta-bernoulli posterior in terms of their sufficient statistics i.e. count of successful trials
    - e.g. currently Ax = b rep for Gaussians, for BB we have finite map {beta names} -> (head_count, tail_count)
    - similarly the *prior* of the beta is not tracked in these leaves, it is stored elsewhere
- We have the possibility to observe beta, which in some sense 'collapses' the posterior rep (head_count, tail_count) and just fixes a p value
    - interesting potential interactions when the observe is conditional 
- In the case where we *don't* directly observe, the contributions from betas occur inside the Flip statements:
    - assign weights to the flip statements that look like a positive/negative weight of:
        {current_beta} -> (head_count + 1, tail_count) => Flip is heads, continue execution
        {current_beta} -> (head_count, tail_count + 1) => Flip is tails, continue execution
- At the end we get some big SPN representation that we recursively do inference on

## SPN representation (for Gaussians and Betas)
- Every node should store:
    - scope (subset of cont. vars it refers to)
- Leaf nodes represents a collection of observations 
    - stores a *specific* concrete realization Ax = b (+ beta obs)
- Sum nodes
    - list of children (which in normalized form should be leaf/product nodes), each with weight 
    - The scope of the sum node is the union of the scopes of the children
- Product nodes
    - list of children (which in normalized form should be leaf/sum nodes) *no weights*
    - The scopes are *disjoint*
    - Scope of product is again union of child scopes
- Operations
    - ADD: 
        - Sum(a, b, ...) + Product = Sum(a, b, ..., Product)
        - Sum(a, b, ...) + Sum(x, y, ...) = Sum(a, b, ..., x, y, ...)
        - Product + Product = Sum(Product, Product)
    - MUL:
        1. IF scopes disjoint
            - A * B = Product(A, B) (also normalizing nested products i.e. if A or B is a product)
        2. ELSE
            - some recursive thing that terminates early when scopes are disjoint (figure it out!)
            (To look at later: vtree - way to constrain possible factorizations so that they "align" more)


# TODO
- beta observes
- hash cons
    - table of spn description to pointer. 
        - spn description = pointers of children
- think about how to handle cases where it might be ok to pass beta as gauss param
- Add Dirichlet processes