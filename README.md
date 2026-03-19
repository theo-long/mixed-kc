# Instructions

A small probabilistic programming language for performing exact inference over programs with both discrete and continuous variables using knowledge compilation.

This project uses `uv`. To run the test programs, first install `uv` then run `uv run pytest`.

The easiest way to write programs is using the DSL in `kc.dsl`. See the `examples` folder for how to write programs and perform inference on them. You can also directly write programs in the IR which is defined in `kc.terms` and `kc.real_values`.

Note: this is a work in progress implementation! Not everything is well documented and things may break.

# Misc. Notes

## TODO
- beta observes
- GPs, DPs
- Gaussian observe linalg optimizations
    - Efficiently check 3 cases:
        1. new lin. indep. observe --> add it
        2. new lin. dep. observe that is compatible --> continue
        3. new mutually incompatible observe --> prune this branch
- hash cons
    - table of spn description to pointer. 
        - spn description = pointers of children
- think about how to handle cases where it might be ok to pass beta as gauss param
- truncated SPN inference