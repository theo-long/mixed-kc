# TODO
- add some documentation and flags for turning features on/off
- add ability to reference inequalities as Boolean variables, and then inequality observes are just handled as usual boolean observes

# Questions
- Are we actually using the BDD library anywhere for WMC, or just to construct BDD?

# Ideas
- Can we allow flip params to be symbolic values and perform inference on them? Point is that we can use KC to compile a formula for P(outcome | flip_thetas) which is some rational function in flip_thetas (since it is P(outcome) / P(observes all hold), both of which are polynomials in flip_thetas).
- Adding in continuous latents that are conjugate

# Things to look at
- SPPL sum product networks