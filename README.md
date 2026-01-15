# TODO
- implement observe for symbolic union
- make sure examples are all working correctly
- add some more type info
- both branches of if/else must return same type

# Questions
- Are we actually using the BDD library anywhere for WMC, or just to construct BDD?

# Ideas
- Ability to actually use inequalities as Boolean type variables. (right now only observes)
- Can we allow flip params to be symbolic values and perform inference on them? Point is that we can use KC to compile a formula for P(outcome | flip_thetas) which is some rational function in flip_thetas (since it is P(outcome) / P(observes all hold), both of which are polynomials in flip_thetas).