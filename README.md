# TODO
- implement observe for symbolic union
- make sure examples are all working correctly
- add some more type info
- both branches of if/else must return same type
- Truncated/Gated Gaussian
    - Everything should be transformed to a GatedGaussian in the first pass
    - In the KC pass is where we handle the GatedGaussian observe logic
    - The reason we need gated gaussian (instead of just doing everything using IfThenElse and Vars representing various flips) is that the latter approach requires us to add a ton of variables at the top level and then reference them further down vs. handling the logic within the GatedGaussian itself.

# Ideas
- Values drawn from a Gaussian but we only ever observe/use booleans from them e.g. N(0, 1) > 0.5