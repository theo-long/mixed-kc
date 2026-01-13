# TODO
- implement observe for symbolic union
- make sure examples are all working correctly
- add some more type info
- both branches of if/else must return same type
- fix the gaussian union problem where we observe *both gaussians* being equal to the same value
    - do this by adding a BDD node that represents "g1 = g2" which is *only* turned on when we actually observe this
    - if we are only dealing with observes directly on variables, this might be ok - however there is a potential issue where we have some kind of pigeonhole principle e.g. we have three gaussians, we observe every pair = 1.0

# Ideas
- Values drawn from a Gaussian but we only ever observe/use booleans from them e.g. N(0, 1) > 0.5