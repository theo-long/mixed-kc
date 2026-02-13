# New Probabilistic DSL Walkthrough

I have implemented a new, ergonomic Python-embedded DSL for writing your probabilistic programs. This DSL allows you to use standard Python variables and operators to build your models, instead of nested `Let` bindings.

## Key Features

1.  **Implicit Variable Binding**: No more `Let("x", ..., Let("y", ...))` spaghetti. Just write `x = ...; y = ...`.
2.  **Operator Overloading**: Use standard `+`, `-`, `*`, `>`, `==` operators on DSL variables.
3.  **Symbolic Gaussians**: You can now use variables as parameters for Gaussians: `gaussian(mean=x, std=1)`.
4.  **Complex Control Flow**: Use `if_else(cond, true_val, false_val)` for branching logic.
5.  **Observations**: Use `observe(condition)` to add constraints.

## Examples

### 1. Basic Gaussian Mixture
```python
from kc import dsl

with dsl.Model() as m:
    b = dsl.flip(0.5, name="b")
    # Clean if-else syntax
    x = dsl.if_else(b, dsl.gaussian(0, 1), dsl.gaussian(0, 2))
    
    # Observe using standard equality
    dsl.observe(x == 1.0)

# Compile to IR
ir = m.compile(b)
```

### 2. Height Measurement (Unknown Units)
Logic that was previously complex is now Python-native:
```python
with dsl.Model() as m:
    true_height = dsl.gaussian(1.7, 0.5, name="h")
    
    # Python loop for multiple measurements
    for obs_val in [1.8, 1.75]:
        # Symbolic mean works automatically!
        # observe(obs_val == N(true_height, 0.1))
        # Note: 'gaussian' creates a new random variable centered at 'true_height'
        measurement_model = dsl.gaussian(true_height, 0.1)
        dsl.observe(measurement_model == obs_val)
```

### 3. Nested Logic
Ported from `test_main.py` (Example `p12`):
```python
with dsl.Model() as m:
    flip1 = dsl.flip(0.5)
    g1 = dsl.gaussian(0, 1)
    
    # Logic on variables
    dsl.observe(g1 > 0.0)
    
    # Branching
    res = dsl.if_else(flip1, g1, dsl.gaussian(0, 1))
    
    m.compile(res)
    m.compile(res)
```

### 4. Measurements with Unknown Units
See `examples/measurements.py` for a full example of inferring a quantity when the unit of measurement (cm, m, feet, inches) is unknown and inferred from the data.
This example demonstrates the usage of `choice` for discrete variables and `switch` for conditional logic based on discrete values.

### 5. Team Combinatorics
See `examples/team_scores.py` for an example of inferring player attributes from match scores, where attributes interact (e.g., doubling strength if a player has a "blue" attribute).


### 6. Bernoulli Priors
You can use `dsl.beta(alpha, beta)` to define a prior over a probability, and pass it to `dsl.flip`.

```python
# p ~ Beta(2, 2)
p = dsl.beta(2.0, 2.0, name="p")
# x ~ Bernoulli(p) 
x = dsl.flip(p, name="x")
```

## files Created
- `kc/dsl.py`: The core implementation.
- `tests/test_dsl.py`: Basic tests.
- `tests/test_dsl_advanced.py`: Advanced examples ported from `test_main.py`.

## Running the Tests
You can run the full suite with:
```bash
uv run pytest tests/test_dsl.py tests/test_dsl_advanced.py
```
