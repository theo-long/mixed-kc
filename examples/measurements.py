from kc import dsl
from kc import run_kc

# Example 1: Noisy measurements with unknown units
# Measurements could be cm, m, feet, inches.
# We want posterior density in metres.


def infer_height(measurements):
    with dsl.Model() as m:
        # Prior on true height in meters
        # e.g. humans are around 1.7m +/- 0.3m
        true_height_m = dsl.gaussian(1.7, 0.3, name="true_height_m")

        # Prior on unit used.
        # We need a categorical choice over 4 options.
        # We simulate this with 2 flips (bits).
        # 00 = CM, 01 = M, 10 = FEET, 11 = INCHES

        u1 = dsl.flip(0.5, name="u1")
        u2 = dsl.flip(0.5, name="u2")

        # Logic to determine scale factor to meters
        # if u1 check u2...

        # Scale factors:
        # CM -> M: 0.01 (u1=F, u2=F)
        # M -> M: 1.0   (u1=F, u2=T)
        # FT -> M: 0.3048 (u1=T, u2=F)
        # IN -> M: 0.0254 (u1=T, u2=T)

        scale_if_u1_false = dsl.if_else(u2, 1.0, 0.01)
        scale_if_u1_true = dsl.if_else(u2, 0.0254, 0.3048)

        scale_to_m = dsl.if_else(u1, scale_if_u1_true, scale_if_u1_false)

        # Measurement noise (e.g. 1cm error, converted to meters roughly?
        # actually noise depends on unit usually? let's assume noise is relative or fixed in measurement space)
        # "noise of measurements" usually implies "I measured X, with error E in that unit".
        # So observed = (true_height / scale) + noise
        # => observed * scale = true_height + noise * scale ?
        # Let's simple model: observed value is Normal(true_height / scale, noise_in_measure_units)

        noise = 0.1  # e.g. 0.1 of a unit error

        for i, val in enumerate(measurements):
            # We predict the measurement in the *unknown unit*
            predicted_val_in_unit = true_height_m / scale_to_m

            # Observe that the actual measurement 'val' is close to predicted
            # Error model: val ~ N(predicted, noise)
            dsl.observe(val == dsl.gaussian(predicted_val_in_unit, noise))

    # Compile the model to get the IR for the true height
    query = m.compile(true_height_m)
    return query


if __name__ == "__main__":
    # Example data: someone is 6 feet tall (approx 1.83m)
    # 6.0 ft, 72.0 inches, 183 cm
    data = [6.0, 6.1]  # Just feet

    print("Compiling model for data:", data)
    ir = infer_height(data)
    print("Compilation successful.")

    # In a real scenario, we would interpret 'ir' to get a density.
    # For now, we print the IR structure or run it if possible.
    # result = run_kc(ir)
    # print("Result BDD:", result)
