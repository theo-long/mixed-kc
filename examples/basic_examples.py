from kc import get_spn
from kc import dsl, run_kc


def discrete_example():
    with dsl.Model() as m:
        f1 = dsl.flip(0.5, name="f1")
        f2 = dsl.flip(0.1, name="f2")
        f3 = dsl.flip(0.8, name="f3")
        x = dsl.ifthenelse(
            f1, f2, dsl.ifthenelse(f3, dsl.const(False), dsl.const(True))
        )
        dsl.observe(x)
    ir = m.compile("f1")
    posterior, Z = run_kc(ir)
    print("----- Discrete Example -----")
    print(f"Posterior probability of f1 == True: {posterior: .2%}")
    print()
    return m


def simple_gaussian_mixture():
    with dsl.Model() as m:
        g1 = dsl.gaussian(0.0, 1.0)
        g2 = dsl.gaussian(0.0, 2.0)
        f1 = dsl.flip(0.5, name="f1")
        x = dsl.ifthenelse(f1, g1, g2)
        dsl.observe(x, 1.0)
    ir = m.compile("f1")
    posterior, Z = run_kc(ir)
    spn = get_spn(ir)
    print("----- Simple Gaussian Mixture -----")
    print("SPN: ")
    print(spn)
    print("Posterior: ")
    print(f"Posterior probability of f1 == True: {posterior: .2%}")
    print()
    return m


def measure_zero_gaussian_example():
    with dsl.Model() as m:
        g1 = dsl.gaussian(0.0, 1.0)
        g2 = dsl.gaussian(0.0, 2.0)
        f1 = dsl.flip(0.5, name="f1")
        x = dsl.ifthenelse(f1, g1, g2)
        dsl.observe(x, 1.0)
        dsl.observe(g1, 1.0)
    ir = m.compile("f1")
    posterior, Z = run_kc(ir)
    spn = get_spn(ir)
    print("----- Measure Zero Gaussian Example -----")
    print("SPN: ")
    print(spn)
    print("Posterior: ")
    print(f"Posterior probability of f1 == True: {posterior: .2%}")
    print()
    return m


def interacting_gaussian_mixture():
    with dsl.Model() as m:
        g1 = dsl.gaussian(0.0, 1.0, name="g1")
        g2 = dsl.gaussian(0.0, 1.0, name="g2")
        x = dsl.ifthenelse(dsl.flip(0.75), g1 + g2, g1 * 2.0)
        dsl.observe(x, 1.0)

    ir = m.compile("g1")
    posterior, Z = run_kc(ir)
    spn = get_spn(ir)
    print("----- Interacting Gaussian Mixture -----")
    print("SPN: ")
    print(spn)
    print("Posterior: ")
    for component in posterior:
        print(component)
    print()
    return m


def beta_example():
    with dsl.Model() as m:
        b = dsl.beta(1, 1, name="b")
        f1 = dsl.flip(b)
        dsl.observe(f1)
    ir = m.compile("b")
    posterior, Z = run_kc(ir)
    spn = get_spn(ir)
    print("----- Beta Example -----")
    print("SPN: ")
    print(spn)
    print("Posterior: ")
    if isinstance(posterior, list):
        for component in posterior:
            print(component)
    else:
        print(posterior)
    print()
    return m


def multiple_beta_example():
    with dsl.Model() as m:
        b = dsl.beta(1, 1, name="b")
        f1 = dsl.flip(b)
        f2 = dsl.flip(b)
        x = dsl.ifthenelse(dsl.flip(0.5), f1, f2)
        dsl.observe(x)
    ir = m.compile("b")
    posterior, Z = run_kc(ir)
    print("----- Multiple Beta Example -----")
    if isinstance(posterior, list):
        for component in posterior:
            print(component)
    else:
        print(posterior)
    print()
    return m


if __name__ == "__main__":
    discrete_example()
    simple_gaussian_mixture()
    measure_zero_gaussian_example()
    interacting_gaussian_mixture()
    beta_example()
    multiple_beta_example()
