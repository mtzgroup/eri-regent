import numpy as np
from scipy import integrate, exp


def integrand(u, t, j):
    return u ** (2.0 * j) * exp(-t * u ** 2.0)


def boys(t, j):
    return integrate.quadrature(integrand, 0.0, 1.0, args=(t, j), tol=1e-15, rtol=1e-15)


if __name__ == "__main__":
    import sys, subprocess, re

    RED = "\033[1;31m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    pattern = re.compile("R000([0-9]+) = (.+)")
    alpha = 12.34
    for t in np.linspace(0.1234, 50.5678, 100):
        output = subprocess.check_output(["regent", "mcmurchie/test_boys.rg", str(t), str(alpha)], cwd="src/")
        parsed = pattern.findall(output)
        actual = np.array([float(v) for _, v in parsed])
        expected = np.array([(-2.0 * alpha) ** int(j) * boys(t, int(j))[0] for j, _ in parsed])
        if not np.allclose(actual, expected, rtol=1e-4):
            error = np.max(np.absolute(actual - expected) / np.absolute(expected))
            sys.stdout.write(RED)
            print("Error in Boys computation")
            sys.stdout.write(RESET)
            print("Relative error = " + str(error) + ", alpha = " + str(alpha) + ", t = " + str(t))
            print("Acutal:")
            print(actual)
            print("Expected:")
            print(expected)
            # sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
