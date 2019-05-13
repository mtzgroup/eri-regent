import numpy as np
from scipy import integrate, exp


def integrand(u, t, j):
    return u ** (2.0 * j) * exp(-t * u ** 2.0)


def boys(t, j):
    return integrate.quadrature(integrand, 0.0, 1.0, args=(t, j), tol=1e-15, rtol=1e-15)


if __name__ == "__main__":
    from subprocess import Popen, PIPE
    import sys, re

    RED = "\033[1;31m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    capture_number = "([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    t_pattern = re.compile("t=" + capture_number)
    r_pattern = re.compile("R000(\d+)=" + capture_number)

    # TODO: Find a reasonable range to test
    input = "\n".join([str(t) for t in np.linspace(0.0012, 123.456, 1000)])
    rg_process = Popen(
        ["regent", "mcmurchie/test_boys.rg"], cwd="src/", stdin=PIPE, stdout=PIPE
    )
    output, _ = rg_process.communicate(input)

    for line in output.rstrip().split("\n"):
        t = float(t_pattern.findall(line)[0])
        r_parsed = r_pattern.findall(line)
        actual = np.array([float(v) for _, v in r_parsed])
        expected = np.array([boys(t, int(j))[0] for j, _ in r_parsed])
        if not np.allclose(actual, expected, atol=1e-12):
            error = np.max(np.absolute(actual - expected))
            sys.stdout.write(RED)
            print("Error in Boys computation!")
            sys.stdout.write(RESET)
            print("Absolute error = " + str(error) + ", t = " + str(t))
            print("Actual:")
            print(actual)
            print("Expected:")
            print(expected)
            rg_process.wait()
            sys.exit(1)

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
    sys.exit(rg_process.wait())
