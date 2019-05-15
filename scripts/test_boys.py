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
    boys_pattern = re.compile("boys(\d+)=" + capture_number)

    for max_j in range(17):
        input = "\n".join([str(t) for t in np.linspace(0, 123.456, 1000)])
        rg_process = Popen(
            ["regent", "mcmurchie/test_boys.rg", str(max_j)],
            cwd="src/",
            stdin=PIPE,
            stdout=PIPE,
        )
        output, _ = rg_process.communicate(input)

        for line in output.rstrip().split("\n"):
            t = float(t_pattern.findall(line)[0])
            boys_parsed = boys_pattern.findall(line)
            actual = np.array([float(v) for _, v in boys_parsed])
            expected = np.array([boys(t, int(j))[0] for j, _ in boys_parsed])
            atol = 1e-14 if t < 25 else 1e-12
            if not np.allclose(actual, expected, atol=atol):
                absolute_error = np.max(np.absolute(actual - expected))
                relative_error = np.max(
                    np.absolute(actual - expected) / np.absolute(expected)
                )
                sys.stdout.write(RED)
                print("Error in Boys computation!")
                sys.stdout.write(RESET)
                print("Absolute error = " + str(absolute_error))
                print("Relative error = " + str(relative_error))
                print("t = " + str(t))
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
