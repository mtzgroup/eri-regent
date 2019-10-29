import numpy as np
from scipy import integrate, exp


def integrand(u, t, j):
    return u ** (2.0 * j) * exp(-t * u ** 2.0)


def boys(t, j):
    tol = np.finfo(np.float64).resolution
    val, err = integrate.quadrature(integrand, 0.0, 1.0, args=(t, j), tol=tol, rtol=tol)
    assert np.max(np.abs(err)) < 2 * tol
    return val


if __name__ == "__main__":
    from subprocess import Popen, PIPE
    import sys, re

    RED = "\033[1;31m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    capture_float = "([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    t_pattern = re.compile("t=" + capture_float)
    boys_pattern = re.compile("boys(\d+)=" + capture_float)

    for max_j in range(17):
        num_inputs = 400
        input = "\n".join(
            ["{:.16f}".format(t) for t in np.linspace(0, 234.567, num_inputs)]
        )
        rg_process = Popen(
            ["regent", "mcmurchie/jfock/test_boys.rg", str(max_j)],
            cwd="src/",
            stdin=PIPE,
            stdout=PIPE,
        )
        output, _ = rg_process.communicate(bytes(input, "utf-8"))

        num_outputs = 0
        for line in str(output).rstrip().split("\\n"):
            try:
                t = np.float64(t_pattern.findall(line)[0])
            except:
                continue
            boys_parsed = boys_pattern.findall(line)
            assert boys_parsed != [], "Did not read in boys values!"
            result = np.array([v for _, v in boys_parsed], dtype=np.float64)
            expected = np.array(
                [boys(t, int(j)) for j, _ in boys_parsed], dtype=np.float64
            )
            if not np.allclose(result, expected, atol=1e-12):
                absolute_error = np.absolute(result - expected)
                relative_error = np.absolute(result - expected) / np.absolute(expected)
                sys.stdout.write(RED)
                print("Error in Boys computation!")
                sys.stdout.write(RESET)
                print(
                    "t = {:.16g}\n"
                    "Max absolute error: {}\n"
                    "Max relative error: {}".format(
                        t, np.max(absolute_error), np.max(relative_error)
                    )
                )
                print("Got:      " + str(result))
                print("Expected: " + str(expected))
                rg_process.wait()
                sys.exit(1)
            num_outputs += 1
        assert num_inputs == num_outputs, "Did not read in all inputs!"

    sys.stdout.write(GREEN)
    print("All tests passed!")
    sys.stdout.write(RESET)
    sys.exit(rg_process.wait())
