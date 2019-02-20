from scipy import integrate, exp

FILE_NAME = "precomputedBoys.h"
FUNCTION_NAME = "getPrecomputedBoys"
ARRAY_NAME = "_precomputedBoys"
# Add 6 because we use a seven term Taylor expansion
LARGEST_J = 16 + 6  # TODO: update this value once it is known


def integrand(u, t, j):
    return u ** (2. * j) * exp(-t * u ** 2.)


def boys(t, j):
    return integrate.quadrature(integrand, 0., 1., args=(t, j), tol=1e-15, rtol=1e-15)


with open(FILE_NAME, "w") as file:
    values = [boys(t / 10.0, j) for t in range(121) for j in range(LARGEST_J + 1)]
    max_abs_error = max([e for _, e in values] + [0.0])
    max_rel_error = max([e / r for r, e in values] + [0.0])
    # TODO: Set precision here
    array_body = ",\n    ".join(["%.32f" % r for r, _ in values])
    header = """#pragma once
#include <assert.h>

double """ + FUNCTION_NAME + """(double t, int j);

const double """ + ARRAY_NAME + "[" + str(len(values)) + """] = {
    """ + array_body + """
};

double """ + FUNCTION_NAME + """(double t, int j) {
    int t_index = 10 * t;
    assert(t_index >= 0 && t_index <= 120);
    assert(j >= 0 && j <= """ + str(LARGEST_J) + """);
    return """ + ARRAY_NAME + "[" + str(LARGEST_J + 1) + """ * t_index + j];
}
"""
    file.write(header)

    print("Wrote header file to " + FILE_NAME)
    print("max absolute error = " + str(max_abs_error))
    print("max relative error = " + str(max_rel_error))
