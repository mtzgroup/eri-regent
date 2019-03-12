import numpy as np
from scipy import integrate, exp

FILE_NAME = "precomputedBoys.h"
# Add 6 because we use a seven term Taylor expansion
LARGEST_J = 4 + 6


def integrand(u, t, j):
    return u ** (2.0 * j) * exp(-t * u ** 2.0)


def boys(t, j):
    return integrate.quadrature(integrand, 0.0, 1.0, args=(t, j), tol=1e-15, rtol=1e-15)


with open(FILE_NAME, "w") as f:
    values, errors = np.transpose(
        [[boys(t / 10.0, j) for j in range(LARGEST_J + 1)] for t in range(121)],
        axes=(2, 0, 1),
    )
    relative_errors = errors / values

    f.write(
"""#pragma once

// Row-major order
// precomputed_boys[t * %d + j] = boys(t / 10.f, j)
const double _precomputed_boys[%d] = {
  %s
};
""" % (LARGEST_J + 1, values.size, ',\n  '.join(map(str, values.flatten())))
    )

    print("Wrote Boys values with shape " + str(values.shape) + " to " + FILE_NAME)
    print("max absolute error = " + str(np.max(errors)))
    print("max relative error = " + str(np.max(relative_errors)))
