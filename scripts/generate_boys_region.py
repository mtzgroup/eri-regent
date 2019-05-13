import numpy as np
from scipy import integrate, exp

import sys

if len(sys.argv) != 2:
    print("Usage: python " + sys.argv[0] + " precomputedBoys.h")
    exit(0)

FILE_NAME = sys.argv[1]
MAX_MOMENTUM = 6
# Add 5 because we use a six term Taylor expansion
LARGEST_J = 2 * MAX_MOMENTUM + 1 + 5


def integrand(u, t, j):
    return u ** (2.0 * j) * exp(-t * u ** 2.0)


def boys(t, j):
    return integrate.quadrature(integrand, 0.0, 1.0, args=(t, j), tol=1e-15, rtol=1e-15)


with open(FILE_NAME, "w") as f:
    values, errors = np.transpose(
        [[boys(t / 10.0, j) for j in range(LARGEST_J + 1)] for t in range(251)],
        axes=(2, 0, 1),
    )
    relative_errors = errors / values

    f.write(
        """#pragma once

const unsigned _precomputed_boys_largest_j = %d;

/*
 * Values of the boys function
 * boys(t, j) = integral from 0 to 1 of u^(2j) exp(-2tu) du
 * in row-major order such that
 * _precomputed_boys[t * (_precomputed_boys_largest_j+1) + j] = boys(t/10.f, j)
 */
const double _precomputed_boys[%d] = {
  %s
};
"""
        % (LARGEST_J, values.size, ",\n  ".join(map(str, values.flatten())))
    )

    print("Wrote Boys values with shape " + str(values.shape) + " to " + FILE_NAME)
    print("max absolute error = " + str(np.max(errors)))
    print("max relative error = " + str(np.max(relative_errors)))
