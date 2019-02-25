import h5py
import numpy as np
from scipy import integrate, exp

FILE_NAME = "precomputedBoys.hdf5"
# Add 6 because we use a seven term Taylor expansion
LARGEST_J = 16 + 6  # TODO: update this value once it is known


def integrand(u, t, j):
    return u ** (2.0 * j) * exp(-t * u ** 2.0)


def boys(t, j):
    return integrate.quadrature(integrand, 0.0, 1.0, args=(t, j), tol=1e-15, rtol=1e-15)


with h5py.File(FILE_NAME, "w") as f:

    # `values` has shape (121, LARGEST_J + 1)
    values, errors = np.transpose(
        [[boys(t / 10.0, j) for j in range(LARGEST_J + 1)] for t in range(121)],
        axes=(2, 0, 1),
    )
    relative_errors = errors / values

    f["data"] = values

    print("Wrote precomputed Boys to " + FILE_NAME)
    print("max absolute error = " + str(np.max(errors)))
    print("max relative error = " + str(np.max(relative_errors)))
