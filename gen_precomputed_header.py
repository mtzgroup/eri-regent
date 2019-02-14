#!/usr/bin/env python3.7

from scipy import integrate, exp

FILE_NAME = "precomputedBoys.h"
FUNCTION_NAME = "getPrecomputedBoys"
ARRAY_NAME = "_precomputedBoys"
LARGEST_J = 4 + 6  # TODO: update this value once it is known


def boys(t, j):
    return integrate.quad(lambda u: u ** (2 * j) * exp(-t * u ** 2), 0, 1)


with open(FILE_NAME, "w") as file:
    values = [boys(t / 10.0, j) for t in range(121) for j in range(LARGEST_J + 1)]
    max_abs_error = max([e for _, e in values] + [0.0])
    max_rel_error = max([e / r for r, e in values] + [0.0])
    # TODO: Set precision here
    array_body = ",\n    ".join([f"{r:.32f}" for r, _ in values])
    header = f"""#pragma once
#include <assert.h>

double {FUNCTION_NAME}(double t, int j);

const double {ARRAY_NAME}[{len(values)}] = {{
    {array_body}
}};

double {FUNCTION_NAME}(double t, int j) {{
    int t_index = 10 * t;
    assert(t_index >= 0 && t_index <= 120 && j >= 0 && j <= {LARGEST_J});
    return {ARRAY_NAME}[{LARGEST_J + 1} * t_index + j];
}}
"""
    file.write(header)

    print(f"Wrote header file to {FILE_NAME}")
    print(f"max absolute error = {max_abs_error}")
    print(f"max relative error = {max_rel_error}")
