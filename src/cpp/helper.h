#pragma once

/**
 * The compiled maximum angular momentum.
 */
#define MAX_MOMENTUM 4

/**
 * Useful equations for indexing.
 */
#define TETRAHEDRAL_NUMBER(N) (((N) * ((N) + 1) * ((N) + 2)) / 6)
#define TRIANGLE_NUMBER(N) (((N) * ((N) + 1)) / 2)

/**
 * Gives the index into the elements of a square.
 */
#define INDEX_SQUARE(Y, X) ((Y) * (MAX_MOMENTUM + 1) + (X))

/**
 * Gives the index into the upper triangular elements of a square.
 * Assumes Y <= X.
 */
#define INDEX_UPPER_TRIANGLE(Y, X) (INDEX_SQUARE(Y, X) - TRIANGLE_NUMBER(Y))
