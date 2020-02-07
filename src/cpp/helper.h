#pragma once
// TODO: Put in eri_regent.h

/**
 * The compiled maximum angular momentum.
 */
#define MAX_MOMENTUM 4

/**
 * Maps a pair of angular momentums to an index.
 * Assumes L1 <= L2
 */
// TODO: Remove
#define L_PAIR_TO_INDEX(L1, L2) ((L2) + (L1)*MAX_MOMENTUM - (L1) * ((L1)-1) / 2)

/**
 * The largest angular momentum index.
 */
#define MAX_MOMENTUM_INDEX L_PAIR_TO_INDEX(MAX_MOMENTUM, MAX_MOMENTUM)

#define TETRAHEDRAL_NUMBER(n) (((n) * ((n) + 1) * ((n) + 2)) / 6)
#define TRIANGLE_NUMBER(n) (((n) * ((n) + 1)) / 2)

/**
 * Gives the index into the elements of a square.
 */
#define INDEX_SQUARE(L1, L2) ((L2) + (L1) * ((MAX_MOMENTUM) + 1))

/**
 * Gives the index into the upper triangular elements of a square.
 * Assumes L1 <= L2.
 */
#define INDEX_UPPER_TRIANGLE(L1, L2)                                           \
  (INDEX_SQUARE(L1, L2) - TRIANGLE_NUMBER(L1))
