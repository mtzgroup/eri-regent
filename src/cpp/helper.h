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

#include <stddef.h>

size_t sizeof_jdata();
size_t sizeof_jdata_array(int L1, int L2);
size_t stride(int L1, int L2);

#include <unistd.h>

#define __TRACE {\
char h[80]; gethostname(h, 80); \
std::cout << h << " " << __FUNCTION__ << " " << __FILE__ << ":" << __LINE__ << std::endl; }
