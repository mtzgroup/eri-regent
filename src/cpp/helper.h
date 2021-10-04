#pragma once
#ifndef ERI_HELPER
#define ERI_HELPER
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
/**
 * binary to decimal
 */
#define BINARY_TO_DECIMAL(L1, L2, L3, L4) (L1*8 + L2*4 + L3*2 + L4*1)

//! Determine number of AO functions in a shell of momentum L.
#define ANGL_FUNCS(L) ((L+1)*(L+2)/2)

//! Determine pair index from two separate momenta                                                                            //! Number of angular shell classes                                                                                           
//! Application wide maximum supported angular momentum                                                                                          
#define ANGL_MAX 2

/*! ANGL_MAX is a zero-based, so we define this for clarity when specifying                                                    *  allocation widths, etc. */
#define ANGL_TYPES (ANGL_MAX + 1)
                   
/*! We assume I<=J! */
#define ANGL_PINDEX(L1, L2) ((L2) + (L1)*( 2*ANGL_TYPES-1-(L1) )/2)

#include <stddef.h>

size_t sizeof_jdata();
size_t sizeof_jdata_array(int L1, int L2);
size_t stride(int L1, int L2);

#include <unistd.h>

#define __TRACE
  // {									\
  //char h[80]; gethostname(h, 80);					\
  //std::cout << h << " " << __FUNCTION__ << " " << __FILE__ << ":" << __LINE__ << std::endl; }
#endif
