#pragma once

#define MAX_MOMENTUM 4

// L1 <= L2
#define L_PAIR_TO_INDEX(L1, L2) ((L2) + (L1)*MAX_MOMENTUM - (L1) * ((L1)-1) / 2)

#define MAX_MOMENTUM_INDEX L_PAIR_TO_INDEX(MAX_MOMENTUM, MAX_MOMENTUM)

// Return the number of atomic orbital functions in shells of momentum 0 to L
// TODO: Find a better name
#define COMPUTE_H(L12) (((L12) + 1) * ((L12) + 2) * ((L12) + 3) / 6)
