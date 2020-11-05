#include "eri_regent.h"
#include "helper.h"

size_t sizeof_jdata() { return 5 * sizeof(double) + sizeof(float); }

size_t sizeof_jdata_array(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return sizeof(double) * TETRAHEDRAL_NUMBER(L1 + L2 + 1);
}

size_t stride(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return sizeof_jdata() + sizeof_jdata_array(L1, L2);
}

