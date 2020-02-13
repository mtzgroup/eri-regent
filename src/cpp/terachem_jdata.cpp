#include "eri_regent.h"
#include "helper.h"

EriRegent::TeraChemJDataList::TeraChemJDataList() {
  for (int i = 0; i < MAX_MOMENTUM_INDEX + 1; i++) {
    num_jbras[i] = 0;
    num_jkets[i] = 0;
  }
}

void EriRegent::TeraChemJDataList::allocate_jbras(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = L_PAIR_TO_INDEX(L1, L2);
  assert(num_jbras[index] == 0);
  if (n > 0) {
    num_jbras[index] = n;
    jbras[index] = calloc(n, stride(L1, L2));
    assert(jbras[index]);
  }
}

void EriRegent::TeraChemJDataList::allocate_jkets(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = L_PAIR_TO_INDEX(L1, L2);
  assert(num_jkets[index] == 0);
  if (n > 0) {
    num_jkets[index] = n;
    jkets[index] = calloc(n, stride(L1, L2));
    assert(jkets[index]);
  }
}

void EriRegent::TeraChemJDataList::free_data() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      if (num_jbras[index] > 0) {
        free(jbras[index]);
      }
      if (num_jkets[index] > 0) {
        free(jkets[index]);
      }
    }
  }
}

int EriRegent::TeraChemJDataList::get_num_jbras(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return num_jbras[L_PAIR_TO_INDEX(L1, L2)];
}

int EriRegent::TeraChemJDataList::get_num_jkets(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return num_jkets[L_PAIR_TO_INDEX(L1, L2)];
}

void EriRegent::TeraChemJDataList::set_jbra(
    int L1, int L2, int i, const EriRegent::TeraChemJData &src) {
  assert(0 <= i && i < get_num_jbras(L1, L2));
  void *dest = (char *)jbras[L_PAIR_TO_INDEX(L1, L2)] + i * stride(L1, L2);
  memcpy(dest, (const void *)&src, sizeof_jdata());
}

void EriRegent::TeraChemJDataList::set_jket(
    int L1, int L2, int i, const EriRegent::TeraChemJData &src) {
  assert(0 <= i && i < get_num_jkets(L1, L2));
  void *dest = (char *)jkets[L_PAIR_TO_INDEX(L1, L2)] + i * stride(L1, L2);
  memcpy(dest, (const void *)&src, sizeof_jdata());
}

const double *EriRegent::TeraChemJDataList::get_output(int L1, int L2, int i) {
  assert(0 <= i && i < get_num_jbras(L1, L2));
  return (double *)((char *)jbras[L_PAIR_TO_INDEX(L1, L2)] +
                    i * stride(L1, L2) + sizeof_jdata());
}

void EriRegent::TeraChemJDataList::set_density(int L1, int L2, int i,
                                               const double *src) {
  assert(0 <= i && i < get_num_jkets(L1, L2));
  void *dest = (char *)jkets[L_PAIR_TO_INDEX(L1, L2)] + i * stride(L1, L2) +
               sizeof_jdata();
  memcpy(dest, (const void *)src, sizeof_jdata_array(L1, L2));
}

size_t EriRegent::TeraChemJDataList::sizeof_jdata() {
  return 5 * sizeof(double) + sizeof(float);
}

size_t EriRegent::TeraChemJDataList::sizeof_jdata_array(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return sizeof(double) * TETRAHEDRAL_NUMBER(L1 + L2 + 1);
}

size_t EriRegent::TeraChemJDataList::stride(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return sizeof_jdata() + sizeof_jdata_array(L1, L2);
}

int EriRegent::TeraChemJDataList::get_largest_momentum() {
  auto max = [](int a, int b) { return (a > b) ? a : b; };
  int largest_momentum = -1;
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      if (get_num_jbras(L1, L2) > 0 || get_num_jkets(L1, L2) > 0) {
        largest_momentum = max(largest_momentum, max(L1, L2));
      }
    }
  }
  return largest_momentum;
}
