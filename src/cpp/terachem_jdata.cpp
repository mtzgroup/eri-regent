#include "eri_regent.h"
#include "helper.h"


EriRegent::TeraChemJDataList::TeraChemJDataList() {
  for (int i = 0; i < TRIANGLE_NUMBER(MAX_MOMENTUM + 1); i++) {
    num_jbras[i] = 0;
    num_jkets[i] = 0;
  }
}

EriRegent::TeraChemJDataList::~TeraChemJDataList() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      if (num_jbras[index] > 0) {
        free(jbras[index]);
      }
      if (num_jkets[index] > 0) {
        free(jkets[index]);
      }
    }
  }
}

void EriRegent::TeraChemJDataList::allocate_jbras(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_UPPER_TRIANGLE(L1, L2);
  assert(num_jbras[index] == 0);
  if (n > 0) {
    num_jbras[index] = n;
    jbras[index] = calloc(n, stride(L1, L2));
    assert(jbras[index]);
  }
}

void EriRegent::TeraChemJDataList::allocate_jkets(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_UPPER_TRIANGLE(L1, L2);
  assert(num_jkets[index] == 0);
  if (n > 0) {
    num_jkets[index] = n;
    jkets[index] = calloc(n, stride(L1, L2));
    assert(jkets[index]);
  }
}

int EriRegent::TeraChemJDataList::get_num_jbras(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return num_jbras[INDEX_UPPER_TRIANGLE(L1, L2)];
}

int EriRegent::TeraChemJDataList::get_num_jkets(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return num_jkets[INDEX_UPPER_TRIANGLE(L1, L2)];
}

void EriRegent::TeraChemJDataList::set_jbra(int L1, int L2, int i, double x,
                                            double y, double z, double eta,
                                            double C, float bound) {
  assert(0 <= i && i < get_num_jbras(L1, L2));
  char *dest = (char *)jbras[INDEX_UPPER_TRIANGLE(L1, L2)] + i * stride(L1, L2);
  {
    double *ptr = (double *)dest;
    ptr[0] = x;
    ptr[1] = y;
    ptr[2] = z;
    ptr[3] = eta;
    ptr[4] = C;
  }
  {
    float *ptr = (float *)(dest + 5 * sizeof(double));
    ptr[0] = bound;
  }
}

void EriRegent::TeraChemJDataList::set_jket(int L1, int L2, int i, double x,
                                            double y, double z, double eta,
                                            double C, float bound) {
  assert(0 <= i && i < get_num_jkets(L1, L2));
  char *dest = (char *)jkets[INDEX_UPPER_TRIANGLE(L1, L2)] + i * stride(L1, L2);
  {
    double *ptr = (double *)dest;
    ptr[0] = x;
    ptr[1] = y;
    ptr[2] = z;
    ptr[3] = eta;
    ptr[4] = C;
  }
  {
    float *ptr = (float *)(dest + 5 * sizeof(double));
    ptr[0] = bound;
  }
}

const double *EriRegent::TeraChemJDataList::get_joutput(int L1, int L2, int i) {
  assert(0 <= i && i < get_num_jbras(L1, L2));
  return (double *)((char *)jbras[INDEX_UPPER_TRIANGLE(L1, L2)] +
                    i * stride(L1, L2) + sizeof_jdata());
}

void EriRegent::TeraChemJDataList::set_jdensity(int L1, int L2, int i,
                                                const double *src) {
  assert(0 <= i && i < get_num_jkets(L1, L2));
  void *dest = (char *)jkets[INDEX_UPPER_TRIANGLE(L1, L2)] +
               i * stride(L1, L2) + sizeof_jdata();
  memcpy(dest, (const void *)src, sizeof_jdata_array(L1, L2));
}

int EriRegent::TeraChemJDataList::get_largest_momentum() {
  int largest_momentum = -1;
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      if (get_num_jbras(L1, L2) > 0 || get_num_jkets(L1, L2) > 0) {
        if (largest_momentum < L1) {
          largest_momentum = L1;
        }
        if (largest_momentum < L2) {
          largest_momentum = L2;
        }
      }
    }
  }
  return largest_momentum;
}
