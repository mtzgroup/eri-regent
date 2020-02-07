#include "eri_regent.h"
#include "helper.h"

size_t sizeof_kpairs() {
  return 11 * sizeof(double) + sizeof(float) + 2 * sizeof(EriRegent::int1d_t);
}

size_t sizeof_kdensity(int L2, int L4) {
  // TODO
  assert(0 <= L2 && L2 <= L4 && L4 <= MAX_MOMENTUM);
  const int H2 = TRIANGLE_NUMBER(L2 + 1);
  const int H4 = TRIANGLE_NUMBER(L4 + 1);
  return H2 * H4 * sizeof(double) + sizeof(float);
}

EriRegent::TeraChemKDataList::TeraChemKDataList() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      num_kpairs[INDEX_SQUARE(L1, L2)] = 0;
    }
  }
  for (int L = 0; L <= MAX_MOMENTUM; L++) {
    num_shells[L] = 0;
  }
}

EriRegent::TeraChemKDataList::~TeraChemKDataList() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      if (num_kpairs[index] > 0) {
        free(kpairs[index]);
      }
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      if (num_shells[L1] > 0 && num_shells[L2] > 0) {
        const int index = INDEX_UPPER_TRIANGLE(L1, L2);
        free(kdensity[index]);
        free(koutput[index]);
      }
    }
  }
}

int EriRegent::TeraChemKDataList::get_num_kpairs(int L1, int L2) {
  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_SQUARE(L1, L2);
  return num_kpairs[index];
}

int EriRegent::TeraChemKDataList::get_num_kdensity(int L2, int L4) {
  //
  return 1;
}

void EriRegent::TeraChemKDataList::allocate_kpairs(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_SQUARE(L1, L2);
  assert(num_kpairs[index] == 0);
  if (n > 0) {
    num_kpairs[index] = n;
    kpairs[index] = calloc(n, sizeof_kpairs());
    assert(kpairs[index]);
  }
}

void EriRegent::TeraChemKDataList::allocate_kdensity(int L2, int L4, int n) {
  assert(0 <= L2 && L2 <= L4 && L4 <= MAX_MOMENTUM);
  const int index = INDEX_UPPER_TRIANGLE(L2, L4);
  assert(num_kdensity[index] == 0);
  if (n > 0) {
    num_kdensity[index] = n;
    kdensity[index] = calloc(n, sizeof_kdensity(L2, L4));
    assert(kdensity[index]);
  }
}

// void EriRegent::TeraChemKDataList::set_num_shells(int[MAX_MOMENTUM + 1]
// shell_counts) {
//   // TODO
//   for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
//     for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
//       if (num_shells[L1] > 0 && num_shells[L2] > 0) {
//         const int index = INDEX_UPPER_TRIANGLE(L2, L4);
//         kdensity[index] = calloc(n, sizeof_kdensity(L2, L4))
//       }
//     }
//   }
// }

void EriRegent::TeraChemKDataList::set_kpair(
    int L1, int L2, int i, double x, double y, double z, double eta, double C,
    float bound, double ishell_x, double ishell_y, double ishell_z,
    double jshell_x, double jshell_y, double jshell_z,
    EriRegent::int1d_t ishell_index, EriRegent::int1d_t jshell_index) {
  assert(0 <= i && i < get_num_kpairs(L1, L2));
  char *dest = (char *)kpairs[INDEX_SQUARE(L1, L2)] + i * sizeof_kpairs();
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
  {
    double *ptr = (double *)(dest + 5 * sizeof(double) + sizeof(float));
    ptr[0] = ishell_x;
    ptr[1] = ishell_y;
    ptr[2] = ishell_z;
    ptr[3] = jshell_x;
    ptr[4] = jshell_y;
    ptr[5] = jshell_z;
  }
  {
    EriRegent::int1d_t *ptr =
        (EriRegent::int1d_t *)(dest + 11 * sizeof(double) + sizeof(float));
    ptr[0] = ishell_index;
    ptr[1] = jshell_index;
  }
}

void EriRegent::TeraChemKDataList::set_kdensity(int L2, int L4, int i,
                                                const double *values,
                                                float bound) {
  assert(0 <= i && i <= get_num_kdensity(L2, L4));
  char *dest = (char *)kdensity[INDEX_UPPER_TRIANGLE(L2, L4)] +
               i * sizeof_kdensity(L2, L4);
  const int H2 = TRIANGLE_NUMBER(L2 + 1);
  const int H4 = TRIANGLE_NUMBER(L4 + 1);
  memcpy((void *)dest, (const void *)values, H2 * H4 * sizeof(double));
  float *bound_ptr = (float *)(dest + H2 * H4 * sizeof(double));
  bound_ptr[0] = bound;
}

const double *EriRegent::TeraChemKDataList::get_koutput(int L1, int L2, int L3,
                                                        int L4, int i) {
  // TODO
  // assert(0 <= i && i <= )
  return NULL;
}

int EriRegent::TeraChemKDataList::get_largest_momentum() {
  int largest_momentum = -1;
  for (int L = 0; L <= MAX_MOMENTUM; L++) {
    if (num_shells[L] > 0) {
      largest_momentum = L;
    }
  }
  return largest_momentum;
}
