#include "eri_regent.h"
#include "helper.h"

size_t sizeof_kpairs() {
  return 11 * sizeof(double) + sizeof(float) + 2 * sizeof(EriRegent::int1d_t);
}

size_t sizeof_kdensity(int L2, int L4) {
  assert(0 <= L2 && L2 <= L4 && L4 <= MAX_MOMENTUM);
  const int H2 = TRIANGLE_NUMBER(L2 + 1);
  const int H4 = TRIANGLE_NUMBER(L4 + 1);
  return H2 * H4 * sizeof(double) + sizeof(float);
}

size_t sizeof_koutput(int L1, int L3) {
  assert(0 <= L1 && L1 <= L3 && L3 <= MAX_MOMENTUM);
  const int H1 = TRIANGLE_NUMBER(L1 + 1);
  const int H3 = TRIANGLE_NUMBER(L3 + 1);
  return H1 * H3 * sizeof(double);
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
      if (num_kbra_prevals[index] > 0) {
        free(kbra_prevals[index]);
      }
      if (num_kket_prevals[index] > 0) {
        free(kket_prevals[index]);
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

int EriRegent::TeraChemKDataList::get_num_kbra_prevals(int L1, int L2) {
  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_SQUARE(L1, L2);
  return num_kbra_prevals[index];
}

int EriRegent::TeraChemKDataList::get_num_kket_prevals(int L1, int L2) {
  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_SQUARE(L1, L2);
  return num_kket_prevals[index];
}

int EriRegent::TeraChemKDataList::get_num_shells(int L) {
  assert(0 <= L && L <= MAX_MOMENTUM);
  return num_shells[L];
}

void EriRegent::TeraChemKDataList::allocate_kpairs(int L1, int L2, int n,
                                                   int _num_kbra_prevals,
                                                   int _num_kket_prevals) {
  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = INDEX_SQUARE(L1, L2);
  assert(num_kpairs[index] == 0);
  if (n > 0) {
    num_kpairs[index] = n;
    kpairs[index] = calloc(n, sizeof_kpairs());
    assert(kpairs[index]);
  }
  if (_num_kbra_prevals > 0) {
    num_kbra_prevals[index] = _num_kbra_prevals;
    kbra_prevals[index] = calloc(n, sizeof(double) * _num_kbra_prevals);
    assert(kbra_prevals[index]);
  }
  if (_num_kket_prevals > 0) {
    num_kket_prevals[index] = _num_kket_prevals;
    kket_prevals[index] = calloc(n, sizeof(double) * _num_kket_prevals);
    assert(kket_prevals[index]);
  }
}

void EriRegent::TeraChemKDataList::allocate_kdensity(int L2, int L4, int n2,
                                                     int n4) {
  assert(0 <= L2 && L2 <= L4 && L4 <= MAX_MOMENTUM);
  if (n2 > 0) {
    assert(num_shells[L2] == 0 || num_shells[L2] == n2);
    num_shells[L2] = n2;
  }
  if (n4 > 0) {
    assert(num_shells[L4] == 0 || num_shells[L4] == n4);
    num_shells[L4] = n4;
  }
  if (n2 > 0 && n4 > 0) {
    const int index = INDEX_UPPER_TRIANGLE(L2, L4);
    kdensity[index] = calloc(n2 * n4, sizeof_kdensity(L2, L4));
    assert(kdensity[index]);
  }
}

void EriRegent::TeraChemKDataList::allocate_all_koutput() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      const int n1 = num_shells[L1];
      const int n3 = num_shells[L3];
      const int n = (get_largest_momentum() + 1) * (get_largest_momentum() + 1);
      if (n1 > 0 && n3 > 0) {
        const int index = INDEX_UPPER_TRIANGLE(L1, L3);
        koutput[index] = calloc(n * n1 * n3, sizeof_koutput(L1, L3));
        assert(koutput[index]);
      }
    }
  }
}

void EriRegent::TeraChemKDataList::set_kpair(
    int L1, int L2, int i, double x, double y, double z, double eta, double C,
    float bound, double PIx, double PIy, double PIz, double PJx, double PJy,
    double PJz, int1d_t ishell_index, int1d_t jshell_index) {
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
    ptr[0] = PIx;
    ptr[1] = PIy;
    ptr[2] = PIz;
    ptr[3] = PJx;
    ptr[4] = PJy;
    ptr[5] = PJz;
  }
  {
    int1d_t *ptr = (int1d_t *)(dest + 11 * sizeof(double) + sizeof(float));
    ptr[0] = ishell_index;
    ptr[1] = jshell_index;
  }
}

void EriRegent::TeraChemKDataList::set_kbra_preval(int L1, int L2, int i, int k,
                                                   double value) {
  assert(0 <= i && i < get_num_kpairs(L1, L2));
  assert(0 <= k && k < get_num_kbra_prevals(L1, L2));
  char *dest = (char *)kbra_prevals[INDEX_SQUARE(L1, L2)] +
               i * sizeof(double) * get_num_kbra_prevals(L1, L2) +
               k * sizeof(double);
  {
    double *ptr = (double *)dest;
    ptr[0] = value;
  }
}

void EriRegent::TeraChemKDataList::set_kket_preval(int L1, int L2, int i, int k,
                                                   double value) {
  assert(0 <= i && i < get_num_kpairs(L1, L2));
  assert(0 <= k && k < get_num_kket_prevals(L1, L2));
  char *dest = (char *)kbra_prevals[INDEX_SQUARE(L1, L2)] +
               i * sizeof(double) * get_num_kket_prevals(L1, L2) +
               k * sizeof(double);
  {
    double *ptr = (double *)dest;
    ptr[0] = value;
  }
}

void EriRegent::TeraChemKDataList::set_kdensity(int L2, int L4,
                                                int bra_jshell_index,
                                                int ket_jshell_index,
                                                const double *src,
                                                float bound) {
  assert(L2 <= L4);
  assert(0 <= bra_jshell_index && bra_jshell_index <= get_num_shells(L2));
  assert(0 <= ket_jshell_index && ket_jshell_index <= get_num_shells(L4));
  char *dest = (char *)kdensity[INDEX_UPPER_TRIANGLE(L2, L4)] +
               (bra_jshell_index * get_num_shells(L4) + ket_jshell_index) *
                   sizeof_kdensity(L2, L4);
  const int H2 = TRIANGLE_NUMBER(L2 + 1);
  const int H4 = TRIANGLE_NUMBER(L4 + 1);
  memcpy((void *)dest, (const void *)src, H2 * H4 * sizeof(double));
  float *bound_ptr = (float *)(dest + H2 * H4 * sizeof(double));
  bound_ptr[0] = bound;
}

const double *EriRegent::TeraChemKDataList::get_koutput(int L1, int L2, int L3,
                                                        int L4,
                                                        int bra_ishell_index,
                                                        int ket_ishell_index) {
  assert(L1 < L3 || (L1 == L3 && L2 <= L4));
  const int n1 = get_num_shells(L1);
  const int n3 = get_num_shells(L3);
  assert(0 <= bra_ishell_index && bra_ishell_index <= n1);
  assert(0 <= ket_ishell_index && ket_ishell_index <= n3);
  const char *koutput1234 = (char *)koutput[INDEX_UPPER_TRIANGLE(L1, L3)] +
                            (L2 + L4 * (get_largest_momentum() + 1)) *
                                sizeof_koutput(L1, L3) * n1 * n3;
  const char *src = koutput1234 + (bra_ishell_index * n3 + ket_ishell_index) *
                                      sizeof_koutput(L1, L3);
  return (const double *)src;
}

void *EriRegent::TeraChemKDataList::get_kpair_data(int L1, int L2) {
  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  return kpairs[INDEX_SQUARE(L1, L2)];
}

void *EriRegent::TeraChemKDataList::get_kbra_preval_data(int L1, int L2) {

  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  return kbra_prevals[INDEX_SQUARE(L1, L2)];
}

void *EriRegent::TeraChemKDataList::get_kket_preval_data(int L1, int L2) {

  assert(0 <= L1 && L1 <= MAX_MOMENTUM);
  assert(0 <= L2 && L2 <= MAX_MOMENTUM);
  return kket_prevals[INDEX_SQUARE(L1, L2)];
}

void *EriRegent::TeraChemKDataList::get_kdensity_data(int L2, int L4) {
  assert(0 <= L2 && L2 <= L4 && L4 <= MAX_MOMENTUM);
  return kdensity[INDEX_UPPER_TRIANGLE(L2, L4)];
}

void *EriRegent::TeraChemKDataList::get_koutput_data(int L1, int L3) {
  assert(0 <= L1 && L1 <= L3 && L3 <= MAX_MOMENTUM);
  return koutput[INDEX_UPPER_TRIANGLE(L1, L3)];
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
