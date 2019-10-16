#pragma once

#include <stdint.h>

struct TeraChemJBra {
  double x;
  double y;
  double z;
  double eta;
  double C;
  float bound;
  double* output;
};

struct TeraChemJKet {
  double x;
  double y;
  double z;
  double eta;
  double C;
  float bound;
  double* density;
};

struct TeraChemJBraList {
  uint8_t L1;
  uint8_t L2;
  size_t length;
  TeraChemJBra* jbras;
};

struct TeraChemJKetList {
  uint8_t L1;
  uint8_t L2;
  size_t length;
  TeraChemJKet* jkets;
};

enum { // Field IDs
  GAMMA_TABLE_FIELD_ID,
};
