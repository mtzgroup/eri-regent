#pragma once

#include <stdint.h>

struct TeraChemJBra {
  double x;
  double y;
  double z;
  double eta;
  double C;
  float bound;
  double *output;
};

struct TeraChemJKet {
  double x;
  double y;
  double z;
  double eta;
  double C;
  float bound;
  double *density;
};

struct TeraChemJBraList {
  size_t length;
  TeraChemJBra *data;
};

struct TeraChemJKetList {
  size_t length;
  TeraChemJKet *data;
};
