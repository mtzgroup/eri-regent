#pragma once

#include "helper.h"

namespace eri_regent {

struct TeraChemJData {
  double x;
  double y;
  double z;
  double eta;
  double C;
  float bound;
};

struct TeraChemJDataList {
  size_t num_jbras[MAX_MOMENTUM_INDEX + 1];
  TeraChemJData* jbras[MAX_MOMENTUM_INDEX + 1];
  double* output[MAX_MOMENTUM_INDEX + 1];

  size_t num_jkets[MAX_MOMENTUM_INDEX + 1];
  TeraChemJData* jkets[MAX_MOMENTUM_INDEX + 1];
  double* density[MAX_MOMENTUM_INDEX + 1];
};

}
