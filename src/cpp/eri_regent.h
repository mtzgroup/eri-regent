#pragma once

#include "helper.h"
#include "legion.h"

class EriRegent {
public:
  EriRegent(int max_momentum, Legion::Context &ctx, Legion::Runtime *runtime);
  ~EriRegent();

  struct TeraChemJData {
    double x;
    double y;
    double z;
    double eta;
    double C;
    float bound;
  };

  /**
   * A list of JBra and JKet data to be passed to `launch_jfock_task`.
   * Use L_PAIR_TO_INDEX(L1, L2) to get the index for these arrays.
   */
  // TODO: Make this a class
  struct TeraChemJDataList {
    int num_jbras[MAX_MOMENTUM_INDEX + 1];
    TeraChemJData *jbras[MAX_MOMENTUM_INDEX + 1];
    double *output[MAX_MOMENTUM_INDEX + 1];

    int num_jkets[MAX_MOMENTUM_INDEX + 1];
    TeraChemJData *jkets[MAX_MOMENTUM_INDEX + 1];
    double *density[MAX_MOMENTUM_INDEX + 1];
  };

  /**
   * Launch the jfock regent task and copies the result into
   * `jdata_list.output`.
   */
  void launch_jfock_task(TeraChemJDataList &jdata_list, float threshold,
                         int parallelism);

private:
  // TODO: Rename `max_momentum`. It should be thought of as a runtime constant
  //       instead of a compile-time constant.
  const int max_momentum;
  Legion::Context ctx;
  Legion::Runtime *runtime;
  Legion::Memory memory;
  Legion::LogicalRegion gamma_table_lr;
  Legion::PhysicalRegion gamma_table_pr;
  Legion::FieldSpace jbra_fspaces[MAX_MOMENTUM_INDEX + 1];
  Legion::FieldSpace jket_fspaces[MAX_MOMENTUM_INDEX + 1];

  void *fill_jbra_data(const TeraChemJData *jbras, int num_jbras, int L12);
  void *fill_jket_data(const TeraChemJData *jkets, int num_jkets,
                       const double *density, int L12);

  void initialize_field_spaces();

  void create_gamma_table_region();

#define JBRA_FIELD_ID(L1, L2, F_NAME) JBRA##L1##L2##_FIELD_##F_NAME##_ID
#define JKET_FIELD_ID(L1, L2, F_NAME) JKET##L1##L2##_FIELD_##F_NAME##_ID

#define JBRA_FIELD_IDS(L1, L2)                                                 \
  JBRA_FIELD_ID(L1, L2, X), JBRA_FIELD_ID(L1, L2, Y),                          \
      JBRA_FIELD_ID(L1, L2, Z), JBRA_FIELD_ID(L1, L2, ETA),                    \
      JBRA_FIELD_ID(L1, L2, C), JBRA_FIELD_ID(L1, L2, BOUND),                  \
      JBRA_FIELD_ID(L1, L2, OUTPUT)

#define JKET_FIELD_IDS(L1, L2)                                                 \
  JKET_FIELD_ID(L1, L2, X), JKET_FIELD_ID(L1, L2, Y),                          \
      JKET_FIELD_ID(L1, L2, Z), JKET_FIELD_ID(L1, L2, ETA),                    \
      JKET_FIELD_ID(L1, L2, C), JKET_FIELD_ID(L1, L2, BOUND),                  \
      JKET_FIELD_ID(L1, L2, DENSITY)

  enum { // Field IDs
    GAMMA_TABLE_FIELD_ID,
    JBRA_FIELD_IDS(0, 0),
    JKET_FIELD_IDS(0, 0),
    JBRA_FIELD_IDS(0, 1),
    JKET_FIELD_IDS(0, 1),
    JBRA_FIELD_IDS(1, 1),
    JKET_FIELD_IDS(1, 1),
    JBRA_FIELD_IDS(0, 2),
    JKET_FIELD_IDS(0, 2),
    JBRA_FIELD_IDS(1, 2),
    JKET_FIELD_IDS(1, 2),
    JBRA_FIELD_IDS(2, 2),
    JKET_FIELD_IDS(2, 2),
    JBRA_FIELD_IDS(0, 3),
    JKET_FIELD_IDS(0, 3),
    JBRA_FIELD_IDS(1, 3),
    JKET_FIELD_IDS(1, 3),
    JBRA_FIELD_IDS(2, 3),
    JKET_FIELD_IDS(2, 3),
    JBRA_FIELD_IDS(3, 3),
    JKET_FIELD_IDS(3, 3),
    JBRA_FIELD_IDS(0, 4),
    JKET_FIELD_IDS(0, 4),
    JBRA_FIELD_IDS(1, 4),
    JKET_FIELD_IDS(1, 4),
    JBRA_FIELD_IDS(2, 4),
    JKET_FIELD_IDS(2, 4),
    JBRA_FIELD_IDS(3, 4),
    JKET_FIELD_IDS(3, 4),
    JBRA_FIELD_IDS(4, 4),
    JKET_FIELD_IDS(4, 4),
  };

  const std::vector<Legion::FieldID> jbra_fields_list[MAX_MOMENTUM_INDEX +
                                                      1] = {
      {JBRA_FIELD_IDS(0, 0)}, {JBRA_FIELD_IDS(0, 1)}, {JBRA_FIELD_IDS(0, 2)},
      {JBRA_FIELD_IDS(0, 3)}, {JBRA_FIELD_IDS(0, 4)}, {JBRA_FIELD_IDS(1, 1)},
      {JBRA_FIELD_IDS(1, 2)}, {JBRA_FIELD_IDS(1, 3)}, {JBRA_FIELD_IDS(1, 4)},
      {JBRA_FIELD_IDS(2, 2)}, {JBRA_FIELD_IDS(2, 3)}, {JBRA_FIELD_IDS(2, 4)},
      {JBRA_FIELD_IDS(3, 3)}, {JBRA_FIELD_IDS(3, 4)}, {JBRA_FIELD_IDS(4, 4)},
  };

  const std::vector<Legion::FieldID> jket_fields_list[MAX_MOMENTUM_INDEX +
                                                      1] = {
      {JKET_FIELD_IDS(0, 0)}, {JKET_FIELD_IDS(0, 1)}, {JKET_FIELD_IDS(0, 2)},
      {JKET_FIELD_IDS(0, 3)}, {JKET_FIELD_IDS(0, 4)}, {JKET_FIELD_IDS(1, 1)},
      {JKET_FIELD_IDS(1, 2)}, {JKET_FIELD_IDS(1, 3)}, {JKET_FIELD_IDS(1, 4)},
      {JKET_FIELD_IDS(2, 2)}, {JKET_FIELD_IDS(2, 3)}, {JKET_FIELD_IDS(2, 4)},
      {JKET_FIELD_IDS(3, 3)}, {JKET_FIELD_IDS(3, 4)}, {JKET_FIELD_IDS(4, 4)},
  };

#undef JBRA_FIELD_IDS
#undef JKET_FIELD_IDS
};
