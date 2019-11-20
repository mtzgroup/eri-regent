#pragma once

#include "helper.h"
#include "legion.h"

class EriRegent {
public:
  /**
   * `gamma_table` must have size 18 x 700 x 5
   */
  EriRegent(Legion::Context &ctx, Legion::Runtime *runtime,
            const double *gamma_table);
  ~EriRegent();

  /**
   * Register Regent tasks defined in eri-regent. This is useful because we
   * don't want to import all of `eri_regent_tasks.h` just to use
   * `eri_regent_tasks_h_register()`.
   */
  static void register_tasks();

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
   */
  class TeraChemJDataList {
    friend class EriRegent;

  public:
    TeraChemJDataList() {
      for (int i = 0; i < MAX_MOMENTUM_INDEX + 1; i++) {
        num_jbras[i] = 0;
        num_jkets[i] = 0;
      }
    }

    /**
     * Allocate `n` jbras/jkets for a given angular momentum pair.
     */
    void allocate_jbras(int L1, int L2, int n);
    void allocate_jkets(int L1, int L2, int n);

    /**
     * Free the allocated memory for all jbras and jkets.
     */
    void free_data();

    /**
     * The number of jbras/jkets for a give angular momentum pair.
     */
    int get_num_jbras(int L1, int L2);
    int get_num_jkets(int L1, int L2);

    /**
     * Copy the data from `src` to jbra/jket `i` for a given angular momentum
     * pair.
     */
    void set_jbra(int L1, int L2, int i, const TeraChemJData &src);
    void set_jket(int L1, int L2, int i, const TeraChemJData &src);

    /**
     * Copy the data from `src` to the density values of jket `i` for a given
     * angular momentum pair. `src` should have length `COMPUTE_H(L1 + L2)`.
     */
    void set_density(int L1, int L2, int i, const double *src);

    /**
     * Get a pointer to the output data of jbra `i` for a given angular
     * momentum. The resulting array has length `COMPUTE_H(L1 + L2)`. Should NOT
     * be free'd.
     */
    const double *get_output(int L1, int L2, int i);

  private:
    /**
     * The size of TeraChemJData. Do not use `sizeof(TeraChemJData)` because
     * that includes padding.
     */
    size_t sizeof_jdata();
    /**
     * The size of the data array for a given angular momentum pair.
     */
    size_t sizeof_jdata_array(int L1, int L2);
    /**
     * The stride of a data entry for a given angular momentum pair.
     */
    size_t stride(int L1, int L2);
    /**
     * The largest angular momentum that has data.
     */
    int get_largest_momentum();

    int num_jbras[MAX_MOMENTUM_INDEX + 1];
    void *jbras[MAX_MOMENTUM_INDEX + 1];

    int num_jkets[MAX_MOMENTUM_INDEX + 1];
    void *jkets[MAX_MOMENTUM_INDEX + 1];
  };

  /**
   * Launch the jfock regent tasks and wait for them to finish.
   */
  void launch_jfock_task(TeraChemJDataList &jdata_list, float threshold,
                         int parallelism);

private:
  Legion::Context ctx;
  Legion::Runtime *runtime;
  Legion::Memory memory;
  Legion::LogicalRegion gamma_table_lr;
  Legion::PhysicalRegion gamma_table_pr;
  Legion::FieldSpace jbra_fspaces[MAX_MOMENTUM_INDEX + 1];
  Legion::FieldSpace jket_fspaces[MAX_MOMENTUM_INDEX + 1];

  void initialize_field_spaces();

/**
 * Generate a name for a given field.
 */
#define JBRA_FIELD_ID(L1, L2, F_NAME) JBRA##L1##L2##_FIELD_##F_NAME##_ID
#define JKET_FIELD_ID(L1, L2, F_NAME) JKET##L1##L2##_FIELD_##F_NAME##_ID

/**
 * Generate a list of all fields for a given fspace.
 */
#define NUM_JBRA_FIELDS (7)
#define NUM_JKET_FIELDS (7)

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

  /**
   * An enum of all the fields so that we can uniquely identify them
   */
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

  /**
   * A vector of field IDs. The order needs to be such that
   * `jbra_fields_list[L_PAIR_TO_INDEX(L1, L2)] = JBRA_FIELD_IDS(L1, L2)`. This
   * is useful because we need to index them at runtime.
   */
  const Legion::FieldID jbra_fields_list[MAX_MOMENTUM_INDEX +
                                         1][NUM_JBRA_FIELDS]{
      {JBRA_FIELD_IDS(0, 0)}, {JBRA_FIELD_IDS(0, 1)}, {JBRA_FIELD_IDS(0, 2)},
      {JBRA_FIELD_IDS(0, 3)}, {JBRA_FIELD_IDS(0, 4)}, {JBRA_FIELD_IDS(1, 1)},
      {JBRA_FIELD_IDS(1, 2)}, {JBRA_FIELD_IDS(1, 3)}, {JBRA_FIELD_IDS(1, 4)},
      {JBRA_FIELD_IDS(2, 2)}, {JBRA_FIELD_IDS(2, 3)}, {JBRA_FIELD_IDS(2, 4)},
      {JBRA_FIELD_IDS(3, 3)}, {JBRA_FIELD_IDS(3, 4)}, {JBRA_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID jket_fields_list[MAX_MOMENTUM_INDEX +
                                         1][NUM_JKET_FIELDS] = {
      {JKET_FIELD_IDS(0, 0)}, {JKET_FIELD_IDS(0, 1)}, {JKET_FIELD_IDS(0, 2)},
      {JKET_FIELD_IDS(0, 3)}, {JKET_FIELD_IDS(0, 4)}, {JKET_FIELD_IDS(1, 1)},
      {JKET_FIELD_IDS(1, 2)}, {JKET_FIELD_IDS(1, 3)}, {JKET_FIELD_IDS(1, 4)},
      {JKET_FIELD_IDS(2, 2)}, {JKET_FIELD_IDS(2, 3)}, {JKET_FIELD_IDS(2, 4)},
      {JKET_FIELD_IDS(3, 3)}, {JKET_FIELD_IDS(3, 4)}, {JKET_FIELD_IDS(4, 4)},
  };

#undef JBRA_FIELD_IDS
#undef JKET_FIELD_IDS
};
