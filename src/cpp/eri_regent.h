#pragma once

#include "eri_regent_tasks.h"
#include "helper.h"
#include "legion.h"

// TODO: Move to EriRegent namespace.
// TODO: Move definition to `eri_regent.cpp`.
// TODO: I might not want this function at all.
/**
 * Starts the legion runtime and then calls the given function. This should be
 * the only function called inside `int main(int argc, char **argv)`.
 *
 * Usage:
 *
 * void original_main(const Task *task, const vector<PhysicalRegion> &regions,
 *                    Context ctx, Runtime *runtime) {
 *   int argc = Runtime::get_input_args().argc;
 *   char **argv = (char **)Runtime::get_input_args().argv;
 *   // Do computations
 * }
 *
 * int main(int argc, char **argv) {
 *   return start_eri_regent_runtime<original_main>(argc, argv);
 * }
 */
template <void (*TASK_PTR)(const Legion::Task *,
                           const std::vector<Legion::PhysicalRegion> &,
                           Legion::Context, Legion::Runtime *)>
int start_eri_regent_runtime(int argc, char **argv) {
  using namespace Legion;
  enum { TOP_LEVEL_TASK_ID }; // Task IDs
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<TASK_PTR>(registrar, "top_level");
  }
  eri_regent_tasks_h_register();
  return Runtime::start(argc, argv);
}

class EriRegent {
public:
  EriRegent(Legion::Context &ctx, Legion::Runtime *runtime,
            const double gamma_table[18][700][5]);
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
   */
  class TeraChemJDataList {
    friend class EriRegent;

  public:
    TeraChemJDataList() : num_jbras{0}, num_jkets{0} {}

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
    void set_jbra(int L1, int L2, int i, TeraChemJData &src);
    void set_jket(int L1, int L2, int i, TeraChemJData &src);

    /**
     * Copy the data from `src` to the density values of jket `i` for a given
     * angular momentum pair.
     */
    void set_density(int L1, int L2, int i, const double *src);

    /**
     * Get a pointer to the output data of jbra `i` for a given angular
     * momentum.
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
   * Launch the jfock regent tasks.
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
