#pragma once

#include "helper.h"
#include "legion.h"
#include "legion/legion_types.h"

using namespace Legion;

class EriRegent {
public:
  // TODO: Confirm that the type of int1d is uint64_t
  /**
   * `int1d` is atype in Regent that we want to use in C++.
   */
  typedef uint64_t int1d_t;

  /**
   * `gamma_table` must have size 18 x 700 x 5
   */
  EriRegent(Runtime* runtime, Context ctx);
  ~EriRegent();

  /**
   * Disable copying.
   */
  EriRegent(EriRegent const &) = delete;
  EriRegent &operator=(EriRegent const &) = delete;

  /**
   * Register Regent tasks defined in eri-regent. Must be called before starting
   * the Legion runtime. This is useful because we don't want to import all of
   * `eri_regent_tasks.h` just to use `eri_regent_tasks_h_register()`.
   */
  static void register_tasks();


  /**
   * A list of JBra and JKet data to be passed to `launch_jfock_task`.
   */
  class TeraChemJDataList {
    friend class EriRegent;

  public:
    TeraChemJDataList();
    ~TeraChemJDataList();

    /**
     * Disable copying.
     */
    TeraChemJDataList(TeraChemJDataList const &) = delete;
    TeraChemJDataList &operator=(TeraChemJDataList const &) = delete;

    /**
     * Allocate `n` jbras/jkets for a given angular momentum pair.
     */
    void allocate_jbras(int L1, int L2, int n);
    void allocate_jkets(int L1, int L2, int n);

    /**
     * Copy the data from `src` to jbra/jket `i` for a given angular momentum
     * pair.
     */
    void set_jbra(int L1, int L2, int i, double x, double y, double z,
                  double eta, double C, float bound);
    void set_jket(int L1, int L2, int i, double x, double y, double z,
                  double eta, double C, float bound);

    /**
     * Copy the data from `src` to the density values of jket `i` for a given
     * angular momentum pair. `src` should have length
     * `TETRAHEDRAL_NUMBER(L1 + L2 + 1)`.
     */
    void set_jdensity(int L1, int L2, int i, const double *src);

    /**
     * Get a pointer to the output data of jbra `i` for a given angular
     * momentum. The resulting array has length
     * `TETRAHEDRAL_NUMBER(L1 + L2 + 1)`. Should NOT be free'd.
     */
    const double *get_joutput(int L1, int L2, int i);

  private:
    /**
     * The largest angular momentum that has data.
     */
    int get_largest_momentum();

    /**
     * The number of jbras/jkets for a give angular momentum pair.
     */
    int get_num_jbras(int L1, int L2);
    int get_num_jkets(int L1, int L2);

    int num_jbras[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
    void *jbras[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];

    int num_jkets[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
    void *jkets[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  };

  /**
   * A list of KFock data to be filled and passed to `launch_kfock_task`.
   */
  class TeraChemKDataList {
    friend class EriRegent;

  public:
    TeraChemKDataList();
    ~TeraChemKDataList();

    /**
     * Disable copying.
     */
    TeraChemKDataList(TeraChemKDataList const &) = delete;
    TeraChemKDataList &operator=(TeraChemKDataList const &) = delete;

    /**
     * Allocate `n` kpairs for a given angular momentum pair.
     */
    void allocate_kpairs(int L1, int L2, int n, int num_kbra_prevals,
                         int num_kket_prevals);

    /**
     * Allocate a density matrix for a given angular momentum pair where `n2` is
     * the number of shells for `L2` and `n4` is the number of shells for `L4`
     */
    void allocate_kdensity(int L2, int L4, int n2, int n4);

    /**
     * Copy the data to kpair `i` for a given angular momentum pair.
     */
    void set_kpair(int L1, int L2, int i, double x, double y, double z,
                   double eta, double C, float bound, double PIx, double PIy,
                   double PIz, double PJx, double PJy, double PJz,
                   int1d_t ishell_index, int1d_t jshell_index);

    void set_kbra_preval(int L1, int L2, int i, int k, double value);
    void set_kket_preval(int L1, int L2, int i, int k, double value);

    /**
     * Copy the data from `src` to the density values for a given
     * shell pair. `src` should have length
     * `TRIANGLE_NUMBER(L2 + 1) * TRIANGLE_NUMBER(L4 + 1)` and should be indexed
     * using `i * TRIANGLE_NUMBER(L4 + 1) + j`.
     */
    void set_kdensity(int L2, int L4, int bra_jshell_index,
                      int ket_jshell_index, const double *src, float bound);

    /**
     * Returns the output values for a given shell pair. The array has length
     * `TRIANGLE_NUMBER(L1 + 1) * TRIANGLE_NUMBER(L3 + 1)` and should be indexed
     * using `i * TRIANGLE_NUMBER(L3 + 1) + j`. Should NOT be free'd.
     */
    const double *get_koutput(int L1, int L2, int L3, int L4,
                              int bra_ishell_index, int ket_ishell_index);

  private:
    /**
     * Only call this method once all kdensities have been allocated.
     */
    void allocate_all_koutput();

    /**
     * The largest angular momentum that has data.
     */
    int get_largest_momentum();

    int get_num_kpairs(int L1, int L2);
    int get_num_kbra_prevals(int L1, int L2);
    int get_num_kket_prevals(int L1, int L2);
    int get_num_shells(int L);
    void *get_kpair_data(int L1, int L2);
    void *get_kbra_preval_data(int L1, int L2);
    void *get_kket_preval_data(int L1, int L2);
    void *get_kdensity_data(int L2, int L4);
    void *get_koutput_data(int L1, int L3);

    int num_kpairs[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
    void *kpairs[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

    int num_kbra_prevals[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
    void *kbra_prevals[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

    int num_kket_prevals[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
    void *kket_prevals[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

    int num_shells[MAX_MOMENTUM + 1];
    void *kdensity[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];

    // TODO: Do not allocate space for lower triangular output entries.
    void *koutput[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  };

  /**
   * Launch the jfock regent tasks and wait for them to finish.
   */
  void launch_jfock_task(TeraChemJDataList &jdata_list, float threshold,
                         int parallelism,
			 int cparallelism);

  /**
   * Launch the kfock regent tasks and wait for them to finish.
   */
  void launch_kfock_task(TeraChemKDataList &kdata_list, float threshold,
                         int parallelism);

private:
  Legion::Context ctx;
  Legion::Runtime *runtime;
  Legion::Memory memory;
  Legion::FieldSpace gamma_table_fspace;
  Legion::IndexSpace gamma_table_ispace;
  Legion::LogicalRegion gamma_table_lr;
  Legion::PhysicalRegion gamma_table_pr;
  Legion::FieldSpace jbra_fspaces[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  Legion::FieldSpace jket_fspaces[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  Legion::FieldSpace kpair_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  Legion::FieldSpace
      kbra_preval_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  Legion::FieldSpace
      kket_preval_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  Legion::FieldSpace kdensity_fspaces[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  Legion::FieldSpace koutput_fspaces[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];

  void initialize_jfock_field_spaces();
  void initialize_kfock_field_spaces();

/**
 * Generate a name for a given field.
 */
#define JBRA_FIELD_ID(L1, L2, F_NAME) JBRA##L1##L2##_FIELD_##F_NAME##_ID
#define JKET_FIELD_ID(L1, L2, F_NAME) JKET##L1##L2##_FIELD_##F_NAME##_ID
#define KPAIR_FIELD_ID(L1, L2, F_NAME) KPAIR##L1##L2##_FIELD##F_NAME##_ID
#define KBRA_PREVAL_FIELD_ID(L1, L2) KBRA_PREVAL##L1##L2##_ID
#define KKET_PREVAL_FIELD_ID(L1, L2) KKET_PREVAL##L1##L2##_ID
#define KDENSITY_FIELD_ID(L2, L4, F_NAME) KDENSITY##L2##L4##_FIELD##F_NAME##_ID
#define KOUTPUT_FIELD_ID(L1, L3, F_NAME) KOUTPUT##L1##L3##_FIELD##F_NAME##_ID

/**
 * Generate a list of all fields for a given fspace.
 */
#define NUM_JBRA_FIELDS (7)
#define NUM_JKET_FIELDS (7)
#define NUM_KPAIR_FIELDS (14)
#define NUM_KBRA_PREVAL_FIELDS (1)
#define NUM_KKET_PREVAL_FIELDS (1)
#define NUM_KDENSITY_FIELDS (2)
#define NUM_KOUTPUT_FIELDS (1)

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

#define KPAIR_FIELD_IDS(L1, L2)                                                \
  KPAIR_FIELD_ID(L1, L2, X), KPAIR_FIELD_ID(L1, L2, Y),                        \
      KPAIR_FIELD_ID(L1, L2, Z), KPAIR_FIELD_ID(L1, L2, ETA),                  \
      KPAIR_FIELD_ID(L1, L2, C), KPAIR_FIELD_ID(L1, L2, BOUND),                \
      KPAIR_FIELD_ID(L1, L2, ISHELL_X), KPAIR_FIELD_ID(L1, L2, ISHELL_Y),      \
      KPAIR_FIELD_ID(L1, L2, ISHELL_Z), KPAIR_FIELD_ID(L1, L2, JSHELL_X),      \
      KPAIR_FIELD_ID(L1, L2, JSHELL_Y), KPAIR_FIELD_ID(L1, L2, JSHELL_Z),      \
      KPAIR_FIELD_ID(L1, L2, ISHELL_INDEX),                                    \
      KPAIR_FIELD_ID(L1, L2, JSHELL_INDEX)

#define KBRA_PREVAL_FIELD_IDS(L1, L2) KBRA_PREVAL_FIELD_ID(L1, L2)

#define KKET_PREVAL_FIELD_IDS(L1, L2) KKET_PREVAL_FIELD_ID(L1, L2)

#define KDENSITY_FIELD_IDS(L2, L4)                                             \
  KDENSITY_FIELD_ID(L2, L4, VALUES), KDENSITY_FIELD_ID(L2, L4, BOUND)

#define KOUTPUT_FIELD_IDS(L1, L3) KOUTPUT_FIELD_ID(L1, L3, VALUES)

  /**
   * An enum containing all the tasks and fields so that we can uniquely
   * identify them
   */
  enum { // Task IDs and Field IDs
    ERI_REGENT_TASK_ID,
    GAMMA_TABLE_FIELD_ID,
    JBRA_FIELD_IDS(0, 0),
    JBRA_FIELD_IDS(0, 1),
    JBRA_FIELD_IDS(0, 2),
    JBRA_FIELD_IDS(0, 3),
    JBRA_FIELD_IDS(0, 4),
    JBRA_FIELD_IDS(1, 1),
    JBRA_FIELD_IDS(1, 2),
    JBRA_FIELD_IDS(1, 3),
    JBRA_FIELD_IDS(1, 4),
    JBRA_FIELD_IDS(2, 2),
    JBRA_FIELD_IDS(2, 3),
    JBRA_FIELD_IDS(2, 4),
    JBRA_FIELD_IDS(3, 3),
    JBRA_FIELD_IDS(3, 4),
    JBRA_FIELD_IDS(4, 4),
    JKET_FIELD_IDS(0, 0),
    JKET_FIELD_IDS(0, 1),
    JKET_FIELD_IDS(0, 2),
    JKET_FIELD_IDS(0, 3),
    JKET_FIELD_IDS(0, 4),
    JKET_FIELD_IDS(1, 1),
    JKET_FIELD_IDS(1, 2),
    JKET_FIELD_IDS(1, 3),
    JKET_FIELD_IDS(1, 4),
    JKET_FIELD_IDS(2, 2),
    JKET_FIELD_IDS(2, 3),
    JKET_FIELD_IDS(2, 4),
    JKET_FIELD_IDS(3, 3),
    JKET_FIELD_IDS(3, 4),
    JKET_FIELD_IDS(4, 4),
    KPAIR_FIELD_IDS(0, 0),
    KPAIR_FIELD_IDS(0, 1),
    KPAIR_FIELD_IDS(0, 2),
    KPAIR_FIELD_IDS(0, 3),
    KPAIR_FIELD_IDS(0, 4),
    KPAIR_FIELD_IDS(1, 0),
    KPAIR_FIELD_IDS(1, 1),
    KPAIR_FIELD_IDS(1, 2),
    KPAIR_FIELD_IDS(1, 3),
    KPAIR_FIELD_IDS(1, 4),
    KPAIR_FIELD_IDS(2, 0),
    KPAIR_FIELD_IDS(2, 1),
    KPAIR_FIELD_IDS(2, 2),
    KPAIR_FIELD_IDS(2, 3),
    KPAIR_FIELD_IDS(2, 4),
    KPAIR_FIELD_IDS(3, 0),
    KPAIR_FIELD_IDS(3, 1),
    KPAIR_FIELD_IDS(3, 2),
    KPAIR_FIELD_IDS(3, 3),
    KPAIR_FIELD_IDS(3, 4),
    KPAIR_FIELD_IDS(4, 0),
    KPAIR_FIELD_IDS(4, 1),
    KPAIR_FIELD_IDS(4, 2),
    KPAIR_FIELD_IDS(4, 3),
    KPAIR_FIELD_IDS(4, 4),

    KBRA_PREVAL_FIELD_IDS(0, 0),
    KBRA_PREVAL_FIELD_IDS(0, 1),
    KBRA_PREVAL_FIELD_IDS(0, 2),
    KBRA_PREVAL_FIELD_IDS(0, 3),
    KBRA_PREVAL_FIELD_IDS(0, 4),
    KBRA_PREVAL_FIELD_IDS(1, 0),
    KBRA_PREVAL_FIELD_IDS(1, 1),
    KBRA_PREVAL_FIELD_IDS(1, 2),
    KBRA_PREVAL_FIELD_IDS(1, 3),
    KBRA_PREVAL_FIELD_IDS(1, 4),
    KBRA_PREVAL_FIELD_IDS(2, 0),
    KBRA_PREVAL_FIELD_IDS(2, 1),
    KBRA_PREVAL_FIELD_IDS(2, 2),
    KBRA_PREVAL_FIELD_IDS(2, 3),
    KBRA_PREVAL_FIELD_IDS(2, 4),
    KBRA_PREVAL_FIELD_IDS(3, 0),
    KBRA_PREVAL_FIELD_IDS(3, 1),
    KBRA_PREVAL_FIELD_IDS(3, 2),
    KBRA_PREVAL_FIELD_IDS(3, 3),
    KBRA_PREVAL_FIELD_IDS(3, 4),
    KBRA_PREVAL_FIELD_IDS(4, 0),
    KBRA_PREVAL_FIELD_IDS(4, 1),
    KBRA_PREVAL_FIELD_IDS(4, 2),
    KBRA_PREVAL_FIELD_IDS(4, 3),
    KBRA_PREVAL_FIELD_IDS(4, 4),
    KKET_PREVAL_FIELD_IDS(0, 0),
    KKET_PREVAL_FIELD_IDS(0, 1),
    KKET_PREVAL_FIELD_IDS(0, 2),
    KKET_PREVAL_FIELD_IDS(0, 3),
    KKET_PREVAL_FIELD_IDS(0, 4),
    KKET_PREVAL_FIELD_IDS(1, 0),
    KKET_PREVAL_FIELD_IDS(1, 1),
    KKET_PREVAL_FIELD_IDS(1, 2),
    KKET_PREVAL_FIELD_IDS(1, 3),
    KKET_PREVAL_FIELD_IDS(1, 4),
    KKET_PREVAL_FIELD_IDS(2, 0),
    KKET_PREVAL_FIELD_IDS(2, 1),
    KKET_PREVAL_FIELD_IDS(2, 2),
    KKET_PREVAL_FIELD_IDS(2, 3),
    KKET_PREVAL_FIELD_IDS(2, 4),
    KKET_PREVAL_FIELD_IDS(3, 0),
    KKET_PREVAL_FIELD_IDS(3, 1),
    KKET_PREVAL_FIELD_IDS(3, 2),
    KKET_PREVAL_FIELD_IDS(3, 3),
    KKET_PREVAL_FIELD_IDS(3, 4),
    KKET_PREVAL_FIELD_IDS(4, 0),
    KKET_PREVAL_FIELD_IDS(4, 1),
    KKET_PREVAL_FIELD_IDS(4, 2),
    KKET_PREVAL_FIELD_IDS(4, 3),
    KKET_PREVAL_FIELD_IDS(4, 4),
    KDENSITY_FIELD_IDS(0, 0),
    KDENSITY_FIELD_IDS(0, 1),
    KDENSITY_FIELD_IDS(0, 2),
    KDENSITY_FIELD_IDS(0, 3),
    KDENSITY_FIELD_IDS(0, 4),
    KDENSITY_FIELD_IDS(1, 1),
    KDENSITY_FIELD_IDS(1, 2),
    KDENSITY_FIELD_IDS(1, 3),
    KDENSITY_FIELD_IDS(1, 4),
    KDENSITY_FIELD_IDS(2, 2),
    KDENSITY_FIELD_IDS(2, 3),
    KDENSITY_FIELD_IDS(2, 4),
    KDENSITY_FIELD_IDS(3, 3),
    KDENSITY_FIELD_IDS(3, 4),
    KDENSITY_FIELD_IDS(4, 4),
    KOUTPUT_FIELD_IDS(0, 0),
    KOUTPUT_FIELD_IDS(0, 1),
    KOUTPUT_FIELD_IDS(0, 2),
    KOUTPUT_FIELD_IDS(0, 3),
    KOUTPUT_FIELD_IDS(0, 4),
    KOUTPUT_FIELD_IDS(1, 1),
    KOUTPUT_FIELD_IDS(1, 2),
    KOUTPUT_FIELD_IDS(1, 3),
    KOUTPUT_FIELD_IDS(1, 4),
    KOUTPUT_FIELD_IDS(2, 2),
    KOUTPUT_FIELD_IDS(2, 3),
    KOUTPUT_FIELD_IDS(2, 4),
    KOUTPUT_FIELD_IDS(3, 3),
    KOUTPUT_FIELD_IDS(3, 4),
    KOUTPUT_FIELD_IDS(4, 4),
  };

  /**
   * A vector of field IDs. The order needs to be such that
   * `jbra_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)] = JBRA_FIELD_IDS(L1, L2)`.
   * This is useful because we need to index them at runtime.
   */
  const Legion::FieldID jbra_fields_list[TRIANGLE_NUMBER(MAX_MOMENTUM +
                                                         1)][NUM_JBRA_FIELDS]{
      {JBRA_FIELD_IDS(0, 0)}, {JBRA_FIELD_IDS(0, 1)}, {JBRA_FIELD_IDS(0, 2)},
      {JBRA_FIELD_IDS(0, 3)}, {JBRA_FIELD_IDS(0, 4)}, {JBRA_FIELD_IDS(1, 1)},
      {JBRA_FIELD_IDS(1, 2)}, {JBRA_FIELD_IDS(1, 3)}, {JBRA_FIELD_IDS(1, 4)},
      {JBRA_FIELD_IDS(2, 2)}, {JBRA_FIELD_IDS(2, 3)}, {JBRA_FIELD_IDS(2, 4)},
      {JBRA_FIELD_IDS(3, 3)}, {JBRA_FIELD_IDS(3, 4)}, {JBRA_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID jket_fields_list[TRIANGLE_NUMBER(
      MAX_MOMENTUM + 1)][NUM_JKET_FIELDS] = {
      {JKET_FIELD_IDS(0, 0)}, {JKET_FIELD_IDS(0, 1)}, {JKET_FIELD_IDS(0, 2)},
      {JKET_FIELD_IDS(0, 3)}, {JKET_FIELD_IDS(0, 4)}, {JKET_FIELD_IDS(1, 1)},
      {JKET_FIELD_IDS(1, 2)}, {JKET_FIELD_IDS(1, 3)}, {JKET_FIELD_IDS(1, 4)},
      {JKET_FIELD_IDS(2, 2)}, {JKET_FIELD_IDS(2, 3)}, {JKET_FIELD_IDS(2, 4)},
      {JKET_FIELD_IDS(3, 3)}, {JKET_FIELD_IDS(3, 4)}, {JKET_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID kpair_fields_list[(MAX_MOMENTUM + 1) *
                                          (MAX_MOMENTUM +
                                           1)][NUM_KPAIR_FIELDS] = {
      {KPAIR_FIELD_IDS(0, 0)}, {KPAIR_FIELD_IDS(0, 1)}, {KPAIR_FIELD_IDS(0, 2)},
      {KPAIR_FIELD_IDS(0, 3)}, {KPAIR_FIELD_IDS(0, 4)}, {KPAIR_FIELD_IDS(1, 0)},
      {KPAIR_FIELD_IDS(1, 1)}, {KPAIR_FIELD_IDS(1, 2)}, {KPAIR_FIELD_IDS(1, 3)},
      {KPAIR_FIELD_IDS(1, 4)}, {KPAIR_FIELD_IDS(2, 0)}, {KPAIR_FIELD_IDS(2, 1)},
      {KPAIR_FIELD_IDS(2, 2)}, {KPAIR_FIELD_IDS(2, 3)}, {KPAIR_FIELD_IDS(2, 4)},
      {KPAIR_FIELD_IDS(3, 0)}, {KPAIR_FIELD_IDS(3, 1)}, {KPAIR_FIELD_IDS(3, 2)},
      {KPAIR_FIELD_IDS(3, 3)}, {KPAIR_FIELD_IDS(3, 4)}, {KPAIR_FIELD_IDS(4, 0)},
      {KPAIR_FIELD_IDS(4, 1)}, {KPAIR_FIELD_IDS(4, 2)}, {KPAIR_FIELD_IDS(4, 3)},
      {KPAIR_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID
      kbra_preval_fields_list[(MAX_MOMENTUM + 1) *
                              (MAX_MOMENTUM + 1)][NUM_KPAIR_FIELDS] = {
          {KBRA_PREVAL_FIELD_IDS(0, 0)}, {KBRA_PREVAL_FIELD_IDS(0, 1)},
          {KBRA_PREVAL_FIELD_IDS(0, 2)}, {KBRA_PREVAL_FIELD_IDS(0, 3)},
          {KBRA_PREVAL_FIELD_IDS(0, 4)}, {KBRA_PREVAL_FIELD_IDS(1, 0)},
          {KBRA_PREVAL_FIELD_IDS(1, 1)}, {KBRA_PREVAL_FIELD_IDS(1, 2)},
          {KBRA_PREVAL_FIELD_IDS(1, 3)}, {KBRA_PREVAL_FIELD_IDS(1, 4)},
          {KBRA_PREVAL_FIELD_IDS(2, 0)}, {KBRA_PREVAL_FIELD_IDS(2, 1)},
          {KBRA_PREVAL_FIELD_IDS(2, 2)}, {KBRA_PREVAL_FIELD_IDS(2, 3)},
          {KBRA_PREVAL_FIELD_IDS(2, 4)}, {KBRA_PREVAL_FIELD_IDS(3, 0)},
          {KBRA_PREVAL_FIELD_IDS(3, 1)}, {KBRA_PREVAL_FIELD_IDS(3, 2)},
          {KBRA_PREVAL_FIELD_IDS(3, 3)}, {KBRA_PREVAL_FIELD_IDS(3, 4)},
          {KBRA_PREVAL_FIELD_IDS(4, 0)}, {KBRA_PREVAL_FIELD_IDS(4, 1)},
          {KBRA_PREVAL_FIELD_IDS(4, 2)}, {KBRA_PREVAL_FIELD_IDS(4, 3)},
          {KBRA_PREVAL_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID
      kket_preval_fields_list[(MAX_MOMENTUM + 1) *
                              (MAX_MOMENTUM + 1)][NUM_KPAIR_FIELDS] = {
          {KKET_PREVAL_FIELD_IDS(0, 0)}, {KKET_PREVAL_FIELD_IDS(0, 1)},
          {KKET_PREVAL_FIELD_IDS(0, 2)}, {KKET_PREVAL_FIELD_IDS(0, 3)},
          {KKET_PREVAL_FIELD_IDS(0, 4)}, {KKET_PREVAL_FIELD_IDS(1, 0)},
          {KKET_PREVAL_FIELD_IDS(1, 1)}, {KKET_PREVAL_FIELD_IDS(1, 2)},
          {KKET_PREVAL_FIELD_IDS(1, 3)}, {KKET_PREVAL_FIELD_IDS(1, 4)},
          {KKET_PREVAL_FIELD_IDS(2, 0)}, {KKET_PREVAL_FIELD_IDS(2, 1)},
          {KKET_PREVAL_FIELD_IDS(2, 2)}, {KKET_PREVAL_FIELD_IDS(2, 3)},
          {KKET_PREVAL_FIELD_IDS(2, 4)}, {KKET_PREVAL_FIELD_IDS(3, 0)},
          {KKET_PREVAL_FIELD_IDS(3, 1)}, {KKET_PREVAL_FIELD_IDS(3, 2)},
          {KKET_PREVAL_FIELD_IDS(3, 3)}, {KKET_PREVAL_FIELD_IDS(3, 4)},
          {KKET_PREVAL_FIELD_IDS(4, 0)}, {KKET_PREVAL_FIELD_IDS(4, 1)},
          {KKET_PREVAL_FIELD_IDS(4, 2)}, {KKET_PREVAL_FIELD_IDS(4, 3)},
          {KKET_PREVAL_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID kdensity_fields_list[TRIANGLE_NUMBER(
      MAX_MOMENTUM + 1)][NUM_KPAIR_FIELDS] = {
      {KDENSITY_FIELD_IDS(0, 0)}, {KDENSITY_FIELD_IDS(0, 1)},
      {KDENSITY_FIELD_IDS(0, 2)}, {KDENSITY_FIELD_IDS(0, 3)},
      {KDENSITY_FIELD_IDS(0, 4)}, {KDENSITY_FIELD_IDS(1, 1)},
      {KDENSITY_FIELD_IDS(1, 2)}, {KDENSITY_FIELD_IDS(1, 3)},
      {KDENSITY_FIELD_IDS(1, 4)}, {KDENSITY_FIELD_IDS(2, 2)},
      {KDENSITY_FIELD_IDS(2, 3)}, {KDENSITY_FIELD_IDS(2, 4)},
      {KDENSITY_FIELD_IDS(3, 3)}, {KDENSITY_FIELD_IDS(3, 4)},
      {KDENSITY_FIELD_IDS(4, 4)},
  };
  const Legion::FieldID koutput_fields_list[TRIANGLE_NUMBER(
      MAX_MOMENTUM + 1)][NUM_KPAIR_FIELDS] = {
      {KOUTPUT_FIELD_IDS(0, 0)}, {KOUTPUT_FIELD_IDS(0, 1)},
      {KOUTPUT_FIELD_IDS(0, 2)}, {KOUTPUT_FIELD_IDS(0, 3)},
      {KOUTPUT_FIELD_IDS(0, 4)}, {KOUTPUT_FIELD_IDS(1, 1)},
      {KOUTPUT_FIELD_IDS(1, 2)}, {KOUTPUT_FIELD_IDS(1, 3)},
      {KOUTPUT_FIELD_IDS(1, 4)}, {KOUTPUT_FIELD_IDS(2, 2)},
      {KOUTPUT_FIELD_IDS(2, 3)}, {KOUTPUT_FIELD_IDS(2, 4)},
      {KOUTPUT_FIELD_IDS(3, 3)}, {KOUTPUT_FIELD_IDS(3, 4)},
      {KOUTPUT_FIELD_IDS(4, 4)},
  };

#undef JBRA_FIELD_IDS
#undef JKET_FIELD_IDS
#undef KPAIR_FIELD_IDS
#undef KBRA_PREVAL_FIELD_IDS
#undef KKET_PREVAL_FIELD_IDS
#undef KDENSITY_FIELD_IDS
#undef KOUTPUT_FIELD_IDS
};
