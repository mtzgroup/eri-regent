#include "eri_regent.h"
#include "eri_regent_tasks.h"
#include "helper.h"
#include "legion.h"

using namespace std;
using namespace Legion;

EriRegent::EriRegent(const double *gamma_table) {
  runtime = Runtime::get_runtime();
  ctx = runtime->begin_implicit_task(ERI_REGENT_TASK_ID,
                                     /*mapper_id=*/0, Processor::LOC_PROC,
                                     "eri_regent_toplevel_task",
                                     /*control_replicable=*/true);
  memory = Machine::MemoryQuery(Machine::get_machine())
               .has_affinity_to(runtime->get_executing_processor(ctx))
               .only_kind(Memory::SYSTEM_MEM)
               .first();

  // Create gamma table region
  const Rect<2> rect({0, 0}, {18 - 1, 700 - 1});
  gamma_table_ispace = runtime->create_index_space(ctx, rect);
  gamma_table_fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc =
        runtime->create_field_allocator(ctx, gamma_table_fspace);
    falloc.allocate_field(5 * sizeof(double), GAMMA_TABLE_FIELD_ID);
  }
  gamma_table_lr = runtime->create_logical_region(ctx, gamma_table_ispace,
                                                  gamma_table_fspace);
  AttachLauncher launcher(EXTERNAL_INSTANCE, gamma_table_lr, gamma_table_lr);
  launcher.attach_array_aos((void *)gamma_table, /*column major*/ false,
                            {GAMMA_TABLE_FIELD_ID}, memory);
  gamma_table_pr = runtime->attach_external_resource(ctx, launcher);

  initialize_jfock_field_spaces();
  initialize_kfock_field_spaces();
}

EriRegent::~EriRegent() {
  runtime->detach_external_resource(ctx, gamma_table_pr);
  runtime->destroy_logical_region(ctx, gamma_table_lr);
  runtime->destroy_field_space(ctx, gamma_table_fspace);
  runtime->destroy_index_space(ctx, gamma_table_ispace);
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      runtime->destroy_field_space(ctx, jbra_fspaces[index]);
      runtime->destroy_field_space(ctx, jket_fspaces[index]);
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      runtime->destroy_field_space(ctx, kpair_fspaces[index]);
    }
  }

  for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
    for (int L4 = L2; L4 <= MAX_MOMENTUM; L4++) {
      const int index = INDEX_UPPER_TRIANGLE(L2, L4);
      runtime->destroy_field_space(ctx, kdensity_fspaces[index]);
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L3);
      runtime->destroy_field_space(ctx, koutput_fspaces[index]);
    }
  }
  runtime->finish_implicit_task(ctx);
}

void EriRegent::register_tasks() { eri_regent_tasks_h_register(); }

void EriRegent::launch_jfock_task(EriRegent::TeraChemJDataList &jdata_list,
                                  float threshold, int parallelism) {
  // Create jbra regions
  LogicalRegion jbras_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion jbras_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace jbras_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      const Rect<1> rect(0, jdata_list.get_num_jbras(L1, L2) - 1);
      jbras_ispace_list[index] = runtime->create_index_space(ctx, rect);
      jbras_lr_list[index] = runtime->create_logical_region(
          ctx, jbras_ispace_list[index], jbra_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jbras_lr_list[index],
                              jbras_lr_list[index]);
      const vector<FieldID> field_list(
          jbra_fields_list[index], jbra_fields_list[index] + NUM_JBRA_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(jdata_list.jbras[index], /*column major*/ false,
                                field_list, memory);
      jbras_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  // Create jket regions
  LogicalRegion jkets_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion jkets_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace jkets_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      const Rect<1> rect(0, jdata_list.get_num_jkets(L1, L2) - 1);
      jkets_ispace_list[index] = runtime->create_index_space(ctx, rect);
      jkets_lr_list[index] = runtime->create_logical_region(
          ctx, jkets_ispace_list[index], jket_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jkets_lr_list[index],
                              jkets_lr_list[index]);
      const vector<FieldID> field_list(
          jket_fields_list[index], jket_fields_list[index] + NUM_JKET_FIELDS);
      launcher.attach_array_aos(jdata_list.jkets[index], /*column major*/ false,
                                field_list, memory);
      jkets_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  jfock_task_launcher launcher;

#define ADD_ARGUMENT_R_JBRAS(L1, L2)                                           \
  {                                                                            \
    const vector<FieldID> field_list(                                          \
        jbra_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)],                        \
        jbra_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)] + NUM_JBRA_FIELDS);     \
    launcher.add_argument_r_jbras##L1##L2(                                     \
        jbras_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)],                           \
        jbras_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)], field_list);              \
  }

  ADD_ARGUMENT_R_JBRAS(0, 0);
  ADD_ARGUMENT_R_JBRAS(0, 1);
  ADD_ARGUMENT_R_JBRAS(0, 2);
  ADD_ARGUMENT_R_JBRAS(0, 3);
  ADD_ARGUMENT_R_JBRAS(0, 4);
  ADD_ARGUMENT_R_JBRAS(1, 1);
  ADD_ARGUMENT_R_JBRAS(1, 2);
  ADD_ARGUMENT_R_JBRAS(1, 3);
  ADD_ARGUMENT_R_JBRAS(1, 4);
  ADD_ARGUMENT_R_JBRAS(2, 2);
  ADD_ARGUMENT_R_JBRAS(2, 3);
  ADD_ARGUMENT_R_JBRAS(2, 4);
  ADD_ARGUMENT_R_JBRAS(3, 3);
  ADD_ARGUMENT_R_JBRAS(3, 4);
  ADD_ARGUMENT_R_JBRAS(4, 4);

#undef ADD_ARGUMENT_R_JBRAS

#define ADD_ARGUMENT_R_JKETS(L1, L2)                                           \
  {                                                                            \
    const vector<FieldID> field_list(                                          \
        jket_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)],                        \
        jket_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)] + NUM_JKET_FIELDS);     \
    launcher.add_argument_r_jkets##L1##L2(                                     \
        jkets_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)],                           \
        jkets_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)], field_list);              \
  }

  ADD_ARGUMENT_R_JKETS(0, 0);
  ADD_ARGUMENT_R_JKETS(0, 1);
  ADD_ARGUMENT_R_JKETS(0, 2);
  ADD_ARGUMENT_R_JKETS(0, 3);
  ADD_ARGUMENT_R_JKETS(0, 4);
  ADD_ARGUMENT_R_JKETS(1, 1);
  ADD_ARGUMENT_R_JKETS(1, 2);
  ADD_ARGUMENT_R_JKETS(1, 3);
  ADD_ARGUMENT_R_JKETS(1, 4);
  ADD_ARGUMENT_R_JKETS(2, 2);
  ADD_ARGUMENT_R_JKETS(2, 3);
  ADD_ARGUMENT_R_JKETS(2, 4);
  ADD_ARGUMENT_R_JKETS(3, 3);
  ADD_ARGUMENT_R_JKETS(3, 4);
  ADD_ARGUMENT_R_JKETS(4, 4);

#undef ADD_ARGUMENT_R_JKETS

  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_parallelism(parallelism);
  launcher.add_argument_largest_momentum(jdata_list.get_largest_momentum());
  Future future = launcher.execute(runtime, ctx);
  future.wait();

  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      runtime->detach_external_resource(ctx, jbras_pr_list[index]);
      runtime->destroy_logical_region(ctx, jbras_lr_list[index]);
      runtime->destroy_index_space(ctx, jbras_ispace_list[index]);
      runtime->detach_external_resource(ctx, jkets_pr_list[index]);
      runtime->destroy_logical_region(ctx, jkets_lr_list[index]);
      runtime->destroy_index_space(ctx, jkets_ispace_list[index]);
    }
  }
}

void EriRegent::launch_kfock_task(EriRegent::TeraChemKDataList &kdata_list,
                                  float threshold, int parallelism) {

  // Create kpair regions
  LogicalRegion kpair_lr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  PhysicalRegion kpair_pr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  IndexSpace kpair_ispace_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      const Rect<1> rect(0, kdata_list.get_num_kpairs(L1, L2) - 1);
      kpair_ispace_list[index] = runtime->create_index_space(ctx, rect);
      kpair_lr_list[index] = runtime->create_logical_region(
          ctx, kpair_ispace_list[index], kpair_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, kpair_lr_list[index],
                              kpair_lr_list[index]);
      const vector<FieldID> field_list(kpair_fields_list[index],
                                       kpair_fields_list[index] +
                                           NUM_KPAIR_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_kpair_data(L1, L2),
                                /*column major*/ false, field_list, memory);
      kpair_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  // Create density regions
  LogicalRegion kdensity_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion kdensity_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace kdensity_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
    for (int L4 = L2; L4 <= MAX_MOMENTUM; L4++) {
      const int index = INDEX_UPPER_TRIANGLE(L2, L4);
      const Rect<2> rect({0, 0}, {kdata_list.get_num_shells(L2) - 1,
                                  kdata_list.get_num_shells(L4) - 1});
      kdensity_ispace_list[index] = runtime->create_index_space(ctx, rect);
      kdensity_lr_list[index] = runtime->create_logical_region(
          ctx, kdensity_ispace_list[index], kdensity_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, kdensity_lr_list[index],
                              kdensity_lr_list[index]);
      const vector<FieldID> field_list(kdensity_fields_list[index],
                                       kdensity_fields_list[index] +
                                           NUM_KDENSITY_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_kdensity_data(L2, L4),
                                /*column major*/ false, field_list, memory);
      kdensity_pr_list[index] =
          runtime->attach_external_resource(ctx, launcher);
    }
  }

  // Create output regions
  kdata_list.allocate_all_koutput();
  LogicalRegion koutput_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion koutput_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace koutput_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      // FIXME: Need to know the regent compiled momentum at runtime.
      int regent_compiled_max_momentum = 2;
      const int index = INDEX_UPPER_TRIANGLE(L1, L3);
      const Rect<3> rect({0, 0, 0}, {(regent_compiled_max_momentum +
                                      1) * (regent_compiled_max_momentum + 1) -
                                         1,
                                     kdata_list.get_num_shells(L1) - 1,
                                     kdata_list.get_num_shells(L3) - 1});
      koutput_ispace_list[index] = runtime->create_index_space(ctx, rect);
      koutput_lr_list[index] = runtime->create_logical_region(
          ctx, koutput_ispace_list[index], koutput_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, koutput_lr_list[index],
                              koutput_lr_list[index]);
      const vector<FieldID> field_list(koutput_fields_list[index],
                                       koutput_fields_list[index] +
                                           NUM_KOUTPUT_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_koutput_data(L1, L3),
                                /*column major*/ false, field_list, memory);
      koutput_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  kfock_task_launcher launcher;

#define ADD_ARGUMENT_R_KPAIRS(L1, L2)                                          \
  {                                                                            \
    const int index = INDEX_SQUARE(L1, L2);                                    \
    const vector<FieldID> field_list(kpair_fields_list[index],                 \
                                     kpair_fields_list[index] +                \
                                         NUM_KPAIR_FIELDS);                    \
    launcher.add_argument_r_pairs##L1##L2(kpair_lr_list[index],                \
                                          kpair_lr_list[index], field_list);   \
  }

  ADD_ARGUMENT_R_KPAIRS(0, 0);
  ADD_ARGUMENT_R_KPAIRS(0, 1);
  ADD_ARGUMENT_R_KPAIRS(0, 2);
  ADD_ARGUMENT_R_KPAIRS(0, 3);
  ADD_ARGUMENT_R_KPAIRS(0, 4);
  ADD_ARGUMENT_R_KPAIRS(1, 0);
  ADD_ARGUMENT_R_KPAIRS(1, 1);
  ADD_ARGUMENT_R_KPAIRS(1, 2);
  ADD_ARGUMENT_R_KPAIRS(1, 3);
  ADD_ARGUMENT_R_KPAIRS(1, 4);
  ADD_ARGUMENT_R_KPAIRS(2, 0);
  ADD_ARGUMENT_R_KPAIRS(2, 1);
  ADD_ARGUMENT_R_KPAIRS(2, 2);
  ADD_ARGUMENT_R_KPAIRS(2, 3);
  ADD_ARGUMENT_R_KPAIRS(2, 4);
  ADD_ARGUMENT_R_KPAIRS(3, 0);
  ADD_ARGUMENT_R_KPAIRS(3, 1);
  ADD_ARGUMENT_R_KPAIRS(3, 2);
  ADD_ARGUMENT_R_KPAIRS(3, 3);
  ADD_ARGUMENT_R_KPAIRS(3, 4);
  ADD_ARGUMENT_R_KPAIRS(4, 0);
  ADD_ARGUMENT_R_KPAIRS(4, 1);
  ADD_ARGUMENT_R_KPAIRS(4, 2);
  ADD_ARGUMENT_R_KPAIRS(4, 3);
  ADD_ARGUMENT_R_KPAIRS(4, 4);

#undef ADD_ARGUMENT_R_KPAIRS

#define ADD_ARGUMENT_R_KDENSITY(L2, L4)                                        \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L2, L4);                            \
    const vector<FieldID> field_list(kdensity_fields_list[index],              \
                                     kdensity_fields_list[index] +             \
                                         NUM_KDENSITY_FIELDS);                 \
    launcher.add_argument_r_density##L2##L4(                                   \
        kdensity_lr_list[index], kdensity_lr_list[index], field_list);         \
  }

  ADD_ARGUMENT_R_KDENSITY(0, 0);
  ADD_ARGUMENT_R_KDENSITY(0, 1);
  ADD_ARGUMENT_R_KDENSITY(0, 2);
  ADD_ARGUMENT_R_KDENSITY(0, 3);
  ADD_ARGUMENT_R_KDENSITY(0, 4);
  ADD_ARGUMENT_R_KDENSITY(1, 1);
  ADD_ARGUMENT_R_KDENSITY(1, 2);
  ADD_ARGUMENT_R_KDENSITY(1, 3);
  ADD_ARGUMENT_R_KDENSITY(1, 4);
  ADD_ARGUMENT_R_KDENSITY(2, 2);
  ADD_ARGUMENT_R_KDENSITY(2, 3);
  ADD_ARGUMENT_R_KDENSITY(2, 4);
  ADD_ARGUMENT_R_KDENSITY(3, 3);
  ADD_ARGUMENT_R_KDENSITY(3, 4);
  ADD_ARGUMENT_R_KDENSITY(4, 4);

#undef ADD_ARGUMENT_R_KDENSITY

#define ADD_ARGUMENT_R_KOUTPUT(L1, L3)                                         \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L1, L3);                            \
    const vector<FieldID> field_list(koutput_fields_list[index],               \
                                     koutput_fields_list[index] +              \
                                         NUM_KOUTPUT_FIELDS);                  \
    launcher.add_argument_r_output##L1##L3(                                    \
        koutput_lr_list[index], koutput_lr_list[index], field_list);           \
  }

  ADD_ARGUMENT_R_KOUTPUT(0, 0);
  ADD_ARGUMENT_R_KOUTPUT(0, 1);
  ADD_ARGUMENT_R_KOUTPUT(0, 2);
  ADD_ARGUMENT_R_KOUTPUT(0, 3);
  ADD_ARGUMENT_R_KOUTPUT(0, 4);
  ADD_ARGUMENT_R_KOUTPUT(1, 1);
  ADD_ARGUMENT_R_KOUTPUT(1, 2);
  ADD_ARGUMENT_R_KOUTPUT(1, 3);
  ADD_ARGUMENT_R_KOUTPUT(1, 4);
  ADD_ARGUMENT_R_KOUTPUT(2, 2);
  ADD_ARGUMENT_R_KOUTPUT(2, 3);
  ADD_ARGUMENT_R_KOUTPUT(2, 4);
  ADD_ARGUMENT_R_KOUTPUT(3, 3);
  ADD_ARGUMENT_R_KOUTPUT(3, 4);
  ADD_ARGUMENT_R_KOUTPUT(4, 4);

#undef ADD_ARGUMENT_R_KOUTPUT

  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_parallelism(parallelism);
  launcher.add_argument_largest_momentum(kdata_list.get_largest_momentum());
  Future future = launcher.execute(runtime, ctx);
  future.wait();

  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      runtime->detach_external_resource(ctx, kpair_pr_list[index]);
      runtime->destroy_logical_region(ctx, kpair_lr_list[index]);
      runtime->destroy_index_space(ctx, kpair_ispace_list[index]);
    }
  }
  for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
    for (int L4 = L2; L4 <= MAX_MOMENTUM; L4++) {
      const int index = INDEX_UPPER_TRIANGLE(L2, L4);
      runtime->detach_external_resource(ctx, kdensity_pr_list[index]);
      runtime->destroy_logical_region(ctx, kdensity_lr_list[index]);
      runtime->destroy_index_space(ctx, kdensity_ispace_list[index]);
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L3);
      runtime->detach_external_resource(ctx, koutput_pr_list[index]);
      runtime->destroy_logical_region(ctx, koutput_lr_list[index]);
      runtime->destroy_index_space(ctx, koutput_ispace_list[index]);
    }
  }
}

void EriRegent::initialize_jfock_field_spaces() {
#define INIT_FSPACES(L1, L2)                                                   \
  {                                                                            \
    const int H = TETRAHEDRAL_NUMBER((L1) + (L2) + 1);                         \
    jbra_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)] =                               \
        runtime->create_field_space(ctx);                                      \
    jket_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)] =                               \
        runtime->create_field_space(ctx);                                      \
    {                                                                          \
      FieldAllocator falloc = runtime->create_field_allocator(                 \
          ctx, jbra_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)]);                    \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, X));         \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, Y));         \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, Z));         \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, ETA));       \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, C));         \
      falloc.allocate_field(sizeof(float), JBRA_FIELD_ID(L1, L2, BOUND));      \
      falloc.allocate_field(H * sizeof(double),                                \
                            JBRA_FIELD_ID(L1, L2, OUTPUT));                    \
    }                                                                          \
    {                                                                          \
      FieldAllocator falloc = runtime->create_field_allocator(                 \
          ctx, jket_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)]);                    \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, X));         \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, Y));         \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, Z));         \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, ETA));       \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, C));         \
      falloc.allocate_field(sizeof(float), JKET_FIELD_ID(L1, L2, BOUND));      \
      falloc.allocate_field(H * sizeof(double),                                \
                            JKET_FIELD_ID(L1, L2, DENSITY));                   \
    }                                                                          \
  }

  INIT_FSPACES(0, 0)
  INIT_FSPACES(0, 1)
  INIT_FSPACES(0, 2)
  INIT_FSPACES(0, 3)
  INIT_FSPACES(0, 4)
  INIT_FSPACES(1, 1)
  INIT_FSPACES(1, 2)
  INIT_FSPACES(1, 3)
  INIT_FSPACES(1, 4)
  INIT_FSPACES(2, 2)
  INIT_FSPACES(2, 3)
  INIT_FSPACES(2, 4)
  INIT_FSPACES(3, 3)
  INIT_FSPACES(3, 4)
  INIT_FSPACES(4, 4)

#undef INIT_FSPACES
}

void EriRegent::initialize_kfock_field_spaces() {
#define INIT_KPAIR_FSPACES(L1, L2)                                             \
  {                                                                            \
    const int index = INDEX_SQUARE(L1, L2);                                    \
    kpair_fspaces[index] = runtime->create_field_space(ctx);                   \
    FieldAllocator falloc =                                                    \
        runtime->create_field_allocator(ctx, kpair_fspaces[index]);            \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, X));          \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, Y));          \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, Z));          \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ETA));        \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, C));          \
    falloc.allocate_field(sizeof(float), KPAIR_FIELD_ID(L1, L2, BOUND));       \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ISHELL_X));   \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ISHELL_Y));   \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ISHELL_Z));   \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, JSHELL_X));   \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, JSHELL_Y));   \
    falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, JSHELL_Z));   \
    falloc.allocate_field(sizeof(int1d_t),                                     \
                          KPAIR_FIELD_ID(L1, L2, ISHELL_INDEX));               \
    falloc.allocate_field(sizeof(int1d_t),                                     \
                          KPAIR_FIELD_ID(L1, L2, JSHELL_INDEX));               \
  }

  INIT_KPAIR_FSPACES(0, 0);
  INIT_KPAIR_FSPACES(0, 1);
  INIT_KPAIR_FSPACES(0, 2);
  INIT_KPAIR_FSPACES(0, 3);
  INIT_KPAIR_FSPACES(0, 4);
  INIT_KPAIR_FSPACES(1, 0);
  INIT_KPAIR_FSPACES(1, 1);
  INIT_KPAIR_FSPACES(1, 2);
  INIT_KPAIR_FSPACES(1, 3);
  INIT_KPAIR_FSPACES(1, 4);
  INIT_KPAIR_FSPACES(2, 0);
  INIT_KPAIR_FSPACES(2, 1);
  INIT_KPAIR_FSPACES(2, 2);
  INIT_KPAIR_FSPACES(2, 3);
  INIT_KPAIR_FSPACES(2, 4);
  INIT_KPAIR_FSPACES(3, 0);
  INIT_KPAIR_FSPACES(3, 1);
  INIT_KPAIR_FSPACES(3, 2);
  INIT_KPAIR_FSPACES(3, 3);
  INIT_KPAIR_FSPACES(3, 4);
  INIT_KPAIR_FSPACES(4, 0);
  INIT_KPAIR_FSPACES(4, 1);
  INIT_KPAIR_FSPACES(4, 2);
  INIT_KPAIR_FSPACES(4, 3);
  INIT_KPAIR_FSPACES(4, 4);

#undef INIT_KPAIR_FSPACES

#define INIT_KDENSITY_FSPACES(L2, L4)                                          \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L2, L4);                            \
    kdensity_fspaces[index] = runtime->create_field_space(ctx);                \
    FieldAllocator falloc =                                                    \
        runtime->create_field_allocator(ctx, kdensity_fspaces[index]);         \
    const int H2 = TRIANGLE_NUMBER((L2) + 1);                                  \
    const int H4 = TRIANGLE_NUMBER((L4) + 1);                                  \
    falloc.allocate_field(H2 *H4 * sizeof(double),                             \
                          KDENSITY_FIELD_ID(L2, L4, VALUES));                  \
    falloc.allocate_field(sizeof(float), KDENSITY_FIELD_ID(L2, L4, BOUND));    \
  }

  INIT_KDENSITY_FSPACES(0, 0);
  INIT_KDENSITY_FSPACES(0, 1);
  INIT_KDENSITY_FSPACES(0, 2);
  INIT_KDENSITY_FSPACES(0, 3);
  INIT_KDENSITY_FSPACES(0, 4);
  INIT_KDENSITY_FSPACES(1, 1);
  INIT_KDENSITY_FSPACES(1, 2);
  INIT_KDENSITY_FSPACES(1, 3);
  INIT_KDENSITY_FSPACES(1, 4);
  INIT_KDENSITY_FSPACES(2, 2);
  INIT_KDENSITY_FSPACES(2, 3);
  INIT_KDENSITY_FSPACES(2, 4);
  INIT_KDENSITY_FSPACES(3, 3);
  INIT_KDENSITY_FSPACES(3, 4);
  INIT_KDENSITY_FSPACES(4, 4);

#undef INIT_KDENSITY_FSPACES

#define INIT_KOUTPUT_FSPACES(L1, L3)                                           \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L1, L3);                            \
    koutput_fspaces[index] = runtime->create_field_space(ctx);                 \
    FieldAllocator falloc =                                                    \
        runtime->create_field_allocator(ctx, koutput_fspaces[index]);          \
    const int H1 = TRIANGLE_NUMBER((L1) + 1);                                  \
    const int H3 = TRIANGLE_NUMBER((L3) + 1);                                  \
    falloc.allocate_field(H1 *H3 * sizeof(double),                             \
                          KOUTPUT_FIELD_ID(L1, L3, VALUES));                   \
  }

  INIT_KOUTPUT_FSPACES(0, 0);
  INIT_KOUTPUT_FSPACES(0, 1);
  INIT_KOUTPUT_FSPACES(0, 2);
  INIT_KOUTPUT_FSPACES(0, 3);
  INIT_KOUTPUT_FSPACES(0, 4);
  INIT_KOUTPUT_FSPACES(1, 1);
  INIT_KOUTPUT_FSPACES(1, 2);
  INIT_KOUTPUT_FSPACES(1, 3);
  INIT_KOUTPUT_FSPACES(1, 4);
  INIT_KOUTPUT_FSPACES(2, 2);
  INIT_KOUTPUT_FSPACES(2, 3);
  INIT_KOUTPUT_FSPACES(2, 4);
  INIT_KOUTPUT_FSPACES(3, 3);
  INIT_KOUTPUT_FSPACES(3, 4);
  INIT_KOUTPUT_FSPACES(4, 4);

#undef INIT_KOUTPUT_FSPACES
}
