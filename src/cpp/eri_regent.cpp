#include "../mcmurchie/gamma_table.h"
#include "eri_regent.h"
#include "helper.h"
#include "jfock_tasks.h"
#include "legion.h"

using namespace std;
using namespace Legion;

EriRegent::EriRegent(Context &ctx, Runtime *runtime)
    : ctx(ctx), runtime(runtime) {
  memory = Machine::MemoryQuery(Machine::get_machine())
               .has_affinity_to(runtime->get_executing_processor(ctx))
               .only_kind(Memory::SYSTEM_MEM)
               .first();

  create_gamma_table_region();

  initialize_field_spaces();
}

EriRegent::~EriRegent() {
  runtime->detach_external_resource(ctx, gamma_table_pr);
  runtime->destroy_logical_region(ctx, gamma_table_lr);
}

void EriRegent::launch_jfock_task(EriRegent::TeraChemJDataList &jdata_list,
                                  float threshold, int parallelism) {
  LogicalRegion jbras_lr_list[MAX_MOMENTUM_INDEX + 1];
  PhysicalRegion jbras_pr_list[MAX_MOMENTUM_INDEX + 1];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      const Rect<1> rect(0, jdata_list.get_num_jbras(L1, L2) - 1);
      const IndexSpace ispace = runtime->create_index_space(ctx, rect);
      jbras_lr_list[index] =
          runtime->create_logical_region(ctx, ispace, jbra_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jbras_lr_list[index],
                              jbras_lr_list[index]);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(jdata_list.jbras[index], /*column major*/ false,
                                jbra_fields_list[index], memory);
      jbras_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  LogicalRegion jkets_lr_list[MAX_MOMENTUM_INDEX + 1];
  PhysicalRegion jkets_pr_list[MAX_MOMENTUM_INDEX + 1];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      const Rect<1> rect(0, jdata_list.get_num_jkets(L1, L2) - 1);
      const IndexSpace ispace = runtime->create_index_space(ctx, rect);
      jkets_lr_list[index] =
          runtime->create_logical_region(ctx, ispace, jket_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jkets_lr_list[index],
                              jkets_lr_list[index]);
      launcher.attach_array_aos(jdata_list.jkets[index], /*column major*/ false,
                                jket_fields_list[index], memory);
      jkets_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  jfock_task_launcher launcher;

#define ADD_ARGUMENT_R_JBRAS(L1, L2)                                           \
  launcher.add_argument_r_jbras##L1##L2(                                       \
      jbras_lr_list[L_PAIR_TO_INDEX(L1, L2)],                                  \
      jbras_lr_list[L_PAIR_TO_INDEX(L1, L2)],                                  \
      jbra_fields_list[L_PAIR_TO_INDEX(L1, L2)])

  ADD_ARGUMENT_R_JBRAS(0, 0);
  ADD_ARGUMENT_R_JBRAS(0, 1);
  ADD_ARGUMENT_R_JBRAS(1, 1);
  ADD_ARGUMENT_R_JBRAS(0, 2);
  ADD_ARGUMENT_R_JBRAS(1, 2);
  ADD_ARGUMENT_R_JBRAS(2, 2);
  ADD_ARGUMENT_R_JBRAS(0, 3);
  ADD_ARGUMENT_R_JBRAS(1, 3);
  ADD_ARGUMENT_R_JBRAS(2, 3);
  ADD_ARGUMENT_R_JBRAS(3, 3);
  ADD_ARGUMENT_R_JBRAS(0, 4);
  ADD_ARGUMENT_R_JBRAS(1, 4);
  ADD_ARGUMENT_R_JBRAS(2, 4);
  ADD_ARGUMENT_R_JBRAS(3, 4);
  ADD_ARGUMENT_R_JBRAS(4, 4);

#undef ADD_ARGUMENT_R_JBRAS

#define ADD_ARGUMENT_R_JKETS(L1, L2)                                           \
  launcher.add_argument_r_jkets##L1##L2(                                       \
      jkets_lr_list[L_PAIR_TO_INDEX(L1, L2)],                                  \
      jkets_lr_list[L_PAIR_TO_INDEX(L1, L2)],                                  \
      jket_fields_list[L_PAIR_TO_INDEX(L1, L2)])

  ADD_ARGUMENT_R_JKETS(0, 0);
  ADD_ARGUMENT_R_JKETS(0, 1);
  ADD_ARGUMENT_R_JKETS(1, 1);
  ADD_ARGUMENT_R_JKETS(0, 2);
  ADD_ARGUMENT_R_JKETS(1, 2);
  ADD_ARGUMENT_R_JKETS(2, 2);
  ADD_ARGUMENT_R_JKETS(0, 3);
  ADD_ARGUMENT_R_JKETS(1, 3);
  ADD_ARGUMENT_R_JKETS(2, 3);
  ADD_ARGUMENT_R_JKETS(3, 3);
  ADD_ARGUMENT_R_JKETS(0, 4);
  ADD_ARGUMENT_R_JKETS(1, 4);
  ADD_ARGUMENT_R_JKETS(2, 4);
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
      const int index = L_PAIR_TO_INDEX(L1, L2);
      runtime->detach_external_resource(ctx, jbras_pr_list[index]);
      runtime->destroy_logical_region(ctx, jbras_lr_list[index]);
      runtime->detach_external_resource(ctx, jkets_pr_list[index]);
      runtime->destroy_logical_region(ctx, jkets_lr_list[index]);
    }
  }
}

void EriRegent::create_gamma_table_region() {
  const Rect<2> rect({0, 0}, {18 - 1, 700 - 1});
  const IndexSpace ispace = runtime->create_index_space(ctx, rect);
  const FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(5 * sizeof(double), GAMMA_TABLE_FIELD_ID);
  }

  gamma_table_lr = runtime->create_logical_region(ctx, ispace, fspace);

  AttachLauncher launcher(EXTERNAL_INSTANCE, gamma_table_lr, gamma_table_lr);

  launcher.attach_array_aos((void *)gamma_table, /*column major*/ false,
                            {GAMMA_TABLE_FIELD_ID}, memory);
  gamma_table_pr = runtime->attach_external_resource(ctx, launcher);
}

void EriRegent::initialize_field_spaces() {
#define INIT_FSPACES(L1, L2)                                                   \
  {                                                                            \
    const int H = COMPUTE_H((L1) + (L2));                                      \
    jbra_fspaces[L_PAIR_TO_INDEX(L1, L2)] = runtime->create_field_space(ctx);  \
    jket_fspaces[L_PAIR_TO_INDEX(L1, L2)] = runtime->create_field_space(ctx);  \
    {                                                                          \
      FieldAllocator falloc = runtime->create_field_allocator(                 \
          ctx, jbra_fspaces[L_PAIR_TO_INDEX(L1, L2)]);                         \
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
          ctx, jket_fspaces[L_PAIR_TO_INDEX(L1, L2)]);                         \
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
  INIT_FSPACES(1, 1)
  INIT_FSPACES(0, 2)
  INIT_FSPACES(1, 2)
  INIT_FSPACES(2, 2)
  INIT_FSPACES(0, 3)
  INIT_FSPACES(1, 3)
  INIT_FSPACES(2, 3)
  INIT_FSPACES(3, 3)
  INIT_FSPACES(0, 4)
  INIT_FSPACES(1, 4)
  INIT_FSPACES(2, 4)
  INIT_FSPACES(3, 4)
  INIT_FSPACES(4, 4)

#undef INIT_FSPACES
}

void EriRegent::TeraChemJDataList::allocate_jbras(int L1, int L2, int n) {
  const int index = L_PAIR_TO_INDEX(L1, L2);
  assert(0 <= index && index <= MAX_MOMENTUM_INDEX);
  if (n > 0) {
    num_jbras[index] = n;
    jbras[index] = calloc(n, stride(L1, L2));
    assert(jbras[index]);
  }
}

void EriRegent::TeraChemJDataList::allocate_jkets(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = L_PAIR_TO_INDEX(L1, L2);
  if (n > 0) {
    num_jkets[index] = n;
    jkets[index] = calloc(n, stride(L1, L2));
    assert(jkets[index]);
  }
}

void EriRegent::TeraChemJDataList::free_data() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      if (num_jbras[index] > 0) {
        free(jbras[index]);
      }
      if (num_jkets[index] > 0) {
        free(jkets[index]);
      }
    }
  }
}

int EriRegent::TeraChemJDataList::get_num_jbras(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return num_jbras[L_PAIR_TO_INDEX(L1, L2)];
}
int EriRegent::TeraChemJDataList::get_num_jkets(int L1, int L2) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  return num_jkets[L_PAIR_TO_INDEX(L1, L2)];
}

EriRegent::TeraChemJData *
EriRegent::TeraChemJDataList::get_jbra_ptr(int L1, int L2, int i) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  assert(0 <= i && i < get_num_jbras(L1, L2));
  return (EriRegent::TeraChemJData *)((char *)jbras[L_PAIR_TO_INDEX(L1, L2)] +
                                      i * stride(L1, L2));
}
EriRegent::TeraChemJData *
EriRegent::TeraChemJDataList::get_jket_ptr(int L1, int L2, int i) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  assert(0 <= i && i < get_num_jkets(L1, L2));
  return (EriRegent::TeraChemJData *)((char *)jkets[L_PAIR_TO_INDEX(L1, L2)] +
                                      i * stride(L1, L2));
}

double *EriRegent::TeraChemJDataList::get_output_ptr(int L1, int L2, int i) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  assert(0 <= i && i < get_num_jbras(L1, L2));
  return (double *)((char *)jbras[L_PAIR_TO_INDEX(L1, L2)] +
                    i * stride(L1, L2) + array_data_offset(L1, L2));
}
double *EriRegent::TeraChemJDataList::get_density_ptr(int L1, int L2, int i) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  assert(0 <= i && i < get_num_jkets(L1, L2));
  return (double *)((char *)jkets[L_PAIR_TO_INDEX(L1, L2)] +
                    i * stride(L1, L2) + array_data_offset(L1, L2));
}

int EriRegent::TeraChemJDataList::stride(int L1, int L2) {
  const int H = COMPUTE_H(L1 + L2);
  return 5 * sizeof(double) + sizeof(float) + H * sizeof(double);
}

int EriRegent::TeraChemJDataList::array_data_offset(int L1, int L2) {
  return 5 * sizeof(double) + sizeof(float);
}

int EriRegent::TeraChemJDataList::get_largest_momentum() {
  int largest_momentum = -1;
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      if (get_num_jbras(L1, L2) > 0 || get_num_jkets(L1, L2) > 0) {
        largest_momentum = max(largest_momentum, max(L1, L2));
      }
    }
  }
  return largest_momentum;
}
