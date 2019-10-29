#include <cstdlib>

#include "../mcmurchie/gamma_table.h"
#include "eri_regent.h"
#include "helper.h"
#include "jfock_tasks.h"
#include "legion.h"

using namespace std;
using namespace Legion;

EriRegent::EriRegent(int max_momentum, Context &ctx, Runtime *runtime)
    : max_momentum(max_momentum), ctx(ctx), runtime(runtime) {
  // TODO: Make sure this line is ok
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
  void *jbras_data_list[MAX_MOMENTUM_INDEX + 1] = {0};
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      jbras_data_list[index] = fill_jbra_data(jdata_list, L1, L2);
      const Rect<1> rect(0, jdata_list.get_num_jbras(L1, L2) - 1);
      const IndexSpace ispace = runtime->create_index_space(ctx, rect);
      jbras_lr_list[index] =
          runtime->create_logical_region(ctx, ispace, jbra_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jbras_lr_list[index],
                              jbras_lr_list[index]);
      launcher.attach_array_aos(jbras_data_list[index], /*column major*/ false,
                                jbra_fields_list[index], memory);
      jbras_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  LogicalRegion jkets_lr_list[MAX_MOMENTUM_INDEX + 1];
  PhysicalRegion jkets_pr_list[MAX_MOMENTUM_INDEX + 1];
  void *jkets_data_list[MAX_MOMENTUM_INDEX + 1] = {0};
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      jkets_data_list[index] = fill_jket_data(jdata_list, L1, L2);
      const Rect<1> rect(0, jdata_list.get_num_jkets(L1, L2) - 1);
      const IndexSpace ispace = runtime->create_index_space(ctx, rect);
      jkets_lr_list[index] =
          runtime->create_logical_region(ctx, ispace, jket_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jkets_lr_list[index],
                              jkets_lr_list[index]);
      launcher.attach_array_aos(jkets_data_list[index], /*column major*/ false,
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
  launcher.add_argument_max_momentum(max_momentum);
  launcher.execute(runtime, ctx);

  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      const int H = COMPUTE_H(L1 + L2);
      const int stride =
          sizeof(double) * 5 + sizeof(float) + sizeof(double) * H;
      const int output_offset = sizeof(double) * 5 + sizeof(float);
      for (int i = 0; i < jdata_list.get_num_jbras(L1, L2); i++) {
        memcpy((void *)jdata_list.get_output(L1, L2, i),
               (const void *)((char *)jbras_data_list[index] + i * stride +
                              output_offset),
               sizeof(double) * H);
      }

      runtime->detach_external_resource(ctx, jbras_pr_list[index]);
      runtime->destroy_logical_region(ctx, jbras_lr_list[index]);
      free(jbras_data_list[index]);
      runtime->detach_external_resource(ctx, jkets_pr_list[index]);
      runtime->destroy_logical_region(ctx, jkets_lr_list[index]);
      free(jkets_data_list[index]);
    }
  }
}

void *EriRegent::fill_jbra_data(EriRegent::TeraChemJDataList &jdata_list,
                                int L1, int L2) {
  const int num_jbras = jdata_list.get_num_jbras(L1, L2);
  const int H = COMPUTE_H(L1 + L2);
  const int stride = sizeof(double) * 5 + sizeof(float) + sizeof(double) * H;
  void *data = calloc(stride, num_jbras);
  for (int i = 0; i < num_jbras; i++) {
    memcpy((void *)((char *)data + i * stride),
           (const void *)jdata_list.get_jbra(L1, L2, i), sizeof(TeraChemJData));
  }
  return data;
}

void *EriRegent::fill_jket_data(EriRegent::TeraChemJDataList &jdata_list,
                                int L1, int L2) {
  const int num_jkets = jdata_list.get_num_jkets(L1, L2);
  const int H = COMPUTE_H(L1 + L2);
  const int stride = sizeof(double) * 5 + sizeof(float) + sizeof(double) * H;
  const int density_offset = sizeof(double) * 5 + sizeof(float);
  void *data = malloc(stride * num_jkets);
  for (int i = 0; i < num_jkets; i++) {
    memcpy((void *)((char *)data + i * stride),
           (const void *)(jdata_list.get_jkets_ptr(L1, L2) + i),
           sizeof(TeraChemJData));
    memcpy((void *)((char *)data + i * stride + density_offset),
           (const void *)jdata_list.get_density(L1, L2, i), sizeof(double) * H);
  }
  return data;
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

  launcher.attach_array_aos((void *)gamma_table, /*column major=*/false,
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
    jbras[index] = (TeraChemJData *)malloc(n * sizeof(TeraChemJData));
    output[index] = (double *)calloc(n * COMPUTE_H(L1 + L2), sizeof(double));
  }
}

void EriRegent::TeraChemJDataList::allocate_jkets(int L1, int L2, int n) {
  assert(0 <= L1 && L1 <= L2 && L2 <= MAX_MOMENTUM);
  const int index = L_PAIR_TO_INDEX(L1, L2);
  if (n > 0) {
    num_jkets[index] = n;
    jkets[index] = (TeraChemJData *)malloc(n * sizeof(TeraChemJData));
    density[index] = (double *)malloc(n * COMPUTE_H(L1 + L2) * sizeof(double));
  }
}

void EriRegent::TeraChemJDataList::free_data() {
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = L_PAIR_TO_INDEX(L1, L2);
      if (num_jbras[index] > 0) {
        free(jbras[index]);
        free(output[index]);
      }
      if (num_jkets[index] > 0) {
        free(jkets[index]);
        free(density[index]);
      }
    }
  }
}
