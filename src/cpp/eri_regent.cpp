#include <cstdlib>

#include "eri_regent.h"
#include "fields.h"
#include "helper.h"
#include "../mcmurchie/gamma_table.h"
#include "jfock_tasks.h"
#include "legion.h"

using namespace std;
using namespace Legion;

namespace eri_regent {

void* fill_jbra_data(const TeraChemJData *jbras, size_t num_jbras, size_t L12);
void* fill_jket_data(const TeraChemJData *jkets, size_t num_jkets,
                     const double *density, size_t L12);


void launch_jfock_task(TeraChemJDataList& jdata_list,
                       LogicalRegion &gamma_table_lr,
                       float threshold, int parallelism,
		                   const FieldSpace *jbra_fspaces,
		                   const FieldSpace *jket_fspaces,
                       Context ctx, Runtime *runtime,
                       const Memory memory) {
  LogicalRegion jbras_lr_list[MAX_MOMENTUM_INDEX + 1];
  PhysicalRegion jbras_pr_list[MAX_MOMENTUM_INDEX + 1];
  void* jbras_data_list[MAX_MOMENTUM_INDEX + 1] = {0};
  for (size_t L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (size_t L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const size_t index = L_PAIR_TO_INDEX(L1, L2);
      jbras_data_list[index] = fill_jbra_data(jdata_list.jbras[index],
                                              jdata_list.num_jbras[index],
					                                    L1 + L2);
      const Rect<1> rect(0, jdata_list.num_jbras[index] - 1);
      const IndexSpace ispace = runtime->create_index_space(ctx, rect);
      jbras_lr_list[index] = runtime->create_logical_region(
                                          ctx, ispace, jbra_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE,
                              jbras_lr_list[index],
		                          jbras_lr_list[index]);
      launcher.attach_array_aos(jbras_data_list[index], /*column major*/false,
		                            jbra_fields_list[index], memory);
      jbras_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  LogicalRegion jkets_lr_list[MAX_MOMENTUM_INDEX + 1];
  PhysicalRegion jkets_pr_list[MAX_MOMENTUM_INDEX + 1];
  void* jkets_data_list[MAX_MOMENTUM_INDEX + 1] = {0};
  for (size_t L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (size_t L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const size_t index = L_PAIR_TO_INDEX(L1, L2);
      jkets_data_list[index] = fill_jket_data(jdata_list.jkets[index],
                                              jdata_list.num_jkets[index],
                                              jdata_list.density[index],
					                                    L1 + L2);
      const Rect<1> rect(0, jdata_list.num_jkets[index] - 1);
      const IndexSpace ispace = runtime->create_index_space(ctx, rect);
      jkets_lr_list[index] = runtime->create_logical_region(
                                          ctx, ispace, jket_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE,
                              jkets_lr_list[index],
		                          jkets_lr_list[index]);
      launcher.attach_array_aos(jkets_data_list[index], /*column major*/false,
		                            jket_fields_list[index], memory);
      jkets_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
    }
  }

  // TODO: use jfock instead of toy task
  toy_task_launcher launcher;

#define ADD_ARGUMENT_R_JBRAS(L1, L2)             \
  launcher.add_argument_r_jbras##L1##L2(         \
    jbras_lr_list[L_PAIR_TO_INDEX(L1, L2)],      \
    jbras_lr_list[L_PAIR_TO_INDEX(L1, L2)],      \
    jbra_fields_list[L_PAIR_TO_INDEX(L1, L2)])

  // ADD_ARGUMENT_R_JBRAS(0, 0);
  ADD_ARGUMENT_R_JBRAS(0, 1);
  // ADD_ARGUMENT_R_JBRAS(1, 1);
  // ADD_ARGUMENT_R_JBRAS(0, 2);
  // ADD_ARGUMENT_R_JBRAS(1, 2);
  // ADD_ARGUMENT_R_JBRAS(2, 2);
  // ADD_ARGUMENT_R_JBRAS(0, 3);
  // ADD_ARGUMENT_R_JBRAS(1, 3);
  // ADD_ARGUMENT_R_JBRAS(2, 3);
  // ADD_ARGUMENT_R_JBRAS(3, 3);
  // ADD_ARGUMENT_R_JBRAS(0, 4);
  // ADD_ARGUMENT_R_JBRAS(1, 4);
  // ADD_ARGUMENT_R_JBRAS(2, 4);
  // ADD_ARGUMENT_R_JBRAS(3, 4);
  // ADD_ARGUMENT_R_JBRAS(4, 4);

#undef ADD_ARGUMENT_R_JBRAS

#define ADD_ARGUMENT_R_JKETS(L1, L2)             \
  launcher.add_argument_r_jkets##L1##L2(         \
    jkets_lr_list[L_PAIR_TO_INDEX(L1, L2)],      \
    jkets_lr_list[L_PAIR_TO_INDEX(L1, L2)],      \
    jket_fields_list[L_PAIR_TO_INDEX(L1, L2)])

  // ADD_ARGUMENT_R_JKETS(0, 0);
  ADD_ARGUMENT_R_JKETS(0, 1);
  // ADD_ARGUMENT_R_JKETS(1, 1);
  // ADD_ARGUMENT_R_JKETS(0, 2);
  // ADD_ARGUMENT_R_JKETS(1, 2);
  // ADD_ARGUMENT_R_JKETS(2, 2);
  // ADD_ARGUMENT_R_JKETS(0, 3);
  // ADD_ARGUMENT_R_JKETS(1, 3);
  // ADD_ARGUMENT_R_JKETS(2, 3);
  // ADD_ARGUMENT_R_JKETS(3, 3);
  // ADD_ARGUMENT_R_JKETS(0, 4);
  // ADD_ARGUMENT_R_JKETS(1, 4);
  // ADD_ARGUMENT_R_JKETS(2, 4);
  // ADD_ARGUMENT_R_JKETS(3, 4);
  // ADD_ARGUMENT_R_JKETS(4, 4);

#undef ADD_ARGUMENT_R_JKETS

  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_parallelism(parallelism);
  launcher.execute(runtime, ctx);

  for (size_t L1 = 0; L1 < MAX_MOMENTUM; L1++) {
    for (size_t L2 = L1; L2 < MAX_MOMENTUM; L2++) {
      const size_t index = L_PAIR_TO_INDEX(L1, L2);
      // const size_t output_offset = sizeof(double) * 5 + sizeof(float);
      // memcpy((void *)jdata_list.output[index],
      //        (const void *)((char *)jbras_data_list[index] + output_offset),
      //        jdata_list.num_jbras[index] * sizeof(double) * COMPUTE_H(L1 + L2));

      destroy_attached_region(jbras_lr_list[index], jbras_pr_list[index],
                              ctx, runtime);
      free(jbras_data_list[index]);
      destroy_attached_region(jkets_lr_list[index], jkets_pr_list[index],
                              ctx, runtime);
      free(jkets_data_list[index]);
    }
  }
}


void* fill_jbra_data(const TeraChemJData *jbras, size_t num_jbras, size_t L12) {
  const size_t H = COMPUTE_H(L12);
  const size_t stride = sizeof(double) * 5 + sizeof(float) + sizeof(double) * H;
  void *data = calloc(stride, num_jbras);
  for (size_t i = 0; i < num_jbras; i++) {
    memcpy((void *)((char *)data + i * stride), (const void*)(jbras + i),
           sizeof(TeraChemJData));
  }
  return data;
}


void* fill_jket_data(const TeraChemJData *jkets, size_t num_jkets,
                     const double* density, size_t L12) {
  const size_t H = COMPUTE_H(L12);
  const size_t stride = sizeof(double) * 5 + sizeof(float) + sizeof(double) * H;
  const size_t density_offset = sizeof(double) * 5 + sizeof(float);
  void *data = calloc(stride, num_jkets);
  for (size_t i = 0; i < num_jkets; i++) {
    memcpy((void *)((char *)data + i * stride), (const void*)(jkets + i),
           sizeof(TeraChemJData));
    memcpy((void *)((char *)data + i * stride + density_offset),
           (const void*)(density + i * H), sizeof(double) * H);
  }
  return data;
}


void create_gamma_table_region(LogicalRegion &lr, PhysicalRegion &pr,
                               Context ctx, Runtime *runtime,
                               const Memory memory) {
  const Rect<2> rect({0, 0}, {18 - 1, 700 - 1});
  const IndexSpace ispace = runtime->create_index_space(ctx, rect);
  const FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(5 * sizeof(double), GAMMA_TABLE_FIELD_ID);
  }

  lr = runtime->create_logical_region(ctx, ispace, fspace);

  AttachLauncher launcher(EXTERNAL_INSTANCE, lr, lr);

  launcher.attach_array_aos((void *)gamma_table, /*column major=*/false,
                            {GAMMA_TABLE_FIELD_ID}, memory);
  pr = runtime->attach_external_resource(ctx, launcher);
}


void destroy_attached_region(LogicalRegion &lr, PhysicalRegion &pr,
                             Context ctx, Runtime *runtime) {
  runtime->detach_external_resource(ctx, pr);
  runtime->destroy_logical_region(ctx, lr);
}


void initialize_field_spaces(FieldSpace jbra_fspaces[MAX_MOMENTUM_INDEX + 1],
                             FieldSpace jket_fspaces[MAX_MOMENTUM_INDEX + 1],
                             Context ctx, Runtime *runtime) {
#define INIT_FSPACES(L1, L2)                                                   \
{                                                                              \
  const int H = COMPUTE_H((L1) + (L2));                                        \
  jbra_fspaces[L_PAIR_TO_INDEX(L1, L2)] = runtime->create_field_space(ctx);    \
  jket_fspaces[L_PAIR_TO_INDEX(L1, L2)] = runtime->create_field_space(ctx);    \
  {                                                                            \
    FieldAllocator falloc = runtime->create_field_allocator(                   \
      ctx, jbra_fspaces[L_PAIR_TO_INDEX(L1, L2)]);                             \
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, X));           \
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, Y));           \
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, Z));           \
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, ETA));         \
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, C));           \
    falloc.allocate_field(sizeof(float), JBRA_FIELD_ID(L1, L2, BOUND));        \
    falloc.allocate_field(H * sizeof(double), JBRA_FIELD_ID(L1, L2, OUTPUT));  \
  }                                                                            \
  {                                                                            \
    FieldAllocator falloc = runtime->create_field_allocator(                   \
      ctx, jket_fspaces[L_PAIR_TO_INDEX(L1, L2)]);                             \
    falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, X));           \
    falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, Y));           \
    falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, Z));           \
    falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, ETA));         \
    falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, C));           \
    falloc.allocate_field(sizeof(float), JKET_FIELD_ID(L1, L2, BOUND));        \
    falloc.allocate_field(H * sizeof(double), JKET_FIELD_ID(L1, L2, DENSITY)); \
  }                                                                            \
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

}
