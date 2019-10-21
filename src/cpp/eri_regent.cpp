// c++ -O2 -Wall -Werror -I[PATH TO legion/runtime] eri_regent.cpp -L[PATH TO legion/bindings/regent] -lregent -L. -ljfock_tasks
// LD_LIBRARY_PATH=[PATH TO legion/bindings/regent]:. ./a.out

#include <iostream>

#include "eri_regent.h"
#include "fields.h"
#include "../mcmurchie/gamma_table.h"
#include "jfock_tasks.h"
#include "legion.h"

using namespace std;
using namespace Legion;


int main(int argc, char **argv) {
  enum {  // Task IDs
    TOP_LEVEL_TASK_ID,
  };

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  jfock_tasks_h_register();
  return Runtime::start(argc, argv);
}


void top_level_task(const Task *task, const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  // TODO: Make sure this line is ok
  const Memory local_sysmem = Machine::MemoryQuery(Machine::get_machine())
      .has_affinity_to(runtime->get_executing_processor(ctx))
      .only_kind(Memory::SYSTEM_MEM)
      .first();

  LogicalRegion gamma_table_lr;
  PhysicalRegion gamma_table_pr;
  create_gamma_table_region(gamma_table_lr, gamma_table_pr, ctx, runtime,
                            local_sysmem);

  TeraChemJBrasList jbras_list = {0};
  // jbras_list.num_jbras01 = 2;
  // jbras_list.jbras01 = jbras01;
  // jbras_list.output01 = output01;
  TeraChemJKetsList jkets_list = {0};
  // jkets_list.num_jkets01 = 2;
  // jkets_list.jkets01 = jkets01;
  // jkets_list.density01 = density01;


  launch_jfock_task(jbras_list, jkets_list, gamma_table_lr, 1.234, 1,
                    ctx, runtime, local_sysmem);

  destroy_attached_region(gamma_table_lr, gamma_table_pr, ctx, runtime);
}


void launch_jfock_task(TeraChemJBrasList& jbras_list,
                       TeraChemJKetsList& jkets_list,
                       LogicalRegion &gamma_table_lr,
                       float threshold, int parallelism,
                       Context ctx, Runtime *runtime,
                       const Memory memory) {
  LogicalRegion jbras01_lr;
  PhysicalRegion jbras01_pr;
  void *jbras01_data = fill_jbra_data(jbras_list.jbras01,
                                      jbras_list.num_jbras01, 0+1);
  create_jbra01_region(jbras01_lr, jbras01_pr,
                       jbras01_data, jbras_list.num_jbras01,
                       ctx, runtime, memory);

  LogicalRegion jkets01_lr;
  PhysicalRegion jkets01_pr;
  void *jkets01_data = fill_jket_data(jkets_list.jkets01,
                                      jkets_list.num_jkets01,
                                      jkets_list.density01, 0+1);
  create_jket01_region(jkets01_lr, jkets01_pr,
                       jkets01_data, jkets_list.num_jkets01,
                       ctx, runtime, memory);

  toy_task_launcher launcher;
  launcher.add_argument_r_jbras01(jbras01_lr, jbras01_lr,
                                  {JBRA_FIELD_IDS(0, 1)});
  launcher.add_argument_r_jkets01(jkets01_lr, jkets01_lr,
                                  {JKET_FIELD_IDS(0, 1)});
  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_parallelism(parallelism);
  launcher.execute(runtime, ctx);

  destroy_attached_region(jbras01_lr, jbras01_pr, ctx, runtime);
  free(jbras01_data);

  destroy_attached_region(jkets01_lr, jkets01_pr, ctx, runtime);
  free(jkets01_data);
}


void* fill_jbra_data(const TeraChemJData *jbras, size_t num_jbras, size_t L12) {
  const size_t H = COMPUTE_H(L12);
  const size_t stride = sizeof(double) * 5 + sizeof(float) + sizeof(double) * H;
  void *data = calloc(stride, num_jbras);
  for (size_t i = 0; i < num_jbras; i++) {
    memcpy((void *)((char *)data + i * stride),
           (const void*)(jbras + i),
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
    memcpy((void *)((char *)data + i * stride),
           (const void*)(jkets + i),
           sizeof(TeraChemJData));
    memcpy((void *)((char *)data + i * stride + density_offset),
           (const void*)(density + i * H),
           sizeof(double) * H);
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
