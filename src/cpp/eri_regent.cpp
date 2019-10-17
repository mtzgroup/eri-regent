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

  // TODO: Get data from terachem and pass here
  launch_jfock_task(NULL, NULL, gamma_table_lr, 1.234, 1, ctx, runtime);

  destroy_attached_region(gamma_table_lr, gamma_table_pr, ctx, runtime);
}


void launch_jfock_task(TeraChemJBraList* jbras_list,
                       TeraChemJKetList* jkets_list,
                       LogicalRegion &gamma_table_lr,
                       float threshold, int parallelism,
                       Context ctx, Runtime *runtime) {
  // TODO: Generate regions for jbras and jkets and pass them as arguments
  toy_task_launcher launcher;
  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_parallelism(parallelism);
  launcher.execute(runtime, ctx);
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
