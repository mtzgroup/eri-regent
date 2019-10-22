// c++ -O2 -Wall -Werror -I[PATH TO legion/runtime] main.cpp eri_regent.cpp -L[PATH TO legion/bindings/regent] -lregent -L. -ljfock_tasks
// LD_LIBRARY_PATH=[PATH TO legion/bindings/regent]:. ./a.out

#include <iostream>

#include "eri_regent.h"
#include "fields.h"
#include "helper.h"
#include "jfock_tasks.h"
#include "legion.h"

// #include "test.h"

using namespace std;
using namespace Legion;
using namespace eri_regent;

void top_level_task(const Task *task, const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  // TODO: Make sure this line is ok
  const Memory local_sysmem = Machine::MemoryQuery(Machine::get_machine())
      .has_affinity_to(runtime->get_executing_processor(ctx))
      .only_kind(Memory::SYSTEM_MEM)
      .first();

  // r_gamma_table is readonly and constant throughout execution
  LogicalRegion gamma_table_lr;
  PhysicalRegion gamma_table_pr;
  create_gamma_table_region(gamma_table_lr, gamma_table_pr, ctx, runtime,
                            local_sysmem);

  FieldSpace jbra_fspaces[MAX_MOMENTUM_INDEX + 1];
  FieldSpace jket_fspaces[MAX_MOMENTUM_INDEX + 1];
  initialize_field_spaces(jbra_fspaces, jket_fspaces, ctx, runtime);

  TeraChemJDataList jdata_list = {0};
  // jdata_list.num_jbras[L_PAIR_TO_INDEX(0, 1)] = 2;
  // jdata_list.jbras[L_PAIR_TO_INDEX(0, 1)] = jbras01;
  // jdata_list.output[L_PAIR_TO_INDEX(0, 1)] = (double *)malloc(sizeof(double) * 2 * 4);
  //
  // jdata_list.num_jkets[L_PAIR_TO_INDEX(0, 1)] = 2;
  // jdata_list.jkets[L_PAIR_TO_INDEX(0, 1)] = jkets01;
  // jdata_list.density[L_PAIR_TO_INDEX(0, 1)] = density01;

  launch_jfock_task(jdata_list, gamma_table_lr, 1.234, 1,
                    jbra_fspaces, jket_fspaces, ctx, runtime, local_sysmem);

  destroy_attached_region(gamma_table_lr, gamma_table_pr, ctx, runtime);
}

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
