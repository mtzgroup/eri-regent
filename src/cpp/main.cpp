/**
 * c++ -O2 -Wall -Werror -I[PATH TO legion/runtime] main.cpp eri_regent.cpp \
 *   -L[PATH TO legion/bindings/regent] -lregent -L. -ljfock_tasks \
 *   LD_LIBRARY_PATH=[PATH TO legion/bindings/regent]:. ./a.out
 */

#include <iostream>

#include "eri_regent.h"
#include "helper.h"
#include "jfock_tasks.h"
#include "legion.h"

// #include "test.h"

using namespace std;
using namespace Legion;

void top_level_task(const Task *task, const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  EriRegent eri_regent(ctx, runtime);

  EriRegent::TeraChemJDataList jdata_list = {0};
  // jdata_list.num_jbras[L_PAIR_TO_INDEX(0, 1)] = 2;
  // jdata_list.jbras[L_PAIR_TO_INDEX(0, 1)] = jbras01;
  // jdata_list.output[L_PAIR_TO_INDEX(0, 1)] = (double *)malloc(sizeof(double)
  // * 2 * 4);
  //
  // jdata_list.num_jkets[L_PAIR_TO_INDEX(0, 1)] = 2;
  // jdata_list.jkets[L_PAIR_TO_INDEX(0, 1)] = jkets01;
  // jdata_list.density[L_PAIR_TO_INDEX(0, 1)] = density01;

  eri_regent.launch_jfock_task(jdata_list, 1.12, 1);
}

int main(int argc, char **argv) {
  enum { // Task IDs
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
