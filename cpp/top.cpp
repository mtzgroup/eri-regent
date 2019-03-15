// c++ -O2 -Wall -Werror -I[PATH TO legion/runtime] top.cpp -L[PATH TO bindings/regent] -lregent -L. -lcoulomb_tasks
// LD_LIBRARY_PATH=[PATH TO legion/bindings/regent]:. ./a.out

#include "coulomb_tasks.h"

#include "legion.h"
#include "legion/legion_c_util.h"

#include <iostream>

using namespace Legion;

const int TID_TOP_LEVEL_TASK = 0;

enum {
  HERMITE_GAUSSIAN_X,
  HERMITE_GAUSSIAN_Y,
  HERMITE_GAUSSIAN_Z,
  HERMITE_GAUSSIAN_ETA,
  HERMITE_GAUSSIAN_L,
  HERMITE_GAUSSIAN_DATA_RECT,
  HERMITE_GAUSSIAN_BOUND
};

const std::vector<FieldID> hermite_gaussian_fields = {HERMITE_GAUSSIAN_X,
                                                      HERMITE_GAUSSIAN_Y,
                                                      HERMITE_GAUSSIAN_Z,
                                                      HERMITE_GAUSSIAN_ETA,
                                                      HERMITE_GAUSSIAN_L,
                                                      HERMITE_GAUSSIAN_DATA_RECT,
                                                      HERMITE_GAUSSIAN_BOUND};

enum {
  PRECOMPUTED_BOYS_DATA
};

const std::vector<FieldID> precomputed_boys_data = {PRECOMPUTED_BOYS_DATA};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  std::cout << "Hi again from Regent\n";

  IndexSpace ispace = runtime->create_index_space(ctx, Domain(Rect<1>(0, 2)));
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_X);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_Y);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_Z);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_ETA);
    falloc.allocate_field(sizeof(int), HERMITE_GAUSSIAN_L);
    falloc.allocate_field(sizeof(legion_rect_1d_t), HERMITE_GAUSSIAN_DATA_RECT);
    falloc.allocate_field(sizeof(float), HERMITE_GAUSSIAN_BOUND);
  }
  LogicalRegion r_gausses = runtime->create_logical_region(ctx, ispace, fspace);

  runtime->fill_field<double>(ctx, r_gausses, r_gausses, HERMITE_GAUSSIAN_X, 0.f);

  legion_rect_1d_t val = {0, 0};
  runtime->fill_field<legion_rect_1d_t>(ctx, r_gausses, r_gausses, HERMITE_GAUSSIAN_DATA_RECT, val);

  coulomb_launcher launcher;
  launcher.add_argument_r_gausses(r_gausses, r_gausses, hermite_gaussian_fields);
  // launcher.add_argument_r_density();
  // launcher.add_argument_r_j_values();
  // launcher.add_argument_r_boys();
  // launcher.add_argument_highest_L(highest_L);
  // launcher.execute(runtime, ctx);
}

int main(int argc, char **argv) {
  std::cout << "Hello Legion\n";
  Runtime::set_top_level_task_id(TID_TOP_LEVEL_TASK);

  TaskVariantRegistrar registrar(TID_TOP_LEVEL_TASK, "top_level");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");

  coulomb_tasks_h_register();
  return Runtime::start(argc, argv);
}
