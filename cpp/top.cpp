// c++ -O2 -Wall -Werror -I[PATH TO legion/runtime] top.cpp -L[PATH TO bindings/regent] -lregent -L. -lcoulomb_tasks
// LD_LIBRARY_PATH=[PATH TO legion/bindings/regent]:. ./a.out

#include "coulomb_tasks.h"
#include "../precomputedBoys.h"

#include "legion.h"
#include "legion/legion_c_util.h"

#include <iostream>

using namespace Legion;

enum {  // Task IDs
  TOP_LEVEL_TASK_ID,
};

enum {  // Field IDs
  HERMITE_GAUSSIAN_X_ID,
  HERMITE_GAUSSIAN_Y_ID,
  HERMITE_GAUSSIAN_Z_ID,
  HERMITE_GAUSSIAN_ETA_ID,
  HERMITE_GAUSSIAN_L_ID,
  HERMITE_GAUSSIAN_DATA_RECT_ID,
  HERMITE_GAUSSIAN_BOUND_ID,
};

enum {
  DOUBLE_VALUE_ID,
};

const std::vector<FieldID> hermite_gaussian_field = {HERMITE_GAUSSIAN_X_ID,
                                                     HERMITE_GAUSSIAN_Y_ID,
                                                     HERMITE_GAUSSIAN_Z_ID,
                                                     HERMITE_GAUSSIAN_ETA_ID,
                                                     HERMITE_GAUSSIAN_L_ID,
                                                     HERMITE_GAUSSIAN_DATA_RECT_ID,
                                                     HERMITE_GAUSSIAN_BOUND_ID};

const std::vector<FieldID> double_field = {DOUBLE_VALUE_ID};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  std::cout << "Hi again from Regent\n";
  int num_gausses = 10;
  int highest_L = 0;

  Rect<1> gausses_rect(0, num_gausses - 1);
  Rect<1> values_rect(0, num_gausses - 1);
  Rect<2> boys_rect({0, 0}, {120, 10});
  IndexSpace i_gausses = runtime->create_index_space(ctx, gausses_rect);
  IndexSpace i_values = runtime->create_index_space(ctx, values_rect);
  IndexSpace i_boys = runtime->create_index_space(ctx, boys_rect);

  std::cout << "Create field spaces\n";
  FieldSpace f_gausses = runtime->create_field_space(ctx);
  FieldSpace f_double = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, f_gausses);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_X_ID);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_Y_ID);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_Z_ID);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_ETA_ID);
    falloc.allocate_field(sizeof(uint64_t), HERMITE_GAUSSIAN_L_ID);
    falloc.allocate_field(sizeof(legion_rect_1d_t), HERMITE_GAUSSIAN_DATA_RECT_ID);
    falloc.allocate_field(sizeof(double), HERMITE_GAUSSIAN_BOUND_ID);
  }
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, f_double);
    falloc.allocate_field(sizeof(double), DOUBLE_VALUE_ID);
  }

  std::cout << "Create logical regions\n";
  LogicalRegion lr_gausses = runtime->create_logical_region(ctx, i_gausses, f_gausses);
  LogicalRegion lr_density = runtime->create_logical_region(ctx, i_values, f_double);
  LogicalRegion lr_j_values = runtime->create_logical_region(ctx, i_values, f_double);
  LogicalRegion lr_boys = runtime->create_logical_region(ctx, i_boys, f_double);

  std::cout << "Create physical regions\n";
  RegionRequirement req_gausses(lr_gausses, READ_WRITE, EXCLUSIVE, lr_gausses);
  req_gausses.add_field(HERMITE_GAUSSIAN_X_ID);
  req_gausses.add_field(HERMITE_GAUSSIAN_Y_ID);
  req_gausses.add_field(HERMITE_GAUSSIAN_Z_ID);
  req_gausses.add_field(HERMITE_GAUSSIAN_ETA_ID);
  req_gausses.add_field(HERMITE_GAUSSIAN_L_ID);
  req_gausses.add_field(HERMITE_GAUSSIAN_DATA_RECT_ID);
  req_gausses.add_field(HERMITE_GAUSSIAN_BOUND_ID);
  InlineLauncher gaussian_launcher(req_gausses);
  PhysicalRegion pr_gausses = runtime->map_region(ctx, gaussian_launcher);

  RegionRequirement req_density(lr_density, READ_WRITE, EXCLUSIVE, lr_density);
  req_density.add_field(DOUBLE_VALUE_ID);
  InlineLauncher density_launcher(req_density);
  PhysicalRegion pr_density = runtime->map_region(ctx, density_launcher);

  RegionRequirement req_boys(lr_boys, READ_WRITE, EXCLUSIVE, lr_boys);
  req_boys.add_field(DOUBLE_VALUE_ID);
  InlineLauncher boys_launcher(req_boys);
  PhysicalRegion pr_boys = runtime->map_region(ctx, boys_launcher);

  pr_gausses.wait_until_valid();
  pr_density.wait_until_valid();
  pr_boys.wait_until_valid();


  std::cout << "Write to regions\n";
  // TODO: FieldAccessor is easy to use but may be slow.
  const FieldAccessor<READ_WRITE, double, 1> gausses_x(pr_gausses, HERMITE_GAUSSIAN_X_ID);
  const FieldAccessor<READ_WRITE, double, 1> gausses_y(pr_gausses, HERMITE_GAUSSIAN_Y_ID);
  const FieldAccessor<READ_WRITE, double, 1> gausses_z(pr_gausses, HERMITE_GAUSSIAN_Z_ID);
  const FieldAccessor<READ_WRITE, double, 1> gausses_eta(pr_gausses, HERMITE_GAUSSIAN_ETA_ID);
  const FieldAccessor<READ_WRITE, uint64_t, 1> gausses_L(pr_gausses, HERMITE_GAUSSIAN_L_ID);
  const FieldAccessor<READ_WRITE, legion_rect_1d_t, 1> gausses_data_rect(pr_gausses, HERMITE_GAUSSIAN_DATA_RECT_ID);
  const FieldAccessor<READ_WRITE, double, 1> gausses_bound(pr_gausses, HERMITE_GAUSSIAN_BOUND_ID);
  const FieldAccessor<READ_WRITE, double, 1> density_value(pr_density, DOUBLE_VALUE_ID);
  const FieldAccessor<READ_WRITE, double, 2> boys_value(pr_boys, DOUBLE_VALUE_ID);

  int L = 0;
  int idx = 0;
  for (PointInRectIterator<1> pir(gausses_rect); pir(); pir++) {
    gausses_x[*pir] = 1.f;
    gausses_y[*pir] = 2.f;
    gausses_z[*pir] = 3.f;
    gausses_eta[*pir] = 4.f;
    gausses_L[*pir] = L;
    int H = (L + 1) * (L + 2) * (L + 3) / 6;
    gausses_data_rect[*pir] = {idx, idx + H - 1};
    gausses_bound[*pir] = 0.f;
    density_value[*pir] = 5.f;
    idx += H;
  }

  idx = 0;
  for (PointInRectIterator<2> pir(boys_rect); pir(); pir++) {
    boys_value[*pir] = _precomputed_boys[idx++];
  }

  runtime->unmap_region(ctx, pr_gausses);
  runtime->unmap_region(ctx, pr_density);
  runtime->unmap_region(ctx, pr_boys);

  std::cout << "Launch task\n";
  coulomb_launcher launcher;
  launcher.add_argument_r_gausses(lr_gausses, lr_gausses, hermite_gaussian_field);
  launcher.add_argument_r_density(lr_density, lr_density, double_field);
  launcher.add_argument_r_j_values(lr_j_values, lr_j_values, double_field);
  launcher.add_argument_r_boys(lr_boys, lr_boys, double_field);
  launcher.add_argument_highest_L(highest_L);
  launcher.execute(runtime, ctx);


  RegionRequirement req_j_values(lr_j_values, READ_ONLY, EXCLUSIVE, lr_j_values);
  req_j_values.add_field(DOUBLE_VALUE_ID);
  InlineLauncher j_values_launcher(req_j_values);
  PhysicalRegion pr_j_values = runtime->map_region(ctx, j_values_launcher);
  pr_j_values.wait_until_valid();
  const FieldAccessor<READ_ONLY, double, 1> j_values(pr_j_values, DOUBLE_VALUE_ID);

  std::vector<double> host_j_values;
  for (PointInRectIterator<1> pir(gausses_rect); pir(); pir++) {
    host_j_values.push_back(j_values[*pir]);
  }
  runtime->unmap_region(ctx, pr_j_values);

  for (auto j : host_j_values) {
    std::cout << j << ", ";
  }
  std::cout << std::endl;
  std::cout << "Finished\n";

  runtime->destroy_logical_region(ctx, lr_gausses);
  runtime->destroy_logical_region(ctx, lr_density);
  runtime->destroy_logical_region(ctx, lr_j_values);
  runtime->destroy_logical_region(ctx, lr_boys);
  runtime->destroy_field_space(ctx, f_gausses);
  runtime->destroy_field_space(ctx, f_double);
  runtime->destroy_index_space(ctx, i_gausses);
  runtime->destroy_index_space(ctx, i_values);
  runtime->destroy_index_space(ctx, i_boys);
}

int main(int argc, char **argv) {
  std::cout << "Hello Legion\n";
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  coulomb_tasks_h_register();
  return Runtime::start(argc, argv);
}
