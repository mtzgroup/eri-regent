/**
 * c++ -O2 -Wall -Werror -I[PATH TO legion/runtime] main.cpp eri_regent.cpp \
 *   -L[PATH TO legion/bindings/regent] -lregent -L. -ljfock_tasks
 * LD_LIBRARY_PATH=[PATH TO legion/bindings/regent]:. ./a.out
 */

#include <iostream>

#include "eri_regent.h"
#include "helper.h"
#include "jfock_tasks.h"
#include "legion.h"
#include "math.h"

using namespace std;
using namespace Legion;

// TODO: Maybe add parse code into class with TeraChemJData
void read_data(FILE *filep, int L1, int L2, int n,
               EriRegent::TeraChemJData *data, double *density = NULL) {
  assert(data);
  const int H = COMPUTE_H(L1 + L2);
  int num_values;
  for (int i = 0; i < n; i++) {
    num_values = fscanf(filep, "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f",
                        &data[i].x, &data[i].y, &data[i].z, &data[i].eta,
                        &data[i].C, &data[i].bound);
    assert(num_values == 6 && "Did not read all values in line!");
    if (density) {
      num_values = fscanf(filep, ",density=");
      assert(num_values == 0 && "Could not find density values!");
      for (int j = 0; j < H; j++) {
        num_values = fscanf(filep, "%lf;", density + i * H + j);
        assert(num_values == 1 && "Did not read all density values!");
      }
    }
    num_values = fscanf(filep, "\n");
    assert(num_values == 0 && "Did not read newline!");
  }
}

int read_data_files(const char *bra_filename, const char *ket_filename,
                    int max_momentum, EriRegent::TeraChemJDataList *data_list) {
  FILE *bra_filep = fopen(bra_filename, "r");
  if (bra_filep == NULL) {
    printf("Unable to open %s!\n", bra_filename);
    return -1;
  }
  FILE *ket_filep = fopen(ket_filename, "r");
  if (ket_filep == NULL) {
    printf("Unable to open %s!\n", ket_filename);
    fclose(bra_filep);
    return -1;
  }

  int num_values, input_L1, input_L2, n;
  for (int L1 = 0; L1 <= max_momentum; L1++) {
    for (int L2 = L1; L2 <= max_momentum; L2++) {
      num_values =
          fscanf(bra_filep, "L1=%d,L2=%d,N=%d\n", &input_L1, &input_L2, &n);
      assert(num_values == 3 && L1 == input_L1 && L2 == input_L2 &&
             "Unexpected angles!");
      data_list->allocate_jbras(L1, L2, n);
      read_data(bra_filep, L1, L2, n, data_list->get_jbras_ptr(L1, L2));

      num_values =
          fscanf(ket_filep, "L1=%d,L2=%d,N=%d\n", &input_L1, &input_L2, &n);
      assert(num_values == 3 && L1 == input_L1 && L2 == input_L2 &&
             "Unexpected angles!");
      data_list->allocate_jkets(L1, L2, n);
      read_data(ket_filep, L1, L2, n, data_list->get_jkets_ptr(L1, L2),
                data_list->get_density_ptr(L1, L2));
    }
  }

  fclose(bra_filep);
  fclose(ket_filep);
  return 0;
}

void verify_output(const char *filename, int max_momentum,
                   EriRegent::TeraChemJDataList &jdata_list, float delta,
                   float epsilon) {
  FILE *filep = fopen(filename, "r");
  int num_values, input_L1, input_L2, n;
  double expected, max_absolue_error = -1, max_relative_error = -1;
  for (int L1 = 0; L1 <= max_momentum; L1++) {
    for (int L2 = L1; L2 <= max_momentum; L2++) {
      const int H = COMPUTE_H(L1 + L2);
      num_values =
          fscanf(filep, "L1=%d,L2=%d,N=%d\n", &input_L1, &input_L2, &n);
      assert(num_values == 3 && L1 == input_L1 && L2 == input_L2 &&
             "Unexpected angles!");
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < H; j++) {
          num_values = fscanf(filep, "%lf\t", &expected);
          double result = jdata_list.get_output(L1, L2, i)[j];
          double absolute_error = fabs(expected - result);
          double relative_error = fabs(absolute_error / expected);
          max_absolue_error = fmax(absolute_error, max_absolue_error);
          max_relative_error = fmax(relative_error, max_relative_error);
          if (isnan(result) || isinf(result) ||
              (absolute_error > delta && relative_error > epsilon)) {
            printf(
                "Value differs at L1 = %d, L2 = %d, JBra[%d].output[%d]:\t"
                "result = %.12f,\texpected = %.12f,\tabsolute_error = %.12g,\t"
                "relative_error = %.12g\n",
                L1, L2, i, j, result, expected, absolute_error, relative_error);
            assert(false);
          }
        }
      }
    }
  }
  printf("Values are correct!\nmax_absolue_error = %.12g\nmax_relative_error = "
         "%.12g\n",
         max_absolue_error, max_relative_error);
}

void top_level_task(const Task *task, const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  // TODO: Take data files as user input
  int max_momentum = 1;
  EriRegent eri_regent(max_momentum, ctx, runtime);

  EriRegent::TeraChemJDataList jdata_list;
  read_data_files("../data/h2o/bras.dat", "../data/h2o/kets.dat", max_momentum,
                  &jdata_list);

  // TODO: Read parameters from file
  float threshold = 0X1.5FD7FEP-37;
  int parallelism = 1;
  eri_regent.launch_jfock_task(jdata_list, threshold, parallelism);

  verify_output("../data/h2o/output.dat", max_momentum, jdata_list, 1e-7, 1e-8);

  jdata_list.free_data();
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
