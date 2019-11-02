/**
 * c++ -O2 -Wall -Werror -I[/path/tolegion/runtime] main.cpp eri_regent.cpp \
 *   -L[/path/to/legion/bindings/regent] -lregent -L. -ljfock_tasks \
 *   -Wl,-rpath,[/path/to/legion/bindings/regent],-rpath,[/path/to/libjfock]
 */

#include <iostream>
#include <unistd.h>

#include "eri_regent.h"
#include "helper.h"
#include "jfock_tasks.h"
#include "legion.h"
#include "math.h"

using namespace std;
using namespace Legion;

float read_parameters(const string filename) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    printf("Unable to open %s!\n", filename.c_str());
    return -1;
  }
  double scalfr, scallr, omega;
  float thresp, thredp;
  int num_values =
      fscanf(filep, "scalfr=%lf\nscallr=%lf\nomega=%lf\nthresp=%f\nthredp=%f\n",
             &scalfr, &scallr, &omega, &thresp, &thredp);
  assert(num_values == 5);
  fclose(filep);
  return thredp;
}

void read_data_files(string bra_filename, string ket_filename,
                     EriRegent::TeraChemJDataList *data_list) {
  FILE *bra_filep = fopen(bra_filename.c_str(), "r");
  if (bra_filep == NULL) {
    printf("Unable to open %s!\n", bra_filename.c_str());
    return;
  }
  FILE *ket_filep = fopen(ket_filename.c_str(), "r");
  if (ket_filep == NULL) {
    printf("Unable to open %s!\n", ket_filename.c_str());
    fclose(bra_filep);
    return;
  }

  int L1, L2, n;

  while (fscanf(bra_filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    data_list->allocate_jbras(L1, L2, n);
    for (int i = 0; i < n; i++) {
      EriRegent::TeraChemJData *jbra = data_list->get_jbra_ptr(L1, L2, i);
      int num_values = fscanf(
          bra_filep, "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f\n", &jbra->x,
          &jbra->y, &jbra->z, &jbra->eta, &jbra->C, &jbra->bound);
      assert(num_values == 6 && "Did not read all values in line!");
    }
  }

  while (fscanf(ket_filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    data_list->allocate_jkets(L1, L2, n);
    const int H = COMPUTE_H(L1 + L2);
    for (int i = 0; i < n; i++) {
      EriRegent::TeraChemJData *jket = data_list->get_jket_ptr(L1, L2, i);
      int num_values =
          fscanf(ket_filep,
                 "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f,density=", &jket->x,
                 &jket->y, &jket->z, &jket->eta, &jket->C, &jket->bound);
      assert(num_values == 6 && "Did not read all values in line!");
      double *density = data_list->get_density_ptr(L1, L2, i);
      for (int j = 0; j < H; j++) {
        num_values = fscanf(ket_filep, "%lf;", density + j);
        assert(num_values == 1 && "Did not read all density values!");
      }
      num_values = fscanf(ket_filep, "\n");
      assert(num_values == 0 && "Did not read newline!");
    }
  }

  fclose(bra_filep);
  fclose(ket_filep);
}

void verify_output(string filename, EriRegent::TeraChemJDataList &jdata_list,
                   float delta, float epsilon) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", filename.c_str());
    return;
  }
  double expected, max_absolue_error = -1, max_relative_error = -1;

  int L1, L2, n;
  while (fscanf(filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    const int H = COMPUTE_H(L1 + L2);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < H; j++) {
        int num_values = fscanf(filep, "%lf\t", &expected);
        assert(num_values == 1 && "Output value not read!");
        double result = jdata_list.get_output_ptr(L1, L2, i)[j];
        double absolute_error = fabs(expected - result);
        double relative_error = fabs(absolute_error / expected);
        max_absolue_error = fmax(absolute_error, max_absolue_error);
        max_relative_error = fmax(relative_error, max_relative_error);
        if (std::isnan(result) || std::isinf(result) ||
            (absolute_error > delta && relative_error > epsilon)) {
          printf("Value differs at L1 = %d, L2 = %d, JBra[%d].output[%d]:\t"
                 "result = %.12f,\texpected = %.12f,\tabsolute_error = %.12g,\t"
                 "relative_error = %.12g\n",
                 L1, L2, i, j, result, expected, absolute_error,
                 relative_error);
          assert(false);
        }
      }
    }
  }
  printf("Values are correct!\nmax_absolue_error = %.12g\nmax_relative_error = "
         "%.12g\n",
         max_absolue_error, max_relative_error);
}

void print_usage_and_abort(int argc, char **argv) {
  fprintf(stderr, "Usage: %s -i [input directory] -p [parallelism]\n", argv[0]);
  assert(false);
}

void top_level_task(const Task *task, const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  int argc = Runtime::get_input_args().argc;
  char **argv = (char **)Runtime::get_input_args().argv;

  int parallelism = 1;
  string input_directory;

  opterr = 0;
  int c;
  while ((c = getopt(argc, argv, "i:p:")) != -1) {
    switch (c) {
    case 'i':
      input_directory = string(optarg);
      break;
    case 'p':
      parallelism = atoi(optarg);
      break;
    default:
      print_usage_and_abort(argc, argv);
    }
  }

  if (parallelism < 1 || input_directory.empty()) {
    print_usage_and_abort(argc, argv);
  }

  string bras_filename = input_directory + "/bras.dat";
  string kets_filename = input_directory + "/kets.dat";
  string parameters_filename = input_directory + "/parameters.dat";
  string output_filename = input_directory + "/output.dat";

  // `EriRegent` should be initialized once at the start of the program.
  EriRegent eri_regent(ctx, runtime);

  // Create a `TeraChemJDataList` and copy data to it.
  EriRegent::TeraChemJDataList jdata_list;
  read_data_files(bras_filename, kets_filename, &jdata_list);
  float threshold = read_parameters(parameters_filename);

  // Launch the Regent tasks and wait for them to finish.
  eri_regent.launch_jfock_task(jdata_list, threshold, parallelism);

  verify_output(output_filename, jdata_list, 1e-7, 1e-8);

  // Free the data.
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
