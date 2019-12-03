/**
 * c++ -O2 -Wall -Werror -I[/path/tolegion/runtime] main.cpp eri_regent.cpp \
 *   -L[/path/to/legion/bindings/regent] -lregent -L. -ljfock_tasks \
 *   -Wl,-rpath,[/path/to/legion/bindings/regent],-rpath,[/path/to/libjfock]
 */

#include <iostream>
#include <unistd.h>

#include "../mcmurchie/gamma_table.h"
#include "eri_regent.h"
#include "helper.h"
#include "legion.h"
#include "math.h"

using namespace std;
using namespace Legion;

float read_parameters(const string filename) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", filename.c_str());
    assert(false);
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
    fprintf(stderr, "Unable to open %s!\n", bra_filename.c_str());
    assert(false);
  }
  FILE *ket_filep = fopen(ket_filename.c_str(), "r");
  if (ket_filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", ket_filename.c_str());
    fclose(bra_filep);
    assert(false);
  }

  int L1, L2, n;

  while (fscanf(bra_filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    data_list->allocate_jbras(L1, L2, n);
    for (int i = 0; i < n; i++) {
      EriRegent::TeraChemJData jbra;
      int num_values =
          fscanf(bra_filep, "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f\n",
                 &jbra.x, &jbra.y, &jbra.z, &jbra.eta, &jbra.C, &jbra.bound);
      assert(num_values == 6 && "Did not read all values in line!");
      data_list->set_jbra(L1, L2, i, jbra);
    }
  }

  while (fscanf(ket_filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    data_list->allocate_jkets(L1, L2, n);
    const int H = COMPUTE_H(L1 + L2);
    for (int i = 0; i < n; i++) {
      EriRegent::TeraChemJData jket;
      int num_values =
          fscanf(ket_filep,
                 "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f,density=", &jket.x,
                 &jket.y, &jket.z, &jket.eta, &jket.C, &jket.bound);
      assert(num_values == 6 && "Did not read all values in line!");
      data_list->set_jket(L1, L2, i, jket);
      double density[COMPUTE_H(L1 + L2)];
      for (int j = 0; j < H; j++) {
        num_values = fscanf(ket_filep, "%lf;", &density[j]);
        assert(num_values == 1 && "Did not read all density values!");
      }
      data_list->set_density(L1, L2, i, density);
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
    assert(false);
  }
  double expected, max_absolue_error = -1, max_relative_error = -1;

  int L1, L2, n;
  while (fscanf(filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    const int H = COMPUTE_H(L1 + L2);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < H; j++) {
        int num_values = fscanf(filep, "%lf\t", &expected);
        assert(num_values == 1 && "Output value not read!");
        double result = jdata_list.get_output(L1, L2, i)[j];
        double absolute_error = fabs(expected - result);
        double relative_error = fabs(absolute_error / expected);
        max_absolue_error = fmax(absolute_error, max_absolue_error);
        max_relative_error = fmax(relative_error, max_relative_error);
        if (std::isnan(result) || std::isinf(result) ||
            (absolute_error > delta && relative_error > epsilon)) {
          printf("Value differs at L1 = %d, L2 = %d, JBra[%d].output[%d]:\n"
                 "result = %.12f\nexpected = %.12f\nabsolute_error = %.12g\n"
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

int main(int argc, char **argv) {
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

  cout << "Verifying with input data from " << input_directory << endl;

  // `register_tasks` should be called once before starting the Legion runtime
  EriRegent::register_tasks();

  // Pass Realm arguments and start the Legion runtime
  Runtime::start(argc, argv, /*background=*/true);

  // `EriRegent` should be initialized once at the start of the program.
  EriRegent eri_regent((const double *)gamma_table);

  // Create a `TeraChemJDataList` and copy data to it.
  EriRegent::TeraChemJDataList jdata_list;
  read_data_files(bras_filename, kets_filename, &jdata_list);
  float threshold = read_parameters(parameters_filename);

  // Launch the Regent tasks and wait for them to finish.
  eri_regent.launch_jfock_task(jdata_list, threshold, parallelism);

  verify_output(output_filename, jdata_list, 1e-11, 1e-12);

  // Free the data.
  jdata_list.free_data();
}
