/**
 * c++ -O2 -Wall -Werror -I[/path/tolegion/runtime] main.cpp eri_regent.cpp \
 *   -L[/path/to/legion/bindings/regent] -lregent -L. -ljfock_tasks \
 *   -Wl,-rpath,[/path/to/legion/bindings/regent],-rpath,[/path/to/libjfock]
 */

#include <chrono>
#include <iostream>
#include <unistd.h>

#include "../mcmurchie/gamma_table.h"
#include "eri_regent.h"
#include "helper.h"
#include "legion.h"
#include "math.h"

using namespace std;
using namespace Legion;
using namespace std::chrono;

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

void read_jdata_files(string bra_filename, string ket_filename,
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
    const int H = TETRAHEDRAL_NUMBER(L1 + L2 + 1);
    for (int i = 0; i < n; i++) {
      EriRegent::TeraChemJData jket;
      int num_values =
          fscanf(ket_filep,
                 "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f,density=", &jket.x,
                 &jket.y, &jket.z, &jket.eta, &jket.C, &jket.bound);
      assert(num_values == 6 && "Did not read all values in line!");
      data_list->set_jket(L1, L2, i, jket);
      double density[TETRAHEDRAL_NUMBER(L1 + L2 + 1)];
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

void verify_jfock_output(string filename,
                         EriRegent::TeraChemJDataList &jdata_list, float delta,
                         float epsilon) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", filename.c_str());
    assert(false);
  }
  double expected, max_absolue_error = -1, max_relative_error = -1;

  int L1, L2, n;
  while (fscanf(filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    const int H = TETRAHEDRAL_NUMBER(L1 + L2 + 1);
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
  fclose(filep);
}

void read_kpairs(string filename, EriRegent::TeraChemKDataList *kdata_list) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", filename.c_str());
    assert(false);
  }

  int L1, L2, n;
  while (fscanf(filep, "L1=%d,L2=%d,N=%d\n", &L1, &L2, &n) == 3) {
    kdata_list->allocate_kpairs(L1, L2, n);
    for (int i = 0; i < n; i++) {
      double x, y, z, eta, C;
      float bound;
      int ishell_idx, jshell_idx;
      double PIx, PIy, PIz, PJx, PJy, PJz;
      int num_values =
          fscanf(filep,
                 "x=%lf,y=%lf,z=%lf,eta=%lf,c=%lf,bound=%f,"
                 "i_shell_idx=%d,j_shell_idx=%d,"
                 "PIx=%lf,PIy=%lf,PIz=%lf,PJx=%lf,PJy=%lf,PJz=%lf\n",
                 &x, &y, &z, &eta, &C, &bound, &ishell_idx, &jshell_idx, &PIx,
                 &PIy, &PIz, &PJx, &PJy, &PJz);
      assert(num_values == 14 && "Did not read all kpair values!");
      kdata_list->set_kpair(L1, L2, i, x, y, z, eta, C, bound, PIx, PIy, PIz,
                            PJx, PJy, PJz, ishell_idx, jshell_idx);
    }
  }
  fclose(filep);
}

void read_kdensity(string filename, EriRegent::TeraChemKDataList *kdata_list) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", filename.c_str());
    assert(false);
  }

  int L2, L4, n2, n4;
  while (fscanf(filep, "L2=%d,L4=%d,N2=%d,N4=%d\n", &L2, &L4, &n2, &n4) == 4) {
    kdata_list->allocate_kdensity(L2, L4, n2, n4);
    for (int bra_jshell_idx = 0; bra_jshell_idx < n2; bra_jshell_idx++) {
      for (int ket_jshell_idx = 0; ket_jshell_idx < n4; ket_jshell_idx++) {
        int i, j;
        int num_values = fscanf(
            filep, "bra_jshell_idx=%d,ket_jshell_idx=%d,values=", &i, &j);
        assert(num_values == 2 && "Did not read all kdensity values!");
        assert(bra_jshell_idx == i && ket_jshell_idx == j);
        double values[TRIANGLE_NUMBER(L2 + 1) * TRIANGLE_NUMBER(L4 + 1)];
        for (int i = 0; i < TRIANGLE_NUMBER(L2 + 1); i++) {
          for (int j = 0; j < TRIANGLE_NUMBER(L4 + 1); j++) {
            const int index = i * TRIANGLE_NUMBER(L4 + 1) + j;
            num_values = fscanf(filep, "%lf,", &values[index]);
            assert(num_values == 1 && "Did not read all kdensity values!");
          }
        }
        float bound;
        num_values = fscanf(filep, "bound=%f\n", &bound);
        assert(num_values == 1 && "Did not read kdensity bound!");
        kdata_list->set_kdensity(L2, L4, bra_jshell_idx, ket_jshell_idx, values,
                                 bound);
      }
    }
  }
  fclose(filep);
}

void verify_kfock_output(string filename,
                         EriRegent::TeraChemKDataList &kdata_list, float delta,
                         float epsilon) {
  FILE *filep = fopen(filename.c_str(), "r");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", filename.c_str());
    assert(false);
  }
  double max_absolue_error = -1, max_relative_error = -1;

  int L1, L2, L3, L4, n1, n3;
  while (fscanf(filep, "L1=%d,L2=%d,L3=%d,L4=%d,N1=%d,N3=%d\n", &L1, &L2, &L3,
                &L4, &n1, &n3) == 6) {
    for (int bra_ishell_idx = 0; bra_ishell_idx < n1; bra_ishell_idx++) {
      for (int ket_ishell_idx = 0; ket_ishell_idx < n3; ket_ishell_idx++) {
        int i, j;
        int num_values = fscanf(
            filep, "bra_ishell_idx=%d,ket_ishell_idx=%d,values=", &i, &j);
        assert(num_values == 2 && "Did not read koutput values!");
        assert(bra_ishell_idx == i && ket_ishell_idx == j);
        for (int i = 0; i < TRIANGLE_NUMBER(L1 + 1); i++) {
          for (int j = 0; j < TRIANGLE_NUMBER(L3 + 1); j++) {
            double expected;
            num_values = fscanf(filep, "%lf,", &expected);
            assert(num_values == 1 && "Did not read koutput value!");
            const int index = i * TRIANGLE_NUMBER(L3 + 1) + j;
            double result = kdata_list.get_koutput(
                L1, L2, L3, L4, bra_ishell_idx, ket_ishell_idx)[index];
            double absolute_error = fabs(expected - result);
            double relative_error = fabs(absolute_error / expected);
            max_absolue_error = fmax(absolute_error, max_absolue_error);
            max_relative_error = fmax(relative_error, max_relative_error);
            if (std::isnan(result) || std::isinf(result) ||
                (absolute_error > delta && relative_error > epsilon)) {
              printf("Value differs at L1234 = %d%d%d%d, "
                     "koutput[%d, %d].values[%d, %d]:\n"
                     "result = %.12f\nexpected = %.12f\n"
                     "absolute_error = %.12g\nrelative_error = %.12g\n",
                     L1, L2, L3, L4, bra_ishell_idx, ket_ishell_idx, i, j,
                     result, expected, absolute_error, relative_error);
              assert(false);
            }
            num_values = fscanf(filep, "\n");
            assert(num_values == 0 && "Did not read line!\n");
          }
        }
      }
    }
  }
  printf("Values are correct!\nmax_absolue_error = %.12g\nmax_relative_error = "
         "%.12g\n",
         max_absolue_error, max_relative_error);
  fclose(filep);
}

void run_jfock_tasks(EriRegent *eri_regent, const string &input_directory,
                     int parallelism, int num_iterations) {
  const bool do_timing = (num_iterations > 0);
  const string bras_filename = input_directory + "/bras.dat";
  const string kets_filename = input_directory + "/kets.dat";
  const string parameters_filename = input_directory + "/parameters.dat";
  const string output_filename = input_directory + "/output.dat";

  // Create a `TeraChemJDataList` and copy data to it.
  EriRegent::TeraChemJDataList jdata_list;
  read_jdata_files(bras_filename, kets_filename, &jdata_list);
  float threshold = read_parameters(parameters_filename);

  // Launch the Regent tasks and wait for them to finish.
  if (do_timing) {
    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
      eri_regent->launch_jfock_task(jdata_list, threshold, parallelism);
    }
    high_resolution_clock::time_point stop_time = high_resolution_clock::now();
    duration<double> elapsed_seconds = stop_time - start_time;

    printf("Ran %d iterations on %s in %lf seconds.\n", num_iterations,
           input_directory.c_str(), elapsed_seconds.count());
    printf("Average runtime: %lf seconds.\n",
           elapsed_seconds.count() / num_iterations);
  } else {
    eri_regent->launch_jfock_task(jdata_list, threshold, parallelism);
    verify_jfock_output(output_filename, jdata_list, 1e-11, 1e-12);
  }

  // Free the data.
  jdata_list.free_data();
}

void run_kfock_tasks(EriRegent *eri_regent, const string &input_directory,
                     int parallelism, int num_iterations) {
  const bool do_timing = (num_iterations > 0);
  const string kpairs_filename = input_directory + "/kfock.dat";
  const string kdensity_filename = input_directory + "/kfock_density.dat";
  const string parameters_filename = input_directory + "/parameters.dat";
  const string output_filename = input_directory + "/kfock_output.dat";

  // Create a `TeraChemKDataList` and copy data to it.
  EriRegent::TeraChemKDataList kdata_list;
  read_kpairs(kpairs_filename, &kdata_list);
  read_kdensity(kdensity_filename, &kdata_list);
  float threshold = read_parameters(parameters_filename);

  // Launch the Regent tasks and wait for them to finish.
  if (do_timing) {
    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
      eri_regent->launch_kfock_task(kdata_list, threshold, parallelism);
    }
    high_resolution_clock::time_point stop_time = high_resolution_clock::now();
    duration<double> elapsed_seconds = stop_time - start_time;

    printf("Ran %d iterations on %s in %lf seconds.\n", num_iterations,
           input_directory.c_str(), elapsed_seconds.count());
    printf("Average runtime: %lf seconds.\n",
           elapsed_seconds.count() / num_iterations);
  } else {
    eri_regent->launch_kfock_task(kdata_list, threshold, parallelism);
    verify_kfock_output(output_filename, kdata_list, 1e-9, 1e-7);
  }
}

void print_usage_and_abort(int argc, char **argv) {
  fprintf(stderr, "Usage: %s -i {dir} -p {value} [-t]\n", argv[0]);
  fprintf(
      stderr,
      "OPTIONS\n"
      "  -a [jfock|kfock] : Run the JFock or KFock algorithm.\n"
      "  -i {dir}         : Use {dir} as the input directory.\n"
      "  -p {value}       : Parallelize {value} ways.\n"
      "  -t {value}       : Run {value} times and report the average\n"
      "                     runtime without verifying the output.\n"
      "  -- {args}        : All following arguments are passed to Realm.\n");
  exit(1);
}

enum FOCK_TYPE { JFOCK, KFOCK };

int main(int argc, char **argv) {

  int realm_argc = argc;
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--", 2) == 0) {
      argc = i;
      break;
    }
  }

  FOCK_TYPE fock_type = JFOCK;
  int parallelism = 1;
  string input_directory;
  int num_iterations = -1;

  opterr = 0;
  int c;
  while ((c = getopt(argc, argv, "a:i:p:t:")) != -1) {
    switch (c) {
    case 'a':
      if (strncmp(optarg, "jfock", 5) == 0) {
        fock_type = JFOCK;
      } else if (strncmp(optarg, "kfock", 5) == 0) {
        fock_type = KFOCK;
      } else {
        print_usage_and_abort(argc, argv);
      }
      break;
    case 'i':
      input_directory = string(optarg);
      break;
    case 'p':
      parallelism = atoi(optarg);
      break;
    case 't':
      num_iterations = atoi(optarg);
      break;
    default:
      print_usage_and_abort(argc, argv);
      break;
    }
  }

  if (parallelism <= 0 || input_directory.empty()) {
    print_usage_and_abort(argc, argv);
  }

  // `register_tasks` should be called once before starting the Legion runtime
  EriRegent::register_tasks();

  // Pass Realm arguments and start the Legion runtime
  Runtime::start(realm_argc, argv, /*background=*/true);

  // `EriRegent` should be initialized once at the start of the program.
  EriRegent *eri_regent = new EriRegent((const double *)gamma_table);

  switch (fock_type) {
  case JFOCK:
    run_jfock_tasks(eri_regent, input_directory, parallelism, num_iterations);
    break;
  case KFOCK:
    run_kfock_tasks(eri_regent, input_directory, parallelism, num_iterations);
    break;
  }

  delete eri_regent;

  // Wait for the Legion runtime to finish after all tasks have been destroyed.
  int exit_code = Runtime::wait_for_shutdown();
  assert(exit_code == 0);
}
