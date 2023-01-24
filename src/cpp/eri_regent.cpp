#include "eri_regent.h"
#include "eri_regent_tasks.h"
#include "helper.h"
#include "legion.h"
#include "fundint.h"
#include <stdio.h>  // for printf (KGJ remove)

using namespace std;
using namespace Legion;


EriRegent::EriRegent(Runtime* runtime_, Context ctx_) {

__TRACE
  runtime = runtime_;
  ctx = ctx_;
__TRACE
#if 0
  ctx = runtime->begin_implicit_task(ERI_REGENT_TASK_ID,
                                     /*mapper_id=*/0, Processor::LOC_PROC,
                                     "eri_regent_toplevel_task",
                                     /*control_replicable=*/true);
#endif
  memory = Machine::MemoryQuery(Machine::get_machine())
               .has_affinity_to(runtime->get_executing_processor(ctx))
               .only_kind(Memory::SYSTEM_MEM)
               .first();

__TRACE
  // Create gamma table region
  const Rect<2> rect({0, 0}, {18 - 1, 700 - 1});
  gamma_table_ispace = runtime->create_index_space(ctx, rect);
  gamma_table_fspace = runtime->create_field_space(ctx);
__TRACE
  {
    FieldAllocator falloc =
        runtime->create_field_allocator(ctx, gamma_table_fspace);
    falloc.allocate_field(5 * sizeof(double), GAMMA_TABLE_FIELD_ID);
  }
  gamma_table_lr = runtime->create_logical_region(ctx, gamma_table_ispace,
                                                  gamma_table_fspace);
__TRACE
  AttachLauncher launcher(EXTERNAL_INSTANCE, gamma_table_lr, gamma_table_lr);
  launcher.attach_array_aos(GammaTable, /*column major*/ false,
                            {GAMMA_TABLE_FIELD_ID}, memory);
  gamma_table_pr = runtime->attach_external_resource(ctx, launcher);
__TRACE

  AcquireLauncher gammaAcquireLauncher(gamma_table_lr, gamma_table_lr, gamma_table_pr);
  gammaAcquireLauncher.add_field(GAMMA_TABLE_FIELD_ID);
  runtime->issue_acquire(ctx, gammaAcquireLauncher);

  initialize_jfock_field_spaces();
  initialize_kfock_field_spaces();
__TRACE
}

EriRegent::~EriRegent() {
  runtime->detach_external_resource(ctx, gamma_table_pr);
  runtime->destroy_logical_region(ctx, gamma_table_lr);
  runtime->destroy_field_space(ctx, gamma_table_fspace);
  runtime->destroy_index_space(ctx, gamma_table_ispace);
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      runtime->destroy_field_space(ctx, jbra_fspaces[index]);
      runtime->destroy_field_space(ctx, jket_fspaces[index]);
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      runtime->destroy_field_space(ctx, kpair_fspaces[index]);
      runtime->destroy_field_space(ctx, kbra_preval_fspaces[index]);
      runtime->destroy_field_space(ctx, kket_preval_fspaces[index]);
    }
  }

  for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
    for (int L4 = L2; L4 <= MAX_MOMENTUM; L4++) {
      const int index = INDEX_UPPER_TRIANGLE(L2, L4);
      runtime->destroy_field_space(ctx, kdensity_fspaces[index]);
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L3);
      runtime->destroy_field_space(ctx, koutput_fspaces[index]);
    }
  }
  //runtime->finish_implicit_task(ctx);
}


void EriRegent::register_tasks() { eri_regent_tasks_h_register(); }


void EriRegent::launch_jfock_task(EriRegent::TeraChemJDataList &jdata_list,
                                  float threshold, int parallelism,
				  int cparallelism) {
  // Create jbra regions
  LogicalRegion jbras_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion jbras_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace jbras_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      if (jdata_list.get_num_jbras(L1, L2) == 0) {
        jbras_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<1>::make_empty());
      } else {
        const Rect<1> rect(0, jdata_list.get_num_jbras(L1, L2) - 1);
        jbras_ispace_list[index] = runtime->create_index_space(ctx, rect);
      }
      jbras_lr_list[index] = runtime->create_logical_region(
          ctx, jbras_ispace_list[index], jbra_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jbras_lr_list[index],
                              jbras_lr_list[index]);
      const vector<FieldID> field_list(
          jbra_fields_list[index], jbra_fields_list[index] + NUM_JBRA_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(jdata_list.jbras[index], /*column major*/ false,
                                field_list, memory);
      jbras_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher jbraAcquireLauncher(jbras_lr_list[index], jbras_lr_list[index], jbras_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          jbraAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, jbraAcquireLauncher);
    }
  }

  // Create jket regions
  LogicalRegion jkets_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion jkets_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace jkets_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      if (jdata_list.get_num_jkets(L1, L2) == 0) {
        jkets_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<1>::make_empty());
      } else {
        const Rect<1> rect(0, jdata_list.get_num_jkets(L1, L2) - 1);
        jkets_ispace_list[index] = runtime->create_index_space(ctx, rect);
      }
      jkets_lr_list[index] = runtime->create_logical_region(
          ctx, jkets_ispace_list[index], jket_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, jkets_lr_list[index],
                              jkets_lr_list[index]);
      const vector<FieldID> field_list(
          jket_fields_list[index], jket_fields_list[index] + NUM_JKET_FIELDS);
      launcher.attach_array_aos(jdata_list.jkets[index], /*column major*/ false,
                                field_list, memory);
      jkets_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher jketAcquireLauncher(jkets_lr_list[index], jkets_lr_list[index], jkets_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          jketAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, jketAcquireLauncher);
    }
  }

  jfock_task_launcher launcher;

#define ADD_ARGUMENT_R_JBRAS(L1, L2)                                           \
  {                                                                            \
    const vector<FieldID> field_list(                                          \
        jbra_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)],                        \
        jbra_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)] + NUM_JBRA_FIELDS);     \
    launcher.add_argument_r_jbras##L1##L2(                                     \
        jbras_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)],                           \
        jbras_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)], field_list);              \
  }

  ADD_ARGUMENT_R_JBRAS(0, 0);
  ADD_ARGUMENT_R_JBRAS(0, 1);
  ADD_ARGUMENT_R_JBRAS(0, 2);
  ADD_ARGUMENT_R_JBRAS(0, 3);
  ADD_ARGUMENT_R_JBRAS(0, 4);
  ADD_ARGUMENT_R_JBRAS(1, 1);
  ADD_ARGUMENT_R_JBRAS(1, 2);
  ADD_ARGUMENT_R_JBRAS(1, 3);
  ADD_ARGUMENT_R_JBRAS(1, 4);
  ADD_ARGUMENT_R_JBRAS(2, 2);
  ADD_ARGUMENT_R_JBRAS(2, 3);
  ADD_ARGUMENT_R_JBRAS(2, 4);
  ADD_ARGUMENT_R_JBRAS(3, 3);
  ADD_ARGUMENT_R_JBRAS(3, 4);
  ADD_ARGUMENT_R_JBRAS(4, 4);

#undef ADD_ARGUMENT_R_JBRAS

#define ADD_ARGUMENT_R_JKETS(L1, L2)                                           \
  {                                                                            \
    const vector<FieldID> field_list(                                          \
        jket_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)],                        \
        jket_fields_list[INDEX_UPPER_TRIANGLE(L1, L2)] + NUM_JKET_FIELDS);     \
    launcher.add_argument_r_jkets##L1##L2(                                     \
        jkets_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)],                           \
        jkets_lr_list[INDEX_UPPER_TRIANGLE(L1, L2)], field_list);              \
  }

  ADD_ARGUMENT_R_JKETS(0, 0);
  ADD_ARGUMENT_R_JKETS(0, 1);
  ADD_ARGUMENT_R_JKETS(0, 2);
  ADD_ARGUMENT_R_JKETS(0, 3);
  ADD_ARGUMENT_R_JKETS(0, 4);
  ADD_ARGUMENT_R_JKETS(1, 1);
  ADD_ARGUMENT_R_JKETS(1, 2);
  ADD_ARGUMENT_R_JKETS(1, 3);
  ADD_ARGUMENT_R_JKETS(1, 4);
  ADD_ARGUMENT_R_JKETS(2, 2);
  ADD_ARGUMENT_R_JKETS(2, 3);
  ADD_ARGUMENT_R_JKETS(2, 4);
  ADD_ARGUMENT_R_JKETS(3, 3);
  ADD_ARGUMENT_R_JKETS(3, 4);
  ADD_ARGUMENT_R_JKETS(4, 4);

#undef ADD_ARGUMENT_R_JKETS

  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_parallelism(parallelism);
  launcher.add_argument_cparallelism(cparallelism);
  launcher.add_argument_largest_momentum(jdata_list.get_largest_momentum());
__TRACE
  Future future = launcher.execute(runtime, ctx);
__TRACE
  future.wait();
__TRACE

  ReleaseLauncher gammaReleaseLauncher(gamma_table_lr, gamma_table_lr, gamma_table_pr);
  gammaReleaseLauncher.add_field(GAMMA_TABLE_FIELD_ID);
  runtime->issue_release(ctx, gammaReleaseLauncher);


  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = L1; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L2);
      ReleaseLauncher jbraReleaseLauncher(jbras_lr_list[index], jbras_lr_list[index], jbras_pr_list[index]);
      const vector<FieldID> jbra_list(
          jbra_fields_list[index], jbra_fields_list[index] + NUM_JBRA_FIELDS);
      vector<FieldID> bfl = jbra_list;
      for(vector<FieldID>::iterator it = bfl.begin();
          it != bfl.end(); ++it) {
          jbraReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, jbraReleaseLauncher);
      ReleaseLauncher jketReleaseLauncher(jkets_lr_list[index], jkets_lr_list[index], jkets_pr_list[index]);
      const vector<FieldID> jket_list(
          jket_fields_list[index], jket_fields_list[index] + NUM_JKET_FIELDS);
      vector<FieldID> kfl = jket_list;
      for(vector<FieldID>::iterator it = kfl.begin();
          it != kfl.end(); ++it) {
          jketReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, jketReleaseLauncher);
      runtime->detach_external_resource(ctx, jbras_pr_list[index]);
      runtime->destroy_logical_region(ctx, jbras_lr_list[index]);
      runtime->destroy_index_space(ctx, jbras_ispace_list[index]);
      runtime->detach_external_resource(ctx, jkets_pr_list[index]);
      runtime->destroy_logical_region(ctx, jkets_lr_list[index]);
      runtime->destroy_index_space(ctx, jkets_ispace_list[index]);
    }
  }
}
#define PSPS 10
#define SSSS 0
void EriRegent::launch_kfock_task(EriRegent::TeraChemKDataList &kdata_list,
                                  float threshold, float kguard, int parallelism) {
  //  std::cout<<__FUNCTION__<<" parallelism "<<parallelism<<std::endl;

   // TODO: read this from a file
   // optimize PSPS and SSSS
  int parray[16];
  int init_val = parallelism*2;
  for (unsigned int i=0; i<16; ++i)
    {
      if ((i==PSPS) || (i==SSSS))
        parray[i] = init_val;
      else
        parray[i] = parallelism;
    }
  // Create kpair regions
  LogicalRegion kpair_lr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  PhysicalRegion kpair_pr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  IndexSpace kpair_ispace_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      //printf("KGJ: get_num_kpairs(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kpairs(L1, L2));
      if (kdata_list.get_num_kpairs(L1, L2) == 0) {
        kpair_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<1>::make_empty());
      } else {
        const Rect<1> rect(0, kdata_list.get_num_kpairs(L1, L2) - 1);
        kpair_ispace_list[index] = runtime->create_index_space(ctx, rect);
      }
      kpair_lr_list[index] = runtime->create_logical_region(
          ctx, kpair_ispace_list[index], kpair_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, kpair_lr_list[index],
                              kpair_lr_list[index]);
      const vector<FieldID> field_list(kpair_fields_list[index],
                                       kpair_fields_list[index] +
                                           NUM_KPAIR_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_kpair_data(L1, L2),
                                /*column major*/ false, field_list, memory);
      kpair_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher kpairAcquireLauncher(kpair_lr_list[index], kpair_lr_list[index], kpair_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          kpairAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, kpairAcquireLauncher);
    }
  }

  // Creat bra_preval regions
  LogicalRegion kbra_preval_lr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  PhysicalRegion kbra_preval_pr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  IndexSpace kbra_preval_ispace_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      //printf("KGJ: get_num_kbra_prevals(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kbra_prevals(L1, L2));
      if (kdata_list.get_num_kpairs(L1, L2) == 0 ||
          kdata_list.get_num_kbra_prevals(L1, L2) == 0) {
        // FIXME: Crashes when trying to create an empty region.
        kbra_preval_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<2>::make_empty());
      } else {
        const Rect<2> rect({0, 0},
                           {kdata_list.get_num_kpairs(L1, L2) - 1,
                            kdata_list.get_num_kbra_prevals(L1, L2) - 1});
        kbra_preval_ispace_list[index] = runtime->create_index_space(ctx, rect);
        /// KGJ: TESTING
        //printf("MEMORY TEST: get_num_kpairs(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kpairs(L1, L2));
        //printf("MEMORY TEST: get_num_kbra_prevals(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kbra_prevals(L1, L2));
        ///
      }
      kbra_preval_lr_list[index] = runtime->create_logical_region(
          ctx, kbra_preval_ispace_list[index], kbra_preval_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, kbra_preval_lr_list[index],
                              kbra_preval_lr_list[index]);
      const vector<FieldID> field_list(kbra_preval_fields_list[index],
                                       kbra_preval_fields_list[index] +
                                           NUM_KBRA_PREVAL_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_kbra_preval_data(L1, L2),
                                /*column major*/ false, field_list, memory);
      kbra_preval_pr_list[index] =
          runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher kbraPrevalAcquireLauncher(kbra_preval_lr_list[index], kbra_preval_lr_list[index], kbra_preval_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          kbraPrevalAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, kbraPrevalAcquireLauncher);
    }
  }

  // Creat ket_preval regions
  LogicalRegion kket_preval_lr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  PhysicalRegion kket_preval_pr_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  IndexSpace kket_preval_ispace_list[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      //printf("KGJ: get_num_kket_prevals(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kket_prevals(L1, L2));
      const int index = INDEX_SQUARE(L1, L2);
      if (kdata_list.get_num_kpairs(L1, L2) == 0 ||
          kdata_list.get_num_kket_prevals(L1, L2) == 0) {
        kket_preval_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<2>::make_empty());
      } else {
        const Rect<2> rect({0, 0},
                           {kdata_list.get_num_kpairs(L1, L2) - 1,
                            kdata_list.get_num_kket_prevals(L1, L2) - 1});
        kket_preval_ispace_list[index] = runtime->create_index_space(ctx, rect);
        /// KGJ: TESTING
        //printf("\nMEMORY TEST: get_num_kpairs(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kpairs(L1, L2));
        //printf("MEMORY TEST: get_num_kket_prevals(%d, %d) = %d \n", L1, L2, kdata_list.get_num_kket_prevals(L1, L2));
        ///
      }
      kket_preval_lr_list[index] = runtime->create_logical_region(
          ctx, kket_preval_ispace_list[index], kket_preval_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, kket_preval_lr_list[index],
                              kket_preval_lr_list[index]);
      const vector<FieldID> field_list(kket_preval_fields_list[index],
                                       kket_preval_fields_list[index] +
                                           NUM_KKET_PREVAL_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_kket_preval_data(L1, L2),
                                /*column major*/ false, field_list, memory);
      kket_preval_pr_list[index] =
          runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher kketPrevalAcquireLauncher(kket_preval_lr_list[index], kket_preval_lr_list[index], kket_preval_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          kketPrevalAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, kketPrevalAcquireLauncher);
    }
  }

  // Create density regions
  LogicalRegion kdensity_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion kdensity_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace kdensity_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
    for (int L4 = L2; L4 <= MAX_MOMENTUM; L4++) {
      //printf("KGJ: get_num_shells(%d) = %d \n", L2, kdata_list.get_num_shells(L2));
      //printf("KGJ: get_num_shells(%d) = %d \n", L4, kdata_list.get_num_shells(L4));
      const int index = INDEX_UPPER_TRIANGLE(L2, L4);
      if (kdata_list.get_num_shells(L2) == 0 ||
          kdata_list.get_num_shells(L4) == 0) {
        kdensity_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<2>::make_empty());
      } else {
        const Rect<2> rect({0, 0}, {kdata_list.get_num_shells(L2) - 1,
                                    kdata_list.get_num_shells(L4) - 1});
        kdensity_ispace_list[index] = runtime->create_index_space(ctx, rect);
      }
      kdensity_lr_list[index] = runtime->create_logical_region(
          ctx, kdensity_ispace_list[index], kdensity_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, kdensity_lr_list[index],
                              kdensity_lr_list[index]);
      const vector<FieldID> field_list(kdensity_fields_list[index],
                                       kdensity_fields_list[index] +
                                           NUM_KDENSITY_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_kdensity_data(L2, L4),
                                /*column major*/ false, field_list, memory);
      kdensity_pr_list[index] =
          runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher kdensityAcquireLauncher(kdensity_lr_list[index], kdensity_lr_list[index], kdensity_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          kdensityAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, kdensityAcquireLauncher);
    }
  }

  // Create output regions
  kdata_list.allocate_all_koutput();
  LogicalRegion koutput_lr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  PhysicalRegion koutput_pr_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  IndexSpace koutput_ispace_list[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      //printf("KGJ: get_num_shells(%d) = %d \n", L1, kdata_list.get_num_shells(L1));
      //printf("KGJ: get_num_shells(%d) = %d \n", L3, kdata_list.get_num_shells(L3));
      //printf("KGJ: get_largest_momentum() = %d \n", kdata_list.get_largest_momentum());
      const int index = INDEX_UPPER_TRIANGLE(L1, L3);
      if (kdata_list.get_num_shells(L1) == 0 ||
          kdata_list.get_num_shells(L3) == 0) {
        koutput_ispace_list[index] =
            runtime->create_index_space(ctx, Rect<3>::make_empty());
      } else {
        const Rect<3> rect({0, 0, 0},
                           {(kdata_list.get_largest_momentum() + 1) *
                                    (kdata_list.get_largest_momentum() + 1) -
                                1,
                            kdata_list.get_num_shells(L1) - 1,
                            kdata_list.get_num_shells(L3) - 1});
        koutput_ispace_list[index] = runtime->create_index_space(ctx, rect);
      }
      koutput_lr_list[index] = runtime->create_logical_region(
          ctx, koutput_ispace_list[index], koutput_fspaces[index]);
      AttachLauncher launcher(EXTERNAL_INSTANCE, koutput_lr_list[index],
                              koutput_lr_list[index]);
      const vector<FieldID> field_list(koutput_fields_list[index],
                                       koutput_fields_list[index] +
                                           NUM_KOUTPUT_FIELDS);
      // TODO: Consider using soa format if it helps performance.
      launcher.attach_array_aos(kdata_list.get_koutput_data(L1, L3),
                                /*column major*/ false, field_list, memory);
      koutput_pr_list[index] = runtime->attach_external_resource(ctx, launcher);
      AcquireLauncher koutputAcquireLauncher(koutput_lr_list[index], koutput_lr_list[index], koutput_pr_list[index]);
      vector<FieldID> fl = field_list;
      for(vector<FieldID>::iterator it = fl.begin();
          it != fl.end(); ++it) {
          koutputAcquireLauncher.add_field(*it);
      }
      runtime->issue_acquire(ctx, koutputAcquireLauncher);
    }
  }

  kfock_task_launcher launcher;

#define ADD_ARGUMENT_R_KPAIRS(L1, L2)                                          \
  {                                                                            \
    const int index = INDEX_SQUARE(L1, L2);                                    \
    {                                                                          \
      const vector<FieldID> field_list(kpair_fields_list[index],               \
                                       kpair_fields_list[index] +              \
                                           NUM_KPAIR_FIELDS);                  \
      launcher.add_argument_r_pairs##L1##L2(kpair_lr_list[index],              \
                                            kpair_lr_list[index], field_list); \
    }                                                                          \
    {                                                                          \
      const vector<FieldID> field_list(kbra_preval_fields_list[index],         \
                                       kbra_preval_fields_list[index] +        \
                                           NUM_KBRA_PREVAL_FIELDS);            \
      launcher.add_argument_r_bra_prevals##L1##L2(                             \
          kbra_preval_lr_list[index], kbra_preval_lr_list[index], field_list); \
    }                                                                          \
    {                                                                          \
      const vector<FieldID> field_list(kket_preval_fields_list[index],         \
                                       kket_preval_fields_list[index] +        \
                                           NUM_KKET_PREVAL_FIELDS);            \
      launcher.add_argument_r_ket_prevals##L1##L2(                             \
          kket_preval_lr_list[index], kket_preval_lr_list[index], field_list); \
    }                                                                          \
  }

  ADD_ARGUMENT_R_KPAIRS(0, 0);
  ADD_ARGUMENT_R_KPAIRS(0, 1);
  ADD_ARGUMENT_R_KPAIRS(0, 2);
  ADD_ARGUMENT_R_KPAIRS(0, 3);
  ADD_ARGUMENT_R_KPAIRS(0, 4);
  ADD_ARGUMENT_R_KPAIRS(1, 0);
  ADD_ARGUMENT_R_KPAIRS(1, 1);
  ADD_ARGUMENT_R_KPAIRS(1, 2);
  ADD_ARGUMENT_R_KPAIRS(1, 3);
  ADD_ARGUMENT_R_KPAIRS(1, 4);
  ADD_ARGUMENT_R_KPAIRS(2, 0);
  ADD_ARGUMENT_R_KPAIRS(2, 1);
  ADD_ARGUMENT_R_KPAIRS(2, 2);
  ADD_ARGUMENT_R_KPAIRS(2, 3);
  ADD_ARGUMENT_R_KPAIRS(2, 4);
  ADD_ARGUMENT_R_KPAIRS(3, 0);
  ADD_ARGUMENT_R_KPAIRS(3, 1);
  ADD_ARGUMENT_R_KPAIRS(3, 2);
  ADD_ARGUMENT_R_KPAIRS(3, 3);
  ADD_ARGUMENT_R_KPAIRS(3, 4);
  ADD_ARGUMENT_R_KPAIRS(4, 0);
  ADD_ARGUMENT_R_KPAIRS(4, 1);
  ADD_ARGUMENT_R_KPAIRS(4, 2);
  ADD_ARGUMENT_R_KPAIRS(4, 3);
  ADD_ARGUMENT_R_KPAIRS(4, 4);

#undef ADD_ARGUMENT_R_KPAIRS

#define ADD_ARGUMENT_R_KDENSITY(L2, L4)                                        \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L2, L4);                            \
    const vector<FieldID> field_list(kdensity_fields_list[index],              \
                                     kdensity_fields_list[index] +             \
                                         NUM_KDENSITY_FIELDS);                 \
    launcher.add_argument_r_density##L2##L4(                                   \
        kdensity_lr_list[index], kdensity_lr_list[index], field_list);         \
  }

  ADD_ARGUMENT_R_KDENSITY(0, 0);
  ADD_ARGUMENT_R_KDENSITY(0, 1);
  ADD_ARGUMENT_R_KDENSITY(0, 2);
  ADD_ARGUMENT_R_KDENSITY(0, 3);
  ADD_ARGUMENT_R_KDENSITY(0, 4);
  ADD_ARGUMENT_R_KDENSITY(1, 1);
  ADD_ARGUMENT_R_KDENSITY(1, 2);
  ADD_ARGUMENT_R_KDENSITY(1, 3);
  ADD_ARGUMENT_R_KDENSITY(1, 4);
  ADD_ARGUMENT_R_KDENSITY(2, 2);
  ADD_ARGUMENT_R_KDENSITY(2, 3);
  ADD_ARGUMENT_R_KDENSITY(2, 4);
  ADD_ARGUMENT_R_KDENSITY(3, 3);
  ADD_ARGUMENT_R_KDENSITY(3, 4);
  ADD_ARGUMENT_R_KDENSITY(4, 4);

#undef ADD_ARGUMENT_R_KDENSITY

#define ADD_ARGUMENT_R_KOUTPUT(L1, L3)                                         \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L1, L3);                            \
    const vector<FieldID> field_list(koutput_fields_list[index],               \
                                     koutput_fields_list[index] +              \
                                         NUM_KOUTPUT_FIELDS);                  \
    launcher.add_argument_r_output##L1##L3(                                    \
        koutput_lr_list[index], koutput_lr_list[index], field_list);           \
  }

  ADD_ARGUMENT_R_KOUTPUT(0, 0);
  ADD_ARGUMENT_R_KOUTPUT(0, 1);
  ADD_ARGUMENT_R_KOUTPUT(0, 2);
  ADD_ARGUMENT_R_KOUTPUT(0, 3);
  ADD_ARGUMENT_R_KOUTPUT(0, 4);
  ADD_ARGUMENT_R_KOUTPUT(1, 1);
  ADD_ARGUMENT_R_KOUTPUT(1, 2);
  ADD_ARGUMENT_R_KOUTPUT(1, 3);
  ADD_ARGUMENT_R_KOUTPUT(1, 4);
  ADD_ARGUMENT_R_KOUTPUT(2, 2);
  ADD_ARGUMENT_R_KOUTPUT(2, 3);
  ADD_ARGUMENT_R_KOUTPUT(2, 4);
  ADD_ARGUMENT_R_KOUTPUT(3, 3);
  ADD_ARGUMENT_R_KOUTPUT(3, 4);
  ADD_ARGUMENT_R_KOUTPUT(4, 4);

#undef ADD_ARGUMENT_R_KOUTPUT

  launcher.add_argument_r_gamma_table(gamma_table_lr, gamma_table_lr,
                                      {GAMMA_TABLE_FIELD_ID});
  launcher.add_argument_threshold(threshold);
  launcher.add_argument_threshold(kguard);
  launcher.add_argument_parallelism0(parray[0]);
  launcher.add_argument_parallelism1(parray[1]);
  launcher.add_argument_parallelism2(parray[2]);
  launcher.add_argument_parallelism3(parray[3]);
  launcher.add_argument_parallelism4(parray[4]);
  launcher.add_argument_parallelism5(parray[5]);
  launcher.add_argument_parallelism6(parray[6]);
  launcher.add_argument_parallelism7(parray[7]);
  launcher.add_argument_parallelism8(parray[8]);
  launcher.add_argument_parallelism9(parray[9]);
  launcher.add_argument_parallelism10(parray[10]);
  launcher.add_argument_parallelism11(parray[11]);
  launcher.add_argument_parallelism12(parray[12]);
  launcher.add_argument_parallelism13(parray[13]);
  launcher.add_argument_parallelism14(parray[14]);
  launcher.add_argument_parallelism15(parray[15]);
  launcher.add_argument_largest_momentum(kdata_list.get_largest_momentum());
__TRACE
  Future future = launcher.execute(runtime, ctx);
__TRACE
  future.wait();
__TRACE

  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
      const int index = INDEX_SQUARE(L1, L2);
      ReleaseLauncher kpairReleaseLauncher(kpair_lr_list[index], kpair_lr_list[index], kpair_pr_list[index]);
      const vector<FieldID> kpair_list(
          kpair_fields_list[index], kpair_fields_list[index] + NUM_KPAIR_FIELDS);
      vector<FieldID> bfl = kpair_list;
      for(vector<FieldID>::iterator it = bfl.begin();
          it != bfl.end(); ++it) {
          kpairReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, kpairReleaseLauncher);

      runtime->detach_external_resource(ctx, kpair_pr_list[index]);
      runtime->destroy_logical_region(ctx, kpair_lr_list[index]);
      runtime->destroy_index_space(ctx, kpair_ispace_list[index]);

      ReleaseLauncher kbraPrevalReleaseLauncher(kbra_preval_lr_list[index], kbra_preval_lr_list[index], kbra_preval_pr_list[index]);
      const vector<FieldID> kbra_preval_list(
          kbra_preval_fields_list[index], kbra_preval_fields_list[index] + NUM_KBRA_PREVAL_FIELDS);
      vector<FieldID> kpfl = kbra_preval_list;
      for(vector<FieldID>::iterator it = kpfl.begin();
          it != kpfl.end(); ++it) {
          kbraPrevalReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, kbraPrevalReleaseLauncher);

      runtime->detach_external_resource(ctx, kbra_preval_pr_list[index]);
      runtime->destroy_logical_region(ctx, kbra_preval_lr_list[index]);
      runtime->destroy_index_space(ctx, kbra_preval_ispace_list[index]);

      ReleaseLauncher kketPrevalReleaseLauncher(kket_preval_lr_list[index], kket_preval_lr_list[index], kket_preval_pr_list[index]);
      const vector<FieldID> kket_preval_list(
          kket_preval_fields_list[index], kket_preval_fields_list[index] + NUM_KKET_PREVAL_FIELDS);
      vector<FieldID> kkfl = kket_preval_list;
      for(vector<FieldID>::iterator it = kkfl.begin();
          it != kkfl.end(); ++it) {
          kketPrevalReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, kketPrevalReleaseLauncher);

      runtime->detach_external_resource(ctx, kket_preval_pr_list[index]);
      runtime->destroy_logical_region(ctx, kket_preval_lr_list[index]);
      runtime->destroy_index_space(ctx, kket_preval_ispace_list[index]);
    }
  }
  for (int L2 = 0; L2 <= MAX_MOMENTUM; L2++) {
    for (int L4 = L2; L4 <= MAX_MOMENTUM; L4++) {
      const int index = INDEX_UPPER_TRIANGLE(L2, L4);

      ReleaseLauncher kdensityReleaseLauncher(kdensity_lr_list[index], kdensity_lr_list[index], kdensity_pr_list[index]);
      const vector<FieldID> kdensity_list(
          kdensity_fields_list[index], kdensity_fields_list[index] + NUM_KDENSITY_FIELDS);
      vector<FieldID> kdfl = kdensity_list;
      for(vector<FieldID>::iterator it = kdfl.begin();
          it != kdfl.end(); ++it) {
          kdensityReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, kdensityReleaseLauncher);

      runtime->detach_external_resource(ctx, kdensity_pr_list[index]);
      runtime->destroy_logical_region(ctx, kdensity_lr_list[index]);
      runtime->destroy_index_space(ctx, kdensity_ispace_list[index]);
    }
  }
  for (int L1 = 0; L1 <= MAX_MOMENTUM; L1++) {
    for (int L3 = L1; L3 <= MAX_MOMENTUM; L3++) {
      const int index = INDEX_UPPER_TRIANGLE(L1, L3);
      ReleaseLauncher koutputReleaseLauncher(koutput_lr_list[index], koutput_lr_list[index], koutput_pr_list[index]);
      const vector<FieldID> koutput_list(
          koutput_fields_list[index], koutput_fields_list[index] + NUM_KOUTPUT_FIELDS);
      vector<FieldID> kofl = koutput_list;
      for(vector<FieldID>::iterator it = kofl.begin();
          it != kofl.end(); ++it) {
          koutputReleaseLauncher.add_field(*it);
      }
      runtime->issue_release(ctx, koutputReleaseLauncher);

      runtime->detach_external_resource(ctx, koutput_pr_list[index]);
      runtime->destroy_logical_region(ctx, koutput_lr_list[index]);
      runtime->destroy_index_space(ctx, koutput_ispace_list[index]);
    }
  }
}

void EriRegent::initialize_jfock_field_spaces() {
#define INIT_FSPACES(L1, L2)                                                   \
  {                                                                            \
    const int H = TETRAHEDRAL_NUMBER((L1) + (L2) + 1);                         \
    jbra_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)] =                               \
        runtime->create_field_space(ctx);                                      \
    jket_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)] =                               \
        runtime->create_field_space(ctx);                                      \
    {                                                                          \
      FieldAllocator falloc = runtime->create_field_allocator(                 \
          ctx, jbra_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)]);                    \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, X));         \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, Y));         \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, Z));         \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, ETA));       \
      falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(L1, L2, C));         \
      falloc.allocate_field(sizeof(float), JBRA_FIELD_ID(L1, L2, BOUND));      \
      falloc.allocate_field(H * sizeof(double),                                \
                            JBRA_FIELD_ID(L1, L2, OUTPUT));                    \
    }                                                                          \
    {                                                                          \
      FieldAllocator falloc = runtime->create_field_allocator(                 \
          ctx, jket_fspaces[INDEX_UPPER_TRIANGLE(L1, L2)]);                    \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, X));         \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, Y));         \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, Z));         \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, ETA));       \
      falloc.allocate_field(sizeof(double), JKET_FIELD_ID(L1, L2, C));         \
      falloc.allocate_field(sizeof(float), JKET_FIELD_ID(L1, L2, BOUND));      \
      falloc.allocate_field(H * sizeof(double),                                \
                            JKET_FIELD_ID(L1, L2, DENSITY));                   \
    }                                                                          \
  }

  INIT_FSPACES(0, 0)
  INIT_FSPACES(0, 1)
  INIT_FSPACES(0, 2)
  INIT_FSPACES(0, 3)
  INIT_FSPACES(0, 4)
  INIT_FSPACES(1, 1)
  INIT_FSPACES(1, 2)
  INIT_FSPACES(1, 3)
  INIT_FSPACES(1, 4)
  INIT_FSPACES(2, 2)
  INIT_FSPACES(2, 3)
  INIT_FSPACES(2, 4)
  INIT_FSPACES(3, 3)
  INIT_FSPACES(3, 4)
  INIT_FSPACES(4, 4)

#undef INIT_FSPACES
}

void EriRegent::initialize_kfock_field_spaces() {
#define INIT_KPAIR_FSPACES(L1, L2)                                             \
  {                                                                            \
    const int index = INDEX_SQUARE(L1, L2);                                    \
    kpair_fspaces[index] = runtime->create_field_space(ctx);                   \
    {                                                                          \
      FieldAllocator falloc =                                                  \
          runtime->create_field_allocator(ctx, kpair_fspaces[index]);          \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, X));        \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, Y));        \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, Z));        \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ETA));      \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, C));        \
      falloc.allocate_field(sizeof(float), KPAIR_FIELD_ID(L1, L2, BOUND));     \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ISHELL_X)); \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ISHELL_Y)); \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, ISHELL_Z)); \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, JSHELL_X)); \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, JSHELL_Y)); \
      falloc.allocate_field(sizeof(double), KPAIR_FIELD_ID(L1, L2, JSHELL_Z)); \
      falloc.allocate_field(sizeof(int1d_t),                                   \
                            KPAIR_FIELD_ID(L1, L2, ISHELL_INDEX));             \
      falloc.allocate_field(sizeof(int1d_t),                                   \
                            KPAIR_FIELD_ID(L1, L2, JSHELL_INDEX));             \
    }                                                                          \
                                                                               \
    kbra_preval_fspaces[index] = runtime->create_field_space(ctx);             \
    {                                                                          \
      FieldAllocator falloc =                                                  \
          runtime->create_field_allocator(ctx, kbra_preval_fspaces[index]);    \
      falloc.allocate_field(sizeof(double), KBRA_PREVAL_FIELD_ID(L1, L2));     \
    }                                                                          \
    kket_preval_fspaces[index] = runtime->create_field_space(ctx);             \
    {                                                                          \
      FieldAllocator falloc =                                                  \
          runtime->create_field_allocator(ctx, kket_preval_fspaces[index]);    \
      falloc.allocate_field(sizeof(double), KKET_PREVAL_FIELD_ID(L1, L2));     \
    }                                                                          \
  }

  INIT_KPAIR_FSPACES(0, 0);
  INIT_KPAIR_FSPACES(0, 1);
  INIT_KPAIR_FSPACES(0, 2);
  INIT_KPAIR_FSPACES(0, 3);
  INIT_KPAIR_FSPACES(0, 4);
  INIT_KPAIR_FSPACES(1, 0);
  INIT_KPAIR_FSPACES(1, 1);
  INIT_KPAIR_FSPACES(1, 2);
  INIT_KPAIR_FSPACES(1, 3);
  INIT_KPAIR_FSPACES(1, 4);
  INIT_KPAIR_FSPACES(2, 0);
  INIT_KPAIR_FSPACES(2, 1);
  INIT_KPAIR_FSPACES(2, 2);
  INIT_KPAIR_FSPACES(2, 3);
  INIT_KPAIR_FSPACES(2, 4);
  INIT_KPAIR_FSPACES(3, 0);
  INIT_KPAIR_FSPACES(3, 1);
  INIT_KPAIR_FSPACES(3, 2);
  INIT_KPAIR_FSPACES(3, 3);
  INIT_KPAIR_FSPACES(3, 4);
  INIT_KPAIR_FSPACES(4, 0);
  INIT_KPAIR_FSPACES(4, 1);
  INIT_KPAIR_FSPACES(4, 2);
  INIT_KPAIR_FSPACES(4, 3);
  INIT_KPAIR_FSPACES(4, 4);

#undef INIT_KPAIR_FSPACES

#define INIT_KDENSITY_FSPACES(L2, L4)                                          \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L2, L4);                            \
    kdensity_fspaces[index] = runtime->create_field_space(ctx);                \
    FieldAllocator falloc =                                                    \
        runtime->create_field_allocator(ctx, kdensity_fspaces[index]);         \
    const int H2 = TRIANGLE_NUMBER((L2) + 1);                                  \
    const int H4 = TRIANGLE_NUMBER((L4) + 1);                                  \
    falloc.allocate_field(H2 *H4 * sizeof(double),                             \
                          KDENSITY_FIELD_ID(L2, L4, VALUES));                  \
    falloc.allocate_field(sizeof(float), KDENSITY_FIELD_ID(L2, L4, BOUND));    \
  }

  INIT_KDENSITY_FSPACES(0, 0);
  INIT_KDENSITY_FSPACES(0, 1);
  INIT_KDENSITY_FSPACES(0, 2);
  INIT_KDENSITY_FSPACES(0, 3);
  INIT_KDENSITY_FSPACES(0, 4);
  INIT_KDENSITY_FSPACES(1, 1);
  INIT_KDENSITY_FSPACES(1, 2);
  INIT_KDENSITY_FSPACES(1, 3);
  INIT_KDENSITY_FSPACES(1, 4);
  INIT_KDENSITY_FSPACES(2, 2);
  INIT_KDENSITY_FSPACES(2, 3);
  INIT_KDENSITY_FSPACES(2, 4);
  INIT_KDENSITY_FSPACES(3, 3);
  INIT_KDENSITY_FSPACES(3, 4);
  INIT_KDENSITY_FSPACES(4, 4);

#undef INIT_KDENSITY_FSPACES

#define INIT_KOUTPUT_FSPACES(L1, L3)                                           \
  {                                                                            \
    const int index = INDEX_UPPER_TRIANGLE(L1, L3);                            \
    koutput_fspaces[index] = runtime->create_field_space(ctx);                 \
    FieldAllocator falloc =                                                    \
        runtime->create_field_allocator(ctx, koutput_fspaces[index]);          \
    const int H1 = TRIANGLE_NUMBER((L1) + 1);                                  \
    const int H3 = TRIANGLE_NUMBER((L3) + 1);                                  \
    falloc.allocate_field(H1 *H3 * sizeof(double),                             \
                          KOUTPUT_FIELD_ID(L1, L3, VALUES));                   \
  }

  INIT_KOUTPUT_FSPACES(0, 0);
  INIT_KOUTPUT_FSPACES(0, 1);
  INIT_KOUTPUT_FSPACES(0, 2);
  INIT_KOUTPUT_FSPACES(0, 3);
  INIT_KOUTPUT_FSPACES(0, 4);
  INIT_KOUTPUT_FSPACES(1, 1);
  INIT_KOUTPUT_FSPACES(1, 2);
  INIT_KOUTPUT_FSPACES(1, 3);
  INIT_KOUTPUT_FSPACES(1, 4);
  INIT_KOUTPUT_FSPACES(2, 2);
  INIT_KOUTPUT_FSPACES(2, 3);
  INIT_KOUTPUT_FSPACES(2, 4);
  INIT_KOUTPUT_FSPACES(3, 3);
  INIT_KOUTPUT_FSPACES(3, 4);
  INIT_KOUTPUT_FSPACES(4, 4);

#undef INIT_KOUTPUT_FSPACES
}
