#pragma once

#include "fields.h"
#include "legion.h"

void top_level_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *runtime);

/**
 * Launch the jfock regent task
 */
void launch_jfock_task(TeraChemJBraList* jbras_list,
                       TeraChemJKetList* jkets_list,
                       Legion::LogicalRegion &gamma_table_lr,
                       float threshold, int parallelism,
                       Legion::Context ctx, Legion::Runtime *runtime);


/**
 * Create a region for the gamma table and fill it with the correct values
 */
void create_gamma_table_region(Legion::LogicalRegion &lr,
                               Legion::PhysicalRegion &pr,
                               Legion::Context ctx, Legion::Runtime *runtime,
                               const Legion::Memory memory);

/**
 * Destroy a region that was populated with `AttachLauncher`
 */
void destroy_attached_region(Legion::LogicalRegion &lr,
                             Legion::PhysicalRegion &pr,
                             Legion::Context ctx, Legion::Runtime *runtime) {
  runtime->detach_external_resource(ctx, pr);
  runtime->destroy_logical_region(ctx, lr);
}


// Return the number of atomic orbital functions in shells of momentum 0 to L
// TODO: Find a better name
#define COMPUTE_H(L12) (((L12) + 1) * ((L12) + 2) * ((L12) + 3) / 6)

#define JBRA_FIELD_ID(L1, L2, FIELD_NAME) JBRA##L1##L2##_FIELD_##FIELD_NAME##_ID
#define JKET_FIELD_ID(L1, L2, FIELD_NAME) JKET##L1##L2##_FIELD_##FIELD_NAME##_ID

#define JBRA_FIELD_IDS(L1, L2)      \
  JBRA_FIELD_ID(L1, L2, X),         \
  JBRA_FIELD_ID(L1, L2, Y),         \
  JBRA_FIELD_ID(L1, L2, Z),         \
  JBRA_FIELD_ID(L1, L2, ETA),       \
  JBRA_FIELD_ID(L1, L2, C),         \
  JBRA_FIELD_ID(L1, L2, BOUND),     \
  JBRA_FIELD_ID(L1, L2, OUTPUT)

#define JKET_FIELD_IDS(L1, L2)      \
  JKET_FIELD_ID(L1, L2, X),         \
  JKET_FIELD_ID(L1, L2, Y),         \
  JKET_FIELD_ID(L1, L2, Z),         \
  JKET_FIELD_ID(L1, L2, ETA),       \
  JKET_FIELD_ID(L1, L2, C),         \
  JKET_FIELD_ID(L1, L2, BOUND),     \
  JKET_FIELD_ID(L1, L2, DENSITY)

enum { // Field IDs
  GAMMA_TABLE_FIELD_ID,
  JBRA_FIELD_IDS(0, 0), JKET_FIELD_IDS(0, 0),
  // JBRA_FIELD_IDS(0, 1), JKET_FIELD_IDS(0, 1),
  // JBRA_FIELD_IDS(1, 1), JKET_FIELD_IDS(1, 1),
  // JBRA_FIELD_IDS(0, 2), JKET_FIELD_IDS(0, 2),
  // JBRA_FIELD_IDS(1, 2), JKET_FIELD_IDS(1, 2),
  // JBRA_FIELD_IDS(2, 2), JKET_FIELD_IDS(2, 2),
  // JBRA_FIELD_IDS(0, 3), JKET_FIELD_IDS(0, 3),
  // JBRA_FIELD_IDS(1, 3), JKET_FIELD_IDS(1, 3),
  // JBRA_FIELD_IDS(2, 3), JKET_FIELD_IDS(2, 3),
  // JBRA_FIELD_IDS(3, 3), JKET_FIELD_IDS(3, 3),
  // JBRA_FIELD_IDS(0, 4), JKET_FIELD_IDS(0, 4),
  // JBRA_FIELD_IDS(1, 4), JKET_FIELD_IDS(1, 4),
  // JBRA_FIELD_IDS(2, 4), JKET_FIELD_IDS(2, 4),
  // JBRA_FIELD_IDS(3, 4), JKET_FIELD_IDS(3, 4),
  // JBRA_FIELD_IDS(4, 4), JKET_FIELD_IDS(4, 4),
};


/**
 * Create a JBra region and fill it with data
 */
void create_jbra00_region(Legion::LogicalRegion &lr,
                          Legion::PhysicalRegion &pr,
                          TeraChemJBraList *jbras, // TODO
                          Legion::Context ctx, Legion::Runtime *runtime,
                          const Legion::Memory memory) {
  using namespace Legion;
  const int H = COMPUTE_H(0 + 0);
  const Rect<1> rect(0, jbras->length - 1);
  const IndexSpace ispace = runtime->create_index_space(ctx, rect);
  const FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(0, 0, X));
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(0, 0, Y));
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(0, 0, Z));
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(0, 0, ETA));
    falloc.allocate_field(sizeof(double), JBRA_FIELD_ID(0, 0, C));
    falloc.allocate_field(sizeof(float), JBRA_FIELD_ID(0, 0, BOUND));
    falloc.allocate_field(H * sizeof(double), JBRA_FIELD_ID(0, 0, OUTPUT));
  }

  lr = runtime->create_logical_region(ctx, ispace, fspace);

  AttachLauncher launcher(EXTERNAL_INSTANCE, lr, lr);

  // TODO: Can alignments help?
  // launcher.attach_array_aos((void *)jbras->data, /*column major=*/false,
  //                           {JBRA_FIELD_IDS(0, 0)}, memory,
  //                           /*alignments=*/NULL);
  // pr = runtime->attach_external_resource(ctx, launcher);
}
