#pragma once

#include "fields.h"
#include "legion.h"

// TODO: Make a new namespace

void top_level_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *runtime);

// TODO: Put launch function in different header
// TODO: Make a new struct for the "regent context"
// TODO: Make something like "init_context" and "destroy_context"
/**
 * Launch the jfock regent task
 */
void launch_jfock_task(TeraChemJDataList &jdata_list,
                       Legion::LogicalRegion &gamma_table_lr,
                       float threshold, int parallelism,
		                   const Legion::FieldSpace *jbra_fspaces,
		                   const Legion::FieldSpace *jket_fspaces,
                       Legion::Context ctx, Legion::Runtime *runtime,
                       const Legion::Memory memory);

/*
 *
 */
void initialize_field_spaces(
  Legion::FieldSpace jbra_fspaces[MAX_MOMENTUM_INDEX + 1],
  Legion::FieldSpace jket_fspaces[MAX_MOMENTUM_INDEX + 1],
  Legion::Context ctx, Legion::Runtime *runtime);

void* fill_jbra_data(const TeraChemJData *jbras, size_t num_jbras, size_t L12);


void* fill_jket_data(const TeraChemJData *jkets, size_t num_jkets,
                     const double *density, size_t L12);


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
  JBRA_FIELD_IDS(0, 1), JKET_FIELD_IDS(0, 1),
  JBRA_FIELD_IDS(1, 1), JKET_FIELD_IDS(1, 1),
  JBRA_FIELD_IDS(0, 2), JKET_FIELD_IDS(0, 2),
  JBRA_FIELD_IDS(1, 2), JKET_FIELD_IDS(1, 2),
  JBRA_FIELD_IDS(2, 2), JKET_FIELD_IDS(2, 2),
  JBRA_FIELD_IDS(0, 3), JKET_FIELD_IDS(0, 3),
  JBRA_FIELD_IDS(1, 3), JKET_FIELD_IDS(1, 3),
  JBRA_FIELD_IDS(2, 3), JKET_FIELD_IDS(2, 3),
  JBRA_FIELD_IDS(3, 3), JKET_FIELD_IDS(3, 3),
  JBRA_FIELD_IDS(0, 4), JKET_FIELD_IDS(0, 4),
  JBRA_FIELD_IDS(1, 4), JKET_FIELD_IDS(1, 4),
  JBRA_FIELD_IDS(2, 4), JKET_FIELD_IDS(2, 4),
  JBRA_FIELD_IDS(3, 4), JKET_FIELD_IDS(3, 4),
  JBRA_FIELD_IDS(4, 4), JKET_FIELD_IDS(4, 4),
};

const std::vector<Legion::FieldID> jbra_fields_list[MAX_MOMENTUM_INDEX + 1] = {
    {JBRA_FIELD_IDS(0, 0)},
    {JBRA_FIELD_IDS(0, 1)},
    {JBRA_FIELD_IDS(0, 2)},
    {JBRA_FIELD_IDS(0, 3)},
    {JBRA_FIELD_IDS(0, 4)},
    {JBRA_FIELD_IDS(1, 1)},
    {JBRA_FIELD_IDS(1, 2)},
    {JBRA_FIELD_IDS(1, 3)},
    {JBRA_FIELD_IDS(1, 4)},
    {JBRA_FIELD_IDS(2, 2)},
    {JBRA_FIELD_IDS(2, 3)},
    {JBRA_FIELD_IDS(2, 4)},
    {JBRA_FIELD_IDS(3, 3)},
    {JBRA_FIELD_IDS(3, 4)},
    {JBRA_FIELD_IDS(4, 4)},
};

const std::vector<Legion::FieldID> jket_fields_list[MAX_MOMENTUM_INDEX + 1] = {
    {JKET_FIELD_IDS(0, 0)},
    {JKET_FIELD_IDS(0, 1)},
    {JKET_FIELD_IDS(0, 2)},
    {JKET_FIELD_IDS(0, 3)},
    {JKET_FIELD_IDS(0, 4)},
    {JKET_FIELD_IDS(1, 1)},
    {JKET_FIELD_IDS(1, 2)},
    {JKET_FIELD_IDS(1, 3)},
    {JKET_FIELD_IDS(1, 4)},
    {JKET_FIELD_IDS(2, 2)},
    {JKET_FIELD_IDS(2, 3)},
    {JKET_FIELD_IDS(2, 4)},
    {JKET_FIELD_IDS(3, 3)},
    {JKET_FIELD_IDS(3, 4)},
    {JKET_FIELD_IDS(4, 4)},
};

#undef JBRA_FIELD_IDS
#undef JKET_FIELD_IDS
