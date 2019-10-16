#pragma once

#include "legion.h"

void top_level_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *runtime);

/**
 * Create a region for the gamma table and fill it with the correct values
 */
void create_gamma_table_region(Legion::LogicalRegion &lr,
                               Legion::PhysicalRegion &pr,
                               Legion::Context ctx, Legion::Runtime *runtime,
                               const Legion::Memory memory);

/**
 * Destroy a gamma table region
 */
void destroy_gamma_table_region(Legion::LogicalRegion &lr,
                                Legion::PhysicalRegion &pr,
                                Legion::Context ctx, Legion::Runtime *runtime) {
  runtime->detach_external_resource(ctx, pr);
  runtime->destroy_logical_region(ctx, lr);
}
