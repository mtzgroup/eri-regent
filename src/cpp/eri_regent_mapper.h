/* Copyright 2020 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __ERI_REGENT_MAPPER_H
#define __ERI_REGENT_MAPPER_H

#include "legion.h"
#include "mappers/logging_wrapper.h"
#include "mappers/default_mapper.h"

#include <stdlib.h>
#include <assert.h>
#include <unistd.h>


namespace Legion {
  namespace Mapping {

    class EriRegentMapper : public DefaultMapper {
    public:
      EriRegentMapper(MapperRuntime *rt, Machine machine, Processor local); 
      virtual ~EriRegentMapper(void);
/*
      void default_policy_select_constraints(MapperContext ctx,
               LayoutConstraintSet &constraints, Memory target_memory,
               const RegionRequirement &req);
*/
/*
      Memory default_policy_select_target_memory(MapperContext ctx,
                          Processor target_proc,
                          const RegionRequirement &req);
*/
/*
      void slice_task(const MapperContext      ctx,
                          const Task&              task,
                          const SliceTaskInput&    input,
                                SliceTaskOutput&   output);
*/
      void replaceMapper();
    };

  }; // namespace Mapping
}; // namespace Legion

#ifdef __cplusplus
extern "C" {
#endif

void register_mappers();

#ifdef __cplusplus
}
#endif


#endif // __ERI_REGENT_MAPPER_H

