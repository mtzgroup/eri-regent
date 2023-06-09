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

#include <unistd.h>
#include <iostream>

#include "eri_regent_mapper.h"
#include "eri_regent.h"

namespace Legion {
  namespace Mapping {

    Logger log_eri_regent_mapper("eri_regent_mapper");

    //--------------------------------------------------------------------------
    EriRegentMapper::EriRegentMapper(MapperRuntime *rt, Machine m, Processor local)
      : DefaultMapper(rt, m, local)
    //--------------------------------------------------------------------------
    {
std::cout << __FUNCTION__ << " constructor" << std::endl;
std::cout << "local_gpus " << local_gpus.size() << std::endl;
    }

    //--------------------------------------------------------------------------
    EriRegentMapper::~EriRegentMapper(void)
    //--------------------------------------------------------------------------
    {
    }

#if 1



    //--------------------------------------------------------------------------
    void DefaultMapper::default_policy_select_constraints(MapperContext ctx,
                     LayoutConstraintSet &constraints, Memory target_memory,
                     const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      // See if we are doing a reduction instance
      if (req.privilege == LEGION_REDUCE)
      {
        // Make reduction fold instances
        constraints.add_constraint(SpecializedConstraint(
                            LEGION_AFFINE_REDUCTION_SPECIALIZE, req.redop))
          .add_constraint(MemoryConstraint(target_memory.kind()));
      }
      else
      {
        // Our base default mapper will try to make instances of containing
        // all fields (in any order) laid out in SOA format to encourage
        // maximum re-use by any tasks which use subsets of the fields
        constraints.add_constraint(SpecializedConstraint())
          .add_constraint(MemoryConstraint(target_memory.kind()));

        if (constraints.field_constraint.field_set.size() == 0)
        {
          // Normal instance creation
          std::vector<FieldID> fields;
          default_policy_select_constraint_fields(ctx, req, fields);
          constraints.add_constraint(FieldConstraint(fields,false/*contiguous*/,
                                                     false/*inorder*/));
        }
        if (constraints.ordering_constraint.ordering.size() == 0)
        {
          IndexSpace is = req.region.get_index_space();
          Domain domain = runtime->get_index_space_domain(ctx, is);
          int dim = domain.get_dim();
          std::vector<DimensionKind> dimension_ordering(dim + 1);
          for (int i = 0; i < dim; ++i)
            dimension_ordering[i] =
              static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
          dimension_ordering[dim] = LEGION_DIM_F;
          constraints.add_constraint(OrderingConstraint(dimension_ordering,
                                                        false/*contigous*/));
        }
      }
    }


#endif

#if 0

    //--------------------------------------------------------------------------
    Memory EriRegentMapper::default_policy_select_target_memory(MapperContext ctx,
                                                   Processor target_proc,
                                                   const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {

      bool prefer_rdma = ((req.tag & DefaultMapper::PREFER_RDMA_MEMORY) != 0);

#if 0
      // TODO: deal with the updates in machine model which will
      //       invalidate this cache
      std::map<Processor,Memory>::iterator it;
      if(req.privilege != LEGION_REDUCE) {
        if (prefer_rdma)
        {
	      it = cached_rdma_target_memory.find(target_proc);
	      if (it != cached_rdma_target_memory.end()) {
            return it->second;
          }
        } else {
          it = cached_target_memory.find(target_proc);
	      if (it != cached_target_memory.end()) {
            return it->second;
          }
        }
      }
#endif

      // Find the visible memories from the processor for the given kind
      Machine::MemoryQuery visible_memories(machine);
      visible_memories.has_affinity_to(target_proc);
      if (visible_memories.count() == 0)
      {
        log_eri_regent_mapper.error("No visible memories from processor " IDFMT "! "
                         "This machine is really messed up!", target_proc.id);
        assert(false);
      }
      // Figure out the memory with the highest-bandwidth
      Memory best_memory = Memory::NO_MEMORY;
      unsigned best_bandwidth = 0;
      Memory best_rdma_memory = Memory::NO_MEMORY;
      unsigned best_rdma_bandwidth = 0;
      std::vector<Machine::ProcessorMemoryAffinity> affinity(1);
      for (Machine::MemoryQuery::iterator it = visible_memories.begin();
            it != visible_memories.end(); it++)
      {

        affinity.clear();
        machine.get_proc_mem_affinity(affinity, target_proc, *it,
				      false /*not just local affinities*/);
        assert(affinity.size() == 1);

        if (!best_memory.exists() || (affinity[0].bandwidth > best_bandwidth)) {
          best_memory = *it;
          best_bandwidth = affinity[0].bandwidth;
        }
        if ((it->kind() == Memory::REGDMA_MEM) &&
	    (!best_rdma_memory.exists() ||
	     (affinity[0].bandwidth > best_rdma_bandwidth))) {
          best_rdma_memory = *it;
          best_rdma_bandwidth = affinity[0].bandwidth;
        }
        if(req.privilege == LEGION_REDUCE) {
          if(it->kind() == Memory::Z_COPY_MEM && it->exists()) {
            best_memory = *it;
            best_bandwidth = affinity[0].bandwidth;
          }
        }
      }

      assert(best_memory.exists());
      if (prefer_rdma)
      {
	    if (!best_rdma_memory.exists()) best_rdma_memory = best_memory;
#if 0
	    cached_rdma_target_memory[target_proc] = best_rdma_memory;
#endif
	    return best_rdma_memory;
      } else {
#if 0
	    cached_target_memory[target_proc] = best_memory;
#endif
	    return best_memory;
      }
    }

#if 0

    void EriRegentMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task,
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output) {
std::cout << "remote gpus " << remote_gpus.size() << " local_gpus " << local_gpus.size() << " remote_procsets " << remote_procsets.size() << std::endl;
      std::vector<VariantID> variants;
      runtime->find_valid_variants(ctx, task.task_id, variants);

      output.slices.resize(input.domain.get_volume());
      unsigned idx = 0;
      Rect<1> rect = input.domain;
      for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
      {
        Rect<1> slice(*pir, *pir);
        output.slices[idx] = TaskSlice(slice,
          remote_gpus[idx % remote_gpus.size()],
          false/*recurse*/, false/*stealable*/);
      }

      default_slice_task(task, local_gpus, remote_gpus,
                       input, output, gpu_slices_cache);

    }

#endif


#endif

  }; // namespace Mapping 
}; // namespace Legion
     
    
   
     //=============================================================================
     // MAPPER REGISTRATION
     //=============================================================================

     static void replaceMapper(Legion::Machine machine,
                           Legion::Runtime* rt,
                           const std::set<Legion::Processor>& local_procs)
     {
       Legion::Mapping::MapperRuntime* mrt = rt->get_mapper_runtime();
       for (Legion::Processor proc : local_procs) {
         Legion::Mapping::EriRegentMapper* mapper = 
           new Legion::Mapping::EriRegentMapper(mrt, machine, proc);
#if 1
         Legion::Mapping::LoggingWrapper* wrapper =
           new Legion::Mapping::LoggingWrapper(mapper);
         rt->replace_default_mapper(wrapper, proc);
#else
         rt->replace_default_mapper(mapper, proc);
#endif
       }
     }
     

#ifdef __cplusplus
extern "C" {
#endif

void register_mappers() {
  Legion::Runtime::add_registration_callback(replaceMapper);
}

#ifdef __cplusplus
}
#endif



