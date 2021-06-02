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
    std::atomic<size_t> EriRegentMapper::kdir = 0;
    std::atomic<size_t> EriRegentMapper::jdir = 0;

    //--------------------------------------------------------------------------
    EriRegentMapper::EriRegentMapper(MapperRuntime *rt, Machine m, Processor local)
      : DefaultMapper(rt, m, local), machine_interface(Utilities::MachineQueryInterface(m))
    //--------------------------------------------------------------------------
    {
      //      std::cout << __FUNCTION__ << " constructor" << std::endl;
      //      std::cout << "local_gpus " << local_gpus.size() << std::endl;
      kdir=jdir=0;
    }

    //--------------------------------------------------------------------------
    EriRegentMapper::~EriRegentMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void EriRegentMapper::slice_task(const MapperContext      ctx,
                                     const Task&              task,
                                     const SliceTaskInput&      input,
                                     SliceTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
      log_eri_regent_mapper.debug("Slice task in eri regent mapper for task %s "
                                  "(ID %lld)", task.get_task_name(),
                                  task.task_id);
      int kfock_mcmurchie_task = strncmp(task.get_task_name(), "KFock", 5);
      int jfock_mcmurchie_task = strncmp(task.get_task_name(), "JFock", 5);
      // 1 - kfock_mc.., jfock_mc.. tasks
      // 2 - gpu target
      Processor::Kind target_kind = task.target_proc.kind();
      bool custom_slice = (kfock_mcmurchie_task == 0) || (jfock_mcmurchie_task == 0);
      bool rotation=true;
      unsigned int dir = 0;

      if (kfock_mcmurchie_task == 0)  dir = EriRegentMapper::kdir++;
      if (jfock_mcmurchie_task == 0)  dir = EriRegentMapper::jdir++;

      if (custom_slice && (target_kind == Processor::TOC_PROC))
        {
          slice_domain(task, input.domain, output.slices, rotation /*rotation */, dir);
          log_eri_regent_mapper.debug() << "custom slice task : " << task.get_task_name()
                                        << " slices = " <<  output.slices.size() << "\n";
        }
      else
        DefaultMapper::slice_task(ctx, task, input, output);

      log_eri_regent_mapper.debug("num slices for task %s (ID %lld)  = %lld", task.get_task_name(),
                                  task.task_id, output.slices.size());
    }

    //--------------------------------------------------------------------------
    void EriRegentMapper::slice_domain(const Task& task,
                                       const Domain &domain,
                                       std::vector<DomainSplit>
                                       &slices,
                                       bool rotation,
                                       unsigned int dir)
    //--------------------------------------------------------------------------
    {
      log_eri_regent_mapper.debug("Slice index space in eri regent mapper for task %s "
                                  "(ID %lld)", task.get_task_name(),
                                  task.get_unique_id());

      Processor::Kind target_kind = task.target_proc.kind();
      std::set<Processor> all_procs;
      if (cached_procs.size() == 0)
        {
          machine.get_all_processors(all_procs);
          machine_interface.filter_processors(machine, target_kind, all_procs);
          std::vector<Processor> procs(all_procs.begin(),all_procs.end());
          for (unsigned i=0; i<all_procs.size(); ++i)
            cached_procs.push_back(procs[i]);
        }
      std::map<TaskID,std::vector<TaskSlice> >::const_iterator finder =
        cached_slices.find(task.task_id);
      if (finder != cached_slices.end()) {
        slices = finder->second;
        log_eri_regent_mapper.debug("Found cached slice  for task %s "
                                    "(ID %lld)", task.get_task_name(),
                                    task.get_unique_id());
        return;
      }

      EriRegentMapper::decompose_index_space(domain, cached_procs,
                                             1 /* splitting factor */,
                                             slices, rotation, dir);
      // cache the slices
      cached_slices[task.task_id] = slices;
    }

    //--------------------------------------------------------------------------
    template <unsigned DIM>
    static void round_robin_point_assign(const Domain &domain, 
                                         const std::vector<Processor> &targets,
                                         unsigned splitting_factor, 
                                         std::vector<EriRegentMapper::DomainSplit>
                                         &slices)
    //--------------------------------------------------------------------------
    {
      Rect<DIM,coord_t> r = domain;

      std::vector<Processor>::const_iterator target_it = targets.begin();
      for(PointInRectIterator<DIM> pir(r); pir(); pir++) 
      {
        // rect containing a single point
        Rect<DIM> subrect(*pir, *pir);
        EriRegentMapper::DomainSplit ds(subrect, 
            *target_it++, false /* recurse */, false /* stealable */);
        slices.push_back(ds);
        if(target_it == targets.end())
          target_it = targets.begin();
      }
    }

    //--------------------------------------------------------------------------
    template <unsigned DIM>
    static void round_robin_point_assign_with_rotation(const Domain &domain, 
                                                       const std::vector<Processor>
                                                       &targets,
                                                       unsigned splitting_factor, 
                                                       std::vector<EriRegentMapper::DomainSplit>
                                                       &slices,
                                                       unsigned int dir)
    //--------------------------------------------------------------------------
    {
      bool forward_dir = true;
      Rect<DIM,coord_t> r = domain;
      unsigned int i = 0;
      if ((dir%2) == 0)
        {
          forward_dir = false;
          i = targets.size()-1;
        }
      log_eri_regent_mapper.debug() << "slice dir =  " << forward_dir << " i = " << i <<  " EriRegentMapper::dir = " << dir << "\n";

      for(PointInRectIterator<DIM> pir(r); pir(); pir++) 
        {
          // rect containing a single point
          Rect<DIM> subrect(*pir, *pir);
          EriRegentMapper::DomainSplit ds(subrect, 
                                          targets[i%targets.size()], 
                                          false /* recurse */,
                                          false /* stealable */);
          log_eri_regent_mapper.debug() << "point = " << *pir << " proc = " << targets[i%targets.size()] << "\n";
          if (forward_dir) ++i;
          slices.push_back(ds);
          if ((i%targets.size()) == 0) {
            if (forward_dir) {
              forward_dir=false;
              i = targets.size()-1;
            }
            else 
              forward_dir=true;
          }
          else if (!forward_dir)
            --i;
        }
    }
    //--------------------------------------------------------------------------
    /*static*/ void EriRegentMapper::decompose_index_space(const Domain &domain, 
                                                           const std::vector<Processor> &targets,
                                                           unsigned splitting_factor, 
                                                           std::vector<DomainSplit> &slices,
                                                           bool rotation,
                                                           unsigned int dir /* backward/forward */)
    //--------------------------------------------------------------------------
    {
      assert(domain.get_dim() == 1);
      if (!rotation)
        round_robin_point_assign<1>(domain, targets, splitting_factor, slices);
      else
        round_robin_point_assign_with_rotation<1>(domain, targets, splitting_factor, slices, dir);
    }

    
#if 0

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
      // std::cout << "remote gpus " << remote_gpus.size() << " local_gpus " << local_gpus.size() << " remote_procsets " << remote_procsets.size() << std::endl;
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
#if 0
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



