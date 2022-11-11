import "regent"

require "helper"
require "mcmurchie.kfock.generate_kfock_integral"

local c = regentlib.c -- TODO: delete this when done printing for debugging

function kfock(r_pairs_list, r_prevals_list, r_labels_list, r_density_list, r_output_list,
               r_gamma_table, threshold, kguard, parallelism, largest_momentum, reverse)
  local statements = terralib.newlist()
  local L_start, L_end, L_stride
  if reverse == 1 then
    L_start = getCompiledMaxMomentum()
    L_end = 0
    L_stride = -1
  else
    L_start = 0
    L_end =  getCompiledMaxMomentum()
    L_stride = 1
  end
  
  for L1 = L_start, L_end, L_stride do -- inclusive
    for L2 = L_start, L_end, L_stride do -- inclusive
      for L3 = L_start, L_end, L_stride do -- inclusive
        for L4 = L_start, L_end, L_stride do -- inclusive
          if L1 < L3 or (L1 == L3 and L2 <= L4) then

            local r_bras = r_pairs_list[L1][L2]
            local r_kets = r_pairs_list[L3][L4]
            local r_bra_prevals = r_prevals_list[L1][L2][1]
            local r_ket_prevals = r_prevals_list[L3][L4][2]
            local r_bra_labels = r_labels_list[L1][L2]
            local r_ket_labels = r_labels_list[L3][L4]
            local r_density
            if L2 <= L4 then
              r_density = r_density_list[L2][L4]
            else
              r_density = r_density_list[L4][L2]
            end
            local r_output = r_output_list[L1][L3]

            -- support variable number of partitions
            -- for S/P orbitals
            -- TODO: extend this to > P orbitals
            --local p = parallelism[15]
            --if L1 <= 1 and L2 <=1 and L3 <=1 and L4 <= 1 then
            --  local pindex = L4*1 + L3*2 + L2*4 + L1*8
            --  p = parallelism[pindex]
            --end
            local p = parallelism -- version for top_kfock.rg

            local k_max = 0
            -- Break up large kernels (e.g. PDPD, DDDD) into separate tasks on 3rd index (k) to decrease compile time
            if (L1 > 0 and L2 > 0 and L3 > 0 and L4 > 0) and (L1 + L2 + L3 + L4 >= 6) then 
              k_max = triangle_number(L3 + 1) - 1
            end
            for k = 0, k_max do -- inclusive
              local kfock_integral = generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4, k)
              if r_bras ~= nil and r_kets ~= nil then
                statements:insert(rquote
                  --fill(r_output.values, [terralib.constant(`arrayof(double, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))]) -- number of comma-sep 0.0's should be size of values 

                  -- set up block loop for bra/ket iShells in a CUDA-friendly way
                  var BSIZEX = 8 -- size of GPU thread block in x dim (ket)
                  var BSIZEY = 8 -- size of GPU thread block in y dim (bra)
                  var gsizex = r_ket_labels[r_ket_labels.ispace.bounds.hi].ishell + 1 -- size of grid in x dim (number of iShells in ket)
                  var gsizey = r_bra_labels[r_bra_labels.ispace.bounds.hi].ishell + 1 -- size of grid in y dim (number of iShells in bra)
                  var gpuparam = region(ispace(int4d, {BSIZEX, BSIZEY, gsizex, gsizey}), int)  -- field type of this region doesn't matter

                  -- TODO: If region is empty, then don't launch a task
                  -- partition output and gpuparam along bra dimension (y dimension)
                  var output_coloring   = ispace(int3d, {1, p, 1})
                  var gpuparam_coloring = ispace(int4d, {1, 1, 1, p})
                  var p_output   = partition(equal, r_output, output_coloring)
                  var p_gpuparam = partition(equal, gpuparam, gpuparam_coloring)
                  __demand(__index_launch)
                  for i = 0, p do -- exclusive
                    kfock_integral(r_bras,        r_kets,
                                   r_bra_prevals, r_ket_prevals,
                                   r_bra_labels,  r_ket_labels,
                                   r_density,     p_output[{0, i, 0}],
                                   r_gamma_table, p_gpuparam[{0, 0, 0, i}].ispace, 
                                   threshold, 1, kguard, 
                                   largest_momentum, BSIZEX, BSIZEY)
                  end
                end)
              end
            end

          end
        end
      end
    end
  end
  return statements
end
