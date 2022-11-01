import "regent"

require "helper"
require "mcmurchie.kfock.generate_kfock_integral"

local c = regentlib.c -- KGJ: delete this

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

            -- KGJ: break up kernels into separate tasks on 3rd (k) index to decrease compile time
            local k_max = 0
            if L1 >= 2 or L2 >= 2 or L3 >= 2 or L4 >=2 then
              k_max = triangle_number(L3+1)-1
            end
            if L1 == 0 and L2 == 0 and L3 == 0 and L4 == 2 then -- SSSD exception
              k_max = 0
            end
            if L1 == 0 and L2 == 0 and L3 == 2 and L4 == 0 then -- SSDS exception
              k_max = 0
            end
            -- support variable number of partitions
            -- for S/P orbitals
            -- TODO: extend this to > P orbitals
            local p = parallelism -- version for top_kfock.rg
            --local p = parallelism[15]
            --if L1 <= 1 and L2 <=1 and L3 <=1 and L4 <= 1 then
            --  local pindex = L4*1 + L3*2 + L2*4 + L1*8
            --  p = parallelism[pindex]
            --end
            for k = 0, k_max do -- inclusive
              local kfock_integral = generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4, k)
              if r_bras ~= nil and r_kets ~= nil then
                statements:insert(rquote
                  -- TODO: If region is empty, then don't launch a task
                  -- partition output along bra dimension (y dimension)
                  var output_coloring = ispace(int3d, {1, p, 1})
                  var p_output = partition(equal, r_output, output_coloring)
                  __demand(__index_launch)
                  for output_color in output_coloring do
                    kfock_integral(r_bras, r_kets,
                                   r_bra_prevals, r_ket_prevals,
                                   r_bra_labels, r_ket_labels,
                                   r_density, p_output[output_color],
                                   r_gamma_table, threshold, 1, kguard, largest_momentum)
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
