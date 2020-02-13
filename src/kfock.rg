import "regent"

require "helper"
require "mcmurchie.kfock.generate_kfock_integral"

function kfock(r_pairs_list, r_density_list, r_output_list,
               r_gamma_table, threshold, parallelism, largest_momentum)
  local statements = terralib.newlist()
  -- TODO: Partition output.
  -- TODO: Reverse the launch order so that large kernels launch first
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = 0, getCompiledMaxMomentum() do -- inclusive
      for L3 = 0, getCompiledMaxMomentum() do -- inclusive
        for L4 = 0, getCompiledMaxMomentum() do -- inclusive
          if L1 < L3 or (L1 == L3 and L2 <= L4) then
            local r_bras, r_kets = r_pairs_list[L1][L2], r_pairs_list[L3][L4]
            local r_density
            if L2 <= L4 then
              r_density = r_density_list[L2][L4]
            else
              r_density = r_density_list[L4][L2]
            end
            local r_output = r_output_list[L1][L3]
            local kfock_integral = generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4)
            if r_bras ~= nil and r_kets ~= nil then
              statements:insert(rquote
                -- TODO: If region is empty, then don't launch a task
                var bra_coloring = ispace(int1d, parallelism)
                var p_bras = partition(equal, r_bras, bra_coloring)
                -- TODO: Partition other regions, too
                -- __demand(__index_launch)
                for bra_color in bra_coloring do
                  kfock_integral(p_bras[bra_color], r_kets,
                                 r_density, r_output,
                                 r_gamma_table, threshold, 1, 1, largest_momentum)
                end
              end)
            end
          end
        end
      end
    end
  end
  return statements
end
