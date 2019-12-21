import "regent"

require "helper"
require "mcmurchie.kfock.generate_kfock_integral"

function kfock(r_pairs_list, r_density_list, r_gamma_table, threshold, parallelism)
  local statements = terralib.newlist()
  -- TODO: Reverse the launch order so that large kernels launch first
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = 0, getCompiledMaxMomentum() do -- inclusive
      for L3 = 0, getCompiledMaxMomentum() do -- inclusive
        for L4 = 0, getCompiledMaxMomentum() do -- inclusive
          local r_bras = r_pairs_list[L1][L2]
          local r_kets = r_pairs_list[L3][L4]
          local kfock_integral = generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4)
          if r_bras ~= nil and r_kets ~= nil then
            statements:insert(rquote
              -- TODO: If region is empty, then don't launch a task
              var bra_coloring = ispace(int1d, parallelism)
              var p_bras = partition(equal, r_bras, bra_coloring)
              assert(false, "Unimplemented")
              -- __demand(__index_launch)
              -- for bra_color in bra_coloring do
              --   -- TODO
              --   kfock_integral(p_bras[bra_color], r_kets, r_gamma_table, threshold)
              -- end
            end)
          end
        end
      end
    end
  end
  return statements
end
