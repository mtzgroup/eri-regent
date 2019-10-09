import "regent"

require "helper"
require "mcmurchie.jfock.generate_jfock_integral"
-- require "rys.generate_integral"

function jfock(r_jbras_list, r_jkets_list, r_gamma_table, threshold, parallelism)
  -- local spin_pattern_init, r_spin_pattern = unpack(generateSpinPatternRegion(max_momentum))
  local statements = terralib.newlist({rquote
    -- TODO: Spin pattern region should not be initialized here.
    -- [spin_pattern_init]
  end})
  for L12 = 0, getCompiledMaxMomentum() do -- inclusive
    local r_jbras = r_jbras_list[L12]
    for L34 = 0, getCompiledMaxMomentum() do -- inclusive
      local r_jkets = r_jkets_list[L34]
      local use_mcmurchie = true
      if use_mcmurchie then
        local jfock_integral = generateTaskMcMurchieJFockIntegral(L12, L34)
        statements:insert(rquote
          var jbra_coloring = ispace(int1d, parallelism)
          var p_jbras = partition(equal, r_jbras, jbra_coloring)
          __demand(__index_launch)
          for jbra_color in jbra_coloring do
            jfock_integral(p_jbras[jbra_color], r_jkets, r_gamma_table, threshold)
          end
        end)
      else -- use Rys
        -- local jfock_integral = generateTaskRysJFockIntegral(L12, L34)
        statements:insert(rquote
          -- var bra_coloring = ispace(int1d, parallelism)
          -- var p_bra_gausses = partition(equal, r_bra_gausses, bra_coloring)
          -- __demand(__index_launch)
          -- for bra_color in bra_coloring do
          --   jfock_integral(p_bra_gausses[bra_color], r_ket_gausses, r_spin_pattern)
          -- end
        end)
      end
    end
  end
  return statements
end
