import "regent"

require "helper"
require "fields"
require "mcmurchie.generate_jfock_integral"
-- require "rys.generate_integral"

function jfock(r_jbras_list, r_jkets_list,
               r_gamma_table, r_output,
               parameters, parallelism)
  -- local spin_pattern_init, r_spin_pattern = unpack(generateSpinPatternRegion(max_momentum))
  local statements = terralib.newlist({rquote
    -- TODO: Spin pattern region should not be initialized here.
    -- [spin_pattern_init]
  end})
  for L12 = 0, getCompiledMaxMomentum() do -- inclusive
    for L34 = 0, getCompiledMaxMomentum() do -- inclusive
      local r_jbras = r_jbras_list[L12]
      local r_jkets = r_jkets_list[L34]
      local use_mcmurchie = true
      if use_mcmurchie then
        local jfock_integral = generateTaskMcMurchieJFockIntegral(L12, L34)
        statements:insert(rquote
          var jbra_coloring = ispace(int1d, parallelism)
          var p_jbras = partition(equal, r_jbras, jbra_coloring)
          __demand(__index_launch)
          for jbra_color in jbra_coloring do
            jfock_integral(p_jbras[jbra_color], r_jkets, r_output, r_gamma_table)
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

-- -- Given two arrays of regions, return a task that computes the jfock operator
-- function jfock(r_jbras_list, r_jkets_list)
--
--   -- Compute the jfock operator and add the result to `r_output`
--   -- `r_output` should be zero'd before this is called
--   local
--   task jfock_task(r_gamma_table : region(ispace(int2d), double[5]),
--                   r_output      : region(ispace(int1d), double),
--                   parameters    : Parameters,
--                   parallelism   : int)
--   where
--     reads(r_gamma_table),
--     reduces +(r_output)
--   do
--     [dispatchIntegrals(r_jbras_list, r_jkets_list,
--                        r_gamma_table, r_output, parameters, parallelism)]
--   end
--   jfock_task:set_name("jfock_task")
--   return jfock_task
-- end
