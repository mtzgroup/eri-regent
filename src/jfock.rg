import "regent"

require "helper"
require "mcmurchie.jfock.generate_jfock_integral"

local c = regentlib.c

function jfock(r_jbras_list, r_jkets_list, r_gamma_table, threshold, parallelism)
  local statements = terralib.newlist()
  -- TODO: Reverse the launch order so that large kernels launch first
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    -- We generate one extra kernel so that we can compute the gradient.
    for L2 = L1, getCompiledMaxMomentum() + 1 do -- inclusive
      for L3 = 0, getCompiledMaxMomentum() do -- inclusive
        for L4 = L3, getCompiledMaxMomentum() do -- inclusive
          local r_jbras = r_jbras_list[L1][L2]
          local r_jkets = r_jkets_list[L3][L4]
          local L12, L34 = L1 + L2, L3 + L4
          local jfock_integral = generateTaskMcMurchieJFockIntegral(L12, L34)
            statements:insert(rquote
c.printf("jfock_integral L1 %d L2 %d L3 %d L4 %d\n", L1, L2, L3, L4);
end)

          if r_jbras ~= nil and r_jkets ~= nil then
            statements:insert(rquote
              -- TODO: If region is empty, then don't launch a task
              var jbra_coloring = ispace(int1d, parallelism)
              var p_jbras = partition(equal, r_jbras, jbra_coloring)
c.printf("launching jfock_integral\n");
              __demand(__index_launch)
              for jbra_color in jbra_coloring do
                jfock_integral(p_jbras[jbra_color], r_jkets, r_gamma_table, threshold)
              end
            end)
statements:insert(rquote c.printf("A\n"); end)
          end
statements:insert(rquote c.printf("B\n"); end)
        end
statements:insert(rquote c.printf("C\n"); end)
      end
statements:insert(rquote c.printf("D\n"); end)
    end
statements:insert(rquote c.printf("E\n"); end)
  end
  statements:insert(rquote c.printf("exiting jfock\n"); end)
  return statements
end
