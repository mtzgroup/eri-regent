import "regent"

require "fields"
require "helper"
require "mcmurchie.generate_R_table"
require "mcmurchie.jfock.generate_kernel_statements"

local rsqrt = regentlib.rsqrt(double)

-- Given a pair of angular momentums, this returns a task
-- to compute electron repulsion integrals between BraKets
-- using the McMurchie algorithm.
local _jfock_integral_cache = {}
function generateTaskMcMurchieJFockIntegral(L12, L34)
  local H12, H34 = tetrahedral_number(L12 + 1), tetrahedral_number(L34 + 1)
  local L_string = LPairToStr[L12]..LPairToStr[L34]
  if _jfock_integral_cache[L_string] ~= nil then
    return _jfock_integral_cache[L_string]
  end

  -- Create a table of Regent variables to hold Hermite polynomials.
  local R = {}
  for N = 0, L12+L34 do -- inclusive
    R[N] = {}
    for L = 0, L12+L34-N do -- inclusive
      R[N][L] = {}
      for M = 0, L12+L34-N-L do -- inclusive
        R[N][L][M] = {}
        for j = 0, L12+L34-N-L-M do -- inclusive
          R[N][L][M][j] = regentlib.newsymbol(double, "R"..N..L..M..j)
        end
      end
    end
  end

  local
  __demand(__leaf)
  __demand(__cuda)
  task jfock_integral(r_jbras       : region(ispace(int1d), getJBra(L12)),
                      r_jkets       : region(ispace(int1d), getJKet(L34)),
                      r_gamma_table : region(ispace(int2d), double[5]),
                      threshold     : float)
  where
    reads(r_jbras.{location, eta, C, bound}, r_jkets, r_gamma_table),
    reduces +(r_jbras.output)
  do
    var jket_idx_bounds_lo : int = r_jkets.ispace.bounds.lo
    var jket_idx_bounds_hi : int = r_jkets.ispace.bounds.hi
    for jbra_idx in r_jbras.ispace do
      var jbra_location : Point = r_jbras[jbra_idx].location
      var jbra_eta : double = r_jbras[jbra_idx].eta
      var jbra_C : double = r_jbras[jbra_idx].C
      var jbra_bound : float = r_jbras[jbra_idx].bound
      -- TODO: `accumulator` should be shared
      var accumulator : double[H12]
      for i = 0, H12 do -- exclusive
        accumulator[i] = 0.0
      end

      for jket_idx = jket_idx_bounds_lo, jket_idx_bounds_hi + 1 do -- exclusive
        var jket = r_jkets[jket_idx]

        var bound : float = jbra_bound * jket.bound
        if bound <= threshold then break end

        var a : double = jbra_location.x - jket.location.x
        var b : double = jbra_location.y - jket.location.y
        var c : double = jbra_location.z - jket.location.z

        var alpha : double = jbra_eta * jket.eta * (1.0 / (jbra_eta + jket.eta))
        var lambda : double = jbra_C * jket.C * rsqrt(jbra_eta + jket.eta)
        var t : double = alpha * (a*a + b*b + c*c);
        [generateStatementsComputeRTable(R, L12+L34+1, t, alpha, lambda,
                                         a, b, c, r_gamma_table)];

        [generateJFockKernelStatements(R, L12, L34, rexpr jket.density end,
                                       accumulator)]
      end

      r_jbras[jbra_idx].output += accumulator
    end
-- regentlib.c.printf(["exiting JFockMcMurchie" .. L_string .. "\n"])
  end
  jfock_integral:set_name("JFockMcMurchie"..L_string)
  _jfock_integral_cache[L_string] = jfock_integral
  return jfock_integral
end
