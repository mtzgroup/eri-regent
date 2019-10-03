import "regent"

require "helper"
require "fields"
require "mcmurchie.generate_R_table"
require "generate_spin_pattern"

local rsqrt = regentlib.rsqrt(double)
local LToStr = {[0]="SS", [1]="SP", [2]="PP", [3]="PD", [4]="DD", [5]="DF", [6]="FF", [7]="FG", [8]="GG"}

-- Given a pair of angular momentums, this returns a task
-- to compute electron repulsion integrals between BraKets
-- using the McMurchie algorithm.
function generateTaskMcMurchieJFockIntegral(L12, L34)
  local H12 = computeH(L12)
  local H34 = computeH(L34)
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


  -- Returns a list of regent statements that implements the McMurchie algorithm
  local function generateJFockKernelStatements(r_jkets, jket_idx, accumulator)
    local density = rexpr r_jkets[jket_idx].density end

    local statements = terralib.newlist()
    local results = {}
    for i = 0, H12-1 do -- inclusive
      results[i] = regentlib.newsymbol(double, "result"..i)
      statements:insert(rquote var [results[i]] = 0.0 end)
    end

    local pattern12 = generateSpinPattern(L12)
    local pattern34 = generateSpinPattern(L34)
    for u = 0, H34-1 do -- inclusive
      for t = 0, H12-1 do -- inclusive
        local Nt, Lt, Mt = unpack(pattern12[t+1])
        local Nu, Lu, Mu = unpack(pattern34[u+1])
        local N, L, M = Nt + Nu, Lt + Lu, Mt + Mu
        if (Nu + Lu + Mu) % 2 == 0 then
          statements:insert(rquote
            [results[t]] += density[u] * [R[N][L][M][0]]
          end)
        else
          statements:insert(rquote
            [results[t]] -= density[u] * [R[N][L][M][0]]
          end)
        end
      end
    end
    for i = 0, H12-1 do -- inclusive
      statements:insert(rquote
        accumulator[i] += [results[i]]
      end)
    end
    return statements
  end

  local
  __demand(__leaf)
  __demand(__cuda)
  task jfock_integral(r_jbras         : region(ispace(int1d), Gaussian),
                      r_jkets         : region(ispace(int1d), getGaussianWithDensity(L34)),
                      r_kernel_output : region(ispace(int1d), double[H12]),
                      r_gamma_table   : region(ispace(int2d), double[5]))
  where
    reads(r_jbras, r_jkets, r_gamma_table),
    reduces +(r_kernel_output)
  do
    var jket_idx_bounds_lo : int = r_jkets.ispace.bounds.lo
    var jket_idx_bounds_hi : int = r_jkets.ispace.bounds.hi
    for jbra_idx in r_jbras.ispace do
      var jbra_x : double = r_jbras[jbra_idx].x
      var jbra_y : double = r_jbras[jbra_idx].y
      var jbra_z : double = r_jbras[jbra_idx].z
      var jbra_eta : double = r_jbras[jbra_idx].eta
      var jbra_C : double = r_jbras[jbra_idx].C
      var jbra_bound : double = r_jbras[jbra_idx].bound
      -- TODO: `accumulator` should be shared
      var accumulator : double[H12]
      for i = 0, H12 do -- exclusive
        accumulator[i] = 0.0
      end

      for jket_idx = jket_idx_bounds_lo, jket_idx_bounds_hi + 1 do -- exclusive
        var jket_x : double = r_jkets[jket_idx].x
        var jket_y : double = r_jkets[jket_idx].y
        var jket_z : double = r_jkets[jket_idx].z
        var jket_eta : double = r_jkets[jket_idx].eta
        var jket_C : double = r_jkets[jket_idx].C
        var jket_bound : double = r_jkets[jket_idx].bound

        -- TODO: Use Gaussian.bound to filter useless loops
        var a : double = jbra_x - jket_x
        var b : double = jbra_y - jket_y
        var c : double = jbra_z - jket_z

        var lambda : double = jbra_C * jket_C * rsqrt(jbra_eta + jket_eta)
        var alpha : double = jbra_eta * jket_eta / (jbra_eta + jket_eta)
        var t : double = alpha * (a*a + b*b + c*c);
        [generateStatementsComputeRTable(R, L12+L34+1, t, alpha, lambda, a, b, c, r_gamma_table)];

        [generateJFockKernelStatements(r_jkets, jket_idx, accumulator)]
      end

      r_kernel_output[jbra_idx] += accumulator
    end
  end
  jfock_integral:set_name("JFockMcMurchie"..LToStr[L12]..LToStr[L34])
  return jfock_integral
end
