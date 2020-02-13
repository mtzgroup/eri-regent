import "regent"

require "fields"
require "helper"
require "mcmurchie.kfock.generate_kernel"
require "mcmurchie.generate_R_table"

local rsqrt = regentlib.rsqrt(double)

local _kfock_integral_cache = {}
function generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4)
  local L_string = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]
  if _kfock_integral_cache[L_string] ~= nil then
    return _kfock_integral_cache[L_string]
  end

  local L12, L34 = L1 + L2, L3 + L4
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
  task kfock_integral(r_bras        : region(ispace(int1d), getKFockPair(L1, L2)),
                      r_kets        : region(ispace(int1d), getKFockPair(L3, L4)),
                      r_density     : region(ispace(int2d), getKFockDensity(L2, L4)),
                      r_output      : region(ispace(int3d), getKFockOutput(L1, L3)),
                      r_gamma_table : region(ispace(int2d), double[5]),
                      threshold : float, threshold2 : float, kguard : float, largest_momentum : int)
  where
    reads(r_bras, r_kets, r_density, r_gamma_table),
    reads writes(r_output)
  do
    var N24 = L2 + L4 * (largest_momentum + 1)
    var ket_idx_bounds_lo : int = r_kets.ispace.bounds.lo
    var ket_idx_bounds_hi : int = r_kets.ispace.bounds.hi
    for bra_idx in r_bras.ispace do
      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do -- exclusive
        var bra = r_bras[bra_idx]
        var ket = r_kets[ket_idx]
        var density : getKFockDensity(L2, L4)
        if L2 <= L4 then
          density = r_density[{bra.jshell_index, ket.jshell_index}]
        else
          density = r_density[{ket.jshell_index, bra.jshell_index}]
        end

        -- TODO: Figure out which threshold to use
        -- TODO: There is another bound to compute
        -- if bra.bound * ket.bound <= threshold then break end

        var a = bra.location.x - ket.location.x
        var b = bra.location.y - ket.location.y
        var c = bra.location.z - ket.location.z

        var alpha = bra.eta * ket.eta * (1.0 / (bra.eta + ket.eta))
        var lambda = bra.C * ket.C * rsqrt(bra.eta + ket.eta)
        var t = alpha * (a*a + b*b + c*c)
        ;[generateStatementsComputeRTable(R, L1+L2+L3+L4+1, t, alpha, lambda,
                                          a, b, c, r_gamma_table)]
        ;[generateKFockKernelStatements(
          R, L1, L2, L3, L4, bra, ket,
          rexpr density.values end,
          rexpr r_output[{N24, bra.ishell_index, ket.ishell_index}].values end
        )]
      end
    end
  end
  kfock_integral:set_name("KFockMcMurchie"..L_string)
  _kfock_integral_cache[L_string] = kfock_integral
  return kfock_integral
end
