import "regent"

require "fields"
require "helper"
-- TODO: Move out of jfock
require "mcmurchie.jfock.generate_R_table"

local rsqrt = regentlib.rsqrt(double)

local _kfock_integral_cache = {}
function generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4)
  local L_string = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]
  if _kfock_integral_cache[L_string] ~= nil then
    return _kfock_integral_cache[L_string]
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
  task kfock_integral(r_bras        : region(ispace(int1d), getKFockPair(L1, L2)),
                      r_kets        : region(ispace(int1d), getKFockPair(L3, L4)),
                      r_density     : region(ispace(int2d), getKFockDensity(L1, L2)),
                      r_output      : region(ispace(int2d), getKFockOutput(L3, L4)),
                      r_gamma_table : region(ispace(int2d), double[5]),
                      threshold : float, threshold2 : float, kguard : float)
  where
    reads(r_bras, r_kets, r_density),
    reduces +(r_output)
  do
    var ket_idx_bounds_lo : int = r_kets.ispace.bounds.lo
    var ket_idx_bounds_hi : int = r_kets.ispace.bounds.hi
    for bra_idx in r_bras.ispace do
      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do -- exclusive
        var bra = r_bras[bra_idx]
        var ket = r_kets[ket_idx]

        var a = bra.location.x - ket.location.x
        var b = bra.location.y - ket.location.y
        var c = bra.location.z - ket.location.z

        var alpha = bra.eta * ket.eta * (1.0 / (bra.eta + ket.eta))
        var lambda = bra.C * ket.C * rsqrt(bra.eta + ket.eta)
        var t = alpha * (a*a + b*b + c*c);
        [generateStatementsComputeRTable(R, L1+L2+L3+L4+1, t, alpha, lambda,
                                         a, b, c, r_gamma_table)];
        -- TODO: Compute output
        assert(false, "Unimplemented")
      end
    end
  end
  kfock_integral:set_name("KFockMcMurchie"..L_string)
  _kfock_integral_cache[L_string] = kfock_integral
  return kfock_integral
end
