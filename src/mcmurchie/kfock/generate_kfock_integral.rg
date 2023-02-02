import "regent"

require "fields"
require "helper"
require "mcmurchie.kfock.generate_kernel"
require "mcmurchie.generate_R_table"

local rsqrt = regentlib.rsqrt(double)

local _kfock_integral_cache = {}
function generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4, k_idx)

  local NL1 = triangle_number(L1 + 1)
  local NL3 = triangle_number(L3 + 1)
  local Nout = NL1 * NL3
  local L_string = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]..k_idx
  if _kfock_integral_cache[L_string] ~= nil then
    return _kfock_integral_cache[L_string]
  end
  --------------------------- Regent Integral Task ----------------------------
  local
  extern
  --__demand(__leaf) 
  --__demand(__cuda) -- NOTE: comment out if printing from kernels (debugging)
  task kfock_integral(r_bras           : region(ispace(int1d), getKFockPair(L1, L2)),
                      r_kets           : region(ispace(int1d), getKFockPair(L3, L4)),
                      r_bra_prevals    : region(ispace(int2d), double),
                      r_ket_prevals    : region(ispace(int2d), double),
                      r_bra_labels     : region(ispace(int1d), getKFockLabel(L1, L2)),
                      r_ket_labels     : region(ispace(int1d), getKFockLabel(L3, L4)),
                      r_density        : region(ispace(int2d), getKFockDensity(L2, L4)),
                      r_output         : region(ispace(int3d), getKFockOutput(L1, L3)),
                      r_gamma_table    : region(ispace(int2d), double[5]),
                      --gpuparam         : ispace(int4d),
                      gpuparam         : region(ispace(int4d), int),
                      threshold        : float, threshold2 : float, kguard : float, 
                      largest_momentum : int, BSIZEX : int, BSIZEY : int)
  where
    reads(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, r_density, r_gamma_table),
    reads writes(r_output.values)
  end
  -----------------------------------------------------------------------------

  kfock_integral:set_name("KFockMcMurchie"..L_string)
  _kfock_integral_cache[L_string] = kfock_integral
  return kfock_integral

end -- end function generateTaskMcMurchieKFockIntegral
