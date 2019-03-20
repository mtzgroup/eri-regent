import "regent"
require("fields")
require("boys")

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

__demand(__leaf)
__demand(__cuda)
task coulombSSSS(r_bra_gausses : region(ispace(int1d), HermiteGaussian),
                 r_ket_gausses : region(ispace(int1d), HermiteGaussian),
                 r_density     : region(ispace(int1d), Double),
                 r_j_values    : region(ispace(int1d), Double),
                 r_boys        : region(ispace(int1d), Double))
where
  reads(r_bra_gausses, r_ket_gausses, r_density, r_boys),
  reduces +(r_j_values),
  r_density * r_j_values,
  r_density * r_boys,
  r_j_values * r_boys
do
  var ket_idx_bounds_lo : int = r_ket_gausses.ispace.bounds.lo
  var ket_idx_bounds_hi : int = r_ket_gausses.ispace.bounds.hi
  for bra_idx in r_bra_gausses.ispace do
    for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do
      var bra = r_bra_gausses[bra_idx]
      var ket = r_ket_gausses[ket_idx]
      -- TODO: Use Gaussian.bound to filter useless loops
      var a : double = bra.x - ket.x
      var b : double = bra.y - ket.y
      var c : double = bra.z - ket.z

      var alpha : double = bra.eta * ket.eta / (bra.eta + ket.eta)
      var t : double = alpha * (a*a+b*b+c*c)
      var R000 : double[1] = __demand(__inline, computeR0000(t, alpha, r_boys))

      var P0 : double = r_density[ket.data_rect.lo].value

      -- TODO: Precompute parts of `lambda`
      var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                      * sqrt(bra.eta + ket.eta))

      r_j_values[bra.data_rect.lo].value += lambda * R000[0] * P0
    end
  end
end
