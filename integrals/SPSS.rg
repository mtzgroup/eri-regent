-- Assumes fspaces and `computeR0001` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

__demand(__leaf)
__demand(__cuda)
task coulombSPSS(r_bra_kets    : region(ispace(int1d), PrimitiveBraKet),
                 r_bra_gausses : region(ispace(int1d), HermiteGaussian),
                 r_ket_gausses : region(ispace(int1d), HermiteGaussian),
                 r_density     : region(ispace(int1d), Double),
                 r_j_values    : region(ispace(int1d), Double),
                 r_boys        : region(ispace(int2d), Double))
where
  reads(r_bra_kets, r_bra_gausses, r_ket_gausses, r_density, r_boys),
  reduces +(r_j_values),
  r_density * r_j_values,
  r_density * r_boys,
  r_j_values * r_boys
do
  for bra_ket in r_bra_kets do
    var bra = r_bra_gausses[bra_ket.bra_idx]
    var ket = r_ket_gausses[bra_ket.ket_idx]
    -- TODO: Use Gaussian.bound to filter useless loops
    var a : double = bra.x - ket.x
    var b : double = bra.y - ket.y
    var c : double = bra.z - ket.z

    var alpha : double = bra.eta * ket.eta / (bra.eta + ket.eta)
    var t : double = alpha * (a*a+b*b+c*c)
    var R000 : double[2] = __demand(__inline, computeR0001(t, alpha, r_boys))

    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var P0 : double = r_density[ket.data_rect.lo].value

    -- TODO: Precompute parts of `lambda`
    var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                    * sqrt(bra.eta + ket.eta))

    r_j_values[bra.data_rect.lo].value += lambda * R000[0] * P0
    r_j_values[bra.data_rect.lo + 1].value += lambda * R1000 * P0
    r_j_values[bra.data_rect.lo + 2].value += lambda * R0100 * P0
    r_j_values[bra.data_rect.lo + 3].value += lambda * R0010 * P0
  end
end
