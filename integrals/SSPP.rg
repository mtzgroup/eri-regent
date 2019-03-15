-- Assumes fspaces and `computeR0002` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

__demand(__leaf)
__demand(__cuda)
task coulombSSPP(r_bra_kets    : region(ispace(int1d), PrimitiveBraKet),
                 r_bra_gausses : region(ispace(int1d), HermiteGaussian),
                 r_ket_gausses : region(ispace(int1d), HermiteGaussian),
                 r_density     : region(ispace(int1d), double),
                 r_j_values    : region(ispace(int1d), double),
                 r_boys        : region(ispace(int2d), PrecomputedBoys))
where
  reads(r_bra_kets, r_bra_gausses, r_ket_gausses, r_density, r_boys),
  reduces +(r_j_values)
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
    var R000 : double[3] = __demand(__inline, computeR0002(t, alpha, r_boys))

    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var R1001 : double = a * R000[2]
    var R0101 : double = b * R000[2]
    var R0011 : double = c * R000[2]

    var R1100 : double = a * R0101
    var R1010 : double = a * R0011
    var R0110 : double = b * R0011

    var R2000 : double = a * R1001 + R000[1]
    var R0200 : double = b * R0101 + R000[1]
    var R0020 : double = c * R0011 + R000[1]

    var P0 : double = r_density[ket.data_rect.lo]
    var P1 : double = r_density[ket.data_rect.lo + 1]
    var P2 : double = r_density[ket.data_rect.lo + 2]
    var P3 : double = r_density[ket.data_rect.lo + 3]
    var P4 : double = r_density[ket.data_rect.lo + 4]
    var P5 : double = r_density[ket.data_rect.lo + 5]
    var P6 : double = r_density[ket.data_rect.lo + 6]
    var P7 : double = r_density[ket.data_rect.lo + 7]
    var P8 : double = r_density[ket.data_rect.lo + 8]
    var P9 : double = r_density[ket.data_rect.lo + 9]

    -- TODO: Precompute parts of `lambda`
    var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                    * sqrt(bra.eta + ket.eta))
    var result : double
    result = R000[0] * P0
    result -= R1000 * P1
    result -= R0100 * P2
    result -= R0010 * P3
    result += R1100 * P4
    result += R1010 * P5
    result += R0110 * P6
    result += R2000 * P7
    result += R0200 * P8
    result += R0020 * P9
    result *= lambda
    r_j_values[bra.data_rect.lo] += result
  end
end
