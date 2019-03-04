-- Assumes fspaces and `computeR000` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

local computeR000 = generateTaskComputeR000(3)

__demand(__cuda)
task coulombSPSP(r_bra_kets    : region(ispace(int1d), PrimitiveBraKet),
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
    var R000 : double[3] = __demand(__inline, computeR000(t, alpha, r_boys))

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

    -- TODO: Precompute parts of `lambda`
    var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                    * sqrt(bra.eta + ket.eta))
    var result : double[4]
    result[0] = R000[0] * P0
    result[1] = R1000 * P0
    result[2] = R0100 * P0
    result[3] = R0010 * P0

    result[0] -= R1000 * P1
    result[1] -= R2000 * P1
    result[2] -= R1100 * P1
    result[3] -= R1010 * P1

    result[0] -= R0100 * P2
    result[1] -= R1100 * P2
    result[2] -= R0200 * P2
    result[3] -= R0110 * P2

    result[0] -= R0010 * P3
    result[1] -= R1010 * P3
    result[2] -= R0110 * P3
    result[3] -= R0020 * P3

    result[0] *= lambda
    result[1] *= lambda
    result[2] *= lambda
    result[3] *= lambda

    r_j_values[bra.data_rect.lo] += result[0]
    r_j_values[bra.data_rect.lo + 1] += result[1]
    r_j_values[bra.data_rect.lo + 2] += result[2]
    r_j_values[bra.data_rect.lo + 3] += result[3]
  end
end
