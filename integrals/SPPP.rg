-- Assumes fspaces and `computeR0003` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

__demand(__leaf)
__demand(__cuda)
task coulombSPPP(r_bra_gausses : region(ispace(int1d), HermiteGaussian),
                 r_ket_gausses : region(ispace(int1d), HermiteGaussian),
                 r_density     : region(ispace(int1d), Double),
                 r_j_values    : region(ispace(int1d), Double),
                 r_boys        : region(ispace(int2d), Double))
where
  reads(r_bra_gausses, r_ket_gausses, r_density, r_boys),
  reduces +(r_j_values),
  r_density * r_j_values,
  r_density * r_boys,
  r_j_values * r_boys
do
  for bra_idx in r_bra_gausses.ispace do
    for ket_idx in r_ket_gausses.ispace do
      var bra = r_bra_gausses[bra_idx]
      var ket = r_ket_gausses[ket_idx]
      -- TODO: Use Gaussian.bound to filter useless loops
      var a : double = bra.x - ket.x
      var b : double = bra.y - ket.y
      var c : double = bra.z - ket.z

      var alpha : double = bra.eta * ket.eta / (bra.eta + ket.eta)
      var t : double = alpha * (a*a+b*b+c*c)
      var R000 : double[4] = __demand(__inline, computeR0003(t, alpha, r_boys))

      var R1000 : double = a * R000[1]
      var R0100 : double = b * R000[1]
      var R0010 : double = c * R000[1]

      var R1001 : double = a * R000[2]
      var R0101 : double = b * R000[2]
      var R0011 : double = c * R000[2]

      var R1002 : double = a * R000[3]
      var R0102 : double = b * R000[3]
      var R0012 : double = c * R000[3]

      var R1100 : double = a * R0101
      var R1010 : double = a * R0011
      var R0110 : double = b * R0011

      var R1101 : double = a * R0102
      var R1011 : double = a * R0012
      var R0111 : double = b * R0012

      var R2000 : double = a * R1001 + R000[1]
      var R0200 : double = b * R0101 + R000[1]
      var R0020 : double = c * R0011 + R000[1]

      var R2001 : double = a * R1002 + R000[2]
      var R0201 : double = b * R0102 + R000[2]
      var R0021 : double = c * R0012 + R000[2]

      var R2100 : double = a * R1101 + R0101
      var R1200 : double = a * R0201
      var R1110 : double = a * R0111

      var R2010 : double = a * R1011 + R0011
      var R1020 : double = a * R0021

      var R0210 : double = b * R0111 + R0011
      var R0120 : double = b * R0021

      var R3000 : double = a * R2001 + R1001 + R1001
      var R0300 : double = b * R0201 + R0101 + R0101
      var R0030 : double = c * R0021 + R0011 + R0011

      var P0 : double = r_density[ket.data_rect.lo].value
      var P1 : double = r_density[ket.data_rect.lo + 1].value
      var P2 : double = r_density[ket.data_rect.lo + 2].value
      var P3 : double = r_density[ket.data_rect.lo + 3].value
      var P4 : double = r_density[ket.data_rect.lo + 4].value
      var P5 : double = r_density[ket.data_rect.lo + 5].value
      var P6 : double = r_density[ket.data_rect.lo + 6].value
      var P7 : double = r_density[ket.data_rect.lo + 7].value
      var P8 : double = r_density[ket.data_rect.lo + 8].value
      var P9 : double = r_density[ket.data_rect.lo + 9].value

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

      result[0] += R1100 * P4
      result[1] += R2100 * P4
      result[2] += R1200 * P4
      result[3] += R1110 * P4

      result[0] += R1010 * P5
      result[1] += R2010 * P5
      result[2] += R1110 * P5
      result[3] += R1020 * P5

      result[0] += R0110 * P6
      result[1] += R1110 * P6
      result[2] += R0210 * P6
      result[3] += R0120 * P6

      result[0] += R2000 * P7
      result[1] += R3000 * P7
      result[2] += R2100 * P7
      result[3] += R2010 * P7

      result[0] += R0200 * P8
      result[1] += R1200 * P8
      result[2] += R0300 * P8
      result[3] += R0210 * P8

      result[0] += R0020 * P9
      result[1] += R1020 * P9
      result[2] += R0120 * P9
      result[3] += R0030 * P9

      result[0] *= lambda
      result[1] *= lambda
      result[2] *= lambda
      result[3] *= lambda

      r_j_values[bra.data_rect.lo].value += result[0]
      r_j_values[bra.data_rect.lo + 1].value += result[1]
      r_j_values[bra.data_rect.lo + 2].value += result[2]
      r_j_values[bra.data_rect.lo + 3].value += result[3]
    end
  end
end
