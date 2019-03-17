import "regent"
require("fields")
require("boys")

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

__demand(__leaf)
__demand(__cuda)
task coulombPPSP(r_bra_gausses : region(ispace(int1d), HermiteGaussian),
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

      -- TODO: Precompute parts of `lambda`
      var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                      * sqrt(bra.eta + ket.eta))

      var result : double[10]
      result[0] = R000[0] * P0
      result[1] = R1000 * P0
      result[2] = R0100 * P0
      result[3] = R0010 * P0
      result[4] = R1100 * P0
      result[5] = R1010 * P0
      result[6] = R0110 * P0
      result[7] = R2000 * P0
      result[8] = R0200 * P0
      result[9] = R0020 * P0

      result[0] -= R1000 * P1
      result[1] -= R2000 * P1
      result[2] -= R1100 * P1
      result[3] -= R1010 * P1
      result[4] -= R2100 * P1
      result[5] -= R2010 * P1
      result[6] -= R1110 * P1
      result[7] -= R3000 * P1
      result[8] -= R1200 * P1
      result[8] -= R1020 * P1

      result[0] -= R0100 * P2
      result[1] -= R1100 * P2
      result[2] -= R0200 * P2
      result[3] -= R0110 * P2
      result[4] -= R1200 * P2
      result[5] -= R1110 * P2
      result[6] -= R0210 * P2
      result[7] -= R2100 * P2
      result[8] -= R0300 * P2
      result[9] -= R0120 * P2

      result[0] -= R0010 * P3
      result[1] -= R1010 * P3
      result[2] -= R0110 * P3
      result[3] -= R0020 * P3
      result[4] -= R1110 * P3
      result[5] -= R1020 * P3
      result[6] -= R0120 * P3
      result[7] -= R2010 * P3
      result[8] -= R0210 * P3
      result[9] -= R0030 * P3

      r_j_values[bra.data_rect.lo].value += lambda * result[0]
      r_j_values[bra.data_rect.lo + 1].value += lambda * result[1]
      r_j_values[bra.data_rect.lo + 2].value += lambda * result[2]
      r_j_values[bra.data_rect.lo + 3].value += lambda * result[3]
      r_j_values[bra.data_rect.lo + 4].value += lambda * result[4]
      r_j_values[bra.data_rect.lo + 5].value += lambda * result[5]
      r_j_values[bra.data_rect.lo + 6].value += lambda * result[6]
      r_j_values[bra.data_rect.lo + 7].value += lambda * result[7]
      r_j_values[bra.data_rect.lo + 8].value += lambda * result[8]
      r_j_values[bra.data_rect.lo + 9].value += lambda * result[9]
    end
  end
end
