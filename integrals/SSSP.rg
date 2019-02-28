-- Assumes fspaces and `computeR000` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

local computeR000 = generateTaskComputeR000(2)

__demand(__cuda)
task coulombSSSP(r_bra_kets    : region(ispace(int1d), PrimitiveBraKet),
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
    var R000 : double[2] = __demand(__inline, computeR000(t, alpha, r_boys))

    var R1000 : double = a * R000[1]
    var R0100 : double = b * R000[1]
    var R0010 : double = c * R000[1]

    var P0 : double = r_density[ket.data_rect.lo]
    var P1 : double = r_density[ket.data_rect.lo + 1]
    var P2 : double = r_density[ket.data_rect.lo + 2]
    var P3 : double = r_density[ket.data_rect.lo + 3]

    -- TODO: Precompute parts of `lambda`
    var lambda : double = 2 * sqrt(pow(M_PI, 5)) / (bra.eta * ket.eta
                                                    * sqrt(bra.eta + ket.eta))
    var result : double
    result = R000[0] * P0
    result -= R1000 * P1
    result -= R0100 * P2
    result -= R0010 * P3
    result *= lambda
    r_j_values[bra.data_rect.lo] += result
  end
end
