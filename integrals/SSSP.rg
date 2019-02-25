-- Assumes fspaces and `computeR000` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

local computeR000 = generateTaskComputeR000(4)

__demand(__cuda)
task coulombSSSP(r_bra_kets    : region(PrimitiveBraKet),
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
    var R000 : double[4] = computeR000(t, alpha, r_boys)

    var R1000 = a * R000[1]
    var R0100 = b * R000[1]
    var R0010 = c * R000[1]

    var P0 = r_density[ket.data_rect.lo]
    var P1 = r_density[ket.data_rect.lo + 1]
    var P2 = r_density[ket.data_rect.lo + 2]
    var P3 = r_density[ket.data_rect.lo + 3]

    r_j_values[bra_ket.bra_idx] += P0 * R000[0] - P1 * R1000
                                                - P2 * R0100
                                                - P3 * R0010
  end
end
