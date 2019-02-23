-- Assumes fspaces and `computeR000` have been previously declared.
import "regent"

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)

local computeR000 = generateTaskComputeR000(1)

__demand(__cuda)
task coulombSSSS(r_bra_kets    : region(PrimitiveBraKet),
                 r_bra_gausses : region(ispace(int1d), HermiteGaussian),
                 r_ket_gausses : region(ispace(int1d), HermiteGaussian),
                 r_density     : region(ispace(int1d), double),
                 r_j_values    : region(ispace(int1d), double))
where
  reads(r_bra_kets, r_bra_gausses, r_ket_gausses, r_density),
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
    var R000 : double[1] = computeR000(t, alpha)

    -- TODO: Precompute parts of `lambda`
    var lambda : double = 2 * sqrt(pow(M_PI, 5)) / (bra.eta * ket.eta
                                                    * sqrt(bra.eta + ket.eta))

    r_j_values[bra_ket.bra_idx] += lambda * R000[0] * r_density[ket.data_rect.lo]
  end
end
