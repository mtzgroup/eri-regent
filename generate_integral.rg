import "regent"
require("fields")
require("boys")

local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = regentlib.sqrt(double)

local computeR000 = {}
for L = 0, 4 do
  computeR000[L] = generateTaskComputeR000(L + 1)
end

function generateRExpression(N, L, M, j, a, b, c, R000)
  local e
  if N == 0 and L == 0 and M == 0 then
    e = rexpr R000[j] end
  elseif N == 0 and L == 0 then
    e = rexpr
      c * [generateRExpression(0, 0, M-1, j+1, a, b, c, R000)]
    end
    if M ~= 1 then
      e = rexpr
        e + (M-1) * [generateRExpression(0, 0, M-2, j+1, a, b, c, R000)]
      end
    end
  elseif N == 0 then
    e = rexpr
      b * [generateRExpression(0, L-1, M, j+1, a, b, c, R000)]
    end
    if L ~= 1 then
      e = rexpr
        e + (L-1) * [generateRExpression(0, L-2, M, j+1, a, b, c, R000)]
      end
    end
  else
    e = rexpr
      a * [generateRExpression(N-1, L, M, j+1, a, b, c, R000)]
    end
    if N ~= 1 then
      e = rexpr
        e + (N-1) * [generateRExpression(N-2, L, M, j+1, a, b, c, R000)]
      end
    end
  end
  return e
end

function generateTaskCoulombIntegral(L12, L34, computeJValues)
  local L = L12 + L34
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  local
  __demand(__leaf)
  __demand(__cuda)
  task integral(r_bra_gausses : region(ispace(int1d), HermiteGaussian),
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
        var R000 : double[L + 1] = __demand(__inline, [computeR000[L]](t, alpha, r_boys))

        var P : double[H34]
        for i = 0, H34 do
          P[i] = r_density[ket.data_rect.lo + i].value
        end

        var result : double[H12] = __demand(__inline, computeJValues(R000, P, a, b, c))

        -- TODO: Precompute `lambda`
        var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                        * sqrt(bra.eta + ket.eta))
        for i = 0, H12 do
          r_j_values[bra.data_rect.lo + i].value += lambda * result[i]
        end
      end
    end
  end
  return integral
end
