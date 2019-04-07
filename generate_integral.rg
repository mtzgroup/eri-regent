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

function generateRExpression(N, L, M, a, b, c, R000)
  assert(N >= 0 and L >= 0 and M >= 0)
  local function aux(N, L, M, j)
    if N == 0 and L == 0 and M == 0 then
      return rexpr R000[j] end
    elseif N == 0 and L == 0 then
      if M == 1 then
        return rexpr c * [aux(0, 0, 0, j+1)] end
      end
      return rexpr c * [aux(0, 0, M-1, j+1)] + (M-1) * [aux(0, 0, M-2, j+1)] end
    elseif N == 0 then
      if L == 1 then
        return rexpr b * [aux(0, 0, M, j+1)] end
      end
      return rexpr b * [aux(0, L-1, M, j+1)] + (L-1) * [aux(0, L-2, M, j+1)] end
    else
      if N == 1 then
        return rexpr a * [aux(0, L, M, j+1)] end
      end
      return rexpr a * [aux(N-1, L, M, j+1)] + (N-1) * [aux(N-2, L, M, j+1)] end
    end
  end
  return aux(N, L, M, 0)
end

function generateJValueStatements(H12, H34, a, b, c, R000, lambda, r_j_values, bra_lo, r_density, ket_lo)

  local function generateExpression(L12, L34)
    -- NOTE: This is based on the format of the input data from TeraChem
    local pattern = {
      {0; 0; 0;};
      {1; 0; 0;};
      {0; 1; 0;};
      {0; 0; 1;};
      {1; 1; 0;};
      {1; 0; 1;};
      {0; 1; 1;};
      {2; 0; 0;};
      {0; 2; 0;};
      {0; 0; 2;};
    }
    local N = pattern[L12 + 1][1] + pattern[L34 + 1][1]
    local L = pattern[L12 + 1][2] + pattern[L34 + 1][2]
    local M = pattern[L12 + 1][3] + pattern[L34 + 1][3]
    local sign
    -- FIXME: I don't understand when `sign` is negative
    if L34 == 0 or L34 > 3 then
      sign = rexpr 1 end
    else
      sign = rexpr -1 end
    end
    return rexpr sign * [generateRExpression(N, L, M, a, b, c, R000)] end
  end

  local statements = terralib.newlist()
  for ket_idx = 0, H34-1 do
    for bra_idx = 0, H12-1 do
      statements:insert(rquote
        r_j_values[bra_lo + bra_idx].value += (lambda * [generateExpression(bra_idx, ket_idx)]
                                                * r_density[ket_lo + ket_idx].value)
      end)
    end
  end
  return statements
end

function generateTaskCoulombIntegral(L12, L34)
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

        -- TODO: Precompute `lambda`
        var lambda : double = 2.0*M_PI*M_PI*sqrt(M_PI) / (bra.eta * ket.eta
                                                        * sqrt(bra.eta + ket.eta))

        var bra_lo = bra.data_rect.lo
        var ket_lo = ket.data_rect.lo
        ;[generateJValueStatements(H12, H34, a, b, c, R000, lambda, r_j_values, bra_lo, r_density, ket_lo)];
      end
    end
  end
  return integral
end
