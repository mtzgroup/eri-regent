import "regent"
require("fields")
require("boys")

local cmath = terralib.includec("math.h")
local sqrt = regentlib.sqrt(double)

local computeR000 = {}
for L = 0, 4 do
  computeR000[L] = generateTaskComputeR000(L + 1)
end

function generateTaskCoulombIntegral(L12, L34)
  local L = L12 + L34
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  local PI_5_2 = math.pow(math.pi, 2.5)

  local function generateRExpression(N, L, M, a, b, c, R000)
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

  local function generateSumStatements(a, b, c, R000, lambda,
                                       r_j_values, j_offset,
                                       r_density, d_offset)
    local result = regentlib.newsymbol(double[H12], "result")
    local P = regentlib.newsymbol(double[H34], "P")
    local statements = terralib.newlist({rquote
      var [result]
      var [P]
    end})
    for i = 0, H12-1 do --inclusive
      statements:insert(rquote
        result[i] = 0
      end)
    end
    for i = 0, H34-1 do --inclusive
      statements:insert(rquote
        P[i] = r_density[d_offset + i].value
      end)
    end
    for u = 0, H34-1 do --inclusive
      for t = 0, H12-1 do -- inclusive
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
        local N = pattern[t + 1][1] + pattern[u + 1][1]
        local L = pattern[t + 1][2] + pattern[u + 1][2]
        local M = pattern[t + 1][3] + pattern[u + 1][3]
        local sign
        -- FIXME: I don't understand when `sign` is negative
        if u == 0 or u > 3 then
          sign = 1
        else
          sign = -1
        end

        statements:insert(rquote
          result[t] += sign * P[u] * [generateRExpression(N, L, M, a, b, c, R000)]
        end)
      end
    end
    for i = 0, H12-1 do -- inclusive
      statements:insert(rquote
        r_j_values[j_offset + i].value += lambda * result[i]
      end)
    end
    return statements
  end

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
        var lambda : double = 2.0 * PI_5_2 / (bra.eta * ket.eta * sqrt(bra.eta + ket.eta))

        var j_offset = bra.data_rect.lo
        var d_offset = ket.data_rect.lo
        ;[generateSumStatements(a, b, c, R000, lambda,
                                r_j_values, j_offset,
                                r_density, d_offset)];
      end
    end
  end
  return integral
end
