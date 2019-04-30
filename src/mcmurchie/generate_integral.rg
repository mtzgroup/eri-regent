import "regent"
require "fields"
require "mcmurchie.boys"
require "generate_spin_pattern"

local LToStr = {[0]="SS", [1]="SP", [2]="PP", [3]="PD", [4]="DD", [5]="FD", [6]="FF"}
local customKernels = require "mcmurchie.kernels.import"

local sqrt = regentlib.sqrt(double)

-- Given a pair of angular momentums, this returns a task
-- to compute electron repulsion integrals between BraKets
-- using the McMurchie algorithm.
function generateTaskMcMurchieIntegral(L12, L34)
  local L = L12 + L34
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  local PI_5_2 = math.pow(math.pi, 2.5)

  -- Returns an expression to recursively compute Hermite polynomials given by
  -- R00MJ = c * R00(M-1)(J+1) + (M-1) * R00(M-2)(J+1)
  -- R0LMJ = b * R0(L-1)M(J+1) + (L-1) * R0(L-2)M(J+1)
  -- RNLMJ = a * R(N-1)LM(J+1) + (N-1) * R(N-2)LM(J+1)
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

  -- Returns a list of regent statements that implements the McMurchie algorithm
  local function generateKernel(a, b, c, R000, lambda, r_j_values, j_offset, r_density, d_offset)
    local customKernel = customKernels[LToStr[L12]..LToStr[L34]]
    if customKernel ~= nil then
      return customKernel(a, b, c, R000, lambda, r_j_values, j_offset, r_density, d_offset)
    end

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
    local pattern = generateSpinPattern(math.max(L12, L34))
    for u = 0, H34-1 do --inclusive
      for t = 0, H12-1 do -- inclusive
        local N = pattern[t + 1][1] + pattern[u + 1][1]
        local L = pattern[t + 1][2] + pattern[u + 1][2]
        local M = pattern[t + 1][3] + pattern[u + 1][3]
        if (pattern[u + 1][1] + pattern[u + 1][2] + pattern[u + 1][3]) % 2 == 0 then
          statements:insert(rquote
            result[t] += P[u] * [generateRExpression(N, L, M, a, b, c, R000)]
          end)
        else
          statements:insert(rquote
            result[t] -= P[u] * [generateRExpression(N, L, M, a, b, c, R000)]
          end)
        end
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
                r_boys        : region(ispace(int2d), double))
  where
    reads(r_bra_gausses, r_ket_gausses, r_density, r_boys),
    reduces +(r_j_values),
    r_density * r_j_values
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
        var R000 : double[L + 1] = __demand(__inline,
          [generateTaskComputeR000(L+1)](t, alpha, r_boys)
        )

        var lambda : double = 2.0 * PI_5_2 / (bra.eta * ket.eta * sqrt(bra.eta + ket.eta))

        var j_offset = bra.data_rect.lo
        var d_offset = ket.data_rect.lo
        ;[generateKernel(a, b, c, R000, lambda, r_j_values, j_offset, r_density, d_offset)];
      end
    end
  end
  integral:set_name("McMurchie"..LToStr[L12]..LToStr[L34])
  return integral
end
