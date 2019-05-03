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
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  local PI_5_2 = math.pow(math.pi, 2.5)
  local R000 = {}

  -- Returns an expression to recursively compute Hermite polynomials given by
  -- R00MJ = c * R00(M-1)(J+1) + (M-1) * R00(M-2)(J+1)
  -- R0LMJ = b * R0(L-1)M(J+1) + (L-1) * R0(L-2)M(J+1)
  -- RNLMJ = a * R(N-1)LM(J+1) + (N-1) * R(N-2)LM(J+1)
  local function generateRExpression(N, L, M, a, b, c)
    assert(N >= 0 and L >= 0 and M >= 0)
    local function aux(N, L, M, j)
      if N == 0 and L == 0 and M == 0 then
        return R000[j]
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

  -- Returns an expression to compute a table of Hermite polynomials given by
  -- R00MJ = c * R00(M-1)(J+1) + (M-1) * R00(M-2)(J+1)
  -- R0LMJ = b * R0(L-1)M(J+1) + (L-1) * R0(L-2)M(J+1)
  -- RNLMJ = a * R(N-1)LM(J+1) + (N-1) * R(N-2)LM(J+1)
  local function generateRTable(a, b, c, statements)
    local levels = L12 + L34 + 1
    -- Create a 4D Lua table. Once populated this will be dim = levels^4
    -- TODO: rewrite for 0-indexed
    -- need to initialize otherwise Lua complains that values are 'nil'
    local R = {}
    for i = 1,levels do
      R[i] = {}
      for j=1,levels do
        R[i][j] = {}
        for k=1,levels do
          R[i][j][k] = {}
          for l=1,levels do
            R[i][j][k][l] = regentlib.newsymbol(double, "R"..i..j..k..l)
          end
        end
      end
    end 
    -- add base cases to j dimension of R table
    for j = 0, levels-1 do -- inclusive
      statements:insert(rquote var [R[1][1][1][j+1]] = [R000[j]] end)
    end
    -- loop through number of recursive steps needed
    for s = 1, levels-1 do -- inclusive
      for j = 0, levels-s-1 do -- inclusive

        -- Use first recursion formula to move down in N dim (skipped if s=1)
        for N = 1, s-1 do -- inclusive
          for L = 0, s-N do -- inclusive  
            local M = s-N-L
            -- use if b/c Lua doesn't realize you don't access [N-1] when N=1
            if N > 1 then
              statements:insert(rquote var [R[N+1][L+1][M+1][j+1]] = a * [R[N][L+1][M+1][j+2]] + (N-1) * [R[N-1][L+1][M+1][j+2]] end)
            else 
              statements:insert(rquote var [R[N+1][L+1][M+1][j+1]] = a * [R[N][L+1][M+1][j+2]] end)
            end
          end
        end

        -- Use second recursion formula to move down in L dim (skipped if s=1)
        for L = 1, s-1 do -- inclusive
          local M = s-L
          -- use if b/c Lua doesn't realize you don't access [L-1] when L=1
          if L > 1 then
            statements:insert(rquote var [R[1][L+1][M+1][j+1]] = b * [R[1][L][M+1][j+2]] + (L-1) * [R[1][L-1][M+1][j+2]] end)
          else
            statements:insert(rquote var [R[1][L+1][M+1][j+1]] = b * [R[1][L][M+1][j+2]] end)
          end
        end

        -- for all s
        -- use if b/c Lua doesn't realize you don't access [s-1] when s=1
        if s > 1 then
          statements:insert(rquote var [R[s+1][1][1][j+1]] = a * [R[s][1][1][j+2]] + (s-1) * [R[s-1][1][1][j+2]] end)
          statements:insert(rquote var [R[1][s+1][1][j+1]] = b * [R[1][s][1][j+2]] + (s-1) * [R[1][s-1][1][j+2]] end) 
          statements:insert(rquote var [R[1][1][s+1][j+1]] = c * [R[1][1][s][j+2]] + (s-1) * [R[1][1][s-1][j+2]] end)
        else
          statements:insert(rquote var [R[s+1][1][1][j+1]] = a * [R[s][1][1][j+2]] end)
          statements:insert(rquote var [R[1][s+1][1][j+1]] = b * [R[1][s][1][j+2]] end)
          statements:insert(rquote var [R[1][1][s+1][j+1]] = c * [R[1][1][s][j+2]] end)
        end

      end
    end

    return R
  end

  -- Returns a list of regent statements that implements the McMurchie algorithm
  local function generateKernel(a, b, c, lambda, r_j_values, j_offset, r_density, d_offset)
    local customKernel = customKernels[LToStr[L12]..LToStr[L34]]
    if customKernel ~= nil then
      return customKernel(a, b, c, R000, lambda, r_j_values, j_offset, r_density, d_offset)
    end

    local statements = terralib.newlist()
    local result, P = {}, {}
    for i = 0, H12-1 do --inclusive
      result[i] = regentlib.newsymbol(double, "result"..i)
      statements:insert(rquote
        var [result[i]] = 0.0
      end)
    end
    for i = 0, H34-1 do --inclusive
      P[i] = regentlib.newsymbol(double, "P"..i)
      statements:insert(rquote
        var [P[i]] = r_density[d_offset + i].value
      end)
    end

    local R = generateRTable(a, b, c, statements)

    local pattern12 = generateSpinPattern(L12)
    local pattern34 = generateSpinPattern(L34)
    for u = 0, H34-1 do --inclusive
      for t = 0, H12-1 do -- inclusive
        local Nt, Lt, Mt = unpack(pattern12[t+1])
        local Nu, Lu, Mu = unpack(pattern34[u+1])
        local N, L, M = Nt + Nu, Lt + Lu, Mt + Mu
        if (Nu + Lu + Mu) % 2 == 0 then
          statements:insert(rquote
            --[result[t]] += [P[u]] * [generateRExpression(N, L, M, a, b, c)]
            [result[t]] += [P[u]] * [R[N+1][L+1][M+1][1]]
          end)
        else
          statements:insert(rquote
            --[result[t]] -= [P[u]] * [generateRExpression(N, L, M, a, b, c)]
            [result[t]] -= [P[u]] * [R[N+1][L+1][M+1][1]]
          end)
        end
      end
    end
    for i = 0, H12-1 do -- inclusive
      statements:insert(rquote
        r_j_values[j_offset + i].value += lambda * [result[i]]
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
        var t : double = alpha * (a*a + b*b + c*c);
        [generateStatementsComputeR000(R000, L12+L34+1, t, alpha, r_boys)]

        var lambda : double = 2.0 * PI_5_2 / (bra.eta * ket.eta * sqrt(bra.eta + ket.eta))

        var j_offset = bra.data_rect.lo
        var d_offset = ket.data_rect.lo;
        [generateKernel(a, b, c, lambda, r_j_values, j_offset, r_density, d_offset)]
      end
    end
  end
  integral:set_name("McMurchie"..LToStr[L12]..LToStr[L34])
  return integral
end
