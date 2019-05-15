import "regent"
require "fields"
require "mcmurchie.generate_R_table"
require "generate_spin_pattern"

local sqrt = regentlib.sqrt(double)
local LToStr = {[0]="SS", [1]="SP", [2]="PP", [3]="PD", [4]="DD", [5]="DF", [6]="FF", [7]="FG", [8]="GG"}


-- Returns a list of regent statements that implements the McMurchie algorithm
local function generateKernelStatements(L12, L34, a, b, c, R,
                                        r_density, d_offset, accumulators)
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6

  local statements = terralib.newlist()
  local results, P = {}, {}
  for i = 0, H12-1 do --inclusive
    results[i] = regentlib.newsymbol(double, "result"..i)
    statements:insert(rquote var [results[i]] = 0.0 end)
  end
  for i = 0, H34-1 do --inclusive
    P[i] = regentlib.newsymbol(double, "P"..i)
    statements:insert(rquote
      var [P[i]] = r_density[d_offset + i].value
    end)
  end

  local pattern12 = generateSpinPattern(L12)
  local pattern34 = generateSpinPattern(L34)
  for u = 0, H34-1 do -- inclusive
    for t = 0, H12-1 do -- inclusive
      local Nt, Lt, Mt = unpack(pattern12[t+1])
      local Nu, Lu, Mu = unpack(pattern34[u+1])
      local N, L, M = Nt + Nu, Lt + Lu, Mt + Mu
      if (Nu + Lu + Mu) % 2 == 0 then
        statements:insert(rquote
          [results[t]] += [P[u]] * [R[N][L][M][0]]
        end)
      else
        statements:insert(rquote
          [results[t]] -= [P[u]] * [R[N][L][M][0]]
        end)
      end
    end
  end
  for i = 0, H12-1 do -- inclusive
    statements:insert(rquote
      [accumulators[i]] += [results[i]]
    end)
  end
  return statements
end


-- Given a pair of angular momentums, this returns a task
-- to compute electron repulsion integrals between BraKets
-- using the McMurchie algorithm.
function generateTaskMcMurchieIntegral(L12, L34)
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  -- Create a table of Regent variables to hold Hermite polynomials.
  local R = {}
  for N = 0, L12+L34 do -- inclusive
    R[N] = {}
    for L = 0, L12+L34-N do -- inclusive
      R[N][L] = {}
      for M = 0, L12+L34-N-L do -- inclusive
        R[N][L][M] = {}
        for j = 0, L12+L34-N-L-M do -- inclusive
          R[N][L][M][j] = regentlib.newsymbol(double, "R"..N..L..M..j)
        end
      end
    end
  end

  local accumulators = {}
  local initializeAccumulatorStatements = terralib.newlist()
  for i = 0, H12-1 do -- inclusive
    accumulators[i] = regentlib.newsymbol(double, "accumulator"..i)
    initializeAccumulatorStatements:insert(rquote
      var [accumulators[i]] = 0.0
    end)
  end


  local function addAccumulators(r_j_values, offset)
    local statements = terralib.newlist()
    for i = 0, H12-1 do -- inclusive
      statements:insert(rquote
        r_j_values[offset + i].value += [accumulators[i]]
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

      var bra = r_bra_gausses[bra_idx];
      [initializeAccumulatorStatements]

      -- FIXME: This region is assumed to be contiguous.
      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do
        var ket = r_ket_gausses[ket_idx]
        -- TODO: Use Gaussian.bound to filter useless loops
        var a : double = bra.x - ket.x
        var b : double = bra.y - ket.y
        var c : double = bra.z - ket.z

        var alpha : double = bra.eta * ket.eta / (bra.eta + ket.eta)
        var t : double = alpha * (a*a + b*b + c*c)
        var lambda : double = bra.C * ket.C / sqrt(bra.eta + ket.eta);
        [generateStatementsComputeRTable(R, L12+L34+1, t, alpha, lambda, r_boys, a, b, c)]

        var offset = ket.data_rect.lo;
        [generateKernelStatements(L12, L34, a, b, c, R, r_density, offset, accumulators)]
      end

      var offset = bra.data_rect.lo;
      [addAccumulators(r_j_values, offset)]
    end
  end
  integral:set_name("McMurchie"..LToStr[L12]..LToStr[L34])
  return integral
end
