import "regent"
require "fields"
require "mcmurchie.generate_R_table"
require "generate_spin_pattern"

local rsqrt = regentlib.rsqrt(double)
local LToStr = {[0]="SS", [1]="SP", [2]="PP", [3]="PD", [4]="DD", [5]="DF", [6]="FF", [7]="FG", [8]="GG"}


-- Returns a list of regent statements that implements the McMurchie algorithm
local function generateKernelStatements(L12, L34, a, b, c, R, r_ket_gausses, ket_idx, accumulator)
  local H12 = (L12 + 1) * (L12 + 2) * (L12 + 3) / 6
  local H34 = (L34 + 1) * (L34 + 2) * (L34 + 3) / 6
  local density = rexpr r_ket_gausses[ket_idx].density end

  local statements = terralib.newlist()
  local results = {}
  for i = 0, H12-1 do --inclusive
    results[i] = regentlib.newsymbol(double, "result"..i)
    statements:insert(rquote var [results[i]] = 0.0 end)
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
          [results[t]] += density[u] * [R[N][L][M][0]]
        end)
      else
        statements:insert(rquote
          [results[t]] -= density[u] * [R[N][L][M][0]]
        end)
      end
    end
  end
  for i = 0, H12-1 do -- inclusive
    statements:insert(rquote
      accumulator[i] += [results[i]]
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


  local
  __demand(__leaf)
  __demand(__cuda)
  task integral(r_bra_gausses : region(ispace(int1d), getHermiteGaussianPacked(L12)),
                r_ket_gausses : region(ispace(int1d), getHermiteGaussianPacked(L34)),
                r_gamma_table : region(ispace(int2d), double[5]))
  where
    reads(r_bra_gausses.{x, y, z, eta, C, bound}),
    reads(r_ket_gausses.{x, y, z, eta, C, density, bound}),
    reads(r_gamma_table),
    reduces +(r_bra_gausses.j)
  do
    var ket_idx_bounds_lo : int = r_ket_gausses.ispace.bounds.lo
    var ket_idx_bounds_hi : int = r_ket_gausses.ispace.bounds.hi
    for bra_idx in r_bra_gausses.ispace do

      var bra_x : double = r_bra_gausses[bra_idx].x
      var bra_y : double = r_bra_gausses[bra_idx].y
      var bra_z : double = r_bra_gausses[bra_idx].z
      var bra_eta : double = r_bra_gausses[bra_idx].eta
      var bra_C : double = r_bra_gausses[bra_idx].C
      var bra_bound : double = r_bra_gausses[bra_idx].bound
      var accumulator : double[H12]
      for i = 0, H12 do -- exclusive
        accumulator[i] = 0.0
      end

      -- FIXME: This region is assumed to be contiguous.
      for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi + 1 do -- exclusive
        var ket_x : double = r_ket_gausses[ket_idx].x
        var ket_y : double = r_ket_gausses[ket_idx].y
        var ket_z : double = r_ket_gausses[ket_idx].z
        var ket_eta : double = r_ket_gausses[ket_idx].eta
        var ket_C : double = r_ket_gausses[ket_idx].C
        var ket_bound : double = r_ket_gausses[ket_idx].bound

        -- TODO: Use Gaussian.bound to filter useless loops
        var a : double = bra_x - ket_x
        var b : double = bra_y - ket_y
        var c : double = bra_z - ket_z

        var rsqrt_etas : double = rsqrt(bra_eta + ket_eta)
        var lambda : double = bra_C * ket_C * rsqrt_etas
        var alpha : double = bra_eta * ket_eta * rsqrt_etas * rsqrt_etas
        var t : double = alpha * (a*a + b*b + c*c);
        [generateStatementsComputeRTable(R, L12+L34+1, t, alpha, lambda, a, b, c, r_gamma_table)];

        [generateKernelStatements(L12, L34, a, b, c, R, r_ket_gausses, ket_idx, accumulator)]
      end

      r_bra_gausses[bra_idx].j += accumulator
    end
  end
  integral:set_name("McMurchie"..LToStr[L12]..LToStr[L34])
  return integral
end
