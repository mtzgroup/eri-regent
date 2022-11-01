import "regent"

require "fields"
require "helper"
require "mcmurchie.kfock.generate_kernel"
require "mcmurchie.generate_R_table"

local rsqrt = regentlib.rsqrt(double)

local _kfock_integral_cache = {}
function generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4, k_idx)
  local Nout = triangle_number(L1 + 1) * triangle_number(L3 + 1)
  local L_string = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]..k_idx
  if _kfock_integral_cache[L_string] ~= nil then
    return _kfock_integral_cache[L_string]
  end

  local L12, L34 = L1 + L2, L3 + L4
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
  __demand(__cuda) -- NOTE: comment out if printing from kernels (debugging)
  task kfock_integral(r_bras        : region(ispace(int1d), getKFockPair(L1, L2)),
                      r_kets        : region(ispace(int1d), getKFockPair(L3, L4)),
                      r_bra_prevals : region(ispace(int2d), double),
                      r_ket_prevals : region(ispace(int2d), double),
                      r_bra_labels  : region(ispace(int1d), getKFockLabel(L1, L2)),
                      r_ket_labels  : region(ispace(int1d), getKFockLabel(L3, L4)),
                      r_density     : region(ispace(int2d), getKFockDensity(L2, L4)),
                      r_output      : region(ispace(int3d), getKFockOutput(L1, L3)),
                      r_gamma_table : region(ispace(int2d), double[5]),
                      threshold : float, threshold2 : float, kguard : float, largest_momentum : int)
  where
    reads(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, r_density, r_gamma_table),
    reduces+(r_output.values)
    --writes(r_output.values)  -- TODO?
  do
    -- for bra label region, bounds determined by output partitioning
    var bra_label_idx_lo : int = r_output.ispace.bounds.lo.y -- y is second dim of r_output (bra_ishell)
    var bra_label_idx_hi : int = r_output.ispace.bounds.hi.y -- y is second dim of r_output (bra_ishell)

    -- Loop over iShells for bra and ket using label regions
    for bra_label_idx = bra_label_idx_lo, bra_label_idx_hi + 1 do -- exclusive
      for ket_label in r_ket_labels do
        var bra_label = r_bra_labels[bra_label_idx]

        -- local accumulator for iShell block
        var accumulator : double[Nout] 
        for i = 0, Nout do -- exclusive
          accumulator[i] = 0.0
        end

        -- determine bra/ket ranges within iShell block from label regions
        var bra_idx_bounds_lo : int = r_bras.ispace.bounds.lo + bra_label.start_index 
        var bra_idx_bounds_hi : int = r_bras.ispace.bounds.lo + bra_label.end_index
        var ket_idx_bounds_lo : int = r_kets.ispace.bounds.lo + ket_label.start_index 
        var ket_idx_bounds_hi : int = r_kets.ispace.bounds.lo + ket_label.end_index

        for bra_idx = bra_idx_bounds_lo, bra_idx_bounds_hi do -- exclusive
          for ket_idx = ket_idx_bounds_lo, ket_idx_bounds_hi do -- exclusive
            var bra = r_bras[bra_idx]
            var ket = r_kets[ket_idx]
            var density : getKFockDensity(L2, L4)
            if L2 <= L4 then
              density = r_density[{bra.jshell_index, ket.jshell_index}]
            else
              density = r_density[{ket.jshell_index, bra.jshell_index}]
            end
 
            -- TODO: remove control flow for optimization?
            var bound : float = bra.bound * ket.bound
            if bound * kguard <= threshold then break end -- regular bound
            if bound * density.bound <= threshold then break end -- density-weighted bound
 
            var a = bra.location.x - ket.location.x
            var b = bra.location.y - ket.location.y
            var c = bra.location.z - ket.location.z
 
            var alpha = bra.eta * ket.eta * (1.0 / (bra.eta + ket.eta))
            var lambda = bra.C * ket.C * rsqrt(bra.eta + ket.eta)
            var t = alpha * (a*a + b*b + c*c)
            ;[generateStatementsComputeRTable(R, L1+L2+L3+L4+1, t, alpha, lambda,
                                              a, b, c, r_gamma_table)]
            ;[generateKFockKernelStatements(
              R, L1, L2, L3, L4, k_idx, bra, ket, r_bra_prevals, r_ket_prevals, bra_idx, ket_idx,
              rexpr density.values end,
              accumulator
            )]
          end
        end
        var N24 = L2 + L4 * (largest_momentum + 1)
        r_output[{N24, bra_label.ishell, ket_label.ishell}].values += accumulator

      end -- end ket iShell 
    end -- end bra iShell

  end
  kfock_integral:set_name("KFockMcMurchie"..L_string)
  _kfock_integral_cache[L_string] = kfock_integral
  return kfock_integral
end
