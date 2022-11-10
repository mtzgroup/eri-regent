import "regent"

require "fields"
require "helper"
require "mcmurchie.kfock.generate_kernel"
require "mcmurchie.generate_R_table"

local rsqrt = regentlib.rsqrt(double)

local _kfock_integral_cache = {}
function generateTaskMcMurchieKFockIntegral(L1, L2, L3, L4, k_idx)
  local NL1 = triangle_number(L1 + 1)
  local NL3 = triangle_number(L3 + 1)
  local Nout = NL1 * NL3
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
  task kfock_integral(r_bras           : region(ispace(int1d), getKFockPair(L1, L2)),
                      r_kets           : region(ispace(int1d), getKFockPair(L3, L4)),
                      r_bra_prevals    : region(ispace(int2d), double),
                      r_ket_prevals    : region(ispace(int2d), double),
                      r_bra_labels     : region(ispace(int1d), getKFockLabel(L1, L2)),
                      r_ket_labels     : region(ispace(int1d), getKFockLabel(L3, L4)),
                      r_density        : region(ispace(int2d), getKFockDensity(L2, L4)),
                      r_output         : region(ispace(int3d), getKFockOutput(L1, L3)),
                      r_gamma_table    : region(ispace(int2d), double[5]),
                      gpuparam         : ispace(int4d),
                      threshold        : float, threshold2 : float, kguard : float, 
                      largest_momentum : int, BSIZEX : int, BSIZEY : int)
  where
    reads(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, r_density, r_gamma_table),
    reads writes(r_output.values)
  do

    -- this loop mimics CUDA threading, looping over iShell blocks (one GPU thread block per iShell block)
    -- then looping over threads
    for thread in gpuparam do
      var thidx = thread.x -- threads in block, 0-7
      var thidy = thread.y -- threads in block, 0-7
      var blidx = thread.z -- ket shell index, size is number of iShells for ket
      var blidy = thread.w -- bra shell index, size is number of iShells for bra

      var continue = true
      -- diagonal kernels (e.g. DPSP or SPSP) can skip the lower triangle of blocks
      if (L1 == L3 and L2 == L4 and blidx < blidy) then continue = false end 
      if continue then

        -- determine bra/ket ranges within iShell block from label regions
        var sizex = r_ket_labels[blidx].end_index - r_ket_labels[blidx].start_index 
        var sizey = r_bra_labels[blidy].end_index - r_bra_labels[blidy].start_index 

        var g_thidy = r_bra_labels[blidy].start_index + thidy
        var s_thidx = r_ket_labels[blidx].start_index + thidx
        var g_thidx = s_thidx

        var stopx = r_ket_labels[blidx].start_index + sizex 
        var stopy = r_bra_labels[blidy].start_index + sizey 

        -- local accumulator for iShell block
        var accumulator : double[Nout] 
        for n = 0, Nout do -- exclusive
          accumulator[n] = 0.0
        end

        repeat
          g_thidx = s_thidx
          var bra_idx = r_bras.ispace.bounds.lo + g_thidy 
          var bra = r_bras[bra_idx]
          var ket_idx = r_kets.ispace.bounds.lo + g_thidx 
          var ket = r_kets[ket_idx]
          var bound : float = bra.bound * ket.bound

          -- TODO: remove control flow for optimization?
          while bound * kguard > threshold do -- regular bound (bra loop)

            var density : getKFockDensity(L2, L4)
            if L2 <= L4 then
              density = r_density[{bra.jshell_index, ket.jshell_index}]
            else
              density = r_density[{ket.jshell_index, bra.jshell_index}]
            end

            if bound * density.bound > threshold then -- density-weighted bound (ket loop)
              ket_idx = r_kets.ispace.bounds.lo + g_thidx 
              ket = r_kets[ket_idx]

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

            g_thidx += BSIZEX
            ket_idx = r_kets.ispace.bounds.lo + g_thidx 
            ket = r_kets[ket_idx]
            bound = bra.bound * ket.bound
          end -- end ket loop

          g_thidy += BSIZEY
        until (s_thidx == g_thidx) -- end bra loop

        -- Scale elements in diag. kernels out here instead of inside generate_kernel (faster)
        if (L1 == L3 and L2 == L4 and blidx == blidy) then -- L1 == L3
          for i = 0, NL1 do -- exclusive
            for k = 0, NL3 do -- exclusive
              if i == k then -- diag. elements of diag. kernels scale output by 1/2
                accumulator[i*NL3+k] = accumulator[i*NL3+k] * 0.5
              elseif k < i then -- lower triangle elements of diag. kernels are 0
                accumulator[i*NL3+k] = 0.0
              end
            end
          end
        end

        var N24 = L2 + L4 * (largest_momentum + 1)
        r_output[{N24, blidy, blidx}].values += accumulator

      end -- end upper triangle block check for diag. kernels
    end -- end gpuparam

  end
  kfock_integral:set_name("KFockMcMurchie"..L_string)
  _kfock_integral_cache[L_string] = kfock_integral
  return kfock_integral
end
