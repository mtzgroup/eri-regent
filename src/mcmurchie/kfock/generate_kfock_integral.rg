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


  --------------------- Helper Metaprogramming Functions ----------------------

  -- Metaprogramming function to obtain correct density indexing
  local
  function generateDensityStatements(L2, L4, bra_j, ket_j, r_density, density)
    local statements = terralib.newlist()
    if L2 <= L4 then
      statements:insert(rquote
        density = r_density[{bra_j, ket_j}]
      end)
    else
      statements:insert(rquote
        density = r_density[{ket_j, bra_j}]
      end)
    end
    return statements
  end

  -- Metaprogramming function to generate bra/ket loops for kernels
  local
  function generateLoopStatements(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, 
                                  r_density, r_gamma_table, accumulator,
                                  thidx, thidy, blidx, blidy, BSIZEX, BSIZEY,
                                  threshold, kguard)

    local statements = terralib.newlist()
    statements:insert(rquote
 
      -- determine bra/ket ranges within iShell block from label regions
      var ket_start_index = r_ket_labels[blidx].start_index
      var bra_start_index = r_bra_labels[blidy].start_index

      var sizex = r_ket_labels[blidx].end_index - ket_start_index 
      var sizey = r_bra_labels[blidy].end_index - bra_start_index 
 
      var g_thidy = bra_start_index + thidy
      var s_thidx = ket_start_index + thidx
      var g_thidx = s_thidx
 
      var stopx = ket_start_index + sizex 
      var stopy = bra_start_index + sizey 
 
      repeat
        g_thidx = s_thidx
        var bra_idx = r_bras.ispace.bounds.lo + g_thidy 
        var bra = r_bras[bra_idx]

        var bra_jshell_index = bra.jshell_index
        var bra_bound = bra.bound
        var bra_location_x = bra.location.x
        var bra_location_y = bra.location.y
        var bra_location_z = bra.location.z
        var bra_eta = bra.eta
        var bra_C = bra.C

        var r_kets_lo = r_kets.ispace.bounds.lo
        var ket_idx = r_kets_lo + g_thidx 
        var ket = r_kets[ket_idx]
        var bound : float = bra.bound * ket.bound
 
        while bound * kguard > threshold do -- regular bound (bra loop)
 
          var density : getKFockDensity(L2, L4)
          ;[generateDensityStatements(L2, L4, rexpr bra_jshell_index end, rexpr ket.jshell_index end, 
                                      r_density, density)]
 
          if bound * density.bound > threshold then -- density-weighted bound (ket loop)
            ket_idx = r_kets_lo + g_thidx 
            ket = r_kets[ket_idx]
 
            var a = bra_location_x - ket.location.x
            var b = bra_location_y - ket.location.y
            var c = bra_location_z - ket.location.z
          
            var ket_eta = ket.eta
 
            var alpha = bra_eta * ket_eta * (1.0 / (bra_eta + ket_eta))
            var lambda = bra_C * ket.C * rsqrt(bra_eta + ket_eta)
            var t = alpha * (a*a + b*b + c*c)

            ;[generateStatementsComputeRTable(R, L1+L2+L3+L4+1, t, alpha, lambda,
                                              a, b, c, r_gamma_table)]

            ;[generateKFockKernelStatements(
              R, L1, L2, L3, L4, k_idx, bra, ket, r_bra_prevals, r_ket_prevals, bra_idx, ket_idx,
              rexpr density.values end,
              accumulator)]
          end
 
          g_thidx += BSIZEX
          ket_idx = r_kets_lo + g_thidx 
          ket = r_kets[ket_idx]
          bound = bra_bound * ket.bound
        end -- end ket loop
 
        g_thidy += BSIZEY
      until (s_thidx == g_thidx) -- end bra loop
 
    end)
    return statements
  end

  -- Metaprogramming function to generate different code statements for diagonal vs. off-diagonal kernels
  local
  function generateStatements(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, 
                              r_density, r_gamma_table, r_output, accumulator,
                              thidx, thidy, blidx, blidy, BSIZEX, BSIZEY,
                              threshold, kguard, largest_momentum)
 
    local statements = terralib.newlist()
 
    -- diagonal kernels (e.g. DPSP or SPSP) can skip the lower triangle of blocks
    if (L1 == L3 and L2 == L4) then
      statements:insert(rquote

        var pivot = r_ket_labels[r_ket_labels.ispace.bounds.hi].ishell 
        --var shift = pivot & 1 -- bitwise and, 1 if nShells is even, 0 if nShells is odd
        var shift : int
        if (int(pivot) % 2 == 1) then -- pivot odd
          shift = 1
        else -- pivot even
          shift = 0
        end
        blidx -= shift 
 
        -- pivot and shift the lower triangle to upper triangle (x --> x in diagram, 6x6 example)
        --   |x1|  |  |  |  |  |  |    -- extra (last) column is to make the shift logic work
        --   |x2|x3|  |  |  |  |  |    -- for cases with even number of columns
        --   |x4|x5|x6|  |  |  |  |
        --   ----------------------
        --   |  |  |  |x6|x5|x4|  |
        --   |  |  |  |  |x3|x2|  |
        --   |  |  |  |  |  |x1|  |
        if blidx < blidy then
          blidx = pivot - blidx - shift
          --blidy = pivot - blidy + (shift ~ 1) -- bitwise xor (~)
          if (int(shift) % 2 == 1) then -- shift odd
            blidy = pivot - blidy + 0 
          else -- shift even
            blidy = pivot - blidy + 1 
          end
        end

        [generateLoopStatements(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, 
                                r_density, r_gamma_table, accumulator,
                                thidx, thidy, blidx, blidy, BSIZEX, BSIZEY,
                                threshold, kguard)]
        if blidx == blidy then
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
      end)

    -- off-diagonal kernels do not need 'continue' control flow
    else 
      statements:insert(rquote
        [generateLoopStatements(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, 
                                r_density, r_gamma_table, accumulator,
                                thidx, thidy, blidx, blidy, BSIZEX, BSIZEY,
                                threshold, kguard)]
        var N24 = L2 + L4 * (largest_momentum + 1)
        r_output[{N24, blidy, blidx}].values += accumulator
      end)
    end
 
    return statements
  end
  -----------------------------------------------------------------------------

  --------------------------- Regent Integral Task ----------------------------
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

      -- local accumulator for iShell block
      var accumulator : double[Nout] 
      for n = 0, Nout do -- exclusive
        accumulator[n] = 0.0
      end

      ;[generateStatements(r_bras, r_kets, r_bra_prevals, r_ket_prevals, r_bra_labels, r_ket_labels, 
                           r_density, r_gamma_table, r_output, accumulator,
                           thidx, thidy, blidx, blidy, BSIZEX, BSIZEY,
                           threshold, kguard, largest_momentum)]
    end -- end gpuparam
  end
  -----------------------------------------------------------------------------

  kfock_integral:set_name("KFockMcMurchie"..L_string)
  _kfock_integral_cache[L_string] = kfock_integral
  return kfock_integral

end -- end function generateTaskMcMurchieKFockIntegral
