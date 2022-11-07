import "regent"

require "helper"
local c = regentlib.c -- KGJ: delete this

-- This function returns the index of elements on an NxN triangle.
-- For N = 3 we get.   For N = 4 we get.
--                             9
--     5                     7 8
--   3 4                   4 5 6
-- 0 1 2                 0 1 2 3
-- Except we don't care about the order of (x, y).
--                       3 6 8 9
-- 2 4 5                 2 5 7 8
-- 1 3 4                 1 4 5 6
-- 0 1 2                 0 1 2 3
local function magic(x, y, N)
  local y, x = math.min(x, y), math.max(x, y)
  return x + y * N - y * (y + 1) / 2
end

-- Similar with N = 3, but in 3d!
local function magic3(x, y, z)
  local pattern = {
    {{0, 1, 2},  -- 0 1 2
     {1, 3, 4},  --   3 4
     {2, 4, 5}}, --     5

    {{1, 3, 4},  --
     {3, 6, 7},  --   6 7
     {4, 7, 8}}, --     8

    {{2, 4, 5},  --
     {4, 7, 8},  --
     {5, 8, 9}}  --     9
  }
  return pattern[x+1][y+1][z+1]
end

-- returns the expeted number of bra preprocessed values
local function num_bra_preproc(L1, L2)
  -- Note: beyond L1=L2=2 is not actually implemented in TeraChem
  local values = {
    { 0,   4,  16,   47}, -- L1 = 0
    { 4,  25,  91,  244}, -- L1 = 1
    {16,  91, 301,  757}, -- L1 = 2
    {47, 244, 757, 1820}  -- L1 = 3
  }
  return values[L1+1][L2+1]
end


-- returns the ket_count for a given k loop (used in D kernels)
local function get_ket_count(L3, L4, k)
  local values = {
    {{  0},            -- SS (not used)
     {  0},            -- SP (not used)
     {  0}},           -- SD

    {{  0,   2,   4},  -- PS
     {  0,   9,  18},  -- PP
     {  0,  31,  62}}, -- PD

    {{  0,   4,   8,  12,  15,  18},  -- DS
     {  0,  18,  36,  54,  68,  82},  -- DP
     {  0,  56, 112, 168, 214, 260}}  -- DD
  }
  return values[L3+1][L4+1][k+1]
end


-- index by [i+1][k+1] since terra tables are one-indexed
-- TODO: extend this past D
local bra_norms = {

  {1.00000000000000000000,  -- i = 0 
   1.00000000000000000000,
   1.00000000000000000000,
   0.57735026918962576452,
   0.57735026918962576452,
   0.57735026918962576452}, 

  {1.00000000000000000000,  -- i = 1 
   1.00000000000000000000,
   1.00000000000000000000,
   0.57735026918962576452,
   0.57735026918962576452,
   0.57735026918962576452}, 

  {1.00000000000000000000,  -- i = 2 
   1.00000000000000000000,
   1.00000000000000000000,
   0.57735026918962576452,
   0.57735026918962576452,
   0.57735026918962576452}, 

  {0.57735026918962576452,  -- i = 3
   0.57735026918962576452,
   0.57735026918962576452,
   0.33333333333333333333,
   0.33333333333333333333,
   0.33333333333333333333}, 

  {0.57735026918962576452,  -- i = 4
   0.57735026918962576452,
   0.57735026918962576452,
   0.33333333333333333333,
   0.33333333333333333333,
   0.33333333333333333333}, 

  {0.57735026918962576452,  -- i = 5
   0.57735026918962576452,
   0.57735026918962576452,
   0.33333333333333333333,
   0.33333333333333333333,
   0.33333333333333333333} 

}

function generateKFockKernelStatements(R, L1, L2, L3, L4, k_idx, bra, ket,
                                       bra_prevals, ket_prevals,
                                       bra_idx, ket_idx,
                                       density, output)

  local statements = terralib.newlist() -- TODO: move this back down after debugging

  local function getR(N, L, M)
    if R[N] == nil or R[N][L] == nil or R[N][L][M] == nil or R[N][L][M][0] == nil then
      return 0
    else
      return R[N][L][M][0]
    end
  end

  local results = {}
  for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
    results[i] = {}
    for k = 0, triangle_number(L3 + 1) - 1 do -- inclusive
      results[i][k] = 0
    end
  end

  --------------------------------------------------------------------------
    -- All kernels above PPPP except SSSD and SSDS.
    --c.printf("Compiling a D kernel...\n\n")

    -- Loop through as follows: [ -2nd- , -3rd- | -1st- , __ ]  Loop index notation: (ij|kl)

    local H12, H34 = tetrahedral_number(L1+L2+1), tetrahedral_number(L3+L4+1)
    
    --c.printf("\nL1 = %1.f  L2 = %1.f  L3 = %1.f  L4 = %1.f\n", L1, L2, L3, L4)
    --c.printf("\nH12 = %1.f  H34 = %1.f\n", H12, H34)

    local patternL1 = generateJFockSpinPatternRestricted(L1)
    local patternL2 = generateJFockSpinPatternRestricted(L2)
    local patternL3 = generateJFockSpinPatternRestricted(L3)

    local pattern12 = generateJFockSpinPatternSorted(L1+L2)
    local pattern34 = generateJFockSpinPatternSorted(L3+L4)

    local ket_count = get_ket_count(L3, L4, k_idx) - 1

    --for k = 0, triangle_number(L3+1)-1 do -- inclusive
      --c.printf("k_idx = %1.f\n", k_idx)
      local largest_ket_count = ket_count + 1 -- move ket preprocessing counter to the next chunk
      local bra_count = 0 -- bra preprocessing counter
      for i = 0, triangle_number(L1+1)-1 do -- inclusive
        --c.printf("    i = %1.f\n", i)
        for j = 0, triangle_number(L2+1)-1 do -- inclusive
          --c.printf("        j = %1.f\n", j)

          ket_count = largest_ket_count -- reset ket preprocessing counter 
          local Ni, Li, Mi = unpack(patternL1[i+1])
          local Nj, Lj, Mj = unpack(patternL2[j+1])
          local Nk, Lk, Mk = unpack(patternL3[k_idx+1])
          local Lfilt = {Ni+Nj, Li+Lj, Mi+Mj}

          -- apply density filter
          local Dpattern = generateJFockSpinPatternRestricted(L4)
          for l = 0, triangle_number(L4+1)-1 do -- inclusive
            Dpattern[l+1][1] = Dpattern[l+1][1] + Nk
            Dpattern[l+1][2] = Dpattern[l+1][2] + Lk
            Dpattern[l+1][3] = Dpattern[l+1][3] + Mk
          end

          -- Write L expressions and ket preprocessing
          local Larr = {} -- Reset L arrays. More are allocated here than necessary
          for x = 0, L1+L2 do -- inclusive
            Larr[x] = {}
            for y = 0, L1+L2 do -- inclusive
              Larr[x][y] = {}
              for z = 0, L1+L2 do -- inclusive
                Larr[x][y][z] = regentlib.newsymbol(double, "Larr"..x..y..z)
                statements:insert(rquote
                  var [Larr[x][y][z]] = 0.0
                end)
              end
            end
          end

          for u = 0, H34-1 do -- inclusive

            local coeff = regentlib.newsymbol(double, "Coeff")
            statements:insert(rquote
              var [coeff] = 0.0
            end)

            for t = 0, H12-1 do -- inclusive
              local Nt, Lt, Mt = unpack(pattern12[t+1])
              local Nu, Lu, Mu = unpack(pattern34[u+1])
              local N, L, M = Nt + Nu, Lt + Lu, Mt + Mu
              -- Handle density complications here
              local den = {}
              if t == 0 then
                for l = 0, triangle_number(L4+1)-1 do -- inclusive
                  local X = Dpattern[l+1][1] - N
                  local Y = Dpattern[l+1][2] - L
                  local Z = Dpattern[l+1][3] - M
                  if X >= 0 and Y >=0 and Z >= 0 then
                    table.insert(den, l)
                  end
                end
                -- if density list is empty, break out of t loop
                if table.getn(den) == 0 then 
                  break 
                end
                -- Now add density contributions
                for l = 0, table.getn(den)-1 do -- inclusive
                  local idx = den[l+1]

                  -- KGJ DEBUGGING
                  if L3 + L4 == 0 then -- No prevals for SS ket
                    statements:insert(rquote
                      [coeff] += [density][j][idx]
                      --[coeff] += 1.0
                    end)
                  else
                    if L2 > L4 then
                      statements:insert(rquote
                        [coeff] += [density][idx][j] * ket_prevals[{ket_idx, ket_count}]
                      end)
                    else
                      statements:insert(rquote
                        [coeff] += [density][j][idx] * ket_prevals[{ket_idx, ket_count}]
                      end)
                    end
                  end
                  -- KGJ DEBUGGING

                  --c.printf("\ncoeff += D[%1.f][%1.f] * ket_prevals[ket_idx, %1.f]\n", j, idx, ket_count)
                  --statements:insert(rquote c.printf("     coeff (%lf) += density[%d][%d] (%lf)  *  ket_prevals[%d, %d] (%lf)\n", [coeff], j, idx, [density][j][idx], ket_idx, ket_count, ket_prevals[{ket_idx, ket_count}]) end)
                  ket_count = ket_count + 1
                end
                -- If you're on last ket level, don't increment ket preprocessing counter
                if N+L+M == L3+L4 then ket_count = ket_count - 1 end
              end


              -- Write L expressions
              if Nt <= Lfilt[1] and Lt <= Lfilt[2] and Mt <= Lfilt[3] then
                if (Nu + Lu + Mu) % 2 == 0 then
                  statements:insert(rquote
                    [Larr[Nt][Lt][Mt]] += [getR(N, L, M)] * [coeff]
                    --c.printf("L%1.f%1.f%1.f += R%1.f%1.f%1.f * coeff\n", Nt, Lt, Mt, N, L, M)
                    --statements:insert(rquote c.printf("           Larr[%d][%d][%d] (%lf) += R[%d][%d][%d] (%lf)  *  coeff (%lf)\n\n", Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], N, L, M, [getR(N, L, M)], [coeff]) end)
                  end)
                else
                  statements:insert(rquote
                    [Larr[Nt][Lt][Mt]] -= [getR(N, L, M)] * [coeff]
                    --c.printf("L%1.f%1.f%1.f -= R%1.f%1.f%1.f * coeff\n", Nt, Lt, Mt, N, L, M)
                    --statements:insert(rquote c.printf("           Larr[%d][%d][%d] (%lf) -= R[%d][%d][%d] (%lf)  *  coeff (%lf)\n\n", Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], N, L, M, [getR(N, L, M)], [coeff]) end)
                  end)
                end
              end
            end
          end

          -- Write bra post-processing and results
          for t = 0, H12-1 do -- inclusive
            local Nt, Lt, Mt = unpack(pattern12[t+1])
            if Nt <= Lfilt[1] and Lt <= Lfilt[2] and Mt <= Lfilt[3] then
              -- if you're on last bra level, use special preprocessed value
              if Nt + Lt + Mt == L1 + L2 then
                if L1 + L2 == 0 then
                  results[i][k_idx] = rexpr [results[i][k_idx]] + [Larr[Nt][Lt][Mt]] end
                  --c.printf("results[%1.f][%1.f] += L%1.f%1.f%1.f\n", i, k_idx, Nt, Lt, Mt)
                  --statements:insert(rquote c.printf("    results[%d][%d] ( %lf) += Larr[%d][%d][%d]\n", i, k_idx, [results[i][k_idx]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]]) end)
                else
                  local tot_bra_preproc = num_bra_preproc(L1,L2)
                  results[i][k_idx] = rexpr [results[i][k_idx]] + [Larr[Nt][Lt][Mt]] * bra_prevals[{bra_idx, tot_bra_preproc - 1}] end
                  --c.printf("results[%1.f][%1.f] += L%1.f%1.f%1.f * bra_prevals[bra_idx, %1.f]\n", i, k_idx, Nt, Lt, Mt, tot_bra_preproc-1)
                  --statements:insert(rquote c.printf("    results[%d][%d] (%lf) += Larr[%d][%d][%d] (%lf)  * bra_prevals[%d, %d] (%lf)\n", i, k_idx, [results[i][k_idx]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], bra_idx, tot_bra_preproc-1, bra_prevals[{bra_idx, tot_bra_preproc-1}]) end)
                end
              else
                results[i][k_idx] = rexpr [results[i][k_idx]] + [Larr[Nt][Lt][Mt]] * bra_prevals[{bra_idx, bra_count}] end 
                --c.printf("results[%1.f][%1.f] += L%1.f%1.f%1.f * bra_prevals[bra_idx, %1.f]\n", i, k_idx, Nt, Lt, Mt, bra_count)
                --statements:insert(rquote c.printf("    results[%d][%d] (%lf) += Larr[%d][%d][%d] (%lf)  * bra_prevals[%d, %d] (%lf)\n", i, k_idx, [results[i][k_idx]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], bra_idx, bra_count, bra_prevals[{bra_idx, bra_count}]) end)
                bra_count = bra_count + 1 
              end
            end
          end

        end
      end -- i loop
    --end -- k loop



  -----------------------------------------------------------------------------

  -- Handle lower D kernel exceptions with k_idx
  local k_min = 0 
  local k_max = triangle_number(L3 + 1) - 1 
  if L1 >= 2 or L2 >= 2 or L3 >= 2 or L4 >=2 then
    k_min = k_idx
    k_max = k_idx
  end
  if L1 == 0 and L2 == 0 and L3 == 0 and L4 == 2 then -- SSSD exception
    k_min = 0
    k_max = triangle_number(L3 + 1) - 1
  end
  if L1 == 0 and L2 == 0 and L3 == 2 and L4 == 0 then -- SSDS exception
    k_min = 0
    k_max = triangle_number(L3 + 1) - 1
  end

  --local statements = terralib.newlist()
  for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
    for k = k_min, k_max do -- inclusive
      local H3  = triangle_number(L3 + 1)
      if L1 == L3 and L2 == L4 then -- Diagonal kernel.
        local factor
        if i < k then -- Upper triangular element.
          factor = 1
        elseif i == k then -- Diagonal element.
          -- NOTE: Diagonal elements of diagonal kernels scale the output
          --       by a factor of 1/2.
          factor = 0.5
        else -- Lower triangular element.
          factor = 0
        end
        statements:insert(rquote
          --if bra.ishell_index < ket.ishell_index then -- Upper triangular element.
          if bra.ishell_index <= ket.ishell_index then -- Upper triangular element.
            [output][i*H3+k] += [results[i][k]] * [bra_norms[i+1][k+1]]
          --elseif bra.ishell_index == ket.ishell_index then -- Diagonal element
          --  [output][i*H3+k] += factor * [results[i][k]] * [bra_norms[i+1][k+1]]
          else -- Lower triangular element.
            -- NOTE: Diagonal kernels skip the lower triangular elements.
            -- no-op
          end
        end)
      else -- Upper triangular kernel.
        statements:insert(rquote
          [output][i*H3+k] += [results[i][k]] * [bra_norms[i+1][k+1]]
        end)
      end
      --statements:insert(rquote c.printf("       output(%d,%d).values(%d,%d) = %lf\n", bra.ishell_index, ket.ishell_index, i, k, [output][i*L3+k]) end)
    end
  end

  return statements
end
