import "regent"

require "helper"
local c = regentlib.c -- TODO: delete this when finished printing for debugging


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

  local statements = terralib.newlist()

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

  -- All kernels are metaprogrammed with the same pattern
  --------------------------------------------------------------------------

  local H12, H34 = tetrahedral_number(L1+L2+1), tetrahedral_number(L3+L4+1)
  
  --c.printf("Compiling a D kernel...\n\n")
  --c.printf("\nL1 = %1.f  L2 = %1.f  L3 = %1.f  L4 = %1.f\n", L1, L2, L3, L4)
  --c.printf("\nH12 = %1.f  H34 = %1.f\n", H12, H34)

  local patternL1 = generateJFockSpinPatternRestricted(L1)
  local patternL2 = generateJFockSpinPatternRestricted(L2)
  local patternL3 = generateJFockSpinPatternRestricted(L3)

  local pattern12 = generateJFockSpinPatternSorted(L1+L2)
  local pattern34 = generateJFockSpinPatternSorted(L3+L4)


  -- Loop through as follows: [ -2nd- , -3rd- | -1st- , __ ]  Loop index notation: (ij|kl)
  -- NOTE: for large kernels, k loop is handled at the level of task launches (to decrease compile time)
  local k_min, k_max 
  if (L1 > 0 and L2 > 0 and L3 > 0 and L4 > 0) and (L1 + L2 + L3 + L4 >= 6) then 
    k_min = k_idx
    k_max = k_idx
  else
    k_min = 0
    k_max = triangle_number(L3+1)-1
  end

  for k = k_min, k_max do -- inclusive
    --c.printf("k = %1.f\n", k)
    local ket_count = get_ket_count(L3, L4, k) - 1

    local largest_ket_count = ket_count + 1 -- move ket preprocessing counter to the next chunk
    local bra_count = 0 -- bra preprocessing counter
    for i = 0, triangle_number(L1+1)-1 do -- inclusive
      --c.printf("    i = %1.f\n", i)
      for j = 0, triangle_number(L2+1)-1 do -- inclusive
        --c.printf("        j = %1.f\n", j)

        ket_count = largest_ket_count -- reset ket preprocessing counter 
        local Ni, Li, Mi = unpack(patternL1[i+1])
        local Nj, Lj, Mj = unpack(patternL2[j+1])
        local Nk, Lk, Mk = unpack(patternL3[k+1])
        local Lfilt = {Ni+Nj, Li+Lj, Mi+Mj}

        -- apply density filter
        local Dpattern = generateJFockSpinPatternRestricted(L4)
        for l = 0, triangle_number(L4+1)-1 do -- inclusive
          Dpattern[l+1][1] = Dpattern[l+1][1] + Nk
          Dpattern[l+1][2] = Dpattern[l+1][2] + Lk
          Dpattern[l+1][3] = Dpattern[l+1][3] + Mk
        end

        local Larr = {} -- Reset L arrays
        for x = 0, L1+L2 do -- inclusive
          Larr[x] = {}
          for y = 0, L1+L2 do -- inclusive
            Larr[x][y] = {}
            for z = 0, L1+L2 do -- inclusive
              Larr[x][y][z] = regentlib.newsymbol(double, "Larr"..x..y..z)
              if (x + y + z) <= L1+L2 then
                statements:insert(rquote
                  var [Larr[x][y][z]] = 0.0
                end)
              end
            end
          end
        end

        local D = {} -- Set density arrays (avoid re-declaring variables)
        for l = 0, triangle_number(L4+1)-1 do -- inclusive
          D[l] = regentlib.newsymbol(double, "D"..j..l)
          if L2 > L4 then
            statements:insert(rquote
              var [D[l]] = [density][l][j]
            end)
          else
            statements:insert(rquote
              var [D[l]] = [density][j][l]
            end)
          end
        end

        local coeff = regentlib.newsymbol(double, "Coeff")
        statements:insert(rquote
          var [coeff] = 0.0
        end)

        ------------------- Write L expressions and ket preprocessing  --------------------
        for u = 0, H34-1 do -- inclusive

          statements:insert(rquote
            [coeff] = 0.0
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

                if L3 + L4 == 0 then -- No prevals for SS ket
                  statements:insert(rquote
                    [coeff] += [D[idx]]
                  end)
                else
                  statements:insert(rquote
                    [coeff] += [D[idx]] * ket_prevals[{ket_idx, ket_count}]
                  end)
                end
                --c.printf("\ncoeff += D[%1.f][%1.f] * ket_prevals[ket_idx, %1.f]\n", j, idx, ket_count)
                --statements:insert(rquote c.printf("     coeff (%lf) += density[%d][%d] (%lf)  *  ket_prevals[%d, %d] (%lf)\n", 
                --                                    [coeff], j, idx, [D[idx]], ket_idx, ket_count, ket_prevals[{ket_idx, ket_count}]) end)
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
                end)
                --c.printf("L%1.f%1.f%1.f += R%1.f%1.f%1.f * coeff\n", Nt, Lt, Mt, N, L, M)
                --statements:insert(rquote c.printf("           Larr[%d][%d][%d] (%lf) += R[%d][%d][%d] (%lf)  *  coeff (%lf)\n\n", 
                --                                   Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], N, L, M, [getR(N, L, M)], [coeff]) end)
              else
                statements:insert(rquote
                  [Larr[Nt][Lt][Mt]] -= [getR(N, L, M)] * [coeff]
                end)
                --c.printf("L%1.f%1.f%1.f -= R%1.f%1.f%1.f * coeff\n", Nt, Lt, Mt, N, L, M)
      	        --statements:insert(rquote c.printf("           Larr[%d][%d][%d] (%lf) -= R[%d][%d][%d] (%lf)  *  coeff (%lf)\n\n", 
                --                                   Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], N, L, M, [getR(N, L, M)], [coeff]) end)
              end
            end
          end -- end t loop
        end -- end u loop

        ----------------------- Write bra post-processing and results ------------------------
        for t = 0, H12-1 do -- inclusive
          local Nt, Lt, Mt = unpack(pattern12[t+1])
          if Nt <= Lfilt[1] and Lt <= Lfilt[2] and Mt <= Lfilt[3] then
            -- if you're on last bra level, use special preprocessed value
            if Nt + Lt + Mt == L1 + L2 then
              if L1 + L2 == 0 then
                results[i][k] = rexpr [results[i][k]] + [Larr[Nt][Lt][Mt]] end
                --c.printf("results[%1.f][%1.f] += L%1.f%1.f%1.f\n", i, k, Nt, Lt, Mt)
                --statements:insert(rquote c.printf("    results[%d][%d] ( %lf) += Larr[%d][%d][%d] ( %lf)\n", 
                --                                   i, k, [results[i][k]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]]) end)
              else
                local tot_bra_preproc = num_bra_preproc(L1,L2)
                results[i][k] = rexpr [results[i][k]] + [Larr[Nt][Lt][Mt]] * bra_prevals[{bra_idx, tot_bra_preproc - 1}] end
                --c.printf("results[%1.f][%1.f] += L%1.f%1.f%1.f * bra_prevals[bra_idx, %1.f]\n", i, k, Nt, Lt, Mt, tot_bra_preproc-1)
                --statements:insert(rquote c.printf("    results[%d][%d] (%lf) += Larr[%d][%d][%d] (%lf)  * bra_prevals[%d, %d] (%lf)\n", 
                --                                   i, k, [results[i][k]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], bra_idx, tot_bra_preproc-1, 
                --                                   bra_prevals[{bra_idx, tot_bra_preproc-1}]) end)
              end
            else
              results[i][k] = rexpr [results[i][k]] + [Larr[Nt][Lt][Mt]] * bra_prevals[{bra_idx, bra_count}] end 
              --c.printf("results[%1.f][%1.f] += L%1.f%1.f%1.f * bra_prevals[bra_idx, %1.f]\n", i, k, Nt, Lt, Mt, bra_count)
              --statements:insert(rquote c.printf("    results[%d][%d] (%lf) += Larr[%d][%d][%d] (%lf)  * bra_prevals[%d, %d] (%lf)\n", 
              --                                   i, k, [results[i][k]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], bra_idx, bra_count, 
              --                                   bra_prevals[{bra_idx, bra_count}]) end)
              bra_count = bra_count + 1 
            end
          end
        end

      end -- j loop
    end -- i loop
  end -- k loop


  -----------------------------------------------------------------------------

  for i = 0, triangle_number(L1+1)-1 do -- inclusive
    for k = k_min, k_max do
      local H3  = triangle_number(L3+1)
      statements:insert(rquote
        [output][i*H3+k] += [results[i][k]] * [bra_norms[i+1][k+1]]
      end)
      --statements:insert(rquote c.printf("       output(%d,%d).values(%d,%d) = %lf\n", 
      --                                   bra.ishell_index, ket.ishell_index, i, k, [output][i*L3+k]) end)
    end
  end

  return statements
end
