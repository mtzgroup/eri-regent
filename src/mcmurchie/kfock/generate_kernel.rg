import "regent"

require "helper"
local c = regentlib.c -- TODO: delete this when finished printing for debugging

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

  local k_min = 0
  local k_max = triangle_number(L3+1)-1

  if L1 <= 1 and L2 <= 1 and L3 <= 1 and L4 <= 1 then
  ------------------------------------------------------------------
  -- Metaprogramming pattern for S and P kernels

    local denomPi = regentlib.newsymbol(double, "DenomPi")
    local denomPj = regentlib.newsymbol(double, "DenomPj")
    local denomQi = regentlib.newsymbol(double, "DenomQi")
    local denomQj = regentlib.newsymbol(double, "DenomQj")
    local Pi, Pj, Qi, Qj = {}, {}, {}, {}
    for x = 0, 2 do  -- inclusive
      Pi[x] = regentlib.newsymbol(double, "Pi"..x)
      Pj[x] = regentlib.newsymbol(double, "Pj"..x)
      Qi[x] = regentlib.newsymbol(double, "Qi"..x)
      Qj[x] = regentlib.newsymbol(double, "Qj"..x)
    end
    if L1 == 1 then
      statements:insert(rquote
        var [denomPi] = 1 / (2 * bra.eta)
        var [Pi[0]] = bra.ishell_location.x 
        var [Pi[1]] = bra.ishell_location.y
        var [Pi[2]] = bra.ishell_location.z 
      end)
    end
    if L2 == 1 then
      statements:insert(rquote
        var [denomPj] = 1 / (2 * bra.eta)
        var [Pj[0]] = bra.jshell_location.x 
        var [Pj[1]] = bra.jshell_location.y
        var [Pj[2]] = bra.jshell_location.z 
      end)
    end
    if L3 == 1 then
      statements:insert(rquote
        var [denomQi] = 1 / (2 * ket.eta)
        var [Qi[0]] = ket.ishell_location.x 
        var [Qi[1]] = ket.ishell_location.y
        var [Qi[2]] = ket.ishell_location.z 
      end)
    end
    if L4 == 1 then
      statements:insert(rquote
        var [denomQj] = 1 / (2 * ket.eta)
        var [Qj[0]] = ket.jshell_location.x 
        var [Qj[1]] = ket.jshell_location.y
        var [Qj[2]] = ket.jshell_location.z 
      end)
    end

    for j = 0, triangle_number(L2 + 1) - 1 do -- inclusive

      local density_triplet = {}
      density_triplet[0] = regentlib.newsymbol(double, "D0")
      density_triplet[1] = regentlib.newsymbol(double, "D1")
      density_triplet[2] = regentlib.newsymbol(double, "D2")
      density_triplet[3] = regentlib.newsymbol(double, "D3")
      if L2 == 0 and L4 == 0 then
        statements:insert(rquote
          var [density_triplet[0]] = [density][0][0]
        end)
      elseif L2 > L4 then -- special case for SPPS kernel
        statements:insert(rquote
          var [density_triplet[0]] = [density][0][j]
        end)
      elseif L2 == 0 then
        statements:insert(rquote
          var density0 = [density][0][0]
          var density1 = [density][0][1]
          var density2 = [density][0][2]
          var [density_triplet[0]] = density0 * [denomQj]
          var [density_triplet[1]] = density1 * [denomQj]
          var [density_triplet[2]] = density2 * [denomQj]
          var [density_triplet[3]] = density0 * [Qj[0]] + density1 * [Qj[1]] + density2 * [Qj[2]]
        end)
      else                     
        statements:insert(rquote
          var density0 = [density][j][0]
          var density1 = [density][j][1]
          var density2 = [density][j][2]
          var [density_triplet[0]] = density0 * [denomQj]
          var [density_triplet[1]] = density1 * [denomQj]
          var [density_triplet[2]] = density2 * [denomQj]
          var [density_triplet[3]] = density0 * [Qj[0]] + density1 * [Qj[1]] + density2 * [Qj[2]]
        end)
      end

      for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
        for k = 0, triangle_number(L3 + 1) - 1 do -- inclusive

          -- Some helpful auxiliary functions.
          local function aux0(j, i)
            local q, r, s = unpack(generateKFockSpinPattern(j)[i+1])
            return rexpr
              [getR(q, r, s)] * [density_triplet[k]]
            end
          end

          local function aux1(j, i)
            local q, r, s = unpack(generateKFockSpinPattern(j)[i+1])
            if L4 == 0 then
              return rexpr
                [getR(q, r, s)] * ([density_triplet[0]])
              end
            else
              return rexpr
                [getR(q, r, s)] * ([density_triplet[3]])
                -  ([getR(q+1, r, s)] * [density_triplet[0]]
                  + [getR(q, r+1, s)] * [density_triplet[1]]
                  + [getR(q, r, s+1)] * [density_triplet[2]])
              end
            end
          end

------------------------------------------------------------------------------
-- NOTE: this is the old version of the S/P metaprogramming, I have left it 
--       here for documentation. Many terms (the ones commented with L = 0)
--       resulted in no-ops (multiplication by zero). The new version below
--       uses rquotes to declare intermediate values instead and avoids the 
--       no-ops (resulting in much cleaner generated code), but looks less 
--       clean on the metaprogramming side.
--
--          results[i][k] = rexpr
--            [results[i][k]] + (
-- TERM 1
--           [Pi[i]] * (                                  -- 1, L1 = 0
--             [Pj[n]] * (                                -- 1, L2 = 0
--                  [Qi[k]] * [aux1(0, 0)]                -- 1, L3 = 0
--                  - [denomQi] * [aux1(1, k)]            -- 0, L3 = 0
--                  + [aux0(0, 0)]                        -- 0, L1 = 0, L3 = 0
--             )
--             + [denomPj] * (                            -- 0, L2 = 0
--               [Qi[k]] * [aux1(1, n)]                   -- 1, L3 = 0
--               - [denomQi] * [aux1(2, magic(k, n, 3))]  -- 0, L3 = 0
--               + [aux0(1, n)]                           -- 0, L1 = 0, L3 = 0
--             )
--           )

-- TERM 2
--           + [denomPi] * (                              -- 0, L1 = 0
--             [Pj[n]] * (                                -- 1, L2 = 0
--               [Qi[k]] * [aux1(1, i)]                   -- 1, L3 = 0
--               - [denomQi] * [aux1(2, magic(i, k, 3))]  -- 0, L3 = 0
--               + [aux0(1, i)]                           -- 0, L1 = 0, L3 = 0
--             )
--             + [denomPj] * (                            -- 0, L2 = 0
--               [Qi[k]] * [aux1(2, magic(i, n, 3))]      -- 1, L3 = 0
--               - [denomQi] * [aux1(3, magic3(i, k, n))] -- 0, L3 = 0
--               + [aux0(2, magic(i, n, 3))]              -- 0, L1 = 0, L3 = 0
--             )
--           )
--            )
--          end
------------------------------------------------------------------------------

          -- TERM 1 ----------------------------------------
          local term1 = regentlib.newsymbol(double, "Term1")
          -- 1a --------------------
          statements:insert(rquote var [term1] = [aux1(0, 0)] end)
          if L3 == 1 then
            statements:insert(rquote [term1] *= [Qi[k]] end)
            statements:insert(rquote [term1] -= [denomQi] * [aux1(1, k)] end)
          end
          if not(L1 == 0 and L3 == 0) and L4 == 1 then
            statements:insert(rquote [term1] += [aux0(0, 0)] end)
          end
          if L2 == 1 then
            statements:insert(rquote [term1] *= [Pj[j]] end)
          end

          -- 1b --------------------
          if L2 == 1 then
            local term1b = regentlib.newsymbol(double, "Term1b")
            statements:insert(rquote var [term1b] = [aux1(1, j)] end)
            if L3 == 1 then
              statements:insert(rquote [term1b] *= [Qi[k]] end)
              statements:insert(rquote [term1b] -= [denomQi] * [aux1(2, magic(k, j, 3))] end)
            end
            if not(L1 == 0 and L3 == 0) and L4 == 1 then
              statements:insert(rquote [term1b] += [aux0(1, j)] end)
            end
            statements:insert(rquote [term1] += [denomPj] * [term1b] end)
          end

          if L1 == 1 then
            statements:insert(rquote [term1] *= [Pi[i]] end)
          end

          results[i][k] = rexpr
            [results[i][k]] + [term1]
          end
          
          -- TERM 2 ----------------------------------------
          if L1 == 1 then
            local term2 = regentlib.newsymbol(double, "Term2")
            -- 2a --------------------
            statements:insert(rquote var [term2] = [aux1(1, i)] end)
            if L3 == 1 then
              statements:insert(rquote [term2] *= [Qi[k]] end)
              statements:insert(rquote [term2] -= [denomQi] *  [aux1(2, magic(i, k, 3))] end)
            end
            if not(L1 == 0 and L3 == 0) and L4 == 1 then
              statements:insert(rquote [term2] += [aux0(1, i)] end)
            end
            if L2 == 1 then
              statements:insert(rquote [term2] *= [Pj[j]] end)
            end
         
            -- 2b --------------------
            if L2 == 1 then
              local term2b = regentlib.newsymbol(double, "Term2b")
              statements:insert(rquote var [term2b] = [aux1(2, magic(i, j, 3))]  end)
              if L3 == 1 then
                statements:insert(rquote [term2b] *= [Qi[k]] end)
                statements:insert(rquote [term2b] -= [denomQi] * [aux1(3, magic3(i, k, j))] end)
              end
              if not(L1 == 0 and L3 == 0) and L4 == 1 then
                statements:insert(rquote [term2b] += [aux0(2, magic(i, j, 3))] end)
              end
              statements:insert(rquote [term2] += [denomPj] * [term2b] end)
            end
         
            statements:insert(rquote [term2] *= [denomPi] end)
         
            results[i][k] = rexpr
              [results[i][k]] + [term2]
            end
          end
          
          -- TERM 3 ----------------------------------------
          if L1 == 1 and L2 == 1 and i == j then
            results[i][k] = rexpr
              [results[i][k]] + [denomPj] * (
                [Qi[k]] * [aux1(0, 0)]  
                - [denomQi] * [aux1(1, k)]
                + [aux0(0, 0)]          
              )
            end
          end

        end -- k loop
      end -- i loop
    end -- n loop

  elseif L1 == 0 and L2 == 0 and L3 == 0 and L4 == 2 then
  -------------------------------------SSSD-------------------------------------
    local denomQj = regentlib.newsymbol(double, "DenomQj")
    local Qj = {}
    for x = 0, 2 do  -- inclusive
      Qj[x] = regentlib.newsymbol(double, "Qj"..x)
    end
    statements:insert(rquote
      var [denomQj] = 1 / (2 * ket.eta)
      var [Qj[0]] = ket.jshell_location.x 
      var [Qj[1]] = ket.jshell_location.y
      var [Qj[2]] = ket.jshell_location.z 
    end)
    local D = {}
    for x = 0, 5 do -- inclusive
      D[x] = regentlib.newsymbol(double, "D0"..x)
      statements:insert(rquote
        var [D[x]] = [density][0][x]
      end)
    end

    local tmp0 = rexpr
      [getR(0, 0, 0)] * (
          [Qj[0]] * [Qj[1]] * [D[0]]
        + [Qj[0]] * [Qj[2]] * [D[1]]
        + [Qj[1]] * [Qj[2]] * [D[2]]
        + [Qj[0]] * [Qj[0]] * [D[3]]
        + [Qj[1]] * [Qj[1]] * [D[4]]
        + [Qj[2]] * [Qj[2]] * [D[5]]
        + ([D[3]] + [D[4]] + [D[5]]) * denomQj
      )
    end

    results[0][0] = rexpr
      tmp0
      - ([D[0]] * [Qj[1]] + [D[1]] * [Qj[2]] + 2 * [D[3]] * [Qj[0]]) * denomQj * [getR(1, 0, 0)]
      - ([D[0]] * [Qj[0]] + [D[2]] * [Qj[2]] + 2 * [D[4]] * [Qj[1]]) * denomQj * [getR(0, 1, 0)]
      - ([D[1]] * [Qj[0]] + [D[2]] * [Qj[1]] + 2 * [D[5]] * [Qj[2]]) * denomQj * [getR(0, 0, 1)]
      + ([D[0]] * denomQj * denomQj * [getR(1, 1, 0)])
      + ([D[1]] * denomQj * denomQj * [getR(1, 0, 1)])
      + ([D[2]] * denomQj * denomQj * [getR(0, 1, 1)])
      + ([D[3]] * denomQj * denomQj * [getR(2, 0, 0)])
      + ([D[4]] * denomQj * denomQj * [getR(0, 2, 0)])
      + ([D[5]] * denomQj * denomQj * [getR(0, 0, 2)])
    end

  elseif L1 == 0 and L2 == 0 and L3 == 2 and L4 == 0 then
  -------------------------------------SSDS-------------------------------------
    local denomQi = regentlib.newsymbol(double, "DenomQi")
    local Qix = regentlib.newsymbol(double, "Qix")
    local Qiy = regentlib.newsymbol(double, "Qiy")
    local Qiz = regentlib.newsymbol(double, "Qiz")
    local D = regentlib.newsymbol(double, "D")
    statements:insert(rquote
      var [denomQi] = 1 / (2 * ket.eta)
      var [Qix] = ket.ishell_location.x 
      var [Qiy] = ket.ishell_location.y
      var [Qiz] = ket.ishell_location.z 
      var [D] = [density][0][0]
    end)

    results[0][0] = rexpr
        [D] * [Qix] * [Qiy] * [getR(0, 0, 0)]
      - [D] * [denomQi] * [Qiy] * [getR(1, 0, 0)]
      - [D] * [Qix] * [denomQi] * [getR(0, 1, 0)]
      + [D] * [denomQi] * [denomQi] * [getR(1, 1, 0)]
    end
    results[0][1] = rexpr
        [D] * [Qix] * [Qiz] * [getR(0, 0, 0)]
      - [D] * [denomQi] * [Qiz] * [getR(1, 0, 0)]
      - [D] * [denomQi] * [Qix] * [getR(0, 0, 1)]
      + [D] * [denomQi] * [denomQi] * [getR(1, 0, 1)]
    end
    results[0][2] = rexpr
        [D] * [Qiy] * [Qiz] * [getR(0, 0, 0)]
      - [D] * [Qiz] * [denomQi] * [getR(0, 1, 0)]
      - [D] * [denomQi] * [Qiy] * [getR(0, 0, 1)]
      + [D] * [denomQi] * [denomQi] * [getR(0, 1, 1)]
    end
    results[0][3] = rexpr
      (
      ([D] * [Qix] * [Qix] + [D] * [denomQi]) * [getR(0, 0, 0)]
      -2 * [D] * [denomQi] * [Qix] * [getR(1, 0, 0)]
      + [D] * [denomQi] * [denomQi] * [getR(2, 0, 0)]
      )
    end
    results[0][4] = rexpr
      (
      ([D] * [Qiy] * [Qiy] + [D] * [denomQi]) * [getR(0, 0, 0)]
      -2 * [D] * [Qiy] * [denomQi] * [getR(0, 1, 0)]
      + [D] * [denomQi] * [denomQi] * [getR(0, 2, 0)]
      )
    end
    results[0][5] = rexpr
      (
      ([D] * [Qiz] * [Qiz] + [D] * [denomQi]) * [getR(0, 0, 0)]
      -2 * [D] * [denomQi] * [Qiz] * [getR(0, 0, 1)]
      + [D] * [denomQi] * [denomQi] * [getR(0, 0, 2)]
      )
    end

  else
  -- Metaprogramming for all D kernels (except SSSD and SSDS) 
  --------------------------------------------------------------------------

    local H12, H34 = tetrahedral_number(L1+L2+1), tetrahedral_number(L3+L4+1)
    
    local patternL1 = generateJFockSpinPatternRestricted(L1)
    local patternL2 = generateJFockSpinPatternRestricted(L2)
    local patternL3 = generateJFockSpinPatternRestricted(L3)
 
    local pattern12 = generateJFockSpinPatternSorted(L1+L2)
    local pattern34 = generateJFockSpinPatternSorted(L3+L4)
 
    -- Loop through as follows: [ -2nd- , -3rd- | -1st- , __ ]  Loop index notation: (ij|kl)
    -- NOTE: for large kernels, k loop is handled at the level of task launches (to decrease compile time)
    if (L1 > 0 and L2 > 0 and L3 > 0 and L4 > 0) and (L1 + L2 + L3 + L4 >= 6) then 
      k_min = k_idx
      k_max = k_idx
    end
 
    for k = k_min, k_max do -- inclusive
      local ket_count = get_ket_count(L3, L4, k) - 1
 
      local largest_ket_count = ket_count + 1 -- move ket preprocessing counter to the next chunk
      local bra_count = 0 -- bra preprocessing counter
      for i = 0, triangle_number(L1+1)-1 do -- inclusive
        for j = 0, triangle_number(L2+1)-1 do -- inclusive
 
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
                  --statements:insert(rquote c.printf("           Larr[%d][%d][%d] (%lf) += R[%d][%d][%d] (%lf)  *  coeff (%lf)\n\n", 
                  --                                   Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], N, L, M, [getR(N, L, M)], [coeff]) end)
                else
                  statements:insert(rquote
                    [Larr[Nt][Lt][Mt]] -= [getR(N, L, M)] * [coeff]
                  end)
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
                  --statements:insert(rquote c.printf("    results[%d][%d] ( %lf) += Larr[%d][%d][%d] ( %lf)\n", 
                  --                                   i, k, [results[i][k]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]]) end)
                else
                  local tot_bra_preproc = num_bra_preproc(L1,L2)
                  results[i][k] = rexpr [results[i][k]] + [Larr[Nt][Lt][Mt]] * bra_prevals[{bra_idx, tot_bra_preproc - 1}] end
                  --statements:insert(rquote c.printf("    results[%d][%d] (%lf) += Larr[%d][%d][%d] (%lf)  * bra_prevals[%d, %d] (%lf)\n", 
                  --                                   i, k, [results[i][k]], Nt, Lt, Mt, [Larr[Nt][Lt][Mt]], bra_idx, tot_bra_preproc-1, 
                  --                                   bra_prevals[{bra_idx, tot_bra_preproc-1}]) end)
                end
              else
                results[i][k] = rexpr [results[i][k]] + [Larr[Nt][Lt][Mt]] * bra_prevals[{bra_idx, bra_count}] end 
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

  end -- end alternate pattern checks


  -----------------------------------------------------------------------------
  -- Add results into output array (same regardless of metaprogramming pattern)

  for i = 0, triangle_number(L1+1)-1 do -- inclusive
    for k = k_min, k_max do
      local H3  = triangle_number(L3+1)
      if L1 < 2 and L3 < 2 then -- avoid unnecessary multiplication by 1
        statements:insert(rquote
          [output][i*H3+k] += [results[i][k]]
        end)
      else
        statements:insert(rquote
          [output][i*H3+k] += [results[i][k]] * [bra_norms[i+1][k+1]]
        end)
      end
      --statements:insert(rquote c.printf("       output(%d,%d).values(%d,%d) = %lf\n", 
      --                                   bra.ishell_index, ket.ishell_index, i, k, [output][i*L3+k]) end)
    end
  end

  return statements
end
