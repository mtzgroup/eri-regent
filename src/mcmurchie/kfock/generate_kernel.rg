import "regent"

require "helper"

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

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket,
                                       bra_prevals, ket_prevals,
                                       bra_idx, ket_idx,
                                       density, output)

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

  -- All kernels up to PPPP are metaprogrammed.
  if L1 <= 1 and L2 <= 1 and L3 <= 1 and L4 <= 1 then
    local Pi = {
      rexpr bra.ishell_location.x end,
      rexpr bra.ishell_location.y end,
      rexpr bra.ishell_location.z end
    }
    local Pj = {
      rexpr bra.jshell_location.x end,
      rexpr bra.jshell_location.y end,
      rexpr bra.jshell_location.z end
    }
    local Qi = {
      rexpr ket.ishell_location.x end,
      rexpr ket.ishell_location.y end,
      rexpr ket.ishell_location.z end
    }
    local Qj = {
      rexpr ket.jshell_location.x end,
      rexpr ket.jshell_location.y end,
      rexpr ket.jshell_location.z end
    }
    local denomPi, denomPj = rexpr 1 / (2 * bra.eta) end, rexpr 1 / (2 * bra.eta) end
    local denomQi, denomQj = rexpr 1 / (2 * ket.eta) end, rexpr 1 / (2 * ket.eta) end
    if L1 == 0 then Pi, denomPi = {1, 0, 0}, 0 end
    if L2 == 0 then Pj, denomPj = {1, 0, 0}, 0 end
    if L3 == 0 then Qi, denomQi = {1, 0, 0}, 0 end
    if L4 == 0 then Qj, denomQj = {1, 0, 0}, 0 end

    for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
      for k = 0, triangle_number(L3 + 1) - 1 do -- inclusive
        for n = 0, 2 do -- inclusive

          local density_triplet
          if L2 == 0 and L4 == 0 then
            density_triplet = {rexpr [density][n][0] end, 0, 0}
          else
            density_triplet = {rexpr [density][n][0] end,
                               rexpr [density][n][1] end,
                               rexpr [density][n][2] end}
          end

          -- Some helpful auxiliary functions.
          local function aux0(n, i)
            local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
            if L1 == 0 and L3 == 0 then
              return 0
            else
              return rexpr
                denomQj * [getR(q, r, s)] * [density_triplet[k+1]]
              end
            end
          end
          local function aux1(n, i)
            local a, b, c = unpack(density_triplet)
            local x, y, z = unpack(Qj)
            local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
            return rexpr
              [getR(q, r, s)] * (a * x + b * y + c * z)
              - denomQj * ([getR(q+1, r, s)] * a
                            + [getR(q, r+1, s)] * b
                            + [getR(q, r, s+1)] * c)
            end
          end

          results[i][k] = rexpr
            [results[i][k]] + (
              [Pi[i+1]] * (
                [Pj[n+1]] * (
                  [Qi[k+1]] * [aux1(0, 0)]
                  - denomQi * [aux1(1, k)]
                  + [aux0(0, 0)]
                )
                + denomPj * (
                  [Qi[k+1]] * [aux1(1, n)]
                  - denomQi * [aux1(2, magic(k, n, 3))]
                  + [aux0(1, n)]
                )
              )
              + denomPi * (
                [Pj[n+1]] * (
                  [Qi[k+1]] * [aux1(1, i)]
                  - denomQi * [aux1(2, magic(i, k, 3))]
                  + [aux0(1, i)]
                )
                + denomPj * (
                  [Qi[k+1]] * [aux1(2, magic(i, n, 3))]
                  - denomQi * [aux1(3, magic3(i, k, n))]
                  + [aux0(2, magic(i, n, 3))]
                )
              )
            )
          end

          if L1 == 1 and L2 == 1 and i == n then
            results[i][k] = rexpr
              [results[i][k]] + denomPj * (
                [Qi[k+1]] * [aux1(0, 0)]
                - denomQi * [aux1(1, k)]
                + [aux0(0, 0)]
              )
            end
          end
        end
      end
    end
  elseif L1 == 0 and L2 == 0 and L3 == 0 and L4 == 2 then
  -------------------------------------SSSD-------------------------------------
    local Qj = {
      rexpr ket.jshell_location.x end,
      rexpr ket.jshell_location.y end,
      rexpr ket.jshell_location.z end
    }
    local denomQj = rexpr 1 / (2 * ket.eta) end

    local tmp0 = rexpr
      [getR(0, 0, 0)] * (
        [Qj[1]] * [Qj[2]] * [density][0][0]
        + [Qj[1]] * [Qj[3]] * [density][0][1]
        + [Qj[2]] * [Qj[3]] * [density][0][2]
        + [Qj[1]] * [Qj[1]] * [density][0][3]
        + [Qj[2]] * [Qj[2]] * [density][0][4]
        + [Qj[3]] * [Qj[3]] * [density][0][5]
        + ([density][0][3] + [density][0][4] + [density][0][5]) * denomQj
      )
    end

    results[0][0] = rexpr
      tmp0
      - ([density][0][0] * [Qj[2]] + [density][0][1] * [Qj[3]] + 2 * [density][0][3] * [Qj[1]]) * denomQj * [getR(1, 0, 0)]
      - ([density][0][0] * [Qj[1]] + [density][0][2] * [Qj[3]] + 2 * [density][0][4] * [Qj[2]]) * denomQj * [getR(0, 1, 0)]
      - ([density][0][1] * [Qj[1]] + [density][0][2] * [Qj[2]] + 2 * [density][0][5] * [Qj[3]]) * denomQj * [getR(0, 0, 1)]
      + ([density][0][0] * denomQj * denomQj * [getR(1, 1, 0)])
      + ([density][0][1] * denomQj * denomQj * [getR(1, 0, 1)])
      + ([density][0][2] * denomQj * denomQj * [getR(0, 1, 1)])
      + ([density][0][3] * denomQj * denomQj * [getR(2, 0, 0)])
      + ([density][0][4] * denomQj * denomQj * [getR(0, 2, 0)])
      + ([density][0][5] * denomQj * denomQj * [getR(0, 0, 2)])
    end

  elseif L1 == 0 and L2 == 0 and L3 == 2 and L4 == 0 then
  -------------------------------------SSDS-------------------------------------
    local Qix = rexpr ket.ishell_location.x end
    local Qiy = rexpr ket.ishell_location.y end
    local Qiz = rexpr ket.ishell_location.z end
    local denomQi = rexpr 1 / (2 * ket.eta) end
    local D = rexpr [density][0][0] end

    results[0][0] = rexpr
      D * Qix * Qiy * [getR(0, 0, 0)]
      - D * denomQi * Qiy * [getR(1, 0, 0)]
      - D * Qix * denomQi * [getR(0, 1, 0)]
      + D * denomQi * denomQi * [getR(1, 1, 0)]
    end
    results[0][1] = rexpr
      D * Qix * Qiz * [getR(0, 0, 0)]
      - D * denomQi * Qiz * [getR(1, 0, 0)]
      - D * denomQi * Qix * [getR(0, 0, 1)]
      + D * denomQi * denomQi * [getR(1, 0, 1)]
    end
    results[0][2] = rexpr
      D * Qiy * Qiz * [getR(0, 0, 0)]
      - D * Qiz * denomQi * [getR(0, 1, 0)]
      - D * denomQi * Qiy * [getR(0, 0, 1)]
      + D * denomQi * denomQi * [getR(0, 1, 1)]
    end
    results[0][3] = rexpr
      (
      (D * Qix * Qix + D * denomQi) * [getR(0, 0, 0)]
      -2 * D * denomQi * Qix * [getR(1, 0, 0)]
      + D * denomQi * denomQi * [getR(2, 0, 0)]
      ) * 0.577350269 -- = normc<2,0,0>(), I'm not sure where this comes from.
    end
    results[0][4] = rexpr
      (
      (D * Qiy * Qiy + D * denomQi) * [getR(0, 0, 0)]
      -2 * D * Qiy * denomQi * [getR(0, 1, 0)]
      + D * denomQi * denomQi * [getR(0, 2, 0)]
      ) * 0.577350269
    end
    results[0][5] = rexpr
      (
      (D * Qiz * Qiz + D * denomQi) * [getR(0, 0, 0)]
      -2 * D * denomQi * Qiz * [getR(0, 0, 1)]
      + D * denomQi * denomQi * [getR(0, 0, 2)]
      ) * 0.577350269
    end

  elseif L1 == 0 and L2 == 0 and L3 == 2 and L4 == 2 then
  -------------------------------------SSDD-------------------------------------

    -- TODO: Does not work for some reason.
    -- results[0][0] = rexpr
    --   [getR(0, 0, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 0}]
    --     + [density][0][1] * ket_prevals[{ket_idx, 1}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 2}]
    --     + [density][0][3] * ket_prevals[{ket_idx, 3}]
    --     + [density][0][4] * ket_prevals[{ket_idx, 4}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 5}]
    --   )
    --   - [getR(1, 0, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 6}]
    --     + [density][0][1] * ket_prevals[{ket_idx, 7}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 8}]
    --     + [density][0][3] * ket_prevals[{ket_idx, 9}]
    --     + [density][0][4] * ket_prevals[{ket_idx, 10}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 11}]
    --   )
    --   - [getR(0, 1, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 12}]
    --     + [density][0][1] * ket_prevals[{ket_idx, 13}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 14}]
    --     + [density][0][3] * ket_prevals[{ket_idx, 15}]
    --     + [density][0][4] * ket_prevals[{ket_idx, 16}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 17}]
    --   )
    --   - [getR(0, 0, 1)] * (
    --     [density][0][1] * ket_prevals[{ket_idx, 18}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 19}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 20}]
    --   )
    --
    --   + [getR(1, 1, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 21}]
    --     + [density][0][1] * ket_prevals[{ket_idx, 22}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 23}]
    --     + [density][0][3] * ket_prevals[{ket_idx, 24}]
    --     + [density][0][4] * ket_prevals[{ket_idx, 25}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 26}]
    --   )
    --   + [getR(1, 0, 1)] * (
    --     [density][0][1] * ket_prevals[{ket_idx, 27}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 28}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 29}]
    --   )
    --   + [getR(0, 1, 1)] * (
    --     [density][0][1] * ket_prevals[{ket_idx, 30}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 31}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 32}]
    --   )
    --   + [getR(2, 0, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 33}]
    --     + [density][0][1] * ket_prevals[{ket_idx, 34}]
    --     + [density][0][3] * ket_prevals[{ket_idx, 35}]
    --   )
    --   + [getR(0, 2, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 36}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 37}]
    --     + [density][0][4] * ket_prevals[{ket_idx, 38}]
    --   )
    --   + [getR(0, 0, 2)] * (
    --     [density][0][5] * ket_prevals[{ket_idx, 39}]
    --   )
    --
    --   - [getR(1, 1, 1)] * (
    --     [density][0][1] * ket_prevals[{ket_idx, 40}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 41}]
    --     + [density][0][5] * ket_prevals[{ket_idx, 42}]
    --   )
    --   - [getR(2, 1, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 43}]
    --     + [density][0][1] * ket_prevals[{ket_idx, 44}]
    --     + [density][0][3] * ket_prevals[{ket_idx, 45}]
    --   )
    --   - [getR(1, 2, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 46}]
    --     + [density][0][2] * ket_prevals[{ket_idx, 47}]
    --     + [density][0][4] * ket_prevals[{ket_idx, 48}]
    --   )
    --   - [getR(2, 0, 1)] * (
    --     [density][0][1] * ket_prevals[{ket_idx, 49}]
    --   )
    --   - [getR(0, 2, 1)] * (
    --     [density][0][2] * ket_prevals[{ket_idx, 50}]
    --   )
    --   - [getR(1, 0, 2)] * (
    --     [density][0][5] * ket_prevals[{ket_idx, 51}]
    --   )
    --   - [getR(0, 1, 2)] * (
    --     [density][0][5] * ket_prevals[{ket_idx, 52}]
    --   )
    --   - [getR(3, 0, 0)] * (
    --     [density][0][3] * ket_prevals[{ket_idx, 53}]
    --   )
    --   - [getR(0, 3, 0)] * (
    --     [density][0][4] * ket_prevals[{ket_idx, 54}]
    --   )
    --
    --   + [getR(2, 1, 1)] * (
    --     [density][0][1] * ket_prevals[{ket_idx, 55}]
    --   )
    --   + [getR(1, 2, 1)] * (
    --     [density][0][2] * ket_prevals[{ket_idx, 55}]
    --   )
    --   + [getR(1, 1, 2)] * (
    --     [density][0][5] * ket_prevals[{ket_idx, 55}]
    --   )
    --   + [getR(2, 2, 0)] * (
    --     [density][0][0] * ket_prevals[{ket_idx, 55}]
    --   )
    --   + [getR(3, 1, 0)] * (
    --     [density][0][3] * ket_prevals[{ket_idx, 55}]
    --   )
    --   + [getR(1, 3, 0)] * (
    --     [density][0][4] * ket_prevals[{ket_idx, 55}]
    --   )
    -- end

  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 2 then
  -------------------------------------PSPD-------------------------------------

    local D00 = rexpr [density][0][0] end
    local D01 = rexpr [density][0][1] end
    local D02 = rexpr [density][0][2] end
    local D03 = rexpr [density][0][3] end
    local D04 = rexpr [density][0][4] end
    local D05 = rexpr [density][0][5] end

    local R0000 = rexpr [getR(0, 0, 0)] end
    local R1000 = rexpr [getR(1, 0, 0)] end
    local R0100 = rexpr [getR(0, 1, 0)] end
    local R0010 = rexpr [getR(0, 0, 1)] end
    local R1100 = rexpr [getR(1, 1, 0)] end
    local R1010 = rexpr [getR(1, 0, 1)] end
    local R0110 = rexpr [getR(0, 1, 1)] end
    local R2000 = rexpr [getR(2, 0, 0)] end
    local R0200 = rexpr [getR(0, 2, 0)] end
    local R0020 = rexpr [getR(0, 0, 2)] end
    local R1110 = rexpr [getR(1, 1, 1)] end
    local R2100 = rexpr [getR(2, 1, 0)] end
    local R2010 = rexpr [getR(2, 0, 1)] end
    local R1200 = rexpr [getR(1, 2, 0)] end
    local R1020 = rexpr [getR(1, 0, 2)] end
    local R2200 = rexpr [getR(2, 2, 0)] end
    local R2110 = rexpr [getR(2, 1, 1)] end
    local R2020 = rexpr [getR(2, 0, 2)] end
    local R3000 = rexpr [getR(3, 0, 0)] end
    local R3100 = rexpr [getR(3, 1, 0)] end
    local R3010 = rexpr [getR(3, 0, 1)] end
    local R4000 = rexpr [getR(4, 0, 0)] end

    local coeff = rexpr
      D00*ket_prevals[{ket_idx, 0}]
      + D01*ket_prevals[{ket_idx, 1}]
      + D02*ket_prevals[{ket_idx, 2}]
      + D03*ket_prevals[{ket_idx, 3}]
      + D04*ket_prevals[{ket_idx, 4}]
      + D05*ket_prevals[{ket_idx, 5}]
    end

    local L000 = rexpr R0000 * coeff end
    local L100 = rexpr R1000 * coeff end

    coeff = rexpr
      D00*ket_prevals[{ket_idx, 6}]
      + D01*ket_prevals[{ket_idx, 7}]
      + D02*ket_prevals[{ket_idx, 8}]
      + D03*ket_prevals[{ket_idx, 9}]
      + D04*ket_prevals[{ket_idx, 10}]
      + D05*ket_prevals[{ket_idx, 11}]
    end

    L000 = rexpr L000 - R1000 * coeff end
    L100 = rexpr L100 - R2000 * coeff end

    coeff = rexpr
      D00*ket_prevals[{ket_idx, 12}]
      + D02*ket_prevals[{ket_idx, 13}]
      + D04*ket_prevals[{ket_idx, 14}]
    end

    L000 = rexpr L000 - R0100 * coeff end
    L100 = rexpr L100 - R1100 * coeff end

    coeff = rexpr
      D01*ket_prevals[{ket_idx,  15}]
      + D02*ket_prevals[{ket_idx,  16}]
      + D05*ket_prevals[{ket_idx,  17}]
    end

    L000 = rexpr L000 - R0010 * coeff end
    L100 = rexpr L100 - R1010 * coeff end

    coeff = rexpr
      D00*ket_prevals[{ket_idx,  18}]
      + D02*ket_prevals[{ket_idx,  19}]
      + D04*ket_prevals[{ket_idx,  20}]
    end

    L000 = rexpr L000 + R1100 * coeff end
    L100 = rexpr L100 + R2100 * coeff end

    coeff = rexpr
      D01*ket_prevals[{ket_idx,  21}]
      + D02*ket_prevals[{ket_idx,  22}]
      + D05*ket_prevals[{ket_idx,  23}]
    end

    L000 = rexpr L000 + R1010 * coeff end
    L100 = rexpr L100 + R2010 * coeff end

    coeff = rexpr
      D02*ket_prevals[{ket_idx,  24}]
    end

    L000 = rexpr L000 + R0110 * coeff end
    L100 = rexpr L100 + R1110 * coeff end

    coeff = rexpr
      D00*ket_prevals[{ket_idx,  25}]
      + D01*ket_prevals[{ket_idx,  26}]
      + D03*ket_prevals[{ket_idx,  27}]
    end

    L000 = rexpr L000 + R2000 * coeff end
    L100 = rexpr L100 + R3000 * coeff end

    coeff = rexpr
      D04*ket_prevals[{ket_idx,  28}]
    end

    L000 = rexpr L000 + R0200 * coeff end
    L100 = rexpr L100 + R1200 * coeff end

    coeff = rexpr
      D05*ket_prevals[{ket_idx,  29}]
    end

    L000 = rexpr L000 + R0020 * coeff end
    L100 = rexpr L100 + R1020 * coeff end

    coeff = rexpr
      D02*ket_prevals[{ket_idx,  30}]
    end

    L000 = rexpr L000 - R1110 * coeff end
    L100 = rexpr L100 - R2110 * coeff end

    coeff = rexpr
      D00*ket_prevals[{ket_idx,  30}]
    end

    L000 = rexpr L000 - R2100 * coeff end
    L100 = rexpr L100 - R3100 * coeff end

    coeff = rexpr
      D04*ket_prevals[{ket_idx,  30}]
    end

    L000 = rexpr L000 - R1200 * coeff end
    L100 = rexpr L100 - R2200 * coeff end

    coeff = rexpr
      D01*ket_prevals[{ket_idx,  30}]
    end

    L000 = rexpr L000 - R2010 * coeff end
    L100 = rexpr L100 - R3010 * coeff end


    coeff = rexpr
      D05*ket_prevals[{ket_idx,  30}]
    end

    L000 = rexpr L000 - R1020 * coeff end
    L100 = rexpr L100 - R2020 * coeff end


    coeff = rexpr
      D03*ket_prevals[{ket_idx,  30}]
    end

    L000 = rexpr L000 - R3000 * coeff end
    L100 = rexpr L100 - R4000 * coeff end


    results[0][0] = rexpr
      L000*bra_prevals[{bra_idx, 0}]
      + L100*bra_prevals[{bra_idx, 3}]
    end

    -- TODO: results[0][0..8] are not yet written.

  else
    -- All kernels above PPPP except SSSD and SSDS.
    -- TODO
  end

  local statements = terralib.newlist()
  for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
    for k = 0, triangle_number(L3 + 1) - 1 do -- inclusive
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
          if bra.ishell_index < ket.ishell_index then -- Upper triangular element.
            [output][i][k] += [results[i][k]]
          elseif bra.ishell_index == ket.ishell_index then -- Diagonal element
            [output][i][k] += factor * [results[i][k]]
          else -- Lower triangular element.
            -- NOTE: Diagonal kernels skip the lower triangular elements.
            -- no-op
          end
        end)
      else -- Upper triangular kernel.
        statements:insert(rquote
          [output][i][k] += [results[i][k]]
        end)
      end
    end
  end

  return statements
end
