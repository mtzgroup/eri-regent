import "regent"

require "helper"

-- This function returns the index of elements on an NxN triangle.
--     5
--   3 4
-- 0 1 2
local function magic(x, y, N)
  local y, x = math.min(x, y), math.max(x, y)
  return x + y * N - y * (y + 1) / 2
end
-- Similar, but in 3d!
local function magic3(x, y, z)
  local pattern = {
    {{0, 1, 2},
     {1, 3, 4},
     {2, 4, 5}},

    {{1, 3, 4},
     {3, 6, 7},
     {4, 7, 8}},

    {{2, 4, 5},
     {4, 7, 8},
     {5, 8, 9}}
  }
  return pattern[x+1][y+1][z+1]
end

function genY(R, P, n, i)
  local x, y, z = unpack(generateKFockSpinPattern(n)[i+1])
  return rexpr [R[x][y][z][0]] * P end
end

function genX(R, P, D, sign, eta, n, i)
  local function getR(N, L, M, j)
    if R[N] == nil or R[N][L] == nil or R[N][L][M] == nil or R[N][L][M][j] == nil then
      return 0
    else
      return R[N][L][M][j]
    end
  end

  if eta == nil then
    sign = 0
    eta = 1
  end

  local a, b, c = unpack(D)
  local x, y, z = unpack(P)
  local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
  return rexpr
    [getR(q, r, s, 0)] * (a * x + b * y + c * z)
      + sign * [getR(q+1, r, s, 0)] * a / (2 * eta)
      + sign * [getR(q, r+1, s, 0)] * b / (2 * eta)
      + sign * [getR(q, r, s+1, 0)] * c / (2 * eta)
  end
end

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket,
                                       density, output)

  local function getR(N, L, M)
    if R[N] == nil or R[N][L] == nil or R[N][L][M] == nil or R[N][L][M][0] == nil then
      return 0
    else
      return R[N][L][M][0]
    end
  end

  local function getBraPi()
    if L1 == 0 then
      return {1, 0, 0}
    else
      return {
        rexpr bra.ishell_location.x end,
        rexpr bra.ishell_location.y end,
        rexpr bra.ishell_location.z end
      }
    end
  end
  local function getBraPj()
    if L2 == 0 then
      return {1, 0, 0}
    else
      return {
        rexpr bra.jshell_location.x end,
        rexpr bra.jshell_location.y end,
        rexpr bra.jshell_location.z end
      }
    end
  end
  local function getKetPi()
    if L3 == 0 then
      return {1, 0, 0}
    else
      return {
        rexpr ket.ishell_location.x end,
        rexpr ket.ishell_location.y end,
        rexpr ket.ishell_location.z end
      }
    end
  end
  local function getKetPj()
    if L4 == 0 then
      return {1, 0, 0}
    else
      return {
        rexpr ket.jshell_location.x end,
        rexpr ket.jshell_location.y end,
        rexpr ket.jshell_location.z end
      }
    end
  end

  local H1, H2 = triangle_number(L1 + 1), triangle_number(L2 + 1)
  local H3, H4 = triangle_number(L3 + 1), triangle_number(L4 + 1)
  local statements = terralib.newlist()
  local results = {}
  for i = 0, H1-1 do -- inclusive
    results[i] = {}
    for k = 0, H3-1 do -- inclusive
      results[i][k] = regentlib.newsymbol(double, "result"..i.."_"..k)
      statements:insert(rquote var [results[i][k]] = 0 end)
    end
  end

  if (L1 == 0 and L2 == 0 and L3 == 0 and L4 == 0)
      or (L1 == 0 and L2 == 0 and L3 == 0 and L4 == 1)
      or (L1 == 0 and L2 == 1 and L3 == 0 and L4 == 1)
      or (L1 == 0 and L2 == 0 and L3 == 1 and L4 == 0) then
    local Pi = getBraPi()
    local Pj = getBraPj()
    local Qi = getKetPi()
    local Qj = getKetPj()
    local denomP, denomQ
    if L2 == 0 then
      denomP = 0
    else
      denomP = rexpr 1 / (2 * bra.eta) end
    end
    if L3 == 0 then
      denomQ = 0
    else
      denomQ = rexpr 1 / (2 * ket.eta) end
    end
    local bra_eta, ket_eta = rexpr bra.eta end, rexpr ket.eta end
    if L3 == 1 and L4 == 0 then
      ket_eta = nil
    end
    for k = 0, H3-1 do -- inclusive
      if L3 == 1 then
        if k == 0 then
          Pj = {1, 0, 0}
        elseif k == 1 then
          Pj = {0, 1, 0}
        else
          Pj = {0, 0, 1}
        end
      end
      local result = rexpr 0 end
      for i = 0, 2 do -- inclusive
        if L2 == 1 and L3 == 0 then
          Qi = {1, 1, 1}
        end
        local D
        if L2 == 0 and L4 == 0 then
          D = {rexpr [density][0][0] end, 0, 0}
        else
          D = {rexpr [density][i][0] end,
               rexpr [density][i][1] end,
               rexpr [density][i][2] end}
        end
        result = rexpr
          result + (
            [Pj[i+1]] * (
              [Qi[i+1]] * [genX(R, Qj, D, -1, ket_eta, 0, 0)]
              - denomQ * [genX(R, Qj, D, -1, ket_eta, 1, i)]
            )
            + denomP * (
              [Qi[i+1]] * [genX(R, Qj, D, -1, ket_eta, 1, i)]
              - denomQ * [genX(R, Qj, D, -1, ket_eta, 2, 0)]
            )
          )
        end
      end
      statements:insert(rquote
        [results[0][k]] = result
      end)
    end

  elseif L1 == 0 and L2 == 1 and L3 == 1 and L4 == 0 then
    -------------------------------- SPPS --------------------------------
    local bra_Pj = getBraPj()
    local ket_Pi = getKetPi()
    local ket_Pj = getKetPj()
    local D = {rexpr [density][0][0] end,
               rexpr [density][0][1] end,
               rexpr [density][0][2] end}
    local eta = rexpr bra.eta end
    local x00 = genX(R, bra_Pj, D, 1, eta, 0, 0)
    local x10 = genX(R, bra_Pj, D, 1, eta, 1, 0)
    local x11 = genX(R, bra_Pj, D, 1, eta, 1, 1)
    local x12 = genX(R, bra_Pj, D, 1, eta, 1, 2)
    statements:insert(rquote
      [results[0][0]] = [ket_Pi[1]] * x00 - x10 / (2 * ket.eta)
      ;[results[0][1]] = [ket_Pi[2]] * x00 - x11 / (2 * ket.eta)
      ;[results[0][2]] = [ket_Pi[3]] * x00 - x12 / (2 * ket.eta)
    end)

  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 0 then
    -------------------------------- PSPS --------------------------------
    local bra_Pi = getBraPi()
    local bra_Pj = getBraPj()
    local ket_Pi = getKetPi()
    local D = {rexpr [density][0][0] end, 0, 0}
    local x00 = genX(R, bra_Pj, D, 0, 1, 0, 0)
    local x10 = genX(R, bra_Pj, D, 0, 1, 1, 0)
    local x11 = genX(R, bra_Pj, D, 0, 1, 1, 1)
    local x12 = genX(R, bra_Pj, D, 0, 1, 1, 2)
    local x20 = genX(R, bra_Pj, D, 0, 1, 2, 0)
    local x21 = genX(R, bra_Pj, D, 0, 1, 2, 1)
    local x22 = genX(R, bra_Pj, D, 0, 1, 2, 2)
    local x23 = genX(R, bra_Pj, D, 0, 1, 2, 3)
    local x24 = genX(R, bra_Pj, D, 0, 1, 2, 4)
    local x25 = genX(R, bra_Pj, D, 0, 1, 2, 5)
    statements:insert(rquote
      [results[0][0]] = (
        [bra_Pi[1]] * ([ket_Pi[1]] * x00 - x10 / (2 * ket.eta))
        + ([ket_Pi[1]] * x10 - x20 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[0][1]] = (
        [bra_Pi[1]] * ([ket_Pi[2]] * x00 - x11 / (2 * ket.eta))
        + ([ket_Pi[2]] * x10 - x21 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[0][2]] = (
        [bra_Pi[1]] * ([ket_Pi[3]] * x00 - x12 / (2 * ket.eta))
        + ([ket_Pi[3]] * x10 - x22 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[1][0]] = (
        [bra_Pi[2]] * ([ket_Pi[1]] * x00 - x10 / (2 * ket.eta))
        + ([ket_Pi[1]] * x11 - x21 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[1][1]] = (
        [bra_Pi[2]] * ([ket_Pi[2]] * x00 - x11 / (2 * ket.eta))
        + ([ket_Pi[2]] * x11 - x23 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[1][2]] = (
        [bra_Pi[2]] * ([ket_Pi[3]] * x00 - x12 / (2 * ket.eta))
        + ([ket_Pi[3]] * x11 - x24 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[2][0]] = (
        [bra_Pi[3]] * ([ket_Pi[1]] * x00 - x10 / (2 * ket.eta))
        + ([ket_Pi[1]] * x12 - x22 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[2][1]] = (
        [bra_Pi[3]] * ([ket_Pi[2]] * x00 - x11 / (2 * ket.eta))
        + ([ket_Pi[2]] * x12 - x24 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[2][2]] = (
        [bra_Pi[3]] * ([ket_Pi[3]] * x00 - x12 / (2 * ket.eta))
        + ([ket_Pi[3]] * x12 - x25 / (2 * ket.eta)) / (2 * bra.eta)
      )
    end)

  elseif L1 == 0 and L2 == 0 and L3 == 1 and L4 == 1 then
    -------------------------------- SSPP --------------------------------
    local ket_Pi = getKetPi()
    local ket_Pj = getKetPj()
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local D = {rexpr [density][0][0] end,
               rexpr [density][0][1] end,
               rexpr [density][0][2] end}
    local eta = rexpr ket.eta end
    local x00 = genX(R, ket_Pj, D, -1, eta, 0, 0)
    local x10 = genX(R, ket_Pj, D, -1, eta, 1, 0)
    local x11 = genX(R, ket_Pj, D, -1, eta, 1, 1)
    local x12 = genX(R, ket_Pj, D, -1, eta, 1, 2)
    statements:insert(rquote
      [results[0][0]] = (
        [ket_Pi[1]] * x00 - x10 / (2 * ket.eta) + [genY(R, P1, 0, 0)]
      )
      ;[results[0][1]] = (
        [ket_Pi[2]] * x00 - x11 / (2 * ket.eta) + [genY(R, P2, 0, 0)]
      )
      ;[results[0][2]] = (
        [ket_Pi[3]] * x00 - x12 / (2 * ket.eta) + [genY(R, P3, 0, 0)]
      )
    end)

  elseif L1 == 0 and L2 == 1 and L3 == 1 and L4 == 1 then
    -------------------------------- SPPP --------------------------------
    local bra_Pj = getBraPj()
    local ket_Pi = getKetPi()
    local ket_Pj = getKetPj()
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local D = {rexpr [density][0][0] end,
               rexpr [density][0][1] end,
               rexpr [density][0][2] end}
    local eta = rexpr ket.eta end
    local x00 = genX(R, ket_Pj, D, -1, eta, 0, 0)
    local x10 = genX(R, ket_Pj, D, -1, eta, 1, 0)
    local x11 = genX(R, ket_Pj, D, -1, eta, 1, 1)
    local x12 = genX(R, ket_Pj, D, -1, eta, 1, 2)
    local x20 = genX(R, ket_Pj, D, -1, eta, 2, 0)
    local x21 = genX(R, ket_Pj, D, -1, eta, 2, 1)
    local x22 = genX(R, ket_Pj, D, -1, eta, 2, 2)
    statements:insert(rquote
      [results[0][0]] = (
        [bra_Pj[1]] * ([ket_Pi[1]] * x00 - x10 / (2 * ket.eta) + [genY(R, P1, 0, 0)])
        + ([ket_Pi[1]] * x10 - x20 / (2 * ket.eta) + [genY(R, P1, 1, 0)]) / (2 * bra.eta)
      )
      ;[results[0][1]] = (
        [bra_Pj[1]] * ([ket_Pi[2]] * x00 - x11 / (2 * ket.eta) + [genY(R, P2, 0, 0)])
        + ([ket_Pi[2]] * x10 - x21 / (2 * ket.eta) + [genY(R, P2, 1, 0)]) / (2 * bra.eta)
      )
      ;[results[0][2]] = (
        [bra_Pj[1]] * ([ket_Pi[3]] * x00 - x12 / (2 * ket.eta) + [genY(R, P3, 0, 0)])
        + ([ket_Pi[3]] * x10 - x22 / (2 * ket.eta) + [genY(R, P3, 1, 0)]) / (2 * bra.eta)
      )
    end)

    local P1 = rexpr [density][1][0] / (2 * ket.eta) end
    local P2 = rexpr [density][1][1] / (2 * ket.eta) end
    local P3 = rexpr [density][1][2] / (2 * ket.eta) end
    local D = {rexpr [density][1][0] end,
               rexpr [density][1][1] end,
               rexpr [density][1][2] end}
    local x00 = genX(R, ket_Pj, D, -1, eta, 0, 0)
    local x10 = genX(R, ket_Pj, D, -1, eta, 1, 0)
    local x11 = genX(R, ket_Pj, D, -1, eta, 1, 1)
    local x12 = genX(R, ket_Pj, D, -1, eta, 1, 2)
    local x21 = genX(R, ket_Pj, D, -1, eta, 2, 1)
    local x23 = genX(R, ket_Pj, D, -1, eta, 2, 3)
    local x24 = genX(R, ket_Pj, D, -1, eta, 2, 4)
    statements:insert(rquote
      [results[0][0]] += (
        [bra_Pj[2]] * ([ket_Pi[1]] * x00 - x10 / (2 * ket.eta) + [genY(R, P1, 0, 0)])
        + ([ket_Pi[1]] * x11 - x21 / (2 * ket.eta) + [genY(R, P1, 1, 1)]) / (2 * bra.eta)
      )
      ;[results[0][1]] += (
        [bra_Pj[2]] * ([ket_Pi[2]] * x00 - x11 / (2 * ket.eta) + [genY(R, P2, 0, 0)])
        + ([ket_Pi[2]] * x11 - x23 / (2 * ket.eta) + [genY(R, P2, 1, 1)]) / (2 * bra.eta)
      )
      ;[results[0][2]] += (
        [bra_Pj[2]] * ([ket_Pi[3]] * x00 - x12 / (2 * ket.eta) + [genY(R, P3, 0, 0)])
        + ([ket_Pi[3]] * x11 - x24 / (2 * ket.eta) + [genY(R, P3, 1, 1)]) / (2 * bra.eta)
      )
    end)

    local P1 = rexpr [density][2][0] / (2 * ket.eta) end
    local P2 = rexpr [density][2][1] / (2 * ket.eta) end
    local P3 = rexpr [density][2][2] / (2 * ket.eta) end
    local D = {rexpr [density][2][0] end,
               rexpr [density][2][1] end,
               rexpr [density][2][2] end}
    local x00 = genX(R, ket_Pj, D, -1, eta, 0, 0)
    local x10 = genX(R, ket_Pj, D, -1, eta, 1, 0)
    local x11 = genX(R, ket_Pj, D, -1, eta, 1, 1)
    local x12 = genX(R, ket_Pj, D, -1, eta, 1, 2)
    local x22 = genX(R, ket_Pj, D, -1, eta, 2, 2)
    local x24 = genX(R, ket_Pj, D, -1, eta, 2, 4)
    local x25 = genX(R, ket_Pj, D, -1, eta, 2, 5)
    statements:insert(rquote
      [results[0][0]] += (
        [bra_Pj[3]] * ([ket_Pi[1]] * x00 - x10 / (2 * ket.eta) + [genY(R, P1, 0, 0)])
        + ([ket_Pi[1]] * x12 - x22 / (2 * ket.eta) + [genY(R, P1, 1, 2)]) / (2 * bra.eta)
      )
      ;[results[0][1]] += (
        [bra_Pj[3]] * ([ket_Pi[2]] * x00 - x11 / (2 * ket.eta) + [genY(R, P2, 0, 0)])
        + ([ket_Pi[2]] * x12 - x24 / (2 * ket.eta) + [genY(R, P2, 1, 2)]) / (2 * bra.eta)
      )
      ;[results[0][2]] += (
        [bra_Pj[3]] * ([ket_Pi[3]] * x00 - x12 / (2 * ket.eta) + [genY(R, P3, 0, 0)])
        + ([ket_Pi[3]] * x12 - x25 / (2 * ket.eta) + [genY(R, P3, 1, 2)]) / (2 * bra.eta)
      )
    end)

  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 1 then
    -------------------------------- PSPP --------------------------------
    local Pi, Pj = getBraPi(), getBraPj()
    local Qi, Qj = getKetPi(), getKetPj()
    local bra_eta, ket_eta = rexpr bra.eta end, rexpr ket.eta end

    for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
      for k = 0, triangle_number(L3 + 1) - 1 do -- inclusive
        local result = rexpr 0 end
        for n = 0, 2 do -- inclusive
          local density_triplet = {rexpr [density][n][0] end,
                                   rexpr [density][n][1] end,
                                   rexpr [density][n][2] end}

          local function aux0(n, i)
            local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
            return rexpr
              [getR(q, r, s)] * [density_triplet[k+1]] / (2 * ket_eta)
            end
          end
          local function aux1(n, i)
            local a, b, c = unpack(density_triplet)
            local x, y, z = unpack(Qj)
            local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
            return rexpr
              [getR(q, r, s)] * (a * x + b * y + c * z)
              - ([getR(q+1, r, s)] * a
                  + [getR(q, r+1, s)] * b
                  + [getR(q, r, s+1)] * c) / (2 * ket_eta)
            end
          end
          result = rexpr
            result + (
              [Pi[i+1]] * (
                [Pj[n+1]] * (
                  [Qi[k+1]] * [aux1(0, 0)]
                  - [aux1(1, k)] / (2 * ket_eta)
                  + [aux0(0, 0)]
                )
                -- + (
                --   [Qi[k+1]] * [aux1(1, n)]
                --   - [aux1(2, magic(k, n, 3))] / (2 * ket_eta)
                --   + [aux0(1, n)]
                -- ) / (2 * bra_eta)
              )
              + (
                [Pj[n+1]] * (
                  [Qi[k+1]] * [aux1(1, i)]
                  - [aux1(2, magic(i, k, 3))] / (2 * ket_eta)
                  + [aux0(1, i)]
                )
                -- + (
                --   [Qi[k+1]] * [aux1(2, magic(i, n, 3))]
                --   - [aux1(3, magic3(i, k, n))] / (2 * ket_eta)
                --   + [aux0(2, magic(i, n, 3))]
                -- ) / (2 * bra_eta)
              ) / (2 * bra_eta)
            )
          end
          -- if i == n then
          --   result = rexpr
          --     result + (
          --       [Qi[k+1]] * [aux1(0, 0)]
          --       - [aux1(1, k)] / (2 * ket_eta)
          --       + [aux0(0, 0)]
          --     ) / (2 * bra_eta)
          --   end
          -- end
        end

        statements:insert(rquote
          [results[i][k]] = result
        end)
      end
    end

  elseif L1 == 1 and L2 == 1 and L3 == 1 and L4 == 1 then
    -------------------------------- PPPP --------------------------------
    local Pi, Pj = getBraPi(), getBraPj()
    local Qi, Qj = getKetPi(), getKetPj()
    local bra_eta, ket_eta = rexpr bra.eta end, rexpr ket.eta end

    for i = 0, triangle_number(L1 + 1) - 1 do -- inclusive
      for k = 0, triangle_number(L3 + 1) - 1 do -- inclusive
        local result = rexpr 0 end
        for n = 0, 2 do -- inclusive
          local density_triplet = {rexpr [density][n][0] end,
                                   rexpr [density][n][1] end,
                                   rexpr [density][n][2] end}

          local function aux0(n, i)
            local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
            return rexpr
              [getR(q, r, s)] * [density_triplet[k+1]] / (2 * ket_eta)
            end
          end
          local function aux1(n, i)
            local a, b, c = unpack(density_triplet)
            local x, y, z = unpack(Qj)
            local q, r, s = unpack(generateKFockSpinPattern(n)[i+1])
            return rexpr
              [getR(q, r, s)] * (a * x + b * y + c * z)
              - ([getR(q+1, r, s)] * a
                  + [getR(q, r+1, s)] * b
                  + [getR(q, r, s+1)] * c) / (2 * ket_eta)
            end
          end
          result = rexpr
            result + (
              [Pi[i+1]] * (
                [Pj[n+1]] * (
                  [Qi[k+1]] * [aux1(0, 0)]
                  - [aux1(1, k)] / (2 * ket_eta)
                  + [aux0(0, 0)]
                )
                + (
                  [Qi[k+1]] * [aux1(1, n)]
                  - [aux1(2, magic(k, n, 3))] / (2 * ket_eta)
                  + [aux0(1, n)]
                ) / (2 * bra_eta)
              )
              + (
                [Pj[n+1]] * (
                  [Qi[k+1]] * [aux1(1, i)]
                  - [aux1(2, magic(i, k, 3))] / (2 * ket_eta)
                  + [aux0(1, i)]
                )
                + (
                  [Qi[k+1]] * [aux1(2, magic(i, n, 3))]
                  - [aux1(3, magic3(i, k, n))] / (2 * ket_eta)
                  + [aux0(2, magic(i, n, 3))]
                ) / (2 * bra_eta)
              ) / (2 * bra_eta)
            )
          end
          if i == n then
            result = rexpr
              result + (
                [Qi[k+1]] * [aux1(0, 0)]
                - [aux1(1, k)] / (2 * ket_eta)
                + [aux0(0, 0)]
              ) / (2 * bra_eta)
            end
          end
        end

        statements:insert(rquote
          [results[i][k]] = result
        end)
      end
    end
  else
    -- assert(false, "Unimplemented KFock kernel!")
  end

  for i = 0, H1-1 do -- inclusive
    for k = 0, H3-1 do -- inclusive
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
