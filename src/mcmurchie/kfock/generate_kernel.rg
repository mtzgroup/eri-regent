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

  if L1 <= 1 and L2 <= 1 and L3 <= 1 and L4 <= 1 then
    -------------------------------- PPPP --------------------------------
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
        local result = rexpr 0 end
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

          result = rexpr
            result + (
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
            result = rexpr
              result + denomPj * (
                [Qi[k+1]] * [aux1(0, 0)]
                - denomQi * [aux1(1, k)]
                + [aux0(0, 0)]
              )
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
