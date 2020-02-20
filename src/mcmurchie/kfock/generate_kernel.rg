import "regent"

require "helper"

function generateXExpression(R, P0, P1, P2, P3, n, i)
  local x, y, z = unpack(generateKFockSpinPattern(n)[i+1])
  if P1 == 0 and P2 == 0 and P3 == 0 then
    return rexpr [R[x][y][z][0]] * P0 end
  else
    return rexpr
      [R[x][y][z][0]] * P0
      - [R[x+1][y][z][0]] * P1
      - [R[x][y+1][z][0]] * P2
      - [R[x][y][z+1][0]] * P3
    end
  end
end

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket,
                                       density, output)
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

  if L1 == 0 and L2 == 0 and L3 == 0 and L4 == 0 then
    -------------------------------- SSSS --------------------------------
    local P0 = rexpr [density][0][0] end
    local x00 = generateXExpression(R, P0, 0, 0, 0, 0, 0)
    statements:insert(rquote
      [results[0][0]] = x00
    end)

  elseif L1 == 0 and L2 == 0 and L3 == 0 and L4 == 1 then
    -------------------------------- SSSP --------------------------------
    local ket_Pj = rexpr ket.jshell_location end
    local P0 = rexpr
      ket_Pj.x * [density][0][0]
      + ket_Pj.y * [density][0][1]
      + ket_Pj.z * [density][0][2]
    end
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    statements:insert(rquote
      [results[0][0]] = x00
    end)

  elseif L1 == 0 and L2 == 1 and L3 == 0 and L4 == 1 then
    -------------------------------- SPSP --------------------------------
    local bra_Pj = rexpr bra.jshell_location end
    local ket_Pj = rexpr ket.jshell_location end
    local P0 = rexpr
      ket_Pj.x * [density][0][0]
      + ket_Pj.y * [density][0][1]
      + ket_Pj.z * [density][0][2]
    end
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    statements:insert(rquote
      [results[0][0]] = bra_Pj.x * x00 + x10 / (2 * bra.eta)
    end)
    local P0 = rexpr
      ket_Pj.x * [density][1][0]
      + ket_Pj.y * [density][1][1]
      + ket_Pj.z * [density][1][2]
    end
    local P1 = rexpr [density][1][0] / (2 * ket.eta) end
    local P2 = rexpr [density][1][1] / (2 * ket.eta) end
    local P3 = rexpr [density][1][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    statements:insert(rquote
      [results[0][0]] += bra_Pj.y * x00 + x11 / (2 * bra.eta)
    end)
    local P0 = rexpr
      ket_Pj.x * [density][2][0]
      + ket_Pj.y * [density][2][1]
      + ket_Pj.z * [density][2][2]
    end
    local P1 = rexpr [density][2][0] / (2 * ket.eta) end
    local P2 = rexpr [density][2][1] / (2 * ket.eta) end
    local P3 = rexpr [density][2][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    statements:insert(rquote
      [results[0][0]] += bra_Pj.z * x00 + x12 / (2 * bra.eta)
    end)

  elseif L1 == 0 and L2 == 0 and L3 == 1 and L4 == 0 then
    -------------------------------- SSPS --------------------------------
    local P0 = rexpr [density][0][0] end
    local x00 = generateXExpression(R, P0, 0, 0, 0, 0, 0)
    local x10 = generateXExpression(R, P0, 0, 0, 0, 1, 0)
    local x11 = generateXExpression(R, P0, 0, 0, 0, 1, 1)
    local x12 = generateXExpression(R, P0, 0, 0, 0, 1, 2)
    statements:insert(rquote
      var ket_Pi = ket.ishell_location
      ;[results[0][0]] = ket_Pi.x * x00 - x10 / (2 * ket.eta)
      ;[results[0][1]] = ket_Pi.y * x00 - x11 / (2 * ket.eta)
      ;[results[0][2]] = ket_Pi.z * x00 - x12 / (2 * ket.eta)
    end)

  elseif L1 == 0 and L2 == 1 and L3 == 1 and L4 == 0 then
    -------------------------------- SPPS --------------------------------
    local bra_Pj = rexpr bra.jshell_location end
    local ket_Pi = rexpr ket.ishell_location end
    local P0 = rexpr
      bra_Pj.x * [density][0][0]
      + bra_Pj.y * [density][0][1]
      + bra_Pj.z * [density][0][2]
    end
    local P1 = rexpr -[density][0][0] / (2 * bra.eta) end
    local P2 = rexpr -[density][0][1] / (2 * bra.eta) end
    local P3 = rexpr -[density][0][2] / (2 * bra.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    statements:insert(rquote
      [results[0][0]] = ket_Pi.x * x00 - x10 / (2 * ket.eta)
      ;[results[0][1]] = ket_Pi.y * x00 - x11 / (2 * ket.eta)
      ;[results[0][2]] = ket_Pi.z * x00 - x12 / (2 * ket.eta)
    end)

  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 0 then
    -------------------------------- PSPS --------------------------------
    local bra_Pi = rexpr bra.ishell_location end
    local ket_Pi = rexpr ket.ishell_location end
    local P0 = rexpr [density][0][0] end
    local x00 = generateXExpression(R, P0, 0, 0, 0, 0, 0)
    local x10 = generateXExpression(R, P0, 0, 0, 0, 1, 0)
    local x11 = generateXExpression(R, P0, 0, 0, 0, 1, 1)
    local x12 = generateXExpression(R, P0, 0, 0, 0, 1, 2)
    local x20 = generateXExpression(R, P0, 0, 0, 0, 2, 0)
    local x21 = generateXExpression(R, P0, 0, 0, 0, 2, 1)
    local x22 = generateXExpression(R, P0, 0, 0, 0, 2, 2)
    local x23 = generateXExpression(R, P0, 0, 0, 0, 2, 3)
    local x24 = generateXExpression(R, P0, 0, 0, 0, 2, 4)
    local x25 = generateXExpression(R, P0, 0, 0, 0, 2, 5)
    statements:insert(rquote
      [results[0][0]] = (
        bra_Pi.x * (ket_Pi.x * x00 - x10 / (2 * ket.eta))
        + (ket_Pi.x * x10 - x20 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[0][1]] = (
        bra_Pi.x * (ket_Pi.y * x00 - x11 / (2 * ket.eta))
        + (ket_Pi.y * x10 - x21 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[0][2]] = (
        bra_Pi.x * (ket_Pi.z * x00 - x12 / (2 * ket.eta))
        + (ket_Pi.z * x10 - x22 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[1][0]] = (
        bra_Pi.y * (ket_Pi.x * x00 - x10 / (2 * ket.eta))
        + (ket_Pi.x * x11 - x21 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[1][1]] = (
        bra_Pi.y * (ket_Pi.y * x00 - x11 / (2 * ket.eta))
        + (ket_Pi.y * x11 - x23 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[1][2]] = (
        bra_Pi.y * (ket_Pi.z * x00 - x12 / (2 * ket.eta))
        + (ket_Pi.z * x11 - x24 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[2][0]] = (
        bra_Pi.z * (ket_Pi.x * x00 - x10 / (2 * ket.eta))
        + (ket_Pi.x * x12 - x22 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[2][1]] = (
        bra_Pi.z * (ket_Pi.y * x00 - x11 / (2 * ket.eta))
        + (ket_Pi.y * x12 - x24 / (2 * ket.eta)) / (2 * bra.eta)
      )
      ;[results[2][2]] = (
        bra_Pi.z * (ket_Pi.z * x00 - x12 / (2 * ket.eta))
        + (ket_Pi.z * x12 - x25 / (2 * ket.eta)) / (2 * bra.eta)
      )
    end)

  elseif L1 == 0 and L2 == 0 and L3 == 1 and L4 == 1 then
    -------------------------------- SSPP --------------------------------
    local ket_Pi = rexpr ket.ishell_location end
    local ket_Pj = rexpr ket.jshell_location end
    local P0 = rexpr
      ket_Pj.x * [density][0][0]
      + ket_Pj.y * [density][0][1]
      + ket_Pj.z * [density][0][2]
    end
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    statements:insert(rquote
      [results[0][0]] = (
        ket_Pi.x * x00
        - x10 / (2 * ket.eta)
        + [generateXExpression(R, P1, 0, 0, 0, 0, 0)]
      )
      ;[results[0][1]] = (
        ket_Pi.y * x00
        - x11 / (2 * ket.eta)
        + [generateXExpression(R, P2, 0, 0, 0, 0, 0)]
      )
      ;[results[0][2]] = (
        ket_Pi.z * x00
        - x12 / (2 * ket.eta)
        + [generateXExpression(R, P3, 0, 0, 0, 0, 0)]
      )
    end)

  elseif L1 == 0 and L2 == 1 and L3 == 1 and L4 == 1 then
    -------------------------------- SPPP --------------------------------
    local bra_Pj = rexpr bra.jshell_location end
    local ket_Pi = rexpr ket.ishell_location end
    local ket_Pj = rexpr ket.jshell_location end
    local P0 = rexpr
      ket_Pj.x * [density][0][0]
      + ket_Pj.y * [density][0][1]
      + ket_Pj.z * [density][0][2]
    end
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    local x20 = generateXExpression(R, P0, P1, P2, P3, 2, 0)
    local x21 = generateXExpression(R, P0, P1, P2, P3, 2, 1)
    local x22 = generateXExpression(R, P0, P1, P2, P3, 2, 2)
    statements:insert(rquote
      [results[0][0]] = (
        bra_Pj.x * (ket_Pi.x * x00 - x10 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 0, 0)])
        + (ket_Pi.x * x10 - x20 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 1, 0)]) / (2 * bra.eta)
      )
      ;[results[0][1]] = (
        bra_Pj.x * (ket_Pi.y * x00 - x11 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 0, 0)])
        + (ket_Pi.y * x10 - x21 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 1, 0)]) / (2 * bra.eta)
      )
      ;[results[0][2]] = (
        bra_Pj.x * (ket_Pi.z * x00 - x12 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 0, 0)])
        + (ket_Pi.z * x10 - x22 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 1, 0)]) / (2 * bra.eta)
      )
    end)

    local P0 = rexpr
      ket_Pj.x * [density][1][0]
      + ket_Pj.y * [density][1][1]
      + ket_Pj.z * [density][1][2]
    end
    local P1 = rexpr [density][1][0] / (2 * ket.eta) end
    local P2 = rexpr [density][1][1] / (2 * ket.eta) end
    local P3 = rexpr [density][1][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    local x21 = generateXExpression(R, P0, P1, P2, P3, 2, 1)
    local x23 = generateXExpression(R, P0, P1, P2, P3, 2, 3)
    local x24 = generateXExpression(R, P0, P1, P2, P3, 2, 4)
    statements:insert(rquote
      [results[0][0]] += (
        bra_Pj.y * (ket_Pi.x * x00 - x10 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 0, 0)])
        + (ket_Pi.x * x11 - x21 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 1, 1)]) / (2 * bra.eta)
      )
      ;[results[0][1]] += (
        bra_Pj.y * (ket_Pi.y * x00 - x11 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 0, 0)])
        + (ket_Pi.y * x11 - x23 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 1, 1)]) / (2 * bra.eta)
      )
      ;[results[0][2]] += (
        bra_Pj.y * (ket_Pi.z * x00 - x12 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 0, 0)])
        + (ket_Pi.z * x11 - x24 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 1, 1)]) / (2 * bra.eta)
      )
    end)

    local P0 = rexpr
      ket_Pj.x * [density][2][0]
      + ket_Pj.y * [density][2][1]
      + ket_Pj.z * [density][2][2]
    end
    local P1 = rexpr [density][2][0] / (2 * ket.eta) end
    local P2 = rexpr [density][2][1] / (2 * ket.eta) end
    local P3 = rexpr [density][2][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    local x22 = generateXExpression(R, P0, P1, P2, P3, 2, 2)
    local x24 = generateXExpression(R, P0, P1, P2, P3, 2, 4)
    local x25 = generateXExpression(R, P0, P1, P2, P3, 2, 5)
    statements:insert(rquote
      [results[0][0]] += (
        bra_Pj.z * (ket_Pi.x * x00 - x10 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 0, 0)])
        + (ket_Pi.x * x12 - x22 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 1, 2)]) / (2 * bra.eta)
      )
      ;[results[0][1]] += (
        bra_Pj.z * (ket_Pi.y * x00 - x11 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 0, 0)])
        + (ket_Pi.y * x12 - x24 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 1, 2)]) / (2 * bra.eta)
      )
      ;[results[0][2]] += (
        bra_Pj.z * (ket_Pi.z * x00 - x12 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 0, 0)])
        + (ket_Pi.z * x12 - x25 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 1, 2)]) / (2 * bra.eta)
      )
    end)

  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 1 then
    -------------------------------- PSPP --------------------------------
    local bra_Pi = rexpr bra.ishell_location end
    local ket_Pi = rexpr ket.ishell_location end
    local ket_Pj = rexpr ket.jshell_location end
    local P0 = rexpr
      ket_Pj.x * [density][0][0]
      + ket_Pj.y * [density][0][1]
      + ket_Pj.z * [density][0][2]
    end
    local P1 = rexpr [density][0][0] / (2 * ket.eta) end
    local P2 = rexpr [density][0][1] / (2 * ket.eta) end
    local P3 = rexpr [density][0][2] / (2 * ket.eta) end
    local x00 = generateXExpression(R, P0, P1, P2, P3, 0, 0)
    local x10 = generateXExpression(R, P0, P1, P2, P3, 1, 0)
    local x11 = generateXExpression(R, P0, P1, P2, P3, 1, 1)
    local x12 = generateXExpression(R, P0, P1, P2, P3, 1, 2)
    local x20 = generateXExpression(R, P0, P1, P2, P3, 2, 0)
    local x21 = generateXExpression(R, P0, P1, P2, P3, 2, 1)
    local x22 = generateXExpression(R, P0, P1, P2, P3, 2, 2)
    local x23 = generateXExpression(R, P0, P1, P2, P3, 2, 3)
    local x24 = generateXExpression(R, P0, P1, P2, P3, 2, 4)
    local x25 = generateXExpression(R, P0, P1, P2, P3, 2, 5)
    statements:insert(rquote
      [results[0][0]] = (
        bra_Pi.x * (ket_Pi.x * x00 - x10 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 0, 0)])
        + (ket_Pi.x * x10 - x20 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 1, 0)]) / (2 * bra.eta)
      )
      ;[results[0][1]] = (
        bra_Pi.x * (ket_Pi.y * x00 - x11 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 0, 0)])
        + (ket_Pi.y * x10 - x21 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 1, 0)]) / (2 * bra.eta)
      )
      ;[results[0][2]] = (
        bra_Pi.x * (ket_Pi.z * x00 - x12 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 0, 0)])
        + (ket_Pi.z * x10 - x22 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 1, 0)]) / (2 * bra.eta)
      )

      ;[results[1][0]] = (
        bra_Pi.y * (ket_Pi.x * x00 - x10 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 0, 0)])
        + (ket_Pi.x * x11 - x21 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 1, 1)]) / (2 * bra.eta)
      )
      ;[results[1][1]] = (
        bra_Pi.y * (ket_Pi.y * x00 - x11 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 0, 0)])
        + (ket_Pi.y * x11 - x23 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 1, 1)]) / (2 * bra.eta)
      )
      ;[results[1][2]] = (
        bra_Pi.y * (ket_Pi.z * x00 - x12 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 0, 0)])
        + (ket_Pi.z * x11 - x24 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 1, 1)]) / (2 * bra.eta)
      )

      ;[results[2][0]] = (
        bra_Pi.z * (ket_Pi.x * x00 - x10 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 0, 0)])
        + (ket_Pi.x * x12 - x22 / (2 * ket.eta) + [generateXExpression(R, P1, 0, 0, 0, 1, 2)]) / (2 * bra.eta)
      )
      ;[results[2][1]] = (
        bra_Pi.z * (ket_Pi.y * x00 - x11 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 0, 0)])
        + (ket_Pi.y * x12 - x24 / (2 * ket.eta) + [generateXExpression(R, P2, 0, 0, 0, 1, 2)]) / (2 * bra.eta)
      )
      ;[results[2][2]] = (
        bra_Pi.z * (ket_Pi.z * x00 - x12 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 0, 0)])
        + (ket_Pi.z * x12 - x25 / (2 * ket.eta) + [generateXExpression(R, P3, 0, 0, 0, 1, 2)]) / (2 * bra.eta)
      )
    end)

  elseif L1 == 1 and L2 == 1 and L3 == 1 and L4 == 1 then
    -------------------------------- PPPP --------------------------------
    statements:insert(rquote
      var bra_denom = 0.5 / bra.eta
      var ket_denom = 0.5 / ket.eta
      var bra_Pi = bra.ishell_location
      var bra_Pj = bra.jshell_location
      var ket_Pi = ket.ishell_location
      var ket_Pj = ket.jshell_location

      var Pxx = [density][0][0]
      var Pxy = [density][0][1]
      var Pxz = [density][0][2]
      var Pyx = [density][1][0]
      var Pyy = [density][1][1]
      var Pyz = [density][1][2]
      var Pzx = [density][2][0]
      var Pzy = [density][2][1]
      var Pzz = [density][2][2]

      var PP0 = Pxx*ket_Pj.x + Pxy*ket_Pj.y + Pxz*ket_Pj.z
      var PP1 = Pxx * ket_denom
      var PP2 = Pxy * ket_denom
      var PP3 = Pxz * ket_denom
      ;[results[0][0]] =
          bra_Pi.x * (
          bra_Pj.x * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1)) +
          bra_denom     * (
          bra_Pj.x * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) -
          ket_denom      * ([R[3][0][0][0]]*PP0 - [R[4][0][0][0]]*PP1 - [R[3][1][0][0]]*PP2 - [R[3][0][1][0]]*PP3) +
          [R[2][0][0][0]] * PP1)) +
          bra_denom     *  (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1)
      ;[results[0][1]] =
          bra_Pi.x * (
          bra_Pj.x * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.x * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) -
          ket_denom      * ([R[2][1][0][0]]*PP0 - [R[3][1][0][0]]*PP1 - [R[2][2][0][0]]*PP2 - [R[2][1][1][0]]*PP3) +
          [R[2][0][0][0]] * PP2)) +
          bra_denom      * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2)
      ;[results[0][2]] =
          bra_Pi.x * (
          bra_Pj.x * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.x * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][1][0]]*PP0 - [R[3][0][1][0]]*PP1 - [R[2][1][1][0]]*PP2 - [R[2][0][2][0]]*PP3) +
          [R[2][0][0][0]] * PP3)) +
          bra_denom      * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3)
      ;[results[1][0]] =
          bra_Pi.y * (
          bra_Pj.x * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1)) +
          bra_denom      * (
          bra_Pj.x * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) -
          ket_denom      * ([R[2][1][0][0]]*PP0 - [R[3][1][0][0]]*PP1 - [R[2][2][0][0]]*PP2 - [R[2][1][1][0]]*PP3) +
          [R[1][1][0][0]] * PP1))
      ;[results[1][1]] =
          bra_Pi.y * (
          bra_Pj.x * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2)) +
          bra_denom * (
          bra_Pj.x * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) -
          ket_denom      * ([R[1][2][0][0]]*PP0 - [R[2][2][0][0]]*PP1 - [R[1][3][0][0]]*PP2 - [R[1][2][1][0]]*PP3) +
          [R[1][1][0][0]] * PP2))
      ;[results[1][2]] =
          bra_Pi.y * (
          bra_Pj.x * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3)) +
          bra_denom * (
          bra_Pj.x * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][1][0]]*PP0 - [R[2][1][1][0]]*PP1 - [R[1][2][1][0]]*PP2 - [R[1][1][2][0]]*PP3) +
          [R[1][1][0][0]] * PP3))
      ;[results[2][0]] =
          bra_Pi.z * (
          bra_Pj.x * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1)) +
          bra_denom * (
          bra_Pj.x * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) -
          ket_denom      * ([R[2][0][1][0]]*PP0 - [R[3][0][1][0]]*PP1 - [R[2][1][1][0]]*PP2 - [R[2][0][2][0]]*PP3) +
          [R[1][0][1][0]] * PP1))
      ;[results[2][1]] =
          bra_Pi.z * (
          bra_Pj.x * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2)) +
          bra_denom * (
          bra_Pj.x * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) -
          ket_denom      * ([R[1][1][1][0]]*PP0 - [R[2][1][1][0]]*PP1 - [R[1][2][1][0]]*PP2 - [R[1][1][2][0]]*PP3) +
          [R[1][0][1][0]] * PP2))
      ;[results[2][2]] =
          bra_Pi.z * (
          bra_Pj.x * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3)) +
          bra_denom * (
          bra_Pj.x * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][2][0]]*PP0 - [R[2][0][2][0]]*PP1 - [R[1][1][2][0]]*PP2 - [R[1][0][3][0]]*PP3) +
          [R[1][0][1][0]] * PP3))

      PP0 = Pyx*ket_Pj.x + Pyy*ket_Pj.y + Pyz*ket_Pj.z
      PP1 = Pyx * ket_denom
      PP2 = Pyy * ket_denom
      PP3 = Pyz * ket_denom
      ;[results[0][0]] +=
          bra_Pi.x * (
          bra_Pj.y * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1)) +
          bra_denom     * (
          bra_Pj.y * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) -
          ket_denom      * ([R[2][1][0][0]]*PP0 - [R[3][1][0][0]]*PP1 - [R[2][2][0][0]]*PP2 - [R[2][1][1][0]]*PP3) +
          [R[1][1][0][0]] * PP1))
      ;[results[0][1]] +=
          bra_Pi.x * (
          bra_Pj.y * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) -
          ket_denom      * ([R[1][2][0][0]]*PP0 - [R[2][2][0][0]]*PP1 - [R[1][3][0][0]]*PP2 - [R[1][2][1][0]]*PP3) +
          [R[1][1][0][0]] * PP2))
      ;[results[0][2]] +=
          bra_Pi.x * (
          bra_Pj.y * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][1][0]]*PP0 - [R[2][1][1][0]]*PP1 - [R[1][2][1][0]]*PP2 - [R[1][1][2][0]]*PP3) +
          [R[1][1][0][0]] * PP3))
      ;[results[1][0]] +=
          bra_Pi.y * (
          bra_Pj.y * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) -
          ket_denom      * ([R[1][2][0][0]]*PP0 - [R[2][2][0][0]]*PP1 - [R[1][3][0][0]]*PP2 - [R[1][2][1][0]]*PP3) +
          [R[0][2][0][0]] * PP1)) +
          bra_denom     *  (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1)
      ;[results[1][1]] +=
          bra_Pi.y * (
          bra_Pj.y * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) -
          ket_denom      * ([R[0][3][0][0]]*PP0 - [R[1][3][0][0]]*PP1 - [R[0][4][0][0]]*PP2 - [R[0][3][1][0]]*PP3) +
          [R[0][2][0][0]] * PP2)) +
          bra_denom     *  (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2)
      ;[results[1][2]] +=
          bra_Pi.y * (
          bra_Pj.y * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) -
          ket_denom      * ([R[0][2][1][0]]*PP0 - [R[1][2][1][0]]*PP1 - [R[0][3][1][0]]*PP2 - [R[0][2][2][0]]*PP3) +
          [R[0][2][0][0]] * PP3)) +
          bra_denom     *  (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3)
      ;[results[2][0]] +=
          bra_Pi.z * (
          bra_Pj.y * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) -
          ket_denom      * ([R[1][1][1][0]]*PP0 - [R[2][1][1][0]]*PP1 - [R[1][2][1][0]]*PP2 - [R[1][1][2][0]]*PP3) +
          [R[0][1][1][0]] * PP1))
      ;[results[2][1]] +=
          bra_Pi.z * (
          bra_Pj.y * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) -
          ket_denom      * ([R[0][2][1][0]]*PP0 - [R[1][2][1][0]]*PP1 - [R[0][3][1][0]]*PP2 - [R[0][2][2][0]]*PP3) +
          [R[0][1][1][0]] * PP2))
      ;[results[2][2]] +=
          bra_Pi.z * (
          bra_Pj.y * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.y * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) -
          ket_denom      * ([R[0][1][2][0]]*PP0 - [R[1][1][2][0]]*PP1 - [R[0][2][2][0]]*PP2 - [R[0][1][3][0]]*PP3) +
          [R[0][1][1][0]] * PP3))

      PP0 = Pzx*ket_Pj.x + Pzy*ket_Pj.y + Pzz*ket_Pj.z
      PP1 = Pzx * ket_denom
      PP2 = Pzy * ket_denom
      PP3 = Pzz * ket_denom
      ;[results[0][0]] +=
          bra_Pi.x * (
          bra_Pj.z * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1)) +
          bra_denom       * (
          bra_Pj.z * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) -
          ket_denom      * ([R[2][0][1][0]]*PP0 - [R[3][0][1][0]]*PP1 - [R[2][1][1][0]]*PP2 - [R[2][0][2][0]]*PP3) +
          [R[1][0][1][0]] * PP1))
      ;[results[0][1]] +=
          bra_Pi.x * (
          bra_Pj.z * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) -
          ket_denom      * ([R[1][1][1][0]]*PP0 - [R[2][1][1][0]]*PP1 - [R[1][2][1][0]]*PP2 - [R[1][1][2][0]]*PP3) +
          [R[1][0][1][0]] * PP2))
      ;[results[0][2]] +=
          bra_Pi.x * (
          bra_Pj.z * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][2][0]]*PP0 - [R[2][0][2][0]]*PP1 - [R[1][1][2][0]]*PP2 - [R[1][0][3][0]]*PP3) +
          [R[1][0][1][0]] * PP3))
      ;[results[1][0]] +=
          bra_Pi.y * (
          bra_Pj.z * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) -
          ket_denom      * ([R[1][1][1][0]]*PP0 - [R[2][1][1][0]]*PP1 - [R[1][2][1][0]]*PP2 - [R[1][1][2][0]]*PP3) +
          [R[0][1][1][0]] * PP1))
      ;[results[1][1]] +=
          bra_Pi.y * (
          bra_Pj.z * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) -
          ket_denom      * ([R[0][2][1][0]]*PP0 - [R[1][2][1][0]]*PP1 - [R[0][3][1][0]]*PP2 - [R[0][2][2][0]]*PP3) +
          [R[0][1][1][0]] * PP2))
      ;[results[1][2]] +=
          bra_Pi.y * (
          bra_Pj.z * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) -
          ket_denom      * ([R[0][1][2][0]]*PP0 - [R[1][1][2][0]]*PP1 - [R[0][2][2][0]]*PP2 - [R[0][1][3][0]]*PP3) +
          [R[0][1][1][0]] * PP3))
      ;[results[2][0]] +=
          bra_Pi.z * (
          bra_Pj.z * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) -
          ket_denom      * ([R[1][0][2][0]]*PP0 - [R[2][0][2][0]]*PP1 - [R[1][1][2][0]]*PP2 - [R[1][0][3][0]]*PP3) +
          [R[0][0][2][0]] * PP1)) +
          bra_denom      * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1)

      ;[results[2][1]] +=
          bra_Pi.z * (
          bra_Pj.z * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) -
          ket_denom      * ([R[0][1][2][0]]*PP0 - [R[1][1][2][0]]*PP1 - [R[0][2][2][0]]*PP2 - [R[0][1][3][0]]*PP3) +
          [R[0][0][2][0]] * PP2)) +
          bra_denom      * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2)
      ;[results[2][2]] +=
          bra_Pi.z * (
          bra_Pj.z * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3)) +
          bra_denom      * (
          bra_Pj.z * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) -
          ket_denom      * ([R[0][0][3][0]]*PP0 - [R[1][0][3][0]]*PP1 - [R[0][1][3][0]]*PP2 - [R[0][0][4][0]]*PP3) +
          [R[0][0][2][0]] * PP3)) +
          bra_denom      * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3)
    end)
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
