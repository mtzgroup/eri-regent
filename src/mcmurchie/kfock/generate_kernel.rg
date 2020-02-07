import "regent"

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
    -- SSSS
    statements:insert(rquote
      [results[0][0]] = [R[0][0][0][0]] * [density][0][0]
    end)
  elseif L1 == 0 and L2 == 0 and L3 == 0 and L4 == 1 then
    -- SSSP
    statements:insert(rquote
      var ket_denom : double = 1.0 / (ket.eta + ket.eta)
      var ket_Pj = ket.jshell_location
      ;[results[0][0]] = (
        ([R[0][0][0][0]] * ket_Pj.x - ket_denom * [R[1][0][0][0]]) * [density][0][0]
        + ([R[0][0][0][0]] * ket_Pj.y - ket_denom * [R[0][1][0][0]]) * [density][0][1]
        + ([R[0][0][0][0]] * ket_Pj.z - ket_denom * [R[0][0][1][0]]) * [density][0][2]
      )
    end)
  elseif L1 == 0 and L2 == 1 and L3 == 0 and L4 == 1 then
    -- SPSP
    statements:insert(rquote
      var bra_denom : double = 1.0 / (bra.eta + bra.eta)
      var ket_denom : double = 1.0 / (ket.eta + ket.eta)
      var ket_Pj = ket.jshell_location
      var bra_Pj = bra.jshell_location
      ;[results[0][0]] = (
        bra_Pj.x * (
          [R[0][0][0][0]] * ([density][0][0] * ket_Pj.x + [density][0][1] * ket_Pj.y + [density][0][2] * ket_Pj.z)
          - [R[1][0][0][0]] * [density][0][0] * ket_denom
          - [R[0][1][0][0]] * [density][0][1] * ket_denom
          - [R[0][0][1][0]] * [density][0][2] * ket_denom
        )
        + bra_denom * (
          [R[1][0][0][0]] * ([density][0][0] * ket_Pj.x + [density][0][1] * ket_Pj.y + [density][0][2] * ket_Pj.z)
          - [R[2][0][0][0]] * [density][0][0] * ket_denom
          - [R[1][1][0][0]] * [density][0][1] * ket_denom
          - [R[1][0][1][0]] * [density][0][2] * ket_denom
        )

        + bra_Pj.y * (
          [R[0][0][0][0]] * ([density][1][0] * ket_Pj.x + [density][1][1] * ket_Pj.y + [density][1][2] * ket_Pj.z)
          - [R[1][0][0][0]] * [density][1][0] * ket_denom
          - [R[0][1][0][0]] * [density][1][1] * ket_denom
          - [R[0][0][1][0]] * [density][1][2] * ket_denom
        )
        + bra_denom * (
          [R[0][1][0][0]] * ([density][1][0] * ket_Pj.x + [density][1][1] * ket_Pj.y + [density][1][2] * ket_Pj.z)
          - [R[1][1][0][0]] * [density][1][0] * ket_denom
          - [R[0][2][0][0]] * [density][1][1] * ket_denom
          - [R[0][1][1][0]] * [density][1][2] * ket_denom
        )

        + bra_Pj.z * (
          [R[0][0][0][0]] * ([density][2][0] * ket_Pj.x + [density][2][1] * ket_Pj.y + [density][2][2] * ket_Pj.z)
          - [R[1][0][0][0]] * [density][2][0] * ket_denom
          - [R[0][1][0][0]] * [density][2][1] * ket_denom
          - [R[0][0][1][0]] * [density][2][2] * ket_denom
        )
        + bra_denom * (
          [R[0][0][1][0]] * ([density][2][0] * ket_Pj.x + [density][2][1] * ket_Pj.y + [density][2][2] * ket_Pj.z)
          - [R[1][0][1][0]] * [density][2][0] * ket_denom
          - [R[0][1][1][0]] * [density][2][1] * ket_denom
          - [R[0][0][2][0]] * [density][2][2] * ket_denom
        )
      )
    end)
  elseif L1 == 0 and L2 == 0 and L3 == 1 and L4 == 0 then
    -- SSPS
    statements:insert(rquote
      var ket_denom = 1.0 / (ket.eta + ket.eta)
      var ket_Pi = ket.ishell_location
      ;[results[0][0]] = (
        [R[0][0][0][0]] * ket_Pi.x - ket_denom * [R[1][0][0][0]]
      ) * [density][0][0]

      ;[results[0][1]] = (
        [R[0][0][0][0]] * ket_Pi.y - ket_denom * [R[0][1][0][0]]
      ) * [density][0][0]

      ;[results[0][2]] = (
        [R[0][0][0][0]] * ket_Pi.z - ket_denom * [R[0][0][1][0]]
      ) * [density][0][0]
    end)
  elseif L1 == 0 and L2 == 1 and L3 == 1 and L4 == 0 then
    -- SPPS
    -- FIXME: This kernel should work, but it doesn't.
    statements:insert(rquote
      var bra_denom = 0.5 / bra.eta
      var ket_denom = 0.5 / ket.eta
      var bra_Pj = bra.jshell_location
      var ket_Pi = ket.ishell_location

      ;[results[0][0]] = (
        [density][0][0] * (
          bra_Pj.x * (ket_Pi.x * [R[0][0][0][0]] - ket_denom * [R[1][0][0][0]])
          + bra_denom * (ket_Pi.x * [R[1][0][0][0]] - ket_denom * [R[2][0][0][0]])
        )
        + [density][0][1] * (
          bra_Pj.y * (ket_Pi.x * [R[0][0][0][0]] - ket_denom * [R[1][0][0][0]])
          + bra_denom * (ket_Pi.x * [R[0][1][0][0]] - ket_denom * [R[1][1][0][0]])
        )
        + [density][0][2] * (
          bra_Pj.z * (ket_Pi.x * [R[0][0][0][0]] - ket_denom * [R[1][0][0][0]])
          + bra_denom * (ket_Pi.x * [R[0][0][1][0]] - ket_denom * [R[1][0][1][0]])
        )
      )

      ;[results[0][1]] = (
        [density][0][0] * (
          bra_Pj.x * (ket_Pi.y * [R[0][0][0][0]] - ket_denom * [R[0][1][0][0]])
          + bra_denom * (ket_Pi.y * [R[1][0][0][0]] - ket_denom * [R[1][1][0][0]])
        )
        + [density][0][1] * (
          bra_Pj.y * (ket_Pi.y * [R[0][0][0][0]] - ket_denom * [R[0][1][0][0]])
          + bra_denom * (ket_Pi.y * [R[0][1][0][0]] - ket_denom * [R[0][2][0][0]])
        )
        + [density][0][2] * (
          bra_Pj.z * (ket_Pi.y * [R[0][0][0][0]] - ket_denom * [R[0][1][0][0]])
          + bra_denom * (ket_Pi.y * [R[0][0][1][0]] - ket_denom * [R[0][1][1][0]])
        )
      )

      ;[results[0][2]] = (
        [density][0][0] * (
          bra_Pj.x * (ket_Pi.z * [R[0][0][0][0]] - ket_denom * [R[0][0][1][0]])
          + bra_denom * (ket_Pi.z * [R[1][0][0][0]] - ket_denom * [R[1][0][1][0]])
        )
        + [density][0][1] * (
          bra_Pj.y * (ket_Pi.z * [R[0][0][0][0]] - ket_denom * [R[0][0][1][0]])
          + bra_denom * (ket_Pi.z * [R[0][1][0][0]] - ket_denom * [R[0][1][1][0]])
        )
        + [density][0][2] * (
          bra_Pj.z * (ket_Pi.z * [R[0][0][0][0]] - ket_denom * [R[0][0][1][0]])
          + bra_denom * (ket_Pi.z * [R[0][0][1][0]] - ket_denom * [R[0][0][2][0]])
        )
      )
    end)
  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 0 then
    -- PSPS
    statements:insert(rquote
      var bra_denom = 0.5 / bra.eta
      var ket_denom = 0.5 / ket.eta
      var bra_Pi = bra.ishell_location
      var ket_Pi = ket.ishell_location

      ;[results[0][0]] = [density][0][0] * (
        bra_Pi.x * (ket_Pi.x * [R[0][0][0][0]] - ket_denom * [R[1][0][0][0]])
        + bra_denom * (ket_Pi.x * [R[1][0][0][0]] - ket_denom * [R[2][0][0][0]])
      )

      ;[results[0][1]] = [density][0][0] * (
        bra_Pi.x * (ket_Pi.y * [R[0][0][0][0]] - ket_denom * [R[0][1][0][0]])
        + bra_denom * (ket_Pi.y * [R[1][0][0][0]] - ket_denom * [R[1][1][0][0]])
      )

      ;[results[0][2]] = [density][0][0] * (
        bra_Pi.x * (ket_Pi.z * [R[0][0][0][0]] - ket_denom * [R[0][0][1][0]])
        + bra_denom * (ket_Pi.z * [R[1][0][0][0]] - ket_denom * [R[1][0][1][0]])
      )

      ;[results[1][0]] = [density][0][0] * (
        bra_Pi.y * (ket_Pi.x * [R[0][0][0][0]] - ket_denom * [R[1][0][0][0]])
        + bra_denom * (ket_Pi.x * [R[0][1][0][0]] - ket_denom * [R[1][1][0][0]])
      )

      ;[results[1][1]] = [density][0][0] * (
        bra_Pi.y * (ket_Pi.y * [R[0][0][0][0]] - ket_denom * [R[0][1][0][0]])
        + bra_denom * (ket_Pi.y * [R[0][1][0][0]] - ket_denom * [R[0][2][0][0]])
      )

      ;[results[1][2]] = [density][0][0] * (
        bra_Pi.y * (ket_Pi.z * [R[0][0][0][0]] - ket_denom * [R[0][0][1][0]])
        + bra_denom * (ket_Pi.z * [R[0][1][0][0]] - ket_denom * [R[0][1][1][0]])
      )

      ;[results[2][0]] = [density][0][0] * (
        bra_Pi.z * (ket_Pi.x * [R[0][0][0][0]] - ket_denom * [R[1][0][0][0]])
        + bra_denom * (ket_Pi.x * [R[0][0][1][0]] - ket_denom * [R[1][0][1][0]])
      )

      ;[results[2][1]] = [density][0][0] * (
        bra_Pi.z * (ket_Pi.y * [R[0][0][0][0]] - ket_denom * [R[0][1][0][0]])
        + bra_denom * (ket_Pi.y * [R[0][0][1][0]] - ket_denom * [R[0][1][1][0]])
      )

      ;[results[2][2]] = [density][0][0] * (
        bra_Pi.z * (ket_Pi.z * [R[0][0][0][0]] - ket_denom * [R[0][0][1][0]])
        + bra_denom * (ket_Pi.z * [R[0][0][1][0]] - ket_denom * [R[0][0][2][0]])
      )
    end)
  elseif L1 == 0 and L2 == 0 and L3 == 1 and L4 == 1 then
    -- SSPP
    statements:insert(rquote
      var ket_denom = 0.5 / ket.eta
      var ket_Pi = ket.ishell_location
      var ket_Pj = ket.jshell_location
      ;[results[0][0]] = (
        [R[0][0][0][0]] * ket_denom * [density][0][0]
        + ket_Pj.x * (
          [R[0][0][0][0]] * (ket_Pi.x * [density][0][0] + ket_Pi.y * [density][0][1] + ket_Pi.z * [density][0][2])
          - [R[1][0][0][0]] * ket_denom * [density][0][0]
          - [R[0][1][0][0]] * ket_denom * [density][0][1]
          - [R[0][0][1][0]] * ket_denom * [density][0][2]
        )
        - ket_denom * (
          [R[1][0][0][0]] * (ket_Pi.x * [density][0][0] + ket_Pi.y * [density][0][1] + ket_Pi.z * [density][0][2])
          - [R[2][0][0][0]] * ket_denom * [density][0][0]
          - [R[1][1][0][0]] * ket_denom * [density][0][1]
          - [R[1][0][1][0]] * ket_denom * [density][0][2]
        )
      )

      ;[results[0][1]] = (
        [R[0][0][0][0]] * ket_denom * [density][0][1]
        + ket_Pj.y * (
          [R[0][0][0][0]] * (ket_Pi.x * [density][0][0] + ket_Pi.y * [density][0][1] + ket_Pi.z * [density][0][2])
          - [R[1][0][0][0]] * ket_denom * [density][0][0]
          - [R[0][1][0][0]] * ket_denom * [density][0][1]
          - [R[0][0][1][0]] * ket_denom * [density][0][2]
        )
        - ket_denom * (
          [R[0][1][0][0]] * (ket_Pi.x * [density][0][0] + ket_Pi.y * [density][0][1] + ket_Pi.z * [density][0][2])
          - [R[1][1][0][0]] * ket_denom * [density][0][0]
          - [R[0][2][0][0]] * ket_denom * [density][0][1]
          - [R[0][1][1][0]] * ket_denom * [density][0][2]
        )
      )

      ;[results[0][2]] = (
        [R[0][0][0][0]] * ket_denom * [density][0][2]
        + ket_Pj.z * (
          [R[0][0][0][0]] * (ket_Pi.x * [density][0][0] + ket_Pi.y * [density][0][1] + ket_Pi.z * [density][0][2])
          - [R[1][0][0][0]] * ket_denom * [density][0][0]
          - [R[0][1][0][0]] * ket_denom * [density][0][1]
          - [R[0][0][1][0]] * ket_denom * [density][0][2]
        )
        - ket_denom * (
          [R[0][0][1][0]] * (ket_Pi.x * [density][0][0] + ket_Pi.y * [density][0][1] + ket_Pi.z * [density][0][2])
          - [R[1][0][1][0]] * ket_denom * [density][0][0]
          - [R[0][1][1][0]] * ket_denom * [density][0][1]
          - [R[0][0][2][0]] * ket_denom * [density][0][2]
        )
      )
    end)
  elseif L1 == 0 and L2 == 1 and L3 == 1 and L4 == 1 then
    -- SPPP
    statements:insert(rquote
      var bra_denom = 0.5 / bra.eta
      var ket_denom = 0.5 / ket.eta
      var bra_Pj = bra.jshell_location
      var ket_Pi = ket.ishell_location
      var ket_Pj = ket.jshell_location

      var PP0 = [density][0][0] * ket_Pi.x + [density][0][1] * ket_Pi.y + [density][0][2] * ket_Pi.z
      var PP1 = [density][0][0] * ket_denom
      var PP2 = [density][0][1] * ket_denom
      var PP3 = [density][0][2] * ket_denom
      ;[results[0][0]] += (
        bra_Pj.x * (
          ket_Pi.x * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[1][0][0][0]] * PP0 - [R[2][0][0][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[1][0][1][0]] * PP3)
          + [R[0][0][0][0]] * PP1
        )
        + bra_denom * (
          ket_Pi.x * ([R[1][0][0][0]] * PP0 - [R[2][0][0][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[1][0][1][0]] * PP3)
          - ket_denom * ([R[2][0][0][0]] * PP0 - [R[3][0][0][0]] * PP1 - [R[2][1][0][0]] * PP2 - [R[2][0][1][0]] * PP3)
          + [R[1][0][0][0]] * PP1
        )
      )
      ;[results[0][1]] += (
        bra_Pj.x * (
          ket_Pi.y * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[0][1][0][0]] * PP0 - [R[1][1][0][0]] * PP1 - [R[0][2][0][0]] * PP2 - [R[0][1][1][0]] * PP3)
          + [R[0][0][0][0]] * PP2
        )
        + bra_denom * (
          ket_Pi.y * ([R[1][0][0][0]] * PP0 - [R[2][0][0][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[1][0][1][0]] * PP3)
          - ket_denom * ([R[1][1][0][0]] * PP0 - [R[2][1][0][0]] * PP1 - [R[1][2][0][0]] * PP2 - [R[1][1][1][0]] * PP3)
          + [R[1][0][0][0]] * PP2
        )
      )
      ;[results[0][2]] += (
        bra_Pj.x * (
          ket_Pi.z * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[0][0][1][0]] * PP0 - [R[1][0][1][0]] * PP1 - [R[0][1][1][0]] * PP2 - [R[0][0][2][0]] * PP3)
          + [R[0][0][0][0]] * PP3
        )
        + bra_denom * (
          ket_Pi.z * ([R[1][0][0][0]] * PP0 - [R[2][0][0][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[1][0][1][0]] * PP3)
          - ket_denom * ([R[1][0][1][0]] * PP0 - [R[2][0][1][0]] * PP1 - [R[1][1][1][0]] * PP2 - [R[1][0][2][0]] * PP3)
          + [R[1][0][0][0]] * PP3
        )
      )

      PP0 = [density][1][0] * ket_Pi.x + [density][1][1] * ket_Pi.y + [density][1][2] * ket_Pi.z
      PP1 = [density][1][0] * ket_denom
      PP2 = [density][1][1] * ket_denom
      PP3 = [density][1][2] * ket_denom
      ;[results[0][0]] += (
        bra_Pj.y * (
          ket_Pi.x * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[1][0][0][0]] * PP0 - [R[2][0][0][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[1][0][1][0]] * PP3)
          + [R[0][0][0][0]] * PP1
        )
        + bra_denom * (
          ket_Pi.x * ([R[0][1][0][0]] * PP0 - [R[1][1][0][0]] * PP1 - [R[0][2][0][0]] * PP2 - [R[0][1][1][0]] * PP3)
          - ket_denom * ([R[1][1][0][0]] * PP0 - [R[2][1][0][0]] * PP1 - [R[1][2][0][0]] * PP2 - [R[1][1][1][0]] * PP3)
          + [R[0][1][0][0]] * PP1
        )
      )
      ;[results[0][1]] += (
        bra_Pj.y * (
          ket_Pi.y * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[0][1][0][0]] * PP0 - [R[1][1][0][0]] * PP1 - [R[0][2][0][0]] * PP2 - [R[0][1][1][0]] * PP3)
          + [R[0][0][0][0]] * PP2
        )
        + bra_denom * (
          ket_Pi.y * ([R[0][1][0][0]] * PP0 - [R[1][1][0][0]] * PP1 - [R[0][2][0][0]] * PP2 - [R[0][1][1][0]] * PP3)
          - ket_denom * ([R[0][2][0][0]] * PP0 - [R[1][2][0][0]] * PP1 - [R[0][3][0][0]] * PP2 - [R[0][2][1][0]] * PP3)
          + [R[0][1][0][0]] * PP2
        )
      )
      ;[results[0][2]] += (
        bra_Pj.y * (
          ket_Pi.z * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[0][0][1][0]] * PP0 - [R[1][0][1][0]] * PP1 - [R[0][1][1][0]] * PP2 - [R[0][0][2][0]] * PP3)
          + [R[0][0][0][0]] * PP3
        )
        + bra_denom * (
          ket_Pi.z * ([R[0][1][0][0]] * PP0 - [R[1][1][0][0]] * PP1 - [R[0][2][0][0]] * PP2 - [R[0][1][1][0]] * PP3)
          - ket_denom * ([R[0][1][1][0]] * PP0 - [R[1][1][1][0]] * PP1 - [R[0][2][1][0]] * PP2 - [R[0][1][2][0]] * PP3)
          + [R[0][1][0][0]] * PP3
        )
      )

      PP0 = [density][2][0] * ket_Pi.x + [density][2][1] * ket_Pi.y + [density][2][2] * ket_Pi.z
      PP1 = [density][2][0] * ket_denom
      PP2 = [density][2][1] * ket_denom
      PP3 = [density][2][2] * ket_denom
      ;[results[0][0]] += (
        bra_Pj.z * (
          ket_Pi.x * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[1][0][0][0]] * PP0 - [R[2][0][0][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[1][0][1][0]] * PP3)
          + [R[0][0][0][0]] * PP1
        )
        + bra_denom * (
          ket_Pi.x * ([R[0][0][1][0]] * PP0 - [R[1][0][1][0]] * PP1 - [R[0][1][1][0]] * PP2 - [R[0][0][2][0]] * PP3)
          - ket_denom * ([R[1][0][1][0]] * PP0 - [R[2][0][1][0]] * PP1 - [R[1][1][1][0]] * PP2 - [R[1][0][2][0]] * PP3)
          + [R[0][0][1][0]] * PP1
        )
      )
      ;[results[0][1]] += (
        bra_Pj.z * (
          ket_Pi.y * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[0][1][0][0]] * PP0 - [R[1][1][0][0]] * PP1 - [R[0][2][0][0]] * PP2 - [R[0][1][1][0]] * PP3)
          + [R[0][0][0][0]] * PP2
        )
        + bra_denom * (
          ket_Pi.y * ([R[0][0][1][0]] * PP0 - [R[1][0][1][0]] * PP1 - [R[0][1][1][0]] * PP2 - [R[0][2][0][0]] * PP3)
          - ket_denom * ([R[0][1][1][0]] * PP0 - [R[1][1][1][0]] * PP1 - [R[0][2][1][0]] * PP2 - [R[0][1][2][0]] * PP3)
          + [R[0][0][1][0]] * PP2
        )
      )
      ;[results[0][2]] += (
        bra_Pj.z * (
          ket_Pi.z * ([R[0][0][0][0]] * PP0 - [R[1][0][0][0]] * PP1 - [R[0][1][0][0]] * PP2 - [R[0][0][1][0]] * PP3)
          - ket_denom * ([R[0][0][1][0]] * PP0 - [R[1][0][1][0]] * PP1 - [R[0][1][1][0]] * PP2 - [R[0][0][2][0]] * PP3)
          + [R[0][0][0][0]] * PP3
        )
        + bra_denom * (
          ket_Pi.z * ([R[0][0][1][0]] * PP0 - [R[1][0][1][0]] * PP1 - [R[1][1][0][0]] * PP2 - [R[0][0][2][0]] * PP3)
          - ket_denom * ([R[0][0][2][0]] * PP0 - [R[1][0][2][0]] * PP1 - [R[0][1][2][0]] * PP2 - [R[0][0][3][0]] * PP3)
          + [R[0][0][1][0]] * PP3
        )
      )
    end)
  elseif L1 == 1 and L2 == 0 and L3 == 1 and L4 == 1 then
    -- PSPP
    statements:insert(rquote
      var bra_denom = 0.5 / bra.eta
      var ket_denom = 0.5 / ket.eta
      var bra_Pi = bra.ishell_location
      var ket_Pi = ket.ishell_location
      var ket_Pj = ket.jshell_location

      var PP0 = [density][0][0]*ket_Pj.x + [density][0][1]*ket_Pj.y + [density][0][2]*ket_Pj.z
      var PP1 = [density][0][0] * ket_denom
      var PP2 = [density][0][1] * ket_denom
      var PP3 = [density][0][2] * ket_denom

      ;[results[0][0]] += (
          bra_Pi.x * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[2][0][0][0]]*PP0 - [R[3][0][0][0]]*PP1 - [R[2][1][0][0]]*PP2 - [R[2][0][1][0]]*PP3) +
          [R[1][0][0][0]] * PP1))

      ;[results[0][1]] += (
          bra_Pi.x * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[1][0][0][0]] * PP2))

      ;[results[0][2]] += (
          bra_Pi.x * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[1][0][0][0]] * PP3))

      ;[results[1][0]] += (
          bra_Pi.y * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[1][1][0][0]]*PP0 - [R[2][1][0][0]]*PP1 - [R[1][2][0][0]]*PP2 - [R[1][1][1][0]]*PP3) +
          [R[0][1][0][0]] * PP1))

      ;[results[1][1]] += (
          bra_Pi.y * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][2][0][0]]*PP0 - [R[1][2][0][0]]*PP1 - [R[0][3][0][0]]*PP2 - [R[0][2][1][0]]*PP3) +
          [R[0][1][0][0]] * PP2))

      ;[results[1][2]] += (
          bra_Pi.y * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][1][0][0]] * PP3))

      ;[results[2][0]] += (
          bra_Pi.z * (
          ket_Pi.x * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[1][0][0][0]]*PP0 - [R[2][0][0][0]]*PP1 - [R[1][1][0][0]]*PP2 - [R[1][0][1][0]]*PP3) +
          [R[0][0][0][0]] * PP1) +
          bra_denom * (
          ket_Pi.x * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[1][0][1][0]]*PP0 - [R[2][0][1][0]]*PP1 - [R[1][1][1][0]]*PP2 - [R[1][0][2][0]]*PP3) +
          [R[0][0][1][0]] * PP1))

      ;[results[2][1]] += (
          bra_Pi.z * (
          ket_Pi.y * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][1][0][0]]*PP0 - [R[1][1][0][0]]*PP1 - [R[0][2][0][0]]*PP2 - [R[0][1][1][0]]*PP3) +
          [R[0][0][0][0]] * PP2) +
          bra_denom * (
          ket_Pi.y * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][1][1][0]]*PP0 - [R[1][1][1][0]]*PP1 - [R[0][2][1][0]]*PP2 - [R[0][1][2][0]]*PP3) +
          [R[0][0][1][0]] * PP2))

      ;[results[2][2]] += (
          bra_Pi.z * (
          ket_Pi.z * ([R[0][0][0][0]]*PP0 - [R[1][0][0][0]]*PP1 - [R[0][1][0][0]]*PP2 - [R[0][0][1][0]]*PP3) -
          ket_denom      * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) +
          [R[0][0][0][0]] * PP3) +
          bra_denom * (
          ket_Pi.z * ([R[0][0][1][0]]*PP0 - [R[1][0][1][0]]*PP1 - [R[0][1][1][0]]*PP2 - [R[0][0][2][0]]*PP3) -
          ket_denom      * ([R[0][0][2][0]]*PP0 - [R[1][0][2][0]]*PP1 - [R[0][1][2][0]]*PP2 - [R[0][0][3][0]]*PP3) +
          [R[0][0][1][0]] * PP3))
    end)
  elseif L1 == 1 and L2 == 1 and L3 == 1 and L4 == 1 then
    -- PPPP
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
    assert(false, "Unimplemented KFock kernel!")
  end

  for i = 0, H1-1 do -- inclusive
    for k = 0, H3-1 do -- inclusive
      if L1 == L3 and L2 == L4 then -- Diagonal kernel.
        local factor
        if i < k then -- Upper triangular element.
          factor = 1
        elseif i == k then -- Diagonal element.
          factor = 0.5
        else -- Lower triangular element.
          factor = 0
        end

        statements:insert(rquote
          if bra.ishell_index < ket.ishell_index then -- Upper triangular element.
            [output][i][k] += [results[i][k]]
          elseif bra.ishell_index == ket.ishell_index then -- Diagonal element
            -- NOTE: Diagonal elements of diagonal kernels scale the output
            --       by a factor of 1/2.
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
