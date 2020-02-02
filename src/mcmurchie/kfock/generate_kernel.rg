import "regent"

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket,
                                       bra_prevals, ket_prevals,
                                       density, output)
  local statements = terralib.newlist()

  if L1 == 0 and L2 == 0 and L3 == 0 and L4 == 0 then
    statements:insert(rquote
      if bra.ishell_index < ket.ishell_index then
        [output][0][0] += [R[0][0][0][0]] * [density][0][0]
      elseif bra.ishell_index == ket.ishell_index then
        [output][0][0] += 0.5 * [R[0][0][0][0]] * [density][0][0]
      else
       -- no-op
      end
    end)
  elseif L1 == 0 and L2 == 0 and L3 == 0 and L4 == 1 then
    statements:insert(rquote
      var denomQ : double = 1.0 / (ket.eta + ket.eta)
      var Pj = ket.jshell_location
      ;[output][0][0] += (
        ([R[0][0][0][0]] * Pj.x - denomQ * [R[1][0][0][0]]) * [density][0][0]
        + ([R[0][0][0][0]] * Pj.y - denomQ * [R[0][1][0][0]]) * [density][0][1]
        + ([R[0][0][0][0]] * Pj.z - denomQ * [R[0][0][1][0]]) * [density][0][2]
      )
    end)
  else
    assert(false, "Unimplemented KFock kernel!")
  end

  return statements
end
