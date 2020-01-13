import "regent"

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket,
                                       bra_prevals, ket_prevals, density)
  local statements = terralib.newlist()
  statements:insert(rquote
    regentlib.assert(false, "Unimplemented kfock kernel")
  end)

  return statements
end
