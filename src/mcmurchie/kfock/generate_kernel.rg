import "regent"

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket, density)
  local statements = terralib.newlist()
  statements:insert(rquote
    regentlib.assert(false, "Unimplemented kfock kernel")
  end)

  return statements
end
