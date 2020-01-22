import "regent"

function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket,
                                       bra_prevals, ket_prevals, density)
  local statements = terralib.newlist()

  -- if L1 == 0 and L2 == 0 and L3 == 0 and L4 == 0 then
  --   statements:insert(rquote
  --     regentlib.assert(false, "Unimplemented kfock kernel")
  --   end)
  -- elseif L1 == 0 and L2 == 0 and L3 == 0 and L4 == 0 then
  --   statements:insert(rquote
  --     regentlib.assert(false, "Unimplemented kfock kernel")
  --   end)
  -- else
  --   statements:insert(rquote
  --     regentlib.assert(false, "Unimplemented kfock kernel")
  --   end)
  -- end

  return statements
end
