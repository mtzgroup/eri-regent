import "regent"

local rsqrt = regentlib.rsqrt(double)

local _kfock_kernel_cache = {}
function generateKFockKernelStatements(R, L1, L2, L3, L4, bra, ket, density)
  local L_string = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]
  if _kfock_kernel_cache[L_string] ~= nil then
    return _kfock_kernel_cache[L_string]
  end

  local statements = teralib.newlist()
  statements:insert(rquote
    assert(false, "Unimplemented")
  end)

  _kfock_kernel_cache[L_string] = statements
  return statements
end
