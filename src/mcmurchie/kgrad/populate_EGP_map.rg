import "regent"
require "fields"
local assert = regentlib.assert

local EGP_header = terralib.includec("mcmurchie/kgrad/EGP_map.h", {"-I", "./"})

local terra getEGPSign(i : int, L1 : int, L2 : int) : int
  if L1 == 0 and L2 == 0 then
    return EGP_header.sign_SS[i]
  elseif L1 == 0 and L2 == 1 then
    return EGP_header.sign_SP[i]
  elseif L1 == 0 and L2 == 2 then
    return EGP_header.sign_SD[i]
  elseif L1 == 1 and L2 == 1 then
    return EGP_header.sign_PP[i]
  elseif L1 == 1 and L2 == 2 then
    return EGP_header.sign_PD[i]
  elseif L1 == 2 and L2 == 2 then
    return EGP_header.sign_DD[i]
  else
    return 0 -- TODO: implement for higher L
  end
end

local terra getEGPStride(i : int, L1 : int, L2 : int) : int
  if L1 == 0 and L2 == 0 then
    return EGP_header.stride_SS[i]
  elseif L1 == 0 and L2 == 1 then
    return EGP_header.stride_SP[i]
  elseif L1 == 0 and L2 == 2 then
    return EGP_header.stride_SD[i]
  elseif L1 == 1 and L2 == 1 then
    return EGP_header.stride_PP[i]
  elseif L1 == 1 and L2 == 2 then
    return EGP_header.stride_PD[i]
  elseif L1 == 2 and L2 == 2 then
    return EGP_header.stride_DD[i]
  else
    return 0 -- TODO: implement for higher L
  end
end

function populateBraEGPMap(region_vars)
  local statements = terralib.newlist()
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      local r_EGPmap = region_vars[L1][L2]
      local map_size = KGradNumBraEGPMap[L1][L2]
      statements:insert(rquote
        assert(L1 <= 2 and L2 <= 2, "Bra EGP map not set for angular momenum above D!")
        var [r_EGPmap] = region(ispace(int1d, map_size), getKGradBraEGPMap(L1, L2)) 
        for i = 0, map_size do -- exclusive
          r_EGPmap[i] = {sign=getEGPSign(i, L1, L2), stride=getEGPStride(i, L1, L2)}
        end
      end)
    end
  end
  return statements
end
