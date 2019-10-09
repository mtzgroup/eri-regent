import "regent"

struct Parameters {
  scalfr              : double;
  scallr              : double;
  omega               : double;
  thresp              : double;
  thredp              : double;
}

local JBraCache = {}
function getJBra(L)
  if JBraCache[L] == nil then
    local H = (L + 1) * (L + 2) * (L + 3) / 6
    local fspace JBra {
      {x, y, z} : double;    -- Location of gaussian
      eta       : double;    -- Exponent of gaussian
      C         : double;
      bound     : double;
      output    : double[H]; -- Result of integrals
    }
    JBraCache[L] = JBra
  end
  return JBraCache[L]
end

local JKetCache = {}
function getJKet(L)
  if JKetCache[L] == nil then
    local H = (L + 1) * (L + 2) * (L + 3) / 6
    local fspace JKet {
      {x, y, z} : double;    -- Location of gaussian
      eta       : double;    -- Exponent of gaussian
      C         : double;
      bound     : double;
      density   : double[H]; -- Preprocessed data
    }
    JKetCache[L] = JKet
  end
  return JKetCache[L]
end
