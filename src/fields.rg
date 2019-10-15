import "regent"

require "helper"

struct Parameters {
  scalfr              : double;
  scallr              : double;
  omega               : double;
  thresp              : float;
  thredp              : float;
}

local JBraCache = {}
function getJBra(L12)
  if JBraCache[L12] == nil then
    local H = computeH(L12)
    local fspace JBra {
      {x, y, z} : double;    -- Location of gaussian
      eta       : double;    -- Exponent of gaussian
      C         : double;
      bound     : float;
      output    : double[H]; -- Result of integrals
    }
    JBraCache[L12] = JBra
  end
  return JBraCache[L12]
end

local JKetCache = {}
function getJKet(L34)
  if JKetCache[L34] == nil then
    local H = computeH(L34)
    local fspace JKet {
      {x, y, z} : double;    -- Location of gaussian
      eta       : double;    -- Exponent of gaussian
      C         : double;
      bound     : float;
      density   : double[H]; -- Preprocessed data
    }
    JKetCache[L34] = JKet
  end
  return JKetCache[L34]
end
