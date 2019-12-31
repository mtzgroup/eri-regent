import "regent"

require "helper"

struct Parameters {
  scalfr : double;
  scallr : double;
  omega  : double;
  thresp : float;
  thredp : float;
}

struct Point {
  x : double;
  y : double;
  z : double;
}

local JBraCache = {}
function getJBra(L12)
  if JBraCache[L12] == nil then
    local H = computeH(L12)
    local fspace JBra {
      location : Point;     -- Location of gaussian
      eta      : double;    -- Exponent of gaussian
      C        : double;
      bound    : float;
      output   : double[H]; -- Result of integrals
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
      location : Point;     -- Location of gaussian
      eta      : double;    -- Exponent of gaussian
      C        : double;
      bound    : float;
      density  : double[H]; -- Preprocessed data
    }
    JKetCache[L34] = JKet
  end
  return JKetCache[L34]
end

local KFockPairCache = {}
function getKFockPair(L1, L2)
  local index = LToStr[L1]..LToStr[L2]
  if KFockPairCache[index] == nil then
    local fspace KFockPair {
      location        : Point;  -- Location of guassian
      eta             : double; -- Exponent of guassian
      C               : double;
      bound           : float;
      ishell_location : Point;
      jshell_location : Point;
      ishell_index    : int1d;
      jshell_index    : int1d;  -- Index for density?
      -- prevals         : double[N];
    }
    KFockPairCache[index] = KFockPair
  end
  return KFockPairCache[index]
end

local KFockDensityCache = {}
function getKFockDensity(L2, L4)
  local index = LToStr[L2]..LToStr[L4]
  if KFockDensityCache[index] == nil then
    local N1 = (L2 + 1) * (L2 + 2) / 2
    local N2 = (L4 + 1) * (L4 + 2) / 2
    local fspace KFockDensity {
      values : double[N1][N2];
      bound  : float;
    }
    KFockDensityCache[index] = KFockDensity
  end
  return KFockDensityCache[index]
end

-- local KFockOutputCache = {}
-- function getKFockOutput()
--   local index
--   if KFockOutputCache[index] == nil then
--     local N
--     local fspace KFockOutput {
--       values : double[N]
--     }
--     KFockOutputCache[index] = KFockOutput
--   end
--   return KFockOutputCache[index]
-- end
