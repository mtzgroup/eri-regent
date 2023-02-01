import "regent"

require "helper"

struct Parameters {
  scalfr : double;
  scallr : double;
  omega  : double;
  thresp : float;
  thredp : float;
  kguard : float;
}

struct Point {
  x : double;
  y : double;
  z : double;
}

local JBraCache = {}
function getJBra(L12)
  if JBraCache[L12] == nil then
    local H = tetrahedral_number(L12 + 1)
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
    local H = tetrahedral_number(L34 + 1)
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

-- TODO?: Create different fspaces for SS, SP, and PP.
--        SS should have not Pi or Pj,
--        SP should only have Pj, and
--        PP should have both Pi and Pj.
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
      ishell_index    : int1d;  -- Index for K output.
      jshell_index    : int1d;  -- Index for density.
    }
    KFockPairCache[index] = KFockPair
  end
  return KFockPairCache[index]
end

local KFockLabelCache = {}
function getKFockLabel(L1, L2)
  local index = LToStr[L1]..LToStr[L2]
  if KFockLabelCache[index] == nil then
    local fspace KFockLabel {
      ishell         : int1d;  -- Which iShell this label belongs to
      start_index    : int1d;   
    }
    KFockLabelCache[index] = KFockLabel
  end
  return KFockLabelCache[index]
end

local KFockDensityCache = {}
function getKFockDensity(L2, L4)
  if L2 > L4 then -- original
    L2, L4 = L4, L2
  end
  local index = LToStr[L2]..LToStr[L4]
  if KFockDensityCache[index] == nil then
    local H2, H4 = triangle_number(L2 + 1), triangle_number(L4 + 1)
    local fspace KFockDensity {
      --values : double[H2][H4];
      values : double[H4][H2]; -- Regent array indexing opposite of C
      bound  : float;
    }
    KFockDensityCache[index] = KFockDensity
  end
  return KFockDensityCache[index]
end

local KFockOutputCache = {}
function getKFockOutput(L1, L3)
  local index = LToStr[L1]..LToStr[L3]
  if KFockOutputCache[index] == nil then
    local H1, H3 = triangle_number(L1 + 1), triangle_number(L3 + 1)
    local fspace KFockOutput {
      values            : double[H3 * H1]; -- flattened 2D array
      bra_ishell_index  : int1d; -- iShell index for L1 (for potential partitioning)
      ket_ishell_index  : int1d; -- iShell index for L3 (for potential partitioning)
    }
    KFockOutputCache[index] = KFockOutput
  end
  return KFockOutputCache[index]
end

local KGradBraCache = {}
function getKGradBra(L12)
  if KGradBraCache[L12] == nil then
    local fspace KGradBra {
      location        : Point;     -- Location of gaussian
      eta             : double;    -- Exponent of gaussian
      C               : double;
      bound           : float;
      ishell_index    : int1d;  -- Index for density fetches
      jshell_index    : int1d;  -- Index for density fetches
    }
    KGradBraCache[L12] = KGradBra
  end
  return KGradBraCache[L12]
end

local KGradOutputCache = {}
function getKGradOutput(L1, L2)
  local index = LToStr[L1]..LToStr[L2]
  if KGradOutputCache[index] == nil then
    local fspace KGradOutput {
      values     : double[6]; -- x,y,z for center A and  x,y,z for center B
      bra_index  : int1d; -- index for L12 (for potential partitioning)
    }
    KGradOutputCache[index] = KGradOutput
  end
  return KGradOutputCache[index]
end

local KGradOutputLongCache = {}
function getKGradOutputLong(L1, L2, L3, L4)
  local index = LToStr[L1]..LToStr[L2]..LToStr[L3]..LToStr[L4]
  if KGradOutputLongCache[index] == nil then
    local fspace KGradOutputLong {
      values     : double[6]; -- x,y,z for center A and  x,y,z for center B
      bra_index  : int1d; -- index for L12 (for potential partitioning)
    }
    KGradOutputLongCache[index] = KGradOutputLong
  end
  return KGradOutputLongCache[index]
end

local KGradBraEGPMapCache = {}
function getKGradBraEGPMap(L1, L2)
  local index = LToStr[L1]..LToStr[L2]
  if KGradBraEGPMapCache[index] == nil then
    local fspace KGradBraEGPMap {
      sign   : int1d; -- sign (+/-) of next braEGP element
      stride : int1d; -- braEGP stride
    }
    KGradBraEGPMapCache[index] = KGradBraEGPMap
  end
  return KGradBraEGPMapCache[index]
end

KFockNumBraPrevals = {
  [0] = {[0] = 0,  [1] = 4,   [2] = 16,  [3] = 47},
  [1] = {[0] = 4,  [1] = 25,  [2] = 91,  [3] = 244},
  [2] = {[0] = 16, [1] = 91,  [2] = 301, [3] = 757},
  [3] = {[0] = 47, [1] = 244, [2] = 757, [3] = 1821},
}

KFockNumKetPrevals = {
  [0] = {[0] = 0,  [1] = 4,   [2] = 16,  [3] = 47},
  [1] = {[0] = 6,  [1] = 27,  [2] = 93,  [3] = 246},
  [2] = {[0] = 21, [1] = 96,  [2] = 306, [3] = 762},
  [3] = {[0] = 56, [1] = 253, [2] = 766, [3] = 1830},
}

-- TODO: extend KGradEGP to F
KGradNumBraEGP = { -- upper triangle
  [0] = {[0] = 5, [1] = 23, [2] = 89},
  [1] = {[0] = 0, [1] = 88, [2] = 386},
  [2] = {[0] = 0, [1] = 0,  [2] = 1494},
}

KGradNumBraEGPMap = { -- upper triangle
  [0] = {[0] = 12, [1] = 66,  [2] = 216},
  [1] = {[0] = 0,  [1] = 336, [2] = 1038},
  [2] = {[0] = 0,  [1] = 0,   [2] = 3072},
}

