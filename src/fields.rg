import "regent"

fspace Double {
  value : double;
}

struct Parameters {
  num_atomic_orbitals : int;
  scalfr              : double;
  scallr              : double;
  omega               : double;
  thresp              : double;
  thredp              : double;
}

-- A field space for a gaussian with no density values.
fspace Gaussian {
  {x, y, z} : double; -- Location of gaussian
  eta       : double; -- Exponent of gaussian
  C         : double; -- FIXME: sqrt(2 * pi^(5/2)) / eta
  bound     : double;
}

-- Generate a field space for a guassian with density values.
local GaussianWithDensityCache = {}
function getGaussianWithDensity(L)
  if GaussianWithDensityCache[L] == nil then
    local H = (L + 1) * (L + 2) * (L + 3) / 6
    local fspace GaussianWithDensity {
      {x, y, z} : double;    -- Location of gaussian
      eta       : double;    -- Exponent of gaussian
      C         : double;    -- FIXME: sqrt(2 * pi^(5/2)) / eta
      bound     : double;
      density   : double[H];
    }
    GaussianWithDensityCache[L] = GaussianWithDensity
  end
  return GaussianWithDensityCache[L]
end
