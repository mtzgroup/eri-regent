import "regent"

fspace HermiteGaussian {
  {x, y, z} : double;  -- Location of Gaussian
  eta       : double;  -- Exponent of Gaussian
  C         : double;  -- sqrt(2 * pi^(5/2)) / eta
  L         : int1d;   -- Angular momentum
  data_rect : rect1d;  -- Gives a range of indices where the number of values
                       -- is given by (L + 1) * (L + 2) * (L + 3) / 6
                       -- If `HermiteGaussian` is interpreted as a "bra", then
                       --`data_rect` refers to the J values. Otherwise, it
                       -- refers to the density matrix values.
  bound     : double;  -- TODO
}

local function generateHermiteGaussian(L)
  local H = (L + 1) * (L + 2) * (L + 3) / 6
  local fspace HermiteGaussianPacked {
    {x, y, z} : double;     -- Location of Gaussian
    eta       : double;     -- Exponent of Gaussian
    C         : double;     -- sqrt(2 * pi^(5/2)) / eta
    density   : double[H];  -- Density matrix
    j         : double[H];  -- J values
    bound     : double;     -- TODO
  }
  return HermiteGaussianPacked
end

local max_momentum = 2
local HermiteGaussianPacked = {}
for L = 0, max_momentum do -- inclusive
  HermiteGaussianPacked[L] = generateHermiteGaussian(L)
end

function getHermiteGaussian(L)
  return HermiteGaussianPacked[L]
end

fspace Double {
  value : double;
}
