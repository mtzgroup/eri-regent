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
--
-- fspace HermiteGaussian {
--   {x, y, z} : double;  -- Location of Gaussian
--   eta       : double;  -- Exponent of Gaussian
--   C         : double;  -- sqrt(2 * pi^(5/2)) / eta
--   L         : int1d;   -- Angular momentum
--   data_rect : rect1d;  -- Gives a range of indices where the number of values
--                        -- is given by (L + 1) * (L + 2) * (L + 3) / 6
--                        -- If `HermiteGaussian` is interpreted as a "bra", then
--                        --`data_rect` refers to the J values. Otherwise, it
--                        -- refers to the density matrix values.
--   bound     : double;  -- TODO
-- }
--
-- local function generateHermiteGaussianPacked(L)
--   local H = (L + 1) * (L + 2) * (L + 3) / 6
--   local fspace HermiteGaussianPacked {
--     {x, y, z} : double;     -- Location of Gaussian
--     eta       : double;     -- Exponent of Gaussian
--     C         : double;     -- sqrt(2 * pi^(5/2)) / eta
--     density   : double[H];  -- Density matrix
--     j         : double[H];  -- J values
--     bound     : double;     -- TODO
--   }
--   return HermiteGaussianPacked
-- end
--
-- local HermiteGaussianPacked = {}
-- function getHermiteGaussianPacked(L)
--   if HermiteGaussianPacked[L] == nil then
--     HermiteGaussianPacked[L] = generateHermiteGaussianPacked(L)
--   end
--   return HermiteGaussianPacked[L]
-- end
--
-- function packHermiteGaussian(L)
--   local H = (L + 1) * (L + 2) * (L + 3) / 6
--   local
--   task packHermiteGaussian(r_gausses : region(ispace(int1d), HermiteGaussian),
--                            r_density : region(ispace(int1d), Double),
--                            r_packed  : region(ispace(int1d), getHermiteGaussianPacked(L)))
--   where
--     reads(r_gausses, r_density),
--     writes(r_packed)
--   do
--     var density : double[H]
--     var i : int = 0
--     for index in r_gausses.ispace do
--       r_packed[i].x = r_gausses[index].x
--       r_packed[i].y = r_gausses[index].y
--       r_packed[i].z = r_gausses[index].z
--       r_packed[i].eta = r_gausses[index].eta
--       r_packed[i].C = r_gausses[index].C
--       r_packed[i].bound = r_gausses[index].bound
--       var data_rect_lo : int = r_gausses[index].data_rect.lo
--       for array_idx = 0, H do -- exclusive
--         density[array_idx] = r_density[data_rect_lo + array_idx].value
--       end
--       r_packed[i].density = density
--       i += 1
--     end
--     var zero : double[H]
--     for i = 0, H do -- exclusive
--       zero[i] = 0.0
--     end
--     fill(r_packed.j, zero)
--   end
--   packHermiteGaussian:set_name("packHermiteGaussian"..L)
--   return packHermiteGaussian
-- end
