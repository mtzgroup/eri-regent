import "regent"

fspace Double {
  value : double;
}

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

local function generateHermiteGaussianPacked(L)
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

local HermiteGaussianPacked = {}
function getHermiteGaussianPacked(L)
  if HermiteGaussianPacked[L] == nil then
    HermiteGaussianPacked[L] = generateHermiteGaussianPacked(L)
  end
  return HermiteGaussianPacked[L]
end

function packHermiteGaussian(L)
  local H = (L + 1) * (L + 2) * (L + 3) / 6
  local
  task packHermiteGaussian(r_gausses : region(ispace(int1d), HermiteGaussian),
                           r_density : region(ispace(int1d), Double),
                           r_packed  : region(ispace(int1d), getHermiteGaussianPacked(L)))
  where
    reads(r_gausses, r_density),
    writes(r_packed)
  do
    regentlib.assert(r_gausses.volume == r_packed.volume, "Packed gaussian does not have correct shape")
    -- FIXME: This copy operation mixes up the values of `eta`
    -- copy(r_gausses.{x, y, z, eta, C}, r_packed.{x, y, z, eta, C})
    var density : double[H]
    var r_gausses_lo : int = r_gausses.bounds.lo
    for i = 0, r_packed.volume do -- exclusive
      r_packed[i].x = r_gausses[r_gausses_lo + i].x
      r_packed[i].y = r_gausses[r_gausses_lo + i].y
      r_packed[i].z = r_gausses[r_gausses_lo + i].z
      r_packed[i].eta = r_gausses[r_gausses_lo + i].eta
      r_packed[i].C = r_gausses[r_gausses_lo + i].C
      r_packed[i].bound = r_gausses[r_gausses_lo + i].bound
      var data_rect_lo : int = r_gausses[r_gausses_lo + i].data_rect.lo
      for array_idx = 0, H do -- exclusive
        density[array_idx] = r_density[data_rect_lo + array_idx].value
      end
      r_packed[i].density = density
    end
    var zero : double[H]
    for i = 0, H do -- exclusive
      zero[i] = 0.0
    end
    fill(r_packed.j, zero)
  end
  packHermiteGaussian:set_name("packHermiteGaussian"..L)
  return packHermiteGaussian
end
