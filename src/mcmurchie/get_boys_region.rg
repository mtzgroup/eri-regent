import "regent"
local root_dir = arg[0]:match(".*/") or "./"
local boysHeader = terralib.includec("mcmurchie/precomputedBoys.h", {"-I", root_dir})
local _precomputedBoys = boysHeader._precomputed_boys
local _precomputed_boys_largest_j = boysHeader._precomputed_boys_largest_j


terra getBoysLargestJ() : int
  return _precomputed_boys_largest_j
end


local
terra _getPrecomputedBoys(t : int, j : int) : double
  return _precomputedBoys[t * (_precomputed_boys_largest_j + 1) + j]
end


task populateBoysRegion(r_boys : region(ispace(int2d), double))
where
  reads writes(r_boys)
do
  -- TODO: Use legion API to populate this region
  for index in r_boys.ispace do
    r_boys[index] = _getPrecomputedBoys(index.x, index.y)
  end
end
