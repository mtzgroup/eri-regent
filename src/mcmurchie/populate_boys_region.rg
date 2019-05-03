import "regent"
local boysHeader = terralib.includec("mcmurchie/precomputedBoys.h", {"-I", "./"})


terra getBoysLargestJ() : int
  return boysHeader._precomputed_boys_largest_j
end


local
terra _getPrecomputedBoys(t : int, j : int) : double
  return boysHeader._precomputed_boys[t * (getBoysLargestJ() + 1) + j]
end


task populateBoysRegion(r_boys : region(ispace(int2d), double))
where
  writes(r_boys)
do
  -- TODO: Use legion API to populate this region
  for index in r_boys.ispace do
    r_boys[index] = _getPrecomputedBoys(index.x, index.y)
  end
end
