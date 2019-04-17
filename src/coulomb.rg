import "regent"
require("fields")
require("mcmurchie.generate_integral")

local c = regentlib.c
local assert = regentlib.assert
-- TODO: Consolidate all variables like this into one location
local max_momentum = 2

-- Dispatches several kernels to compute a block of BraKets.
-- All Bras and all Kets are assumed to have the same angular momentum.
-- TODO: Decide at runtime to run McMurchie or Rys.
local function dispatchIntegrals(bra_L_color, ket_L_color,
                                 bra_coloring, ket_coloring,
                                 p_bra_gausses, p_ket_gausses,
                                 p_density, p_j_values, r_boys)
  local statements = terralib.newlist()
  for L12 = 0, max_momentum do -- inclusive
    for L34 = 0, max_momentum do -- inclusive
      statements:insert(rquote
        if [int](bra_L_color) == L12 and [int](ket_L_color) == L34 then
          for bra_color in bra_coloring do
            __demand(__parallel)
            for ket_color in ket_coloring do
              [generateTaskMcMurchieIntegral(L12, L34)](
                p_bra_gausses[bra_color], p_ket_gausses[ket_color],
                p_density[ket_color], p_j_values[bra_color], r_boys
              )
            end
          end
        end
      end)
    end
  end
  return statements
end

-- Computes fancy two-electron repulsion integrals
--
-- @param r_gausses Region of contracted gaussians. These are assumed to be
--                  ordered by angular momentum so that all gaussians of a
--                  particular angular momentum are contiguous.
-- @param r_density Region of density matrix values that are referenced
--                  from r_gausses.
-- @param r_j_values Region of j values that will be overwritten.
-- @param r_boys Region of precomputed values of the Boys integral.
-- @param highets_L The highest angular momentum of the contracted guassians.
task coulomb(r_gausses  : region(ispace(int1d), HermiteGaussian),
             r_density  : region(ispace(int1d), Double),
             r_j_values : region(ispace(int1d), Double),
             r_boys     : region(ispace(int1d), Double),
             highest_L : int, parallelism : int)
where
  reads(r_gausses, r_density, r_boys),
  reads writes(r_j_values),
  r_density * r_j_values,
  r_density * r_boys,
  r_j_values * r_boys
do
  fill(r_j_values.value, 0.0)
  regentlib.assert(highest_L <= max_momentum, "Please compile for higher angular momentum.")

  var L_coloring = ispace(int1d, highest_L + 1)
  var p_gausses = partition(r_gausses.L, L_coloring)
  var p_density = image(r_density, p_gausses, r_gausses.data_rect)
  var p_j_values = image(r_j_values, p_gausses, r_gausses.data_rect)

  for bra_L_color in L_coloring do
    for ket_L_color in L_coloring do
      -- TODO: Need to decide how much parallelism to give to each integral
      var bra_coloring = ispace(int1d, parallelism)
      var ket_coloring = ispace(int1d, parallelism)
      var p_bra_gausses = partition(equal, p_gausses[bra_L_color], bra_coloring)
      var p_ket_gausses = partition(equal, p_gausses[ket_L_color], ket_coloring)
      var p_density = image(p_density[ket_L_color], p_ket_gausses, r_gausses.data_rect)
      var p_j_values = image(p_j_values[bra_L_color], p_bra_gausses, r_gausses.data_rect)

      ;[dispatchIntegrals(bra_L_color, ket_L_color,
                          bra_coloring, ket_coloring,
                          p_bra_gausses, p_ket_gausses,
                          p_density, p_j_values, r_boys)];
    end
  end
end
