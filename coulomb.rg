import "regent"
require("fields")
local integralTypes = {
  "SSSS", "SSSP", "SSPP", "SPSS", "SPSP", "SPPP", "PPSS", "PPSP", "PPPP"
}
for _, type in pairs(integralTypes) do
  require("integrals."..type)
end

local c = regentlib.c
local assert = regentlib.assert

-- Computes fancy two-electron repulsion integrals
--
-- @param r_gausses Region of contracted gaussians.
-- @param r_density Region of density matrix values that are referenced
--                  from r_gausses.
-- @param r_j_values Region of j values that will be overwritten.
-- @param r_boys Region of precomputed values of the Boys integral.
-- @param highets_L The highest angular momentum of the contracted guassians.
task coulomb(r_gausses  : region(ispace(int1d), HermiteGaussian),
             r_density  : region(ispace(int1d), Double),
             r_j_values : region(ispace(int1d), Double),
             r_boys     : region(ispace(int2d), Double),
             highest_L  : int)
where
  reads(r_gausses, r_density, r_boys),
  reads writes(r_j_values),
  r_density * r_j_values,
  r_density * r_boys,
  r_j_values * r_boys
do
  fill(r_j_values.value, 0.0)

  -- First partition by angular momentum
  var L_coloring = ispace(int1d, highest_L + 1)
  var p_gausses = partition(r_gausses.L, L_coloring)
  var p_density = image(r_density, p_gausses, r_gausses.data_rect)
  var p_j_values = image(r_j_values, p_gausses, r_gausses.data_rect)

  for bra_color in L_coloring do
    for ket_color in L_coloring do
      -- TODO: Need to decide how much parallelism to give to each integral
      var coloring = ispace(int1d, 2)
      -- TODO: Is it ok to partition here? Or is there some way to partition
      --       all block types all at once?
      -- One idea to partition integrals: equally partition bras and don't
      -- partition kets at all.
      var p_bra_gausses = partition(equal, p_gausses[bra_color], coloring)
      var r_ket_gausses = p_gausses[ket_color]
      var r_density = p_density[ket_color]
      var p_j_values = image(p_j_values[bra_color], p_bra_gausses, r_gausses.data_rect)

      -- TODO: Metaprogram
      var block_type = [int2d]{bra_color, ket_color}
      if block_type == [int2d]{0, 0} then
        __demand(__parallel)
        for color in coloring do
          coulombSSSS(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{0, 1} then
        __demand(__parallel)
        for color in coloring do
          coulombSSSP(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{0, 2} then
        __demand(__parallel)
        for color in coloring do
          coulombSSPP(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{1, 0} then
        __demand(__parallel)
        for color in coloring do
          coulombSPSS(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{1, 1} then
        __demand(__parallel)
        for color in coloring do
          coulombSPSP(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{1, 2} then
        __demand(__parallel)
        for color in coloring do
          coulombSPPP(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{2, 0} then
        __demand(__parallel)
        for color in coloring do
          coulombPPSS(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{2, 1} then
        __demand(__parallel)
        for color in coloring do
          coulombPPSP(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      elseif block_type == [int2d]{2, 2} then
        __demand(__parallel)
        for color in coloring do
          coulombPPPP(p_bra_gausses[color], r_ket_gausses,
                      r_density, p_j_values[color], r_boys)
        end
      else
        c.printf("Block type = {%d, %d}\n", block_type.x, block_type.y)
        assert(false, "Block type not implemented")
      end
    end
  end
end
