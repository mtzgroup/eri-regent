import "regent"
require "fields"
require "mcmurchie.generate_integral"
require "rys.generate_integral"

local max_momentum = 2

-- Generate code to dispatch two-electron repulsion integrals
local function dispatchIntegrals(r_gausses, p_gausses, p_density, p_j_values, r_boys, highest_L, parallelism)
  -- local lua_spin_pattern = generateSpinPatternRegion(max_momentum)
  local statements = terralib.newlist({rquote
    -- [lua_spin_pattern.initialize]
  end})
  for L12 = 0, max_momentum do -- inclusive
    for L34 = 0, max_momentum do -- inclusive
      local use_mcmurchie = true
      if use_mcmurchie then
        statements:insert(rquote
          if L12 <= highest_L and L34 <= highest_L then
            -- var bra_coloring = ispace(int1d, parallelism)
            -- var ket_coloring = ispace(int1d, parallelism)
            -- var p_bra_gausses = partition(equal, p_gausses[L12], bra_coloring)
            -- var p_ket_gausses = partition(equal, p_gausses[L34], ket_coloring)
            -- var p_density = image(p_density[L34], p_ket_gausses, r_gausses.data_rect)
            -- var p_j_values = image(p_j_values[L12], p_bra_gausses, r_gausses.data_rect)
            -- for bra_color in bra_coloring do
            --   __demand(__parallel)
            --   for ket_color in ket_coloring do
            --     [generateTaskMcMurchieIntegral(L12, L34)](
            --       p_bra_gausses[bra_color], p_ket_gausses[ket_color],
            --       p_density[ket_color], p_j_values[bra_color], r_boys
            --     )
            --   end
            -- end
            [generateTaskMcMurchieIntegral(L12, L34)](p_gausses[L12], p_gausses[L34],
                                                      p_density[L34], p_j_values[L12],
                                                      r_boys)
          end
        end)
      else -- use Rys
        statements:insert(rquote
          -- if L12 <= highest_L and L34 <= highest_L then
          --   var bra_coloring = ispace(int1d, parallelism)
          --   var ket_coloring = ispace(int1d, parallelism)
          --   var p_bra_gausses = partition(equal, p_gausses[L12], bra_coloring)
          --   var p_ket_gausses = partition(equal, p_gausses[L34], ket_coloring)
          --   var p_density = image(p_density[L34], p_ket_gausses, r_gausses.data_rect)
          --   var p_j_values = image(p_j_values[L12], p_bra_gausses, r_gausses.data_rect)
          --   for bra_color in bra_coloring do
          --     __demand(__parallel)
          --     for ket_color in ket_coloring do
          --       [generateTaskRysIntegral(L12, L34)](
          --         p_bra_gausses[bra_color], p_ket_gausses[ket_color],
          --         p_density[ket_color], p_j_values[bra_color],
          --         [lua_spin_pattern.r_spin_pattern], 1, 1
          --       )
          --     end
          --   end
          -- end
        end)
      end
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
  regentlib.assert(highest_L <= max_momentum,
                   "Please compile for higher angular momentum.")
  regentlib.assert(r_boys.volume >= 121 * (2*highest_L+7),
                   "Please generate more precomputed boys values.")
  fill(r_j_values.value, 0.0)

  var L_coloring = ispace(int1d, highest_L + 1)
  var p_gausses = partition(r_gausses.L, L_coloring)
  var p_density = image(r_density, p_gausses, r_gausses.data_rect)
  var p_j_values = image(r_j_values, p_gausses, r_gausses.data_rect)

  ;[dispatchIntegrals(r_gausses, p_gausses, p_density, p_j_values, r_boys, highest_L, parallelism)];
end
