import "regent"
require "fields"
require "mcmurchie.generate_integral"
require "mcmurchie.populate_gamma_table"
require "rys.generate_integral"

local max_momentum
if arg[1] == "--max_momentum" then
  max_momentum = tonumber(arg[2])
else
  max_momentum = 2
end

-- Generate code to dispatch two-electron repulsion integrals
local function dispatchIntegrals(p_gausses, r_density, r_j_values, r_gamma_table)
  -- local lua_spin_pattern = generateSpinPatternRegion(max_momentum)
  local statements = terralib.newlist({rquote
    -- TODO: Spin pattern region should not be initialized here.
    -- [lua_spin_pattern.initialize]
  end})

  local r_gausses_packed = {}
  for L = 0, max_momentum do -- inclusive
    r_gausses_packed[L] = regentlib.newsymbol("r_gausses_packed"..L)
    statements:insert(rquote
      var [r_gausses_packed[L]] = region(ispace(int1d, p_gausses[L].volume), getHermiteGaussianPacked(L));
      [packHermiteGaussian(L)](p_gausses[L], r_density, [r_gausses_packed[L]])
    end)
  end

  for L12 = 0, max_momentum do -- inclusive
    for L34 = 0, max_momentum do -- inclusive
      local r_bra_gausses = r_gausses_packed[L12]
      local r_ket_gausses = r_gausses_packed[L34]
      local use_mcmurchie = true
      if use_mcmurchie then
        local integral = generateTaskMcMurchieIntegral(L12, L34)
        local parallelism = 1
        statements:insert(rquote
          var bra_coloring = ispace(int1d, parallelism)
          var p_bra_gausses = partition(equal, r_bra_gausses, bra_coloring)
          __demand(__parallel)
          for bra_color in bra_coloring do
            integral(p_bra_gausses[bra_color], r_ket_gausses, r_gamma_table)
          end
        end)
      else -- use Rys
        local integral = generateTaskRysIntegral(L12, L34)
        statements:insert(rquote
          -- var bra_coloring = ispace(int1d, 1)
          -- var ket_coloring = ispace(int1d, 1)
          -- var p_bra_gausses = partition(equal, r_bra_gausses, bra_coloring)
          -- var p_ket_gausses = partition(equal, r_ket_gausses, ket_coloring)
          -- for bra_color in bra_coloring do
          --   __demand(__parallel)
          --   for ket_color in ket_coloring do
          --     integral(p_bra_gausses[bra_color], p_ket_gausses[ket_color],
          --              [lua_spin_pattern.r_spin_pattern], 1, 1)
          --   end
          -- end
        end)
      end
    end
  end

  for L = 0, max_momentum do -- inclusive
    local H = (L + 1) * (L + 2) * (L + 3) / 6
    local r_packed = r_gausses_packed[L]
    statements:insert(rquote
      var p_gausses_lo : int = p_gausses[L].bounds.lo
      for i = 0, r_packed.volume do -- exclusive
        var data_rect_lo : int = p_gausses[L][p_gausses_lo + i].data_rect.lo
        for array_idx = 0, H do -- exclusive
          r_j_values[data_rect_lo + array_idx].value = r_packed[i].j[array_idx]
        end
      end
    end)
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
-- @param highest_L The highest angular momentum of the contracted guassians.
-- FIXME: All `HermiteGaussian`s in `r_gausses` with the same angular momentum
--        are assumed to be contiguous. Either check this with an `assert` or
--        create a new region where this is the case.
task coulomb(r_gausses  : region(ispace(int1d), HermiteGaussian),
             r_density  : region(ispace(int1d), Double),
             r_j_values : region(ispace(int1d), Double),
             highest_L  : int)
where
  reads(r_gausses, r_density),
  writes(r_j_values),
  r_density * r_j_values
do
  regentlib.assert(highest_L == max_momentum, "Please compile for correct angular momentum")
  -- TODO: Eventually populating r_gamma_table should be done outside `coulomb`.
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)

  var p_gausses = partition(r_gausses.L, ispace(int1d, highest_L + 1));

  [dispatchIntegrals(p_gausses, r_density, r_j_values, r_gamma_table)]
end
