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
local function dispatchIntegrals(p_gausses, r_density, r_j_values, r_gamma_table, parallelism)
  -- local lua_spin_pattern = generateSpinPatternRegion(max_momentum)
  local statements = terralib.newlist({rquote
    -- TODO: Spin pattern region should not be initialized here.
    -- [lua_spin_pattern.initialize]
  end})

  local r_gausses_packed = {}
  for L = 0, max_momentum do -- inclusive
    local H = (L + 1) * (L + 2) * (L + 3) / 6
    local r_packed = regentlib.newsymbol("r_gausses_packed"..L)
    statements:insert(rquote
      var [r_packed] = region(ispace(int1d, p_gausses[L].volume), getHermiteGaussian(L))
      -- FIXME: This copy operation mixes up the values of `eta`
      -- copy((p_gausses[L]).{x, y, z, eta, bound}, [r_packed[L]].{x, y, z, eta, bound})
      var p_gausses_lo : int = p_gausses[L].bounds.lo
      for i = 0, r_packed.volume do -- exclusive
        r_packed[i].x = p_gausses[L][p_gausses_lo + i].x
        r_packed[i].y = p_gausses[L][p_gausses_lo + i].y
        r_packed[i].z = p_gausses[L][p_gausses_lo + i].z
        r_packed[i].eta = p_gausses[L][p_gausses_lo + i].eta
        r_packed[i].C = p_gausses[L][p_gausses_lo + i].C
        r_packed[i].bound = p_gausses[L][p_gausses_lo + i].bound
        var data_rect_lo : int = p_gausses[L][p_gausses_lo + i].data_rect.lo
        for array_idx = 0, H do -- exclusive
          r_packed[i].density[array_idx] = r_density[data_rect_lo + array_idx].value
        end
      end
      var zero : double[H]
      for i = 0, H do -- exclusive
        zero[i] = 0.0
      end
      fill(r_packed.j, zero)
    end)
    r_gausses_packed[L] = r_packed
  end


  for L12 = 0, max_momentum do -- inclusive
    for L34 = 0, max_momentum do -- inclusive
      local use_mcmurchie = true
      if use_mcmurchie then
        statements:insert(rquote
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
          --       p_density[ket_color], p_j_values[bra_color], r_gamma_table
          --     )
          --   end
          -- end
          [generateTaskMcMurchieIntegral(L12, L34)]([r_gausses_packed[L12]], [r_gausses_packed[L34]],
                                                    r_gamma_table)
        end)
      else -- use Rys
        statements:insert(rquote
          -- var bra_coloring = ispace(int1d, parallelism)
          -- var ket_coloring = ispace(int1d, parallelism)
          -- var p_bra_gausses = partition(equal, p_gausses[L12], bra_coloring)
          -- var p_ket_gausses = partition(equal, p_gausses[L34], ket_coloring)
          -- var p_density = image(p_density[L34], p_ket_gausses, r_gausses.data_rect)
          -- var p_j_values = image(p_j_values[L12], p_bra_gausses, r_gausses.data_rect)
          -- for bra_color in bra_coloring do
          --   __demand(__parallel)
          --   for ket_color in ket_coloring do
          --     [generateTaskRysIntegral(L12, L34)](
          --       p_bra_gausses[bra_color], p_ket_gausses[ket_color],
          --       p_density[ket_color], p_j_values[bra_color],
          --       [lua_spin_pattern.r_spin_pattern], 1, 1
          --     )
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
-- @param highets_L The highest angular momentum of the contracted guassians.
-- FIXME: All `HermiteGaussian`s in `r_gausses` with the same angular momentum
--        are assumed to be contiguous. Either check this with an `assert` or
--        create a new region where this is the case.
task coulomb(r_gausses  : region(ispace(int1d), HermiteGaussian),
             r_density  : region(ispace(int1d), Double),
             r_j_values : region(ispace(int1d), Double),
             highest_L : int, parallelism : int)
where
  reads(r_gausses, r_density),
  writes(r_j_values),
  r_density * r_j_values
do
  regentlib.assert(highest_L == max_momentum,
                   "Please compile for higher angular momentum.")
  -- TODO: Eventually populating r_gamma_table should be done outside `coulomb`.
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)

  var p_gausses = partition(r_gausses.L, ispace(int1d, highest_L + 1));

  [dispatchIntegrals(p_gausses, r_density, r_j_values, r_gamma_table, parallelism)]
end
