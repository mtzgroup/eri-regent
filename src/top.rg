import "regent"

local Config = require "config"
require "parse_files"
require "mcmurchie.populate_gamma_table"
require "jfock"

local c = regentlib.c
local assert = regentlib.assert
-- local fabs = regentlib.fabs(double)

-- task writeOutput(r_gausses  : region(ispace(int1d), HermiteGaussian),
--                  r_j_values : region(ispace(int1d), Double),
--                  config     : Config)
-- where
--   reads(r_gausses, r_j_values)
-- do
--   if config.output_filename[0] ~= 0 then
--     if config.verbose then c.printf("Writing output\n") end
--     var file = c.fopen(config.output_filename, "w")
--     -- c.fprintf(file, "%d\n\n", config.num_gausses)
--     for i in r_gausses.ispace do
--       var bra = r_gausses[i]
--       -- c.fprintf(file, "%d %.6f %.12f %.12f %.12f ", bra.L, bra.eta, bra.x, bra.y, bra.z)
--       for j = [int](bra.data_rect.lo), [int](bra.data_rect.hi) + 1 do
--         c.fprintf(file, "%.12f ", r_j_values[j].value)
--       end
--       c.fprintf(file, "\n")
--     end
--     c.fclose(file)
--   end
-- end

-- task verifyOutput(r_gausses       : region(ispace(int1d), HermiteGaussian),
--                   r_j_values      : region(ispace(int1d), Double),
--                   r_true_j_values : region(ispace(int1d), Double),
--                   config          : Config)
-- where
--   reads(r_gausses, r_j_values, r_true_j_values)
-- do
--   if config.true_values_filename[0] == 0 then
--     return
--   end
--   if config.verbose then c.printf("Verifying output\n") end
--   var max_error : double = 0.0
--   var num_incorrect = 0
--   for gauss_idx in r_gausses.ispace do
--     var gaussian = r_gausses[gauss_idx]
--     for i = [int](gaussian.data_rect.lo), [int](gaussian.data_rect.hi + 1) do
--       var actual : double = r_j_values[i].value
--       var expected : double = r_true_j_values[i].value
--       var error : double = fabs(actual - expected)
--       if [bool](c.isnan(actual)) or [bool](c.isinf(actual)) or error > 1e-8 then
--         c.printf("Value differs at gaussian = %d, L = %d, i = %d: actual = %.12f, expected = %.12f\n",
--                  gauss_idx, gaussian.L, i - [int](gaussian.data_rect.lo), actual, expected)
--         num_incorrect += 1
--       end
--       if error > max_error then
--         max_error = error
--       end
--     end
--   end
--   if config.verbose or num_incorrect > 0 then
--     c.printf("%d/%d incorrect values found\n", num_incorrect, r_j_values.ispace.volume)
--     c.printf("Max error = %.12f\n", max_error)
--   end
--   if num_incorrect > 0 then
--     c.exit(1)
--   end
-- end

local r_jbras_list, r_jkets_list = {}, {}
for i = 0, getCompiledMaxMomentum() do -- inclusive
  r_jbras_list[i] = regentlib.newsymbol("r_jbras"..i)
  r_jkets_list[i] = regentlib.newsymbol("r_jkets"..i)
end

function dumpRegionSizes(name, r)
  local statements = terralib.newlist()
  for i = 0, #r do -- inclusive
    statements:insert(rquote
      c.printf("* \t\t%s  %20d *\n", [LToStr[i]], [r[i]].volume)
    end)
  end
  return statements
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  config:dump()

  -- Read regions and parameters from file --
  -------------------------------------------
  ;[writeGaussiansToRegions(rexpr config.bras_filename end, r_jbras_list)];
  [writeGaussiansWithDensityToRegions(rexpr config.kets_filename end, r_jkets_list)]
  var data : double[5]
  readParametersFile(config.parameters_filename, data)
  -- TODO: Add number of atomic orbitals
  var parameters = [Parameters]{
    num_atomic_orbitals = 19, -- TODO
    scalfr = data[0],
    scallr = data[1],
    omega = data[2],
    thresp = data[3],
    thredp = data[4],
  }
  -------------------------------------------

  -- Generate region for the gamma table --
  -----------------------------------------
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)
  -----------------------------------------

  -- Create output region --
  --------------------------
  var N = parameters.num_atomic_orbitals * parameters.num_atomic_orbitals
  var r_output = region(ispace(int1d, N), double)
  fill(r_output, 0)
  --------------------------

  c.printf("******************************************\n")
  c.printf("*    Two-Electron Repulsion Integrals    *\n")
  c.printf("*                                        *\n")
  c.printf("* Max Angular Momentum : %15u *\n", config.max_momentum)
  c.printf("* # Atomic Orbitals : %18u *\n", parameters.num_atomic_orbitals)
  c.printf("* Parallelism : %24u *\n", config.parallelism)
  c.printf("* Number of Bras                         *\n");
  [dumpRegionSizes("Bras", r_jbras_list)]
  c.printf("* Number of Kets                         *\n");
  [dumpRegionSizes("Kets", r_jkets_list)]
  c.printf("******************************************\n")

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()
  __fence(__execution, __block) -- Make sure we only time the computation

  -- Compute results --
  ---------------------
  ;[jfock(r_jbras_list, r_jkets_list,
          r_gamma_table, r_output,
          parameters, rexpr config.parallelism end)]
  ---------------------

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  -- writeOutput(r_gausses, r_j_values, config)
  -- verifyOutput(r_gausses, r_j_values, r_true_j_values, config)
end

regentlib.start(toplevel)
