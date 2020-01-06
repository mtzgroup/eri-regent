import "regent"

local Config = require "config"
require "helper"
require "parse_files"
require "mcmurchie.populate_gamma_table"
require "kfock"

local assert = regentlib.assert
local c = regentlib.c

local r_pairs_list, r_density_list = {}, {}
for L1 = 0, getCompiledMaxMomentum() do -- inclusive
  r_pairs_list[L1], r_density_list[L1] = {}, {}
  for L2 = L1, getCompiledMaxMomentum() do -- inclusive
    r_density_list[L1][L2] = regentlib.newsymbol("r_kfock_density"..L1..L2)
  end
  for L2 = 0, getCompiledMaxMomentum() do -- inclusive
    r_pairs_list[L1][L2] = regentlib.newsymbol("r_kfock_pairs"..L1..L2)
  end
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  config:dump()

  -- Read regions and parameters from file --
  -------------------------------------------
  var kfock_filename: int8[512]
  var kfock_density_filename : int8[512]
  c.sprintf([&int8](kfock_filename), "%s/kfock_sym.dat", config.input_directory)
  c.sprintf([&int8](kfock_density_filename), "%s/kfock_sym_density.dat", config.input_directory)

  ;[writeKFockToRegions(rexpr kfock_filename end, r_pairs_list)]
  ;[writeKFockDensityToRegions(rexpr kfock_density_filename end, r_density_list)]

  var data : double[5]
  readParametersFile(config.parameters_filename, data)
  var parameters = [Parameters]{
    scalfr = data[0],
    scallr = data[1],
    omega = data[2],
    thresp = data[3],
    thredp = data[4],
  }
  -------------------------------------------

  -- Generate region for the output and gamma table --
  -----------------------------------------
  var r_output = region(ispace(int2d, {6, 6}), double)
  fill(r_output, 0.0)
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)
  -----------------------------------------

  c.printf("******************************************\n")
  c.printf("*    Two-Electron Repulsion Integrals    *\n")
  c.printf("*                 KFock                  *\n")
  c.printf("* Parallelism: %25u *\n", config.parallelism)
  c.printf("******************************************\n")

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()
  __fence(__execution, __block) -- Make sure we only time the computation

  -- Compute results --
  ---------------------
  var threshold = parameters.thredp
  var parallelism = config.parallelism;
  [kfock(r_pairs_list, r_density_list, r_output, r_gamma_table, threshold, parallelism)]
  ---------------------

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)
  __fence(__execution, __block) -- Make sure we only time the computation


  -- Write or verify output --
  ----------------------------
  var output_filename = config.output_filename
  if output_filename[0] ~= 0 then
    assert(false, "Unimplemented")
  end
  var verify_filename = config.verify_filename
  if verify_filename[0] ~= 0 then
    verifyKFockOutput(r_output, 1e-7, 1e-8, verify_filename)
  end
  ----------------------------
end

regentlib.start(toplevel)
