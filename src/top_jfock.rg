import "regent"

local Config = require "config"
require "helper"
require "jfock"
require "mcmurchie.populate_gamma_table"
require "parse_files"

local assert = regentlib.assert
local c = regentlib.c

local r_jbras_list, r_jkets_list = {}, {}
for L1 = 0, getCompiledMaxMomentum() do -- inclusive
  r_jbras_list[L1], r_jkets_list[L1] = {}, {}
  for L2 = L1, getCompiledMaxMomentum() do -- inclusive
    r_jbras_list[L1][L2] = regentlib.newsymbol("r_jbras"..L1..L2)
    r_jkets_list[L1][L2] = regentlib.newsymbol("r_jkets"..L1..L2)
  end
end

function dumpRegionSizes(name, r)
  local statements = terralib.newlist()
  for L1 = 0, getCompiledMaxMomentum() do -- inclusive
    for L2 = L1, getCompiledMaxMomentum() do -- inclusive
      statements:insert(rquote
        c.printf("* \t\t%s%s  %20d *\n", [LToStr[L1]], [LToStr[L2]],
                                         [r[L1][L2]].volume)
      end)
    end
  end
  return statements
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  config:dump()

  -- Read regions and parameters from file --
  -------------------------------------------
  var bras_filename : int8[512]
  var kets_filename : int8[512]
  c.sprintf([&int8](bras_filename), "%s/jfock_bras.dat", config.input_directory)
  c.sprintf([&int8](kets_filename), "%s/jfock_kets.dat", config.input_directory)

  ;[writeJBrasToRegions(rexpr bras_filename end, r_jbras_list)]
  ;[writeJKetsToRegions(rexpr kets_filename end, r_jkets_list)]
  var data : double[6]
  readParametersFile(config.parameters_filename, data)
  var parameters = [Parameters]{
    scalfr = data[0],
    scallr = data[1],
    omega = data[2],
    thresp = data[3],
    thredp = data[4],
    kguard = data[5],
  }
  -------------------------------------------

  -- Generate region for the gamma table --
  -----------------------------------------
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)
  -----------------------------------------

  c.printf("******************************************\n")
  c.printf("*    Two-Electron Repulsion Integrals    *\n")
  c.printf("*                 JFock                  *\n")
  c.printf("* Parallelism: %25u *\n", config.parallelism)
  c.printf("* Number of Bras:                        *\n");
  [dumpRegionSizes("Bras", r_jbras_list)]
  c.printf("* Number of Kets:                        *\n");
  [dumpRegionSizes("Kets", r_jkets_list)]
  c.printf("******************************************\n")

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()
  __fence(__execution, __block) -- Make sure we only time the computation

  -- Compute results --
  ---------------------
  var threshold = parameters.thredp
  var parallelism = config.parallelism;
  [jfock(r_jbras_list, r_jkets_list, r_gamma_table, threshold, parallelism, parallelism)]
  ---------------------

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)
  __fence(__execution, __block) -- Make sure we only time the computation


  -- Write or verify output --
  ----------------------------
  var output_filename = config.output_filename
  if output_filename[0] ~= 0 then
    [writeJFockOutput(r_jbras_list, output_filename)]
  end
  var verify_filename = config.verify_filename
  if verify_filename[0] ~= 0 then
    [verifyJFockOutput(r_jbras_list, 1e-7, 1e-8, verify_filename)]
  end
  ----------------------------

  -- Timing --
  ------------
  if config.num_trials > 1 then
    c.printf("\nCollecting timing info...\n")
    __fence(__execution, __block) -- Make sure we only time the computation
    ts_start = c.legion_get_current_time_in_micros()
    __fence(__execution, __block) -- Make sure we only time the computation
    for i = 0, config.num_trials do
      [jfock(r_jbras_list, r_jkets_list, r_gamma_table, threshold, parallelism, parallelism)]
    end
    __fence(__execution, __block) -- Make sure we only time the computation
    ts_stop = c.legion_get_current_time_in_micros()
    c.printf("Coulomb operator, avg. time over %d trials: %.4f sec \n", 
              config.num_trials, (ts_stop - ts_start) * 1e-6/float(config.num_trials))
    __fence(__execution, __block) -- Make sure we only time the computation
  end

end

regentlib.start(toplevel)
