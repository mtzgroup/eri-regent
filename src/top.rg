import "regent"

local Config = require "config"
require "helper"
require "jfock"
require "mcmurchie.populate_gamma_table"
require "parse_files"

local c = regentlib.c
local assert = regentlib.assert

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
  ;[writeJBrasToRegions(rexpr config.bras_filename end, r_jbras_list)]
  ;[writeJKetsToRegions(rexpr config.kets_filename end, r_jkets_list)]
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

  -- Generate region for the gamma table --
  -----------------------------------------
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)
  -----------------------------------------

  c.printf("******************************************\n")
  c.printf("*    Two-Electron Repulsion Integrals    *\n")
  c.printf("*                                        *\n")
  c.printf("* Max Angular Momentum : %15u *\n", config.max_momentum)
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
  var threshold = parameters.thredp
  var parallelism = config.parallelism;
  [jfock(r_jbras_list, r_jkets_list, r_gamma_table, threshold, parallelism)]
  ---------------------

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)


  -- Write or verify output --
  ----------------------------
  var output_filename = config.output_filename
  if output_filename[0] ~= 0 then
    [writeOutput(r_jbras_list, output_filename)]
  end
  var verify_filename = config.verify_filename
  if verify_filename[0] ~= 0 then
    [verifyOutput(r_jbras_list, 1e-8, verify_filename)]
  end
  ----------------------------
end

regentlib.start(toplevel)
