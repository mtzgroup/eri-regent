import "regent"

local Config = require "config"
require "helper"
require "parse_files"
require "mcmurchie.populate_gamma_table"
require "kfock"

local assert = regentlib.assert
local c = regentlib.c

-- Iterate over the full square
local r_pairs_list, r_labels_list, r_prevals_list = {}, {}, {}
for L1 = 0, getCompiledMaxMomentum() do -- inclusive
  r_pairs_list[L1] = {}
  r_labels_list[L1] = {}
  r_prevals_list[L1] = {}
  for L2 = 0, getCompiledMaxMomentum() do -- inclusive
    r_pairs_list[L1][L2] = regentlib.newsymbol("r_kfock_pairs"..L1..L2)
    r_labels_list[L1][L2] = regentlib.newsymbol("r_kfock_labels"..L1..L2)
    r_prevals_list[L1][L2] = {
      regentlib.newsymbol("r_kfock_bra_prevals"..L1..L2),
      regentlib.newsymbol("r_kfock_ket_prevals"..L1..L2)
    }
  end
end

-- Iterate over the upper triangle.
local r_density_list, r_output_list = {}, {}
for L1 = 0, getCompiledMaxMomentum() do -- inclusive
  r_density_list[L1], r_output_list[L1] = {}, {}
  for L2 = L1, getCompiledMaxMomentum() do -- inclusive
    r_density_list[L1][L2] = regentlib.newsymbol("r_density"..L1..L2)
    r_output_list[L1][L2] = regentlib.newsymbol("r_output"..L1..L2)
  end
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  config:dump()

  -- Read regions and parameters from file --
  -------------------------------------------
  var kfock_filename: int8[512]
  var kfock_labels_filename : int8[512]
  var kfock_density_filename : int8[512]
  c.sprintf([&int8](kfock_filename), "%s/kfock.dat", config.input_directory)
  c.sprintf([&int8](kfock_labels_filename), "%s/kfock_labels.dat", config.input_directory)
  c.sprintf([&int8](kfock_density_filename), "%s/kfock_density.dat", config.input_directory)

  ;[writeKFockToRegions(rexpr kfock_filename end, r_pairs_list, r_prevals_list)]
  ;[writeKFockLabelsToRegions(rexpr kfock_labels_filename end, r_labels_list)]
  ;[writeKFockDensityToRegions(rexpr kfock_density_filename end,
                               r_density_list, r_output_list)]

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

  -- Generate region for gamma table --
  -------------------------------------
  var r_gamma_table = region(ispace(int2d, {18, 700}), double[5])
  populateGammaTable(r_gamma_table)
  -------------------------------------

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
  var kguard = parameters.kguard
  var parallelism = config.parallelism;
  var largest_momentum = [getCompiledMaxMomentum()]
  ;[kfock(r_pairs_list, r_prevals_list, r_labels_list, r_density_list, r_output_list,
          r_gamma_table, threshold, kguard, parallelism, largest_momentum)]
  ---------------------
  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  __fence(__execution, __block) -- Make sure we only time the computation

  -- Write or verify output --
  ----------------------------
  var output_filename = config.output_filename
  if output_filename[0] ~= 0 then
    assert(false, "Unimplemented")
  end
  var verify_filename = config.verify_filename
  if verify_filename[0] ~= 0 then
    [verifyKFockOutput(r_output_list, 1e-7, 1e-8, verify_filename)]
  end
  ----------------------------
  c.printf("Exchange operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  -- Timing --
  ------------
  if config.num_trials > 1 then
    c.printf("\nCollecting timing info...\n")
    __fence(__execution, __block) -- Make sure we only time the computation
    ts_start = c.legion_get_current_time_in_micros()
    __fence(__execution, __block) -- Make sure we only time the computation
    for i = 0, config.num_trials do
      [kfock(r_pairs_list, r_prevals_list, r_labels_list, r_density_list, r_output_list,
              r_gamma_table, threshold, kguard, parallelism, largest_momentum)]
    end
    __fence(__execution, __block) -- Make sure we only time the computation
    ts_stop = c.legion_get_current_time_in_micros()
    c.printf("Exchange operator, avg. time over %d trials: %.4f sec \n", 
              config.num_trials, (ts_stop - ts_start) * 1e-6/float(config.num_trials))
    __fence(__execution, __block) -- Make sure we only time the computation
  end
end

if os.getenv('SAVEOBJ') == '1' then
  local exe = os.getenv('OBJNAME') or "kfock_regent"
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = {"-L" .. out_dir, "-lm"}
  regentlib.c.printf("Saving executable to %s ... \n", exe)
  regentlib.saveobj(toplevel, exe, "executable", nil, link_flags)
else
  regentlib.start(toplevel)
end
