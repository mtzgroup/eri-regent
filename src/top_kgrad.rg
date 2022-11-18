import "regent"

local Config = require "config"
require "helper"
require "parse_files"
require "mcmurchie.populate_gamma_table"
require "mcmurchie.kgrad.populate_EGP_map"
--require "kgrad" -- TODO: uncomment when kgrad.rg file is complete

local assert = regentlib.assert
local c = regentlib.c

-- Iterate over the full square for ket
local r_kets_list, r_ketlabels_list, r_ketprevals_list = {}, {}, {}
for L1 = 0, getCompiledMaxMomentum() do -- inclusive
  r_kets_list[L1]       = {}
  r_ketlabels_list[L1]  = {}
  r_ketprevals_list[L1] = {}
  for L2 = 0, getCompiledMaxMomentum() do -- inclusive
    r_kets_list[L1][L2]       = regentlib.newsymbol("r_kgrad_kets"..L1..L2)
    r_ketlabels_list[L1][L2]  = regentlib.newsymbol("r_kgrad_ket_labels"..L1..L2)
    r_ketprevals_list[L1][L2] = regentlib.newsymbol("r_kgrad_ket_prevals"..L1..L2)
  end
end

-- Iterate over the upper triangle for bra, densities, and output
local r_bras_list, r_braEGP_list, r_braEGPmap_list = {}, {}, {}
local r_denik_list, r_denjl_list, r_output_list = {}, {}, {}
for L1 = 0, getCompiledMaxMomentum() do -- inclusive
  r_bras_list[L1]      = {}
  r_braEGP_list[L1]    = {}
  r_braEGPmap_list[L1] = {}
  r_denik_list[L1]     = {}
  r_denjl_list[L1]     = {}
  r_output_list[L1]    = {}
  for L2 = L1, getCompiledMaxMomentum() do -- inclusive
    r_bras_list[L1][L2]      = regentlib.newsymbol("r_kgrad_bras"..L1..L2)
    r_braEGP_list[L1][L2]    = regentlib.newsymbol("r_kgrad_braEGP"..L1..L2)
    r_braEGPmap_list[L1][L2] = regentlib.newsymbol("r_braEGPmap"..L1..L2)
    r_denik_list[L1][L2]     = regentlib.newsymbol("r_denik"..L1..L2)
    r_denjl_list[L1][L2]     = regentlib.newsymbol("r_denjl"..L1..L2)
    r_output_list[L1][L2]    = regentlib.newsymbol("r_output"..L1..L2)
  end
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  config:dump()

  -- Read regions and parameters from file --
  -------------------------------------------
  var kgrad_bras_filename: int8[512]
  var kgrad_kets_filename: int8[512]
  var kgrad_labels_filename : int8[512]
  var kgrad_denik_filename : int8[512]
  var kgrad_denjl_filename : int8[512]
  c.sprintf([&int8](kgrad_bras_filename), "%s/kgrad_bras.dat", config.input_directory)
  c.sprintf([&int8](kgrad_kets_filename), "%s/kgrad_kets.dat", config.input_directory)
  c.sprintf([&int8](kgrad_labels_filename), "%s/kfock_labels.dat", config.input_directory) -- kgrad labels are the same as kfock
  c.sprintf([&int8](kgrad_denik_filename), "%s/kgrad_denik.dat", config.input_directory)
  c.sprintf([&int8](kgrad_denjl_filename), "%s/kgrad_denjl.dat", config.input_directory)

  ;[writeKGradBrasToRegions(rexpr kgrad_bras_filename end, r_bras_list, r_braEGP_list, r_output_list)]
  ;[writeKGradKetsToRegions(rexpr kgrad_kets_filename end, r_kets_list, r_ketprevals_list)]
  ;[writeKFockLabelsToRegions(rexpr kgrad_labels_filename end, r_ketlabels_list)] -- kgrad labels are the same as kfock
  ;[writeKGradDensityToRegions(rexpr kgrad_denik_filename end, r_denik_list)]
  ;[writeKGradDensityToRegions(rexpr kgrad_denjl_filename end, r_denjl_list)]

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

  -- Generate region for bra EGP map --
  -------------------------------------
  ;[populateBraEGPMap(r_braEGPmap_list)]
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
  var parallelism = config.parallelism;
  var largest_momentum = [getCompiledMaxMomentum()]
--  ;[kgrad(r_bras_list, r_braEGP_list, 
--          r_kets_list, r_ketprevals_list, r_ketlabels_list, 
--          r_denik_list, r_denjl_list, r_output_list,
--          r_braEGPmap_list, r_gamma_table, 
--          threshold, parallelism, largest_momentum)]
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
    [verifyKGradOutput(r_output_list, 1e-7, 1e-8, verify_filename)]
  end
  ----------------------------
  c.printf("Exchange gradient operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  -- Timing --
  ------------
  if config.num_trials > 1 then
    c.printf("\nCollecting timing info...\n")
    __fence(__execution, __block) -- Make sure we only time the computation
    ts_start = c.legion_get_current_time_in_micros()
    __fence(__execution, __block) -- Make sure we only time the computation
    for i = 0, config.num_trials do
--      [kgrad(r_bras_list, r_braEGP_list, 
--             r_kets_list, r_ketprevals_list, r_ketlabels_list, 
--             r_denik_list, r_denjl_list, r_output_list,
--             r_braEGPmap_list, r_gamma_table, 
--             threshold, parallelism, largest_momentum)]
    end
    __fence(__execution, __block) -- Make sure we only time the computation
    ts_stop = c.legion_get_current_time_in_micros()
    c.printf("Exchange gradient operator, avg. time over %d trials: %.4f sec \n", 
              config.num_trials, (ts_stop - ts_start) * 1e-6/float(config.num_trials))
    __fence(__execution, __block) -- Make sure we only time the computation
  end
end

regentlib.start(toplevel)
