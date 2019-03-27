import "regent"

require("coulomb")
local Config = require("config")
local c = regentlib.c
local assert = regentlib.assert
local fabs = regentlib.fabs(double)
local root_dir = arg[0]:match(".*/") or "./"
local _precomputedBoys = terralib.includec("precomputedBoys.h",
                                          {"-I", root_dir})._precomputed_boys

terra getPrecomputedBoys(idx : int) : double
  return _precomputedBoys[idx]
end

terra fgets(line : &int8, n : int, file : &c.FILE) : bool
  return c.fgets(line, n, file) ~= nil
end

terra readline(line : &int8, H : int)
  -- "L eta x y z [density values]"
  var values : &double = [&double](c.malloc(sizeof(double) * (H + 5)))
  var beginC : &int8 = line
  var endC : &int8
  for i = 0, H + 5 do
    values[i] = c.strtod(beginC, &endC)
    beginC = endC
  end
  return values
end

-- Populates regions from input file
task populateData(r_gausses       : region(ispace(int1d), HermiteGaussian),
                  r_density       : region(ispace(int1d), Double),
                  r_true_j_values : region(ispace(int1d), Double),
                  config          : Config)
where
  reads writes(r_gausses, r_density, r_true_j_values),
  r_density * r_true_j_values
do
  var file = c.fopen(config.input_filename, "r")
  var line : int8[512]
  fgets(line, 512, file)  -- Skip first line
  var i : int1d = 0
  var density_idx : int1d = 0
  while fgets(line, 512, file) do
    if c.strncmp(line, "\n", 1) ~= 0 and c.strncmp(line, "\r\n", 2) ~= 0 then
      var data : int[1]
      c.sscanf([&int8](line), "%d", data)
      var L : int = data[0]
      var H : int = (L + 1) * (L + 2) * (L + 3) / 6
      var values : &double = readline(line, H)
      var eta : double = values[1]
      var x : double = values[2]
      var y : double = values[3]
      var z : double = values[4]
      r_gausses[i] = {x=x, y=y, z=z, eta=eta, L=L,
                      data_rect={density_idx, density_idx+H-1}, bound=0}
      for j = 0, H do
        r_density[density_idx].value = values[j + 5]
        density_idx = density_idx + 1
      end
      c.free(values)
      i = i + 1
    end
  end
  -- c.printf("%d %d\n", density_idx, config.num_data_values)
  assert([int](density_idx) == config.num_data_values, "Wrong number of data values")
  c.fclose(file)

  if config.true_values_filename[0] ~= 0 then
    var file = c.fopen(config.true_values_filename, "r")
    var line : int8[512]
    fgets(line, 512, file)  -- Skip first line
    var i : int1d = 0
    var j_idx: int1d = 0
    while fgets(line, 512, file) do
      if c.strncmp(line, "\n", 1) ~= 0 and c.strncmp(line, "\r\n", 2) ~= 0 then
        var data : int[1]
        c.sscanf([&int8](line), "%d", data)
        var L : int = data[0]
        var H : int = (L + 1) * (L + 2) * (L + 3) / 6
        var values : &double = readline(line, H)
        for j = 0, H do
          r_true_j_values[j_idx].value = values[j + 5]
          j_idx = j_idx + 1
        end
        c.free(values)
        i = i + 1
      end
    end
    assert([int](j_idx) == config.num_data_values, "Wrong number of j values")
    c.fclose(file)
  end
end

task write_output(r_gausses  : region(ispace(int1d), HermiteGaussian),
                  r_j_values : region(ispace(int1d), Double),
                  config     : Config)
where
  reads(r_gausses, r_j_values)
do
  if config.output_filename[0] ~= 0 then
    c.printf("Writing output\n")
    var file = c.fopen(config.output_filename, "w")
    -- c.fprintf(file, "%d\n\n", config.num_gausses)
    for i in r_gausses.ispace do
      var bra = r_gausses[i]
      -- c.fprintf(file, "%d %.6f %.12f %.12f %.12f ", bra.L, bra.eta, bra.x, bra.y, bra.z)
      for j = [int](bra.data_rect.lo), [int](bra.data_rect.hi) + 1 do
        c.fprintf(file, "%.12f ", r_j_values[j].value)
      end
      c.fprintf(file, "\n")
    end
    c.fclose(file)
  end
end

task verify_output(r_gausses       : region(ispace(int1d), HermiteGaussian),
                   r_j_values      : region(ispace(int1d), Double),
                   r_true_j_values : region(ispace(int1d), Double),
                   config          : Config)
where
  reads(r_gausses, r_j_values, r_true_j_values)
do
  if config.true_values_filename[0] == 0 then
    return
  end
  c.printf("Verifying output\n")
  var max_error : double = 0.0
  var num_incorrect = 0
  for gauss_idx in r_gausses.ispace do
    var gaussian = r_gausses[gauss_idx]
    for i = [int](gaussian.data_rect.lo), [int](gaussian.data_rect.hi + 1) do
      var error : double = fabs(r_j_values[i].value - r_true_j_values[i].value)
      if error > 1e-10 then
        c.printf("Value differs at gaussian = %d, L = %d, i = %d: actual = %.12f, expected = %.12f\n",
                 gauss_idx, gaussian.L, i - [int](gaussian.data_rect.lo),
                 r_j_values[i].value, r_true_j_values[i].value)
        num_incorrect += 1
      end
      if error > max_error then
        max_error = error
      end
    end
  end
  c.printf("%d/%d incorrect values found\n", num_incorrect, r_j_values.ispace.volume)
  c.printf("Max error = %.12f\n", max_error)
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  -- Estimate the memory footprint
  var memory_footprint : int = (
      config.num_gausses * [ terralib.sizeof(HermiteGaussian) ]
        + 2 * config.num_data_values * [ terralib.sizeof(Double) ]
    ) / 1024 / 1024
  c.printf("**********************************************\n")
  c.printf("*      Two-Electron Repulsion Integrals      *\n")
  c.printf("*                                            *\n")
  c.printf("* Highest Angular Momentum : %15u *\n", config.highest_L)
  c.printf("* # Hermite Gaussians      : %15u *\n", config.num_gausses)
  c.printf("* # Data values            : %15u *\n", config.num_data_values)
  c.printf("* Memory footprint         : %12u Mb *\n", memory_footprint)
  c.printf("**********************************************\n")

  var r_gausses = region(ispace(int1d, config.num_gausses), HermiteGaussian)
  var r_density = region(ispace(int1d, config.num_data_values), Double)
  var r_j_values = region(ispace(int1d, config.num_data_values), Double)
  var r_true_j_values = region(ispace(int1d, config.num_data_values), Double)

  var r_boys = region(ispace(int1d, 121 * 11), Double)
  -- TODO: Use legion API to populate this region
  for index in r_boys.ispace do
    r_boys[index].value = getPrecomputedBoys(index)
  end

  c.printf("Reading input file\n")
  populateData(r_gausses, r_density, r_true_j_values, config)

  c.printf("Launching integrals\n")
  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()

  coulomb(r_gausses, r_density, r_j_values, r_boys, config.highest_L)

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  write_output(r_gausses, r_j_values, config)
  verify_output(r_gausses, r_j_values, r_true_j_values, config)
end

regentlib.start(toplevel)
