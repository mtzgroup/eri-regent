import "regent"

local Config = require("config")
local c = regentlib.c
local assert = regentlib.assert

fspace HermiteGaussian {
  {x, y, z} : double;  -- Location of Gaussian
  eta       : double;  -- Exponent of Gaussian
  L         : int;     -- Angular momentum
  data_rect : rect1d;  -- Gives a range of indices where the number of values
                       -- is given by (L + 1) * (L + 2) * (L + 3) / 6
                       -- If `HermiteGaussian` is interpreted as a "bra", then
                       --`data_rect` refers to the J values. Otherwise, it
                       -- refers to the density matrix values.
  bound : float;       -- TODO
}

fspace PrimitiveBraKet {
  bra_idx    : int1d;
  ket_idx    : int1d;
  block_type : int2d; -- Gives the type of integral to compute by {L12, L34}
                      -- where L12 <= angular momentum of bra
                      --       L34 <= angular momentum of ket
}

fspace PrecomputedBoys {
  data : double;
}

require("boys")
-- Import integrals after declaring fspaces and `computeR000`
integralTypes = {
  "SSSS",
  "SSSP", -- "SSPP", "SPSP", "SPPP", "PPPP",
}
for _, type in pairs(integralTypes) do
  require("integrals."..type)
end

-- Takes a group of BraKets that all have the same angular momentum pair.
-- Computes the necessary integrals and adds the values into `r_j_values`.
task coulomb(r_bra_kets    : region(ispace(int1d), PrimitiveBraKet),
             r_bra_gausses : region(ispace(int1d), HermiteGaussian),
             r_ket_gausses : region(ispace(int1d), HermiteGaussian),
             r_density     : region(ispace(int1d), double),
             r_j_values    : region(ispace(int1d), double),
             r_boys        : region(ispace(int2d), PrecomputedBoys),
             parallelism   : int)
where
  reads(r_bra_kets, r_bra_gausses, r_ket_gausses, r_density, r_boys),
  reduces +(r_j_values)
do
  var coloring = ispace(int1d, parallelism)
  var p_bra_kets = partition(equal, r_bra_kets, coloring)
  var p_bra_gausses = image(r_bra_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_ket_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_density = image(r_density, p_ket_gausses, r_ket_gausses.data_rect)

  var r_j_partials = region(r_j_values.ispace, double)
  fill(r_j_partials, 0)
  var p_j_partials = image(r_j_partials, p_bra_gausses, r_bra_gausses.data_rect)

  var block_type = r_bra_kets[0].block_type
  if block_type.x == 0 and block_type.y == 0 then
    __demand(__parallel)
    for color in coloring do
      coulombSSSS(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type.x == 0 and block_type.y == 1 then
    __demand(__parallel)
    for color in coloring do
      coulombSSSP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  else
    assert(false, "Block type not implemented")
  end

  for i in r_j_values.ispace do
    r_j_values[i] += r_j_partials[i]
  end
end

terra fgets(line : &int8, n : int, file : &c.FILE) : bool
  return c.fgets(line, n, file) ~= nil
end

-- Reads n doubles from `str` separated by spaces and puts them into `values`
terra sgetnd(str : &int8, n : int)
  var values : &double = [&double](c.malloc(sizeof(double) * n))
  var token : &int8 = c.strtok(str, " ")
  var i : int = 0
  while token ~= nil do
    assert(i < n, "Too many values found!\n")
    values[i] = c.atof(token)
    token = c.strtok(nil, " ")
    i = i + 1
  end
  return values
end

task populateData(r_bra_kets : region(ispace(int1d), PrimitiveBraKet),
                  r_gausses  : region(ispace(int1d), HermiteGaussian),
                  r_density  : region(ispace(int1d), double),
                  r_j_values : region(ispace(int1d), double),
                  config     : Config)
where
  reads writes(r_bra_kets, r_gausses, r_density, r_j_values)
do
  fill(r_j_values, 0)

  var datai : int[1]
  var data : double[5]
  var density_str : int8[256]
  var file = c.fopen(config.input_filename, "r")
  var line : int8[512]
  fgets(line, 512, file)  -- Skip first line
  var i : int1d = 0
  var density_idx : int1d = 0
  while fgets(line, 512, file) do
    if c.strncmp(line, "\n", 1) ~= 0 and c.strncmp(line, "\r\n", 2) ~= 0 then
      -- "L eta x y z [density values]"
      c.sscanf([&int8](line), "%d %lf %lf %lf %lf %256[0-9.eE- ]",
                              datai, data, data+1, data+2, data+3, density_str)
      var L : int = datai[0]
      var eta : double = data[0]
      var x : double = data[1]
      var y : double = data[2]
      var z : double = data[3]
      var H : int = (L + 1) * (L + 2) * (L + 3) / 6
      r_gausses[i] = {x=x, y=y, z=z, eta=eta, L=L,
                      data_rect={density_idx, density_idx+H-1}, bound=0}
      var values : &double = sgetnd(density_str, H)
      for j = 0, H do
        r_density[density_idx] = values[j]
        density_idx = density_idx + 1
      end
      c.free(values)
      i = i + 1
    end
  end
  c.fclose(file)

  var bra_ket_idx : int = 0
  for bra_idx in r_gausses.ispace do
    for ket_idx in r_gausses.ispace do
      -- if bra_idx <= ket_idx then  -- FIXME: Do I need all bra_kets?
        var block_type : int2d = {r_gausses[bra_idx].L, r_gausses[ket_idx].L}
        r_bra_kets[bra_ket_idx] = {bra_idx=bra_idx, ket_idx=ket_idx,
                                   block_type=block_type}
      -- end
    end
  end
end

task write_output(r_j_values : region(ispace(int1d), double), config : Config)
where
  reads(r_j_values)
do
  if config.output_filename[0] ~= 0 then
    var file = c.fopen(config.output_filename, "w")
    for i in r_j_values.ispace do
      c.fprintf(file, "%.12f\n", r_j_values[i])
    end
    c.fclose(file)
  end
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  c.printf("**********************************************\n")
  c.printf("*      Two-Electron Repulsion Integrals      *\n")
  c.printf("*                                            *\n")
  c.printf("* Highest Angular Momentum : %15u *\n", config.highest_L)
  c.printf("* # Hermite Gaussians      : %15u *\n", config.num_gausses)
  c.printf("* # BraKets                : %15u *\n", config.num_bra_kets)
  c.printf("* # Data values            : %15u *\n", config.num_data_values)
  c.printf("* # Parallel Tasks         : %15u *\n", config.parallelism)
  c.printf("**********************************************\n")

  var r_bra_kets = region(ispace(int1d, config.num_bra_kets), PrimitiveBraKet)
  var r_gausses = region(ispace(int1d, config.num_gausses), HermiteGaussian)
  var r_density = region(ispace(int1d, config.num_data_values), double)
  var r_j_values = region(ispace(int1d, config.num_data_values), double)

  populateData(r_bra_kets, r_gausses, r_density, r_j_values, config)

  -- TODO: Need to decide how much parallelism to give to each block
  var block_coloring = ispace(int2d, {config.highest_L+1, config.highest_L+1})
  var p_bra_kets = partition(r_bra_kets.block_type, block_coloring)
  var p_bra_gausses = image(r_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_density = image(r_density, p_ket_gausses, r_gausses.data_rect)
  var p_j_values = image(r_j_values, p_bra_gausses, r_gausses.data_rect)

  var r_boys = region(ispace(int2d, {121, 23}), PrecomputedBoys)
  attach(hdf5, r_boys.data, "precomputedBoys.hdf5", regentlib.file_read_only)
  acquire(r_boys)

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()

  __demand(__parallel)
  for block_type in block_coloring do
    coulomb(p_bra_kets[block_type],
            p_bra_gausses[block_type], p_ket_gausses[block_type],
            p_density[block_type], p_j_values[block_type], r_boys, 1)
  end

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  release(r_boys)
  detach(hdf5, r_boys.data)

  write_output(r_j_values, config)
end

regentlib.start(toplevel)
