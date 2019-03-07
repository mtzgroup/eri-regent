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
                      -- where L12 <- angular momentum of bra
                      --       L34 <- angular momentum of ket
}

fspace PrecomputedBoys {
  data : double;
}

require("boys")
computeR0000 = generateTaskComputeR000(1)
computeR0001 = generateTaskComputeR000(2)
computeR0002 = generateTaskComputeR000(3)
computeR0003 = generateTaskComputeR000(4)
computeR0004 = generateTaskComputeR000(5)
-- Must import integrals after declaring fspaces and `computeR000*`
integralTypes = {
  "SSSS",
  "SSSP", "SSPP", "SPSS", "SPSP", "SPPP", "PPSS", "PPSP", "PPPP"
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
             parallelism   : int,
             block_type    : int2d)
where
  reads(r_bra_kets, r_bra_gausses, r_ket_gausses, r_density, r_boys),
  reduces +(r_j_values)
do
  if r_bra_kets.volume == 0 then return end

  var coloring = ispace(int1d, parallelism)
  var p_bra_kets = partition(equal, r_bra_kets, coloring)
  var p_bra_gausses = image(r_bra_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_ket_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_density = image(r_density, p_ket_gausses, r_ket_gausses.data_rect)

  var r_j_partials = region(r_j_values.ispace, double)
  fill(r_j_partials, 0.0)
  var p_j_partials = image(r_j_partials, p_bra_gausses, r_bra_gausses.data_rect)
  if block_type == [int2d]{0, 0} then
    __demand(__parallel)
    for color in coloring do
      coulombSSSS(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{0, 1} then
    __demand(__parallel)
    for color in coloring do
      coulombSSSP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{0, 2} then
    __demand(__parallel)
    for color in coloring do
      coulombSSPP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{1, 0} then
    __demand(__parallel)
    for color in coloring do
      coulombSPSS(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{1, 1} then
    __demand(__parallel)
    for color in coloring do
      coulombSPSP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{1, 2} then
    __demand(__parallel)
    for color in coloring do
      coulombSPPP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{2, 0} then
    __demand(__parallel)
    for color in coloring do
      coulombPPSS(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{2, 1} then
    __demand(__parallel)
    for color in coloring do
      coulombPPSP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  elseif block_type == [int2d]{2, 2} then
    __demand(__parallel)
    for color in coloring do
      coulombPPPP(p_bra_kets[color],
                  p_bra_gausses[color], p_ket_gausses[color],
                  p_density[color], p_j_partials[color], r_boys)
    end
  else
    c.printf("Block type = {%d, %d}\n", block_type.x, block_type.y)
    assert(false, "Block type not implemented")
  end

  for i in r_j_values.ispace do
    r_j_values[i] += r_j_partials[i]
  end
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
task populateData(r_bra_kets      : region(ispace(int1d), PrimitiveBraKet),
                  r_gausses       : region(ispace(int1d), HermiteGaussian),
                  r_density       : region(ispace(int1d), double),
                  r_j_values      : region(ispace(int1d), double),
                  r_true_j_values : region(ispace(int1d), double),
                  config          : Config)
where
  reads writes(r_bra_kets, r_gausses, r_density, r_j_values, r_true_j_values)
do
  fill(r_j_values, 0.0)

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
        r_density[density_idx] = values[j + 5]
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
          r_true_j_values[j_idx] = values[j + 5]
          j_idx = j_idx + 1
        end
        c.free(values)
        i = i + 1
      end
    end
    assert([int](j_idx) == config.num_data_values, "Wrong number of j values")
    c.fclose(file)
  end

  var bra_ket_idx : int = 0
  for bra_idx in r_gausses.ispace do
    for ket_idx in r_gausses.ispace do
      var block_type : int2d = {r_gausses[bra_idx].L, r_gausses[ket_idx].L}
      r_bra_kets[bra_ket_idx] = {bra_idx=bra_idx, ket_idx=ket_idx,
                                 block_type=block_type}
      bra_ket_idx += 1
    end
  end
  assert(bra_ket_idx == config.num_bra_kets, "Wrong number of BraKets")
end

task write_output(r_gausses  : region(ispace(int1d), HermiteGaussian),
                  r_j_values : region(ispace(int1d), double),
                  config     : Config)
where
  reads(r_gausses, r_j_values)
do
  if config.output_filename[0] ~= 0 then
    var file = c.fopen(config.output_filename, "w")
    -- c.fprintf(file, "%d\n\n", config.num_gausses)
    for i in r_gausses.ispace do
      var bra = r_gausses[i]
      -- c.fprintf(file, "%d %.6f %.12f %.12f %.12f ", bra.L, bra.eta, bra.x, bra.y, bra.z)
      for j = [int](bra.data_rect.lo), [int](bra.data_rect.hi) + 1 do
        c.fprintf(file, "%.12f ", r_j_values[j])
      end
      c.fprintf(file, "\n")
    end
    c.fclose(file)
  end
end

task verify_output(r_gausses       : region(ispace(int1d), HermiteGaussian),
                   r_j_values      : region(ispace(int1d), double),
                   r_true_j_values : region(ispace(int1d), double),
                   config          : Config)
where
  reads(r_gausses, r_j_values, r_true_j_values)
do
  if config.true_values_filename[0] == 0 then
    return
  end
  var max_error : double = 0.0
  for gauss_idx in r_gausses.ispace do
    var gaussian = r_gausses[gauss_idx]
    for i = [int](gaussian.data_rect.lo), [int](gaussian.data_rect.hi + 1) do
      var error : double = r_j_values[i] - r_true_j_values[i]
      if error > 1e-10 or error < -1e-10 then
        c.printf("Value differs at gaussian = %d, L = %d, i = %d: actual = %.12f, expected = %.12f\n",
                 gauss_idx, gaussian.L, i - [int](gaussian.data_rect.lo), r_j_values[i], r_true_j_values[i])
      end
      if error > max_error then
        max_error = error
      end
    end
  end
  c.printf("Max error = %.12f\n", max_error)
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
  var r_true_j_values = region(ispace(int1d, config.num_data_values), double)

  populateData(r_bra_kets, r_gausses, r_density, r_j_values, r_true_j_values, config)

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
            p_density[block_type], p_j_values[block_type], r_boys, 1, block_type)
  end

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  release(r_boys)
  detach(hdf5, r_boys.data)

  write_output(r_gausses, r_j_values, config)
  verify_output(r_gausses, r_j_values, r_true_j_values, config)
end

regentlib.start(toplevel)
