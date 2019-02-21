import "regent"

local Config = require("config")
local c = regentlib.c
local assert = regentlib.assert

fspace HermiteGaussian {
  x     : double;
  y     : double;
  z     : double;
  eta   : double;      -- Exponent of Gaussian
  L     : int;         -- Angular momentum
  d_start_idx : int1d; -- Preprocessed density matrix elements.
                       -- Number of values is given by
                       -- (L + 1) * (L + 2) * (L + 3) / 6
                       -- Can I use `legion_domain_t` here?
  bound : float;       -- TODO
}

fspace PrimitiveBraKet {
  bra_idx    : int1d;
  ket_idx    : int1d;
  block_type : int2d;
}

require("boys")
-- Import integrals after declaring fspaces and `computeR000`
for _, type in pairs({"SSSS", "SSSP"}) do
  require("integrals."..type)
end

task coulomb(r_gausses  : region(ispace(int1d), HermiteGaussian),
             r_density  : region(ispace(int1d), double),
             r_j_values : region(ispace(int1d), double),
             r_bra_kets : region(PrimitiveBraKet),
             block_type : int2d, parallelism : int)
where
  reads(r_gausses, r_density, r_bra_kets), reads writes(r_j_values)
do
  var coloring = ispace(int1d, parallelism)
  var p_bra_kets = partition(equal, r_bra_kets, coloring)
  var p_bra_gausses = image(r_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_gausses = p_bra_gausses | p_ket_gausses
  var p_j_values = image(r_j_values, p_bra_kets, r_bra_kets.bra_idx)
  -- TODO: partition r_density

  if block_type.x == 0 and block_type.y == 0 then
    -- FIXME: Cannot parallelize due to reduce in `j_values`
    -- __demand(__parallel)
    for color in coloring do
      coulombSSSS(p_gausses[color], r_density,
                  p_j_values[color], p_bra_kets[color])
    end
  elseif block_type.x == 0 and block_type.y == 1 then
    assert(false, "Block type not implemented")
    -- __demand(__parallel)
    for color in coloring do
      coulombSSSP(p_gausses[color], r_density,
                  p_j_values[color], p_bra_kets[color])
    end
  else
    assert(false, "Block type not implemented")
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

task populateData(r_gausses  : region(ispace(int1d), HermiteGaussian),
                  r_density  : region(ispace(int1d), double),
                  r_j_values : region(ispace(int1d), double),
                  r_bra_kets : region(PrimitiveBraKet),
                  file_name  : &int8)
where
  reads writes(r_gausses, r_density, r_j_values, r_bra_kets)
do
  fill(r_j_values, 0)

  var datai : int[1]
  var data : double[5]
  var density_str : int8[256]
  var file = c.fopen(file_name, "r")
  var line : int8[512]
  fgets(line, 512, file)  -- Skip first line
  var i : int1d = 0
  var density_idx : int1d = 0
  while fgets(line, 512, file) do
    if c.strncmp(line, "\n", 1) ~= 0 and c.strncmp(line, "\r\n", 2) ~= 0 then
      -- "L eta x y z [density values]"
      c.sscanf([&int8](line), "%d %lf %lf %lf %lf %256[0-9.eE- ]",
                              datai, data, data+1, data+2, data+3, density_str)
      r_gausses[i].x = data[1]
      r_gausses[i].y = data[2]
      r_gausses[i].z = data[3]
      r_gausses[i].eta = data[0]
      r_gausses[i].L = datai[0]
      r_gausses[i].d_start_idx = density_idx
      r_gausses[i].bound = 0  -- TODO
      var L = r_gausses[i].L
      var H : int = (L + 1) * (L + 2) * (L + 3) / 6
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

  var bra_ket_ispace = c.legion_physical_region_get_logical_region(
                              __physical(r_bra_kets)[0]).index_space
  var itr = c.legion_index_iterator_create(__runtime(), __context(), bra_ket_ispace)
  for bra_idx in r_gausses.ispace do
    for ket_idx in r_gausses.ispace do
      -- if bra_idx <= ket_idx then  -- FIXME: Do I need all bra_kets?
        var bra_ket_ptr = c.legion_index_iterator_next(itr)
        var block_type : int2d = {r_gausses[bra_idx].L, r_gausses[ket_idx].L}
        r_bra_kets[bra_ket_ptr] = {bra_idx, ket_idx, block_type}
      -- end
    end
  end
  c.legion_index_iterator_destroy(itr)
end

task write_output(r_j_values : region(ispace(int1d), double), filename : &int8)
where
  reads(r_j_values)
do
  if filename[0] ~= 0 then
    var file = c.fopen(filename, "w")
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
  c.printf("* # Hermite Gaussians      : %15u *\n", config.num_gausses)
  c.printf("* # BraKets                : %15u *\n", config.num_bra_kets)
  c.printf("* # Density Values         : %15u *\n", config.num_density_values)
  c.printf("* Highest Angular Momentum : %15u *\n", config.highest_L)
  c.printf("* # Parallel Tasks         : %15u *\n", config.parallelism)
  c.printf("**********************************************\n")

  var r_gausses = region(ispace(int1d, config.num_gausses), HermiteGaussian)
  var r_density_matrix = region(ispace(int1d, config.num_density_values), double)
  var r_j_values = region(ispace(int1d, config.num_gausses), double)
  var r_bra_kets = region(ispace(ptr, config.num_bra_kets), PrimitiveBraKet)

  populateData(r_gausses, r_density_matrix, r_j_values, r_bra_kets, config.input_filename)

  -- TODO: Need to decide how much parallelism to give to each block
  var block_coloring = ispace(int2d, {config.highest_L+1, config.highest_L+1})
  var p_bra_kets = partition(r_bra_kets.block_type, block_coloring)
  var p_bra_gausses = image(r_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_gausses = p_bra_gausses | p_ket_gausses
  -- FIXME: j data also has a variable length like `density_matrix`
  var p_j_values = image(r_j_values, p_bra_kets, r_bra_kets.bra_idx)
  -- var p_density_matrix = image(r_density_matrix, p_gausses, r_gausses.d_start_idx)
  -- FIXME: Need to manually color density matrix.

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()

  -- FIXME: Cannot parallelize due to reduce in `j_values`
  -- __demand(__parallel)
  for block_type in block_coloring do
    coulomb(p_gausses[block_type], r_density_matrix,
            p_j_values[block_type], p_bra_kets[block_type],
            block_type, 1)
  end

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  write_output(r_j_values, config.output_filename)
end

regentlib.start(toplevel)
