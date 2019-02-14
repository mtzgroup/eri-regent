import "regent"

local Config = require("config")
local c = regentlib.c
local assert = regentlib.assert
local cmath = terralib.includec("math.h")
local M_PI = cmath.M_PI
local sqrt = cmath.sqrt
local pow = cmath.pow
local floor = cmath.floor
local exp = cmath.exp
-- FIXME: Hack to allow docker to see header file
local getPrecomputedBoys = terralib.includec("precomputedBoys.h", {"-I", "/coulomb"}).getPrecomputedBoys

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
  bound : float;
}

fspace PrimitiveBraKet {
  bra_idx : int1d;
  ket_idx : int1d;
}

terra computeR000(t : double, alpha : double, R000 : &double, length : int)
  -- assert(t > 0, "FIXME: Not sure if I need to handle this case.")

  if 0 <= t and t < 12 then
    var t_est : double = floor(10.0 * t + 0.5) / 10.0
    R000[length-1] = 0
    var factor = 1
    for k = 0, 7 do
      var f : double = getPrecomputedBoys(t_est, length-1+k) * factor
      R000[length-1] = R000[length-1] + f
      factor = factor * (t_est - t) / (k + 1)
    end
    for j = length-2, -1, -1 do
      R000[j] = (2 * t * R000[j+1] + exp(-1.0 * t)) / (2.0 * j + 1)
    end
  else
    assert(length <= 16, "Only accurate for j <= 16")
    R000[0] = sqrt(M_PI) / (2.0 * sqrt(t))
    var g : double
    if t < 15 then
      g = 0.4999489092 - 0.2473631686 / t + 0.321180909 / (t * t)
            - 0.3811559346 / (t * t * t)
    elseif t < 18 then
      g = 0.4998436875 - 0.24249438 / t + 0.24642845 / (t * t);
    elseif t < 24 then
      g = 0.499093162 - 0.2152832 / t
    elseif t < 30 then
      g = 0.490
    end

    if t < 30 then
      R000[0] = R000[0] - exp(-1.0 * t) * g / t
    end

    for j = 1, length do
      R000[j] = 2.0 / t * ((2 * j + 1) * R000[j-1] - exp(-1.0 * t))
    end
  end

  var factor : double = 1
  for j = 0, length do
    R000[j] = R000[j] * factor
    factor = factor * -2 * alpha
  end
end

__demand(__cuda)
task coulombSSSS(r_gausses  : region(ispace(int1d), HermiteGaussian),
                 r_density  : region(ispace(int1d), double),
                 r_j_values : region(ispace(int1d), double),
                 r_bra_kets : region(PrimitiveBraKet))
where
  reads(r_gausses, r_density, r_bra_kets), reduces +(r_j_values)
do
  var R000 : double[1]
  for bra_ket in r_bra_kets do
    var bra = r_gausses[bra_ket.bra_idx]
    var ket = r_gausses[bra_ket.ket_idx]
    -- TODO: Use Gaussian.bound to filter useless loops
    var a : double = bra.x - ket.x
    var b : double = bra.y - ket.y
    var c : double = bra.z - ket.z

    var alpha : double = bra.eta * ket.eta / (bra.eta + ket.eta)
    var t : double = alpha * (a*a+b*b+c*c)
    computeR000(t, alpha, R000, 1)

    r_j_values[bra_ket.bra_idx] += R000[0] * r_density[ket.d_start_idx]
  end
end

__demand(__cuda)
task coulombSSSP(r_gausses  : region(ispace(int1d), HermiteGaussian),
                 r_density  : region(ispace(int1d), double),
                 r_j_values : region(ispace(int1d), double),
                 r_bra_kets : region(PrimitiveBraKet))
where
  reads(r_gausses, r_density, r_bra_kets), reduces +(r_j_values)
do
  var R000 : double[4]
  for bra_ket in r_bra_kets do
    var bra = r_gausses[bra_ket.bra_idx]
    var ket = r_gausses[bra_ket.ket_idx]
    -- TODO: Use Gaussian.bound to filter useless loops
    var a : double = bra.x - ket.x
    var b : double = bra.y - ket.y
    var c : double = bra.z - ket.z

    var alpha : double = bra.eta * ket.eta / (bra.eta + ket.eta)
    var t : double = alpha * (a*a+b*b+c*c)
    computeR000(t, alpha, R000, 4)

    var R1000 = a * R000[1]
    var R0100 = b * R000[1]
    var R0010 = c * R000[1]

    var P0 = r_density[ket.d_start_idx]
    var P1 = r_density[ket.d_start_idx + 1]
    var P2 = r_density[ket.d_start_idx + 2]
    var P3 = r_density[ket.d_start_idx + 3]

    r_j_values[bra_ket.bra_idx] += P0 * R000[0] - P1 * R1000
                                                - P2 * R0100
                                                - P3 * R0010
  end
end

task coulomb(r_gausses  : region(ispace(int1d), HermiteGaussian),
             r_density  : region(ispace(int1d), double),
             r_j_values : region(ispace(int1d), double),
             r_bra_kets : region(PrimitiveBraKet),
             block_type : int, parallelism : int)
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

  if block_type == 0 then
    -- FIXME: Cannot parallelize due to reduce in `r_density_matrix`
    -- __demand(__parallel)
    for color in coloring do
      coulombSSSS(p_gausses[color], r_density,
                  p_j_values[color], p_bra_kets[color])
    end
  elseif block_type == 1 then
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
      i = i + 1
      var L = r_gausses[i].L
      var H : int = (L + 1) * (L + 2) * (L + 3) / 6
      var values : &double = sgetnd(density_str, H)
      for j = 0, H do
        r_density[density_idx] = values[j]
        density_idx = density_idx + 1
      end
      c.free(values)
    end
  end
  c.fclose(file)

  var block_coloring : legion_coloring_t = c.legion_coloring_create()
  var bra_ket_ispace = c.legion_physical_region_get_logical_region(
                              __physical(r_bra_kets)[0]).index_space
  var itr = c.legion_index_iterator_create(__runtime(), __context(),
                                           bra_ket_ispace)
  for bra_idx in r_gausses.ispace do
    for ket_idx in r_gausses.ispace do
      if bra_idx <= ket_idx then
        -- Create a PrimitiveBraKet and give it a block color
        var bra_ket_ptr = c.legion_index_iterator_next(itr)
        var block = r_gausses[bra_idx].L + r_gausses[ket_idx].L
        c.legion_coloring_add_point(block_coloring, block, bra_ket_ptr)
        r_bra_kets[bra_ket_ptr] = {bra_idx, ket_idx}
      end
    end
  end
  c.legion_index_iterator_destroy(itr)
  return block_coloring
end

task write_output(r_j_values : region(ispace(int1d), double), filename : &int8)
where
  reads(r_j_values)
do
  if filename[0] ~= 0 then
    var file = c.fopen(filename, "w")
    for i in r_j_values.ispace do
      c.fprintf(file, "%lf\n", r_j_values[i])
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
  c.printf("* # Blocks                 : %15u *\n", config.num_blocks)
  c.printf("* # Parallel Tasks         : %15u *\n", config.parallelism)
  c.printf("**********************************************\n")

  var r_gausses = region(ispace(int1d, config.num_gausses), HermiteGaussian)
  var r_density_matrix = region(ispace(int1d, config.num_density_values), double)
  var r_j_values = region(ispace(int1d, config.num_gausses), double)
  var r_bra_kets = region(ispace(ptr, config.num_bra_kets), PrimitiveBraKet)

  var block_coloring = populateData(r_gausses, r_density_matrix, r_j_values,
                                    r_bra_kets, config.input_filename)

  -- TODO: Need to decide how much parallelism to give to each block
  var p_bra_kets = partition(disjoint, r_bra_kets, block_coloring)
  var p_bra_gausses = image(r_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_gausses = p_bra_gausses | p_ket_gausses
  var p_j_values = image(r_j_values, p_bra_kets, r_bra_kets.bra_idx)
  -- var p_density_matrix = image(r_density_matrix, p_gausses, r_gausses.d_start_idx)
  -- FIXME: Need to manually color density matrix.

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_start = c.legion_get_current_time_in_micros()

  -- FIXME: Cannot parallelize due to reduce in `r_density_matrix`
  -- __demand(__parallel)
  for block = 0, config.num_blocks do
    coulomb(p_gausses[block], r_density_matrix,
            p_j_values[block], p_bra_kets[block],
            block, 2)
  end

  __fence(__execution, __block) -- Make sure we only time the computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Coulomb operator: %.4f sec\n", (ts_stop - ts_start) * 1e-6)

  c.legion_coloring_destroy(block_coloring)

  write_output(r_j_values, config.output_filename)
end

regentlib.start(toplevel)


-- TODO: Better naming for symbols
-- local aSymbol = symbol(double)
-- local bSymbol = symbol(double)
-- local cSymbol = symbol(double)
-- local R000Symbol = symbol(&double)
-- -- TODO: Maybe this could be memoized since it does not directly depend on
-- --       a, b, c, or j.
-- function genRNLMJExpr(N, L, M)
--   local J = N + L + M
--   local function aux(N, L, M)
--     if N < 0 or L < 0 or M < 0 then
--       return `0
--     elseif N == 0 and L == 0 and M == 0 then
--       return `R000Symbol[J]
--     elseif N == 0 and L == 0 then
--       local first, second = aux(0, 0, M - 1), aux(0, 0, M - 2)
--       return `aSymbol * first + (M - 1) * second
--     elseif N == 0 then
--       local first, second = aux(0, L - 1, M), aux(0, L - 2, M)
--       return `bSymbol * first + (L - 1) * second
--     else
--       local first, second = aux(N - 1, L, M), aux(N - 2, L, M)
--       return `cSymbol * first + (N - 1) * second
--     end
--   end
--   return aux(N, L, M)
-- end
-- terra computeR(a : double, b : double, c : double, N : int, L : int, M : int)
--   var [aSymbol] = a
--   var [bSymbol] = b
--   var [cSymbol] = c
--   var [R000Symbol] = computeR000(0, 3) -- TODO: Actually compute this
--   return [ genRNLMJExpr(1, 2, 1) ]
-- end
