import "regent"
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
             parallelism : int, L12 : int, L34 : int)
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
  var block_type : int2d = {L12, L34}
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

task coulomb_start(r_bra_kets : region(ispace(int1d), PrimitiveBraKet),
                   r_gausses  : region(ispace(int1d), HermiteGaussian),
                   r_density  : region(ispace(int1d), double),
                   r_j_values : region(ispace(int1d), double),
                   r_boys     : region(ispace(int2d), PrecomputedBoys),
                   highest_L  : int)
where
  reads(r_bra_kets, r_gausses, r_density, r_boys),
  reads writes(r_j_values)
do
  -- TODO: Need to decide how much parallelism to give to each block
  var block_coloring = ispace(int2d, {highest_L+1, highest_L+1})
  var p_bra_kets = partition(r_bra_kets.block_type, block_coloring)
  var p_bra_gausses = image(r_gausses, p_bra_kets, r_bra_kets.bra_idx)
  var p_ket_gausses = image(r_gausses, p_bra_kets, r_bra_kets.ket_idx)
  var p_density = image(r_density, p_ket_gausses, r_gausses.data_rect)
  var p_j_values = image(r_j_values, p_bra_gausses, r_gausses.data_rect)

  -- TODO: Why are these tasks dependent?
  -- __demand(__parallel)
  for block_type in block_coloring do
    coulomb(p_bra_kets[block_type],
            p_bra_gausses[block_type], p_ket_gausses[block_type],
            p_density[block_type], p_j_values[block_type], r_boys,
            1, block_type.x, block_type.y)
  end
end

-- TODO: Is there a way to enable this only when it is directly called by regent?
-- local root_dir = arg[0]:match(".*/") or "./"
-- regentlib.save_tasks(root_dir.."embed_tasks.h", root_dir.."libembed_tasks.so")
