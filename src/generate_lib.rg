import "regent"

require "fields"
require "helper"
require "jfock"
require "kfock"

--------------------------------------------------------------------------------
-- Launch jfock tasks
--------------------------------------------------------------------------------
-- r_jbras[L1][L2] - Regions of JBras with angular momentum pair L1, L2
-- r_jkets[L1][L2] - Regions of JKets with angular momentum pair L1, L2
-- r_gamma_table   - A region holding a lookup table needed by the kernels
-- threshold       - A threshold to determine if a BraKet needs to contribute
--                   to output
-- parallelism     - How many ways should divide the work of each kernel
-- max_momentum    - A sanity check to make sure eri-regent was compiled with a
--                   large enough momentum
--------------------------------------------------------------------------------
task jfock_task(r_jbras00 : region(ispace(int1d), getJBra(0+0)),
                r_jbras01 : region(ispace(int1d), getJBra(0+1)),
                r_jbras11 : region(ispace(int1d), getJBra(1+1)),
                r_jbras02 : region(ispace(int1d), getJBra(0+2)),
                r_jbras12 : region(ispace(int1d), getJBra(1+2)),
                r_jbras22 : region(ispace(int1d), getJBra(2+2)),
                r_jbras03 : region(ispace(int1d), getJBra(0+3)),
                r_jbras13 : region(ispace(int1d), getJBra(1+3)),
                r_jbras23 : region(ispace(int1d), getJBra(2+3)),
                r_jbras33 : region(ispace(int1d), getJBra(3+3)),
                r_jbras04 : region(ispace(int1d), getJBra(0+4)),
                r_jbras14 : region(ispace(int1d), getJBra(1+4)),
                r_jbras24 : region(ispace(int1d), getJBra(2+4)),
                r_jbras34 : region(ispace(int1d), getJBra(3+4)),
                r_jbras44 : region(ispace(int1d), getJBra(4+4)),
                r_jkets00 : region(ispace(int1d), getJKet(0+0)),
                r_jkets01 : region(ispace(int1d), getJKet(0+1)),
                r_jkets11 : region(ispace(int1d), getJKet(1+1)),
                r_jkets02 : region(ispace(int1d), getJKet(0+2)),
                r_jkets12 : region(ispace(int1d), getJKet(1+2)),
                r_jkets22 : region(ispace(int1d), getJKet(2+2)),
                r_jkets03 : region(ispace(int1d), getJKet(0+3)),
                r_jkets13 : region(ispace(int1d), getJKet(1+3)),
                r_jkets23 : region(ispace(int1d), getJKet(2+3)),
                r_jkets33 : region(ispace(int1d), getJKet(3+3)),
                r_jkets04 : region(ispace(int1d), getJKet(0+4)),
                r_jkets14 : region(ispace(int1d), getJKet(1+4)),
                r_jkets24 : region(ispace(int1d), getJKet(2+4)),
                r_jkets34 : region(ispace(int1d), getJKet(3+4)),
                r_jkets44 : region(ispace(int1d), getJKet(4+4)),
                r_gamma_table : region(ispace(int2d, {18, 700}), double[5]),
                threshold : float, parallelism : int, largest_momentum : int)
where
  reads writes(
    r_jbras00,
    r_jbras01,
    r_jbras11,
    r_jbras02,
    r_jbras12,
    r_jbras22,
    r_jbras03,
    r_jbras13,
    r_jbras23,
    r_jbras33,
    r_jbras04,
    r_jbras14,
    r_jbras24,
    r_jbras34,
    r_jbras44
  ),
  reads (
    r_gamma_table,
    r_jkets00,
    r_jkets01,
    r_jkets11,
    r_jkets02,
    r_jkets12,
    r_jkets22,
    r_jkets03,
    r_jkets13,
    r_jkets23,
    r_jkets33,
    r_jkets04,
    r_jkets14,
    r_jkets24,
    r_jkets34,
    r_jkets44
  )
do
  regentlib.assert(largest_momentum <= [getCompiledMaxMomentum()] + 1,
      "Please recompile eri-regent with a larger max momentum!");
  [jfock(
    {
      [0]={[0]=r_jbras00, [1]=r_jbras01, [2]=r_jbras02, [3]=r_jbras03,
           [4]=r_jbras04},
      [1]={[1]=r_jbras11, [2]=r_jbras12, [3]=r_jbras13, [4]=r_jbras14},
      [2]={[2]=r_jbras22, [3]=r_jbras23, [4]=r_jbras24},
      [3]={[3]=r_jbras33, [4]=r_jbras34},
      [4]={[4]=r_jbras44},
    },
    {
      [0]={[0]=r_jkets00, [1]=r_jkets01, [2]=r_jkets02, [3]=r_jkets03,
           [4]=r_jkets04},
      [1]={[1]=r_jkets11, [2]=r_jkets12, [3]=r_jkets13, [4]=r_jkets14},
      [2]={[2]=r_jkets22, [3]=r_jkets23, [4]=r_jkets24},
      [3]={[3]=r_jkets33, [4]=r_jkets34},
      [4]={[4]=r_jkets44},
    },
    r_gamma_table, threshold, parallelism)]
end

--------------------------------------------------------------------------------
-- Launch kfock tasks
--------------------------------------------------------------------------------
-- r_pairs[L1][L2]   - Regions of KFock pairs with angular momentum pair L1, L2
-- r_density[L2][L4] - Regions of density values with angular momentum pair L24
-- r_output[L1][L3]  - Regions of output values with angular momentum pair L13
-- r_gamma_table     - A region holding a lookup table needed by the kernels
-- threshold         - A threshold to determine if a KFock pair needs to
--                     contribute to output
-- parallelism       - How many ways should divide the work of each kernel
-- max_momentum      - A sanity check to make sure eri-regent was compiled with
--                     a large enough momentum
--------------------------------------------------------------------------------
task kfock_task(r_pairs00     : region(ispace(int1d), getKFockPair(0, 0)),
                -- r_pairs01     : region(ispace(int1d), getKFockPair(1, 0)),
                -- r_pairs10     : region(ispace(int1d), getKFockPair(0, 1)),
                -- r_pairs11     : region(ispace(int1d), getKFockPair(1, 1)),
                r_density00   : region(ispace(int2d), getKFockDensity(0, 0)),
                -- r_density01   : region(ispace(int2d), getKFockDensity(0, 1)),
                -- r_density11   : region(ispace(int2d), getKFockDensity(1, 1)),
                r_output00    : region(ispace(int3d), getKFockOutput(0, 0)),
                -- r_output01    : region(ispace(int2d), getKFockOutput(0, 1)),
                -- r_output11    : region(ispace(int2d), getKFockOutput(1, 1)),
                r_gamma_table : region(ispace(int2d, {18, 700}), double[5]),
                threshold : float, parallelism : int, largest_momentum : int)
where
  reads (
    r_pairs00,
    -- r_pairs01,
    -- r_pairs10,
    -- r_pairs11,
    r_density00,
    -- r_density01,
    -- r_density11,
    r_gamma_table
  ),
  reads writes(
    r_output00
    -- r_output01,
    -- r_output11
  )
do
  regentlib.assert(largest_momentum <= [getCompiledMaxMomentum()],
                   "Please recompile eri-regent with a larger max momentum!");
  [kfock(
    {
      [0]={[0]=r_pairs00}
    },
    {
      [0]={[0]=r_density00}
    },
    {
      [0]={[0]=r_output00}
    },
    r_gamma_table, threshold, parallelism)]
end

local header, lib = nil, nil
for i, arg_value in ipairs(arg) do
  if arg[i] == "--lib" then
    lib = arg[i+1]
  elseif arg[i] == "--header" then
    header = arg[i+1]
  end
end
assert(header ~= nil and lib ~= nil,
       "Must give library and header path `--lib [path] --header [path]`")

regentlib.save_tasks(header, lib)
print("Generated header at "..header.." and library at "..lib)
