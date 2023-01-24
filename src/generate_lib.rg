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
__forbid(__inner)
task jfock_task(r_jbras00 : region(ispace(int1d), getJBra(0+0)),
                r_jbras01 : region(ispace(int1d), getJBra(0+1)),
                r_jbras02 : region(ispace(int1d), getJBra(0+2)),
                r_jbras03 : region(ispace(int1d), getJBra(0+3)),
                r_jbras04 : region(ispace(int1d), getJBra(0+4)),
                r_jbras11 : region(ispace(int1d), getJBra(1+1)),
                r_jbras12 : region(ispace(int1d), getJBra(1+2)),
                r_jbras13 : region(ispace(int1d), getJBra(1+3)),
                r_jbras14 : region(ispace(int1d), getJBra(1+4)),
                r_jbras22 : region(ispace(int1d), getJBra(2+2)),
                r_jbras23 : region(ispace(int1d), getJBra(2+3)),
                r_jbras24 : region(ispace(int1d), getJBra(2+4)),
                r_jbras33 : region(ispace(int1d), getJBra(3+3)),
                r_jbras34 : region(ispace(int1d), getJBra(3+4)),
                r_jbras44 : region(ispace(int1d), getJBra(4+4)),
                r_jkets00 : region(ispace(int1d), getJKet(0+0)),
                r_jkets01 : region(ispace(int1d), getJKet(0+1)),
                r_jkets02 : region(ispace(int1d), getJKet(0+2)),
                r_jkets03 : region(ispace(int1d), getJKet(0+3)),
                r_jkets04 : region(ispace(int1d), getJKet(0+4)),
                r_jkets11 : region(ispace(int1d), getJKet(1+1)),
                r_jkets12 : region(ispace(int1d), getJKet(1+2)),
                r_jkets13 : region(ispace(int1d), getJKet(1+3)),
                r_jkets14 : region(ispace(int1d), getJKet(1+4)),
                r_jkets22 : region(ispace(int1d), getJKet(2+2)),
                r_jkets23 : region(ispace(int1d), getJKet(2+3)),
                r_jkets24 : region(ispace(int1d), getJKet(2+4)),
                r_jkets33 : region(ispace(int1d), getJKet(3+3)),
                r_jkets34 : region(ispace(int1d), getJKet(3+4)),
                r_jkets44 : region(ispace(int1d), getJKet(4+4)),
                r_gamma_table : region(ispace(int2d, {18, 700}), double[5]),
                threshold : float, parallelism : int, cparallelism: int, largest_momentum : int)
where
  reads writes(
    r_jbras00, r_jbras01, r_jbras02, r_jbras03, r_jbras04,
    r_jbras11, r_jbras12, r_jbras13, r_jbras14,
    r_jbras22, r_jbras23, r_jbras24,
    r_jbras33, r_jbras34,
    r_jbras44
  ),
  reads (
    r_gamma_table,
    r_jkets00, r_jkets01, r_jkets02, r_jkets03, r_jkets04,
    r_jkets11, r_jkets12, r_jkets13, r_jkets14,
    r_jkets22, r_jkets23, r_jkets24,
    r_jkets33, r_jkets34,
    r_jkets44
  )
do
  regentlib.assert(largest_momentum <= [getCompiledMaxMomentum()] + 1,
      "Please recompile eri-regent with a larger max momentum!");
  [jfock(
    {
      [0]={[0]=r_jbras00, [1]=r_jbras01, [2]=r_jbras02, [3]=r_jbras03, [4]=r_jbras04},
      [1]={[1]=r_jbras11, [2]=r_jbras12, [3]=r_jbras13, [4]=r_jbras14},
      [2]={[2]=r_jbras22, [3]=r_jbras23, [4]=r_jbras24},
      [3]={[3]=r_jbras33, [4]=r_jbras34},
      [4]={[4]=r_jbras44},
    },
    {
      [0]={[0]=r_jkets00, [1]=r_jkets01, [2]=r_jkets02, [3]=r_jkets03, [4]=r_jkets04},
      [1]={[1]=r_jkets11, [2]=r_jkets12, [3]=r_jkets13, [4]=r_jkets14},
      [2]={[2]=r_jkets22, [3]=r_jkets23, [4]=r_jkets24},
      [3]={[3]=r_jkets33, [4]=r_jkets34},
      [4]={[4]=r_jkets44},
    },
    r_gamma_table, threshold, parallelism, cparallelism)]
end

--------------------------------------------------------------------------------
-- Launch kfock tasks with varying parallelism for S/P Mcmurchie task launches
--------------------------------------------------------------------------------
-- r_pairs[L1][L2]   - Regions of KFock pairs with angular momentum pair L1, L2
-- r_bra_prevals     - Regions of prevals (Hermite->Cartesian conversions) for bra
-- r_ket_prevals     - Regions of prevals (Cartesian->Hermite conversions) for ket
-- r_labels[L1][L2]  - Regions of KFock labels with angular momentum pair L1, L2
-- r_density[L2][L4] - Regions of density values with angular momentum pair L24
-- r_output[L1][L3]  - Regions of output values with angular momentum pair L13
-- r_gamma_table     - A region holding a lookup table needed by the kernels
-- threshold         - A threshold to determine if a KFock pair needs to
--                     contribute to output
-- kguard            - A density-based threshold
-- parallelism       - How many ways should divide the work of each kernel
-- max_momentum      - A sanity check to make sure eri-regent was compiled with
--                     a large enough momentum
--------------------------------------------------------------------------------
__forbid(__inner)
task kfock_task(r_pairs00     : region(ispace(int1d), getKFockPair(0, 0)),
                r_pairs01     : region(ispace(int1d), getKFockPair(0, 1)),
                r_pairs02     : region(ispace(int1d), getKFockPair(0, 2)),
                r_pairs03     : region(ispace(int1d), getKFockPair(0, 3)),
                r_pairs04     : region(ispace(int1d), getKFockPair(0, 4)),
                r_pairs10     : region(ispace(int1d), getKFockPair(1, 0)),
                r_pairs11     : region(ispace(int1d), getKFockPair(1, 1)),
                r_pairs12     : region(ispace(int1d), getKFockPair(1, 2)),
                r_pairs13     : region(ispace(int1d), getKFockPair(1, 3)),
                r_pairs14     : region(ispace(int1d), getKFockPair(1, 4)),
                r_pairs20     : region(ispace(int1d), getKFockPair(2, 0)),
                r_pairs21     : region(ispace(int1d), getKFockPair(2, 1)),
                r_pairs22     : region(ispace(int1d), getKFockPair(2, 2)),
                r_pairs23     : region(ispace(int1d), getKFockPair(2, 3)),
                r_pairs24     : region(ispace(int1d), getKFockPair(2, 4)),
                r_pairs30     : region(ispace(int1d), getKFockPair(3, 0)),
                r_pairs31     : region(ispace(int1d), getKFockPair(3, 1)),
                r_pairs32     : region(ispace(int1d), getKFockPair(3, 2)),
                r_pairs33     : region(ispace(int1d), getKFockPair(3, 3)),
                r_pairs34     : region(ispace(int1d), getKFockPair(3, 4)),
                r_pairs40     : region(ispace(int1d), getKFockPair(4, 0)),
                r_pairs41     : region(ispace(int1d), getKFockPair(4, 1)),
                r_pairs42     : region(ispace(int1d), getKFockPair(4, 2)),
                r_pairs43     : region(ispace(int1d), getKFockPair(4, 3)),
                r_pairs44     : region(ispace(int1d), getKFockPair(4, 4)),
                r_bra_prevals00 : region(ispace(int2d), double),
                r_bra_prevals01 : region(ispace(int2d), double),
                r_bra_prevals02 : region(ispace(int2d), double),
                r_bra_prevals03 : region(ispace(int2d), double),
                r_bra_prevals04 : region(ispace(int2d), double),
                r_bra_prevals10 : region(ispace(int2d), double),
                r_bra_prevals11 : region(ispace(int2d), double),
                r_bra_prevals12 : region(ispace(int2d), double),
                r_bra_prevals13 : region(ispace(int2d), double),
                r_bra_prevals14 : region(ispace(int2d), double),
                r_bra_prevals20 : region(ispace(int2d), double),
                r_bra_prevals21 : region(ispace(int2d), double),
                r_bra_prevals22 : region(ispace(int2d), double),
                r_bra_prevals23 : region(ispace(int2d), double),
                r_bra_prevals24 : region(ispace(int2d), double),
                r_bra_prevals30 : region(ispace(int2d), double),
                r_bra_prevals31 : region(ispace(int2d), double),
                r_bra_prevals32 : region(ispace(int2d), double),
                r_bra_prevals33 : region(ispace(int2d), double),
                r_bra_prevals34 : region(ispace(int2d), double),
                r_bra_prevals40 : region(ispace(int2d), double),
                r_bra_prevals41 : region(ispace(int2d), double),
                r_bra_prevals42 : region(ispace(int2d), double),
                r_bra_prevals43 : region(ispace(int2d), double),
                r_bra_prevals44 : region(ispace(int2d), double),
                r_ket_prevals00 : region(ispace(int2d), double),
                r_ket_prevals01 : region(ispace(int2d), double),
                r_ket_prevals02 : region(ispace(int2d), double),
                r_ket_prevals03 : region(ispace(int2d), double),
                r_ket_prevals04 : region(ispace(int2d), double),
                r_ket_prevals10 : region(ispace(int2d), double),
                r_ket_prevals11 : region(ispace(int2d), double),
                r_ket_prevals12 : region(ispace(int2d), double),
                r_ket_prevals13 : region(ispace(int2d), double),
                r_ket_prevals14 : region(ispace(int2d), double),
                r_ket_prevals20 : region(ispace(int2d), double),
                r_ket_prevals21 : region(ispace(int2d), double),
                r_ket_prevals22 : region(ispace(int2d), double),
                r_ket_prevals23 : region(ispace(int2d), double),
                r_ket_prevals24 : region(ispace(int2d), double),
                r_ket_prevals30 : region(ispace(int2d), double),
                r_ket_prevals31 : region(ispace(int2d), double),
                r_ket_prevals32 : region(ispace(int2d), double),
                r_ket_prevals33 : region(ispace(int2d), double),
                r_ket_prevals34 : region(ispace(int2d), double),
                r_ket_prevals40 : region(ispace(int2d), double),
                r_ket_prevals41 : region(ispace(int2d), double),
                r_ket_prevals42 : region(ispace(int2d), double),
                r_ket_prevals43 : region(ispace(int2d), double),
                r_ket_prevals44 : region(ispace(int2d), double),
                r_labels00     : region(ispace(int1d), getKFockLabel(0, 0)),
                r_labels01     : region(ispace(int1d), getKFockLabel(0, 1)),
                r_labels02     : region(ispace(int1d), getKFockLabel(0, 2)),
                r_labels03     : region(ispace(int1d), getKFockLabel(0, 3)),
                r_labels04     : region(ispace(int1d), getKFockLabel(0, 4)),
                r_labels10     : region(ispace(int1d), getKFockLabel(1, 0)),
                r_labels11     : region(ispace(int1d), getKFockLabel(1, 1)),
                r_labels12     : region(ispace(int1d), getKFockLabel(1, 2)),
                r_labels13     : region(ispace(int1d), getKFockLabel(1, 3)),
                r_labels14     : region(ispace(int1d), getKFockLabel(1, 4)),
                r_labels20     : region(ispace(int1d), getKFockLabel(2, 0)),
                r_labels21     : region(ispace(int1d), getKFockLabel(2, 1)),
                r_labels22     : region(ispace(int1d), getKFockLabel(2, 2)),
                r_labels23     : region(ispace(int1d), getKFockLabel(2, 3)),
                r_labels24     : region(ispace(int1d), getKFockLabel(2, 4)),
                r_labels30     : region(ispace(int1d), getKFockLabel(3, 0)),
                r_labels31     : region(ispace(int1d), getKFockLabel(3, 1)),
                r_labels32     : region(ispace(int1d), getKFockLabel(3, 2)),
                r_labels33     : region(ispace(int1d), getKFockLabel(3, 3)),
                r_labels34     : region(ispace(int1d), getKFockLabel(3, 4)),
                r_labels40     : region(ispace(int1d), getKFockLabel(4, 0)),
                r_labels41     : region(ispace(int1d), getKFockLabel(4, 1)),
                r_labels42     : region(ispace(int1d), getKFockLabel(4, 2)),
                r_labels43     : region(ispace(int1d), getKFockLabel(4, 3)),
                r_labels44     : region(ispace(int1d), getKFockLabel(4, 4)),
                r_density00   : region(ispace(int2d), getKFockDensity(0, 0)),
                r_density01   : region(ispace(int2d), getKFockDensity(0, 1)),
                r_density02   : region(ispace(int2d), getKFockDensity(0, 2)),
                r_density03   : region(ispace(int2d), getKFockDensity(0, 3)),
                r_density04   : region(ispace(int2d), getKFockDensity(0, 4)),
                r_density11   : region(ispace(int2d), getKFockDensity(1, 1)),
                r_density12   : region(ispace(int2d), getKFockDensity(1, 2)),
                r_density13   : region(ispace(int2d), getKFockDensity(1, 3)),
                r_density14   : region(ispace(int2d), getKFockDensity(1, 4)),
                r_density22   : region(ispace(int2d), getKFockDensity(2, 2)),
                r_density23   : region(ispace(int2d), getKFockDensity(2, 3)),
                r_density24   : region(ispace(int2d), getKFockDensity(2, 4)),
                r_density33   : region(ispace(int2d), getKFockDensity(3, 3)),
                r_density34   : region(ispace(int2d), getKFockDensity(3, 4)),
                r_density44   : region(ispace(int2d), getKFockDensity(4, 4)),
                r_output00    : region(ispace(int3d), getKFockOutput(0, 0)),
                r_output01    : region(ispace(int3d), getKFockOutput(0, 1)),
                r_output02    : region(ispace(int3d), getKFockOutput(0, 2)),
                r_output03    : region(ispace(int3d), getKFockOutput(0, 3)),
                r_output04    : region(ispace(int3d), getKFockOutput(0, 4)),
                r_output11    : region(ispace(int3d), getKFockOutput(1, 1)),
                r_output12    : region(ispace(int3d), getKFockOutput(1, 2)),
                r_output13    : region(ispace(int3d), getKFockOutput(1, 3)),
                r_output14    : region(ispace(int3d), getKFockOutput(1, 4)),
                r_output22    : region(ispace(int3d), getKFockOutput(2, 2)),
                r_output23    : region(ispace(int3d), getKFockOutput(2, 3)),
                r_output24    : region(ispace(int3d), getKFockOutput(2, 4)),
                r_output33    : region(ispace(int3d), getKFockOutput(3, 3)),
                r_output34    : region(ispace(int3d), getKFockOutput(3, 4)),
                r_output44    : region(ispace(int3d), getKFockOutput(4, 4)),
                r_gamma_table : region(ispace(int2d, {18, 700}), double[5]),
                threshold    : float,
                kguard       : float,
                parallelism0 : int,
                parallelism1 : int,
		parallelism2 : int,
		parallelism3 : int,
		parallelism4 : int,
		parallelism5 : int,
		parallelism6 : int,
		parallelism7 : int,
		parallelism8 : int,
		parallelism9 : int,
		parallelism10 : int,
		parallelism11 : int,
		parallelism12 : int,
		parallelism13 : int,
		parallelism14 : int,
		parallelism15 : int,
                largest_momentum : int)
where
  reads (
    r_pairs00, r_pairs01, r_pairs02, r_pairs03, r_pairs04,
    r_pairs10, r_pairs11, r_pairs12, r_pairs13, r_pairs14,
    r_pairs20, r_pairs21, r_pairs22, r_pairs23, r_pairs24,
    r_pairs30, r_pairs31, r_pairs32, r_pairs33, r_pairs34,
    r_pairs40, r_pairs41, r_pairs42, r_pairs43, r_pairs44,
    r_bra_prevals00, r_bra_prevals01, r_bra_prevals02, r_bra_prevals03, r_bra_prevals04,
    r_bra_prevals10, r_bra_prevals11, r_bra_prevals12, r_bra_prevals13, r_bra_prevals14,
    r_bra_prevals20, r_bra_prevals21, r_bra_prevals22, r_bra_prevals23, r_bra_prevals24,
    r_bra_prevals30, r_bra_prevals31, r_bra_prevals32, r_bra_prevals33, r_bra_prevals34,
    r_bra_prevals40, r_bra_prevals41, r_bra_prevals42, r_bra_prevals43, r_bra_prevals44,
    r_ket_prevals00, r_ket_prevals01, r_ket_prevals02, r_ket_prevals03, r_ket_prevals04,
    r_ket_prevals10, r_ket_prevals11, r_ket_prevals12, r_ket_prevals13, r_ket_prevals14,
    r_ket_prevals20, r_ket_prevals21, r_ket_prevals22, r_ket_prevals23, r_ket_prevals24,
    r_ket_prevals30, r_ket_prevals31, r_ket_prevals32, r_ket_prevals33, r_ket_prevals34,
    r_ket_prevals40, r_ket_prevals41, r_ket_prevals42, r_ket_prevals43, r_ket_prevals44,
    r_labels00, r_labels01, r_labels02, r_labels03, r_labels04,
    r_labels10, r_labels11, r_labels12, r_labels13, r_labels14,
    r_labels20, r_labels21, r_labels22, r_labels23, r_labels24,
    r_labels30, r_labels31, r_labels32, r_labels33, r_labels34,
    r_labels40, r_labels41, r_labels42, r_labels43, r_labels44,
    r_density00, r_density01, r_density02, r_density03, r_density04,
    r_density11, r_density12, r_density13, r_density14,
    r_density22, r_density23, r_density24,
    r_density33, r_density34,
    r_density44,
    r_gamma_table
  ),
  reads writes(
    r_output00, r_output01, r_output02, r_output03, r_output04,
    r_output11, r_output12, r_output13, r_output14,
    r_output22, r_output23, r_output24,
    r_output33, r_output34,
    r_output44
  )
do
  regentlib.assert(largest_momentum <= [getCompiledMaxMomentum()],
                   "Please recompile eri-regent with a larger max momentum!");
  [kfock(
    {
      [0]={[0]=r_pairs00, [1]=r_pairs01, [2]=r_pairs02, [3]=r_pairs03, [4]=r_pairs04},
      [1]={[0]=r_pairs10, [1]=r_pairs11, [2]=r_pairs12, [3]=r_pairs13, [4]=r_pairs14},
      [2]={[0]=r_pairs20, [1]=r_pairs21, [2]=r_pairs22, [3]=r_pairs23, [4]=r_pairs24},
      [3]={[0]=r_pairs30, [1]=r_pairs31, [2]=r_pairs32, [3]=r_pairs33, [4]=r_pairs34},
      [4]={[0]=r_pairs40, [1]=r_pairs41, [2]=r_pairs42, [3]=r_pairs43, [4]=r_pairs44}
    },
    {
      [0]={[0]={r_bra_prevals00, r_ket_prevals00}, [1]={r_bra_prevals01, r_ket_prevals01},
           [2]={r_bra_prevals02, r_ket_prevals02}, [3]={r_bra_prevals03, r_ket_prevals03},
           [4]={r_bra_prevals04, r_ket_prevals04}},
      [1]={[0]={r_bra_prevals10, r_ket_prevals10}, [1]={r_bra_prevals11, r_ket_prevals11},
           [2]={r_bra_prevals12, r_ket_prevals12}, [3]={r_bra_prevals13, r_ket_prevals13},
           [4]={r_bra_prevals14, r_ket_prevals14}},
      [2]={[0]={r_bra_prevals20, r_ket_prevals20}, [1]={r_bra_prevals21, r_ket_prevals21}, 
           [2]={r_bra_prevals22, r_ket_prevals22}, [3]={r_bra_prevals23, r_ket_prevals23}, 
           [4]={r_bra_prevals24, r_ket_prevals24}},
      [3]={[0]={r_bra_prevals30, r_ket_prevals30}, [1]={r_bra_prevals31, r_ket_prevals31},
           [2]={r_bra_prevals32, r_ket_prevals32}, [3]={r_bra_prevals33, r_ket_prevals33},
           [4]={r_bra_prevals34, r_ket_prevals34}},
      [4]={[0]={r_bra_prevals40, r_ket_prevals40}, [1]={r_bra_prevals41, r_ket_prevals41},
           [2]={r_bra_prevals42, r_ket_prevals42}, [3]={r_bra_prevals43, r_ket_prevals43},
           [4]={r_bra_prevals44, r_ket_prevals44}}
    },
    {
      [0]={[0]=r_labels00, [1]=r_labels01, [2]=r_labels02, [3]=r_labels03, [4]=r_labels04},
      [1]={[0]=r_labels10, [1]=r_labels11, [2]=r_labels12, [3]=r_labels13, [4]=r_labels14},
      [2]={[0]=r_labels20, [1]=r_labels21, [2]=r_labels22, [3]=r_labels23, [4]=r_labels24},
      [3]={[0]=r_labels30, [1]=r_labels31, [2]=r_labels32, [3]=r_labels33, [4]=r_labels34},
      [4]={[0]=r_labels40, [1]=r_labels41, [2]=r_labels42, [3]=r_labels43, [4]=r_labels44}
    },
    {
      [0]={[0]=r_density00, [1]=r_density01, [2]=r_density02, [3]=r_density03, [4]=r_density04},
      [1]={[1]=r_density11, [2]=r_density12, [3]=r_density13, [4]=r_density14},
      [2]={[2]=r_density22, [3]=r_density23, [4]=r_density24},
      [3]={[3]=r_density33, [4]=r_density34},
      [4]={[4]=r_density00}
    },
    {
      [0]={[0]=r_output00, [1]=r_output01, [2]=r_output02, [3]=r_output03, [4]=r_output04},
      [1]={[1]=r_output11, [2]=r_output12, [3]=r_output13, [4]=r_output14},
      [2]={[2]=r_output22, [3]=r_output23, [4]=r_output24},
      [3]={[3]=r_output33, [4]=r_output34},
      [4]={[4]=r_output44}
    },
    r_gamma_table, threshold, kguard,
    {[0]=parallelism0, [1]=parallelism1,
     [2]=parallelism2, [3]=parallelism3,
     [4]=parallelism4, [5]=parallelism5,
     [6]=parallelism6, [7]=parallelism7, 
     [8]=parallelism8, [9]=parallelism9,
     [10]=parallelism10,[11]=parallelism11,
     [12]=parallelism12, [13]=parallelism13,
     [14]=parallelism14, [15]=parallelism15
    },
    largest_momentum, 1)]
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
