# eri-regent

Standalone Regent code that calculates two-electron repulsion integrals (ERIs) to form the J and K matrices and their gradients, a building blocks in many quantum chemistry methods. Regent enables straightforward execution on multiple architectures: CPUs, GPUs, and multiple nodes.

Code can be run as standalone (with input files from TeraChem) or integrated in a TeraChem build (currently using the `legion_updates` branch of TeraChem).

Authors: K. Grace Johnson, Ellis Hoag, Seema Mirchandaney, Alex Aiken, Todd J. Martinez


## Setup

Running Regent code requires installing the Legion programming system. The dependencies of Legion are:
- Linux, macOS, or another Unix
- A C++ 98 (or newer) compiler (GCC, Clang, Intel, or PGI) and GNU Make
- Python (for Regent and build, profiling, and debugging tools)
- *Optional*: CUDA 5.0 or newer (for NVIDIA GPUs)
- *Optional*: GASNet (for networking, see installation instructions here: https://legion.stanford.edu/gasnet/)

### Legion Installation
Optional arguments for compiling with GPU support (CUDA) and multi-node support (GASNet) are commented. 

```bash
git clone https://gitlab.com/StanfordLegion/legion.git -b hijack_registration_hack
export LEGION_SRC=$PWD/legion
export CC=<path to icc or gcc, etc.>
export CXX=<path to icc or g++, etc.>
#export GASNET=<path to gasnet install directory>             # Optional, multi-node
#export CONDUIT=ibv                                           # Optional, multi-node (also set as `mpi`)
#export GPU_ARCH=maxwell                                      # Optional, GPU (set to desired arch.)
#export USE_CUDA=1                                            # Optional, GPU
#export CUDA_BIN_PATH=$CUDA_HOME                              # Optional, GPU
$LEGION_SRC/language/scripts/setup_env.py --cmake  \
    --terra-url https://github.com/StanfordLegion/terra.git \
    --terra-branch luajit2.1 \
    --extra="-DCMAKE_INSTALL_PREFIX=$LEGION_INSTALL_PATH" \
    --extra="-DCMAKE_BUILD_TYPE=Release" \
    --extra="-DLegion_HIJACK_CUDART=OFF" \
#   --extra="--with-gasnet ${GASNET}" \                       # Optional, multi-node
#   --extra="--cuda" \                                        # Optional, GPU
export REGENT=$LEGION_SRC/language/regent.py
alias regent=$REGENT
```

Further information on Legion installation can be found here: https://legion.stanford.edu/starting/


### Submodule of TeraChem

Normally this project (eri-regent) is a submodule of the larger TeraChem project. If this is your intended use, clone `eri-regent` inside of the `regintbox` folder in TeraChem and the project will be compiled as part of the TeraChem build:

```bash
# Pull/checkout the code
cd terachem/
git checkout legion_updates
cd regintbox/src/
git clone https://github.com/mtzgroup/j-eri-regent
cd ../../
# Set the following environment variables and configure TeraChem's make:
export RG_MAX_MOMENTUM=P # options are S, P, D, F (see below)
export LEGIONHOME=[path to legion install directory]
export LEGIONSRC=[path to legion source directory]
./configure --uselegion
make
```

The eri-regent project is compiled into a .so and there is a C++ interface in `terachem/regintbox/src` that calls to routines in this library.

See below for an example build script for the Fire computer cluster.
```bash
module load MPICH/3.2.1-GCC-7.2.0-2.29 # TODO: You need this to compile Regent kernels but it messes up something in THCbox..
module load OpenMM/7.4.2-intel-2017.8.262-CUDA-11.0.2
module load protobuf/3.14.0
ml CUDA/11.0.2
ml intel/2017.8.262

export LD_LIBRARY_PATH=/global/software/CUDA/11.0.2/lib:${LD_LIBRARY_PATH}
export GPU_ARCH=maxwell,volta,ampere
export DEBUG=0
export RG_MAX_MOMENTUM=P
export LEGIONHOME=/home/kgjohn/code/terachem-legion-builds/MPI/legion2_install
export LEGIONSRC=/home/kgjohn/code/terachem-legion-builds/MPI/legion2

./configure --uselegion --nodftbplus --nocilk

export REGENT=$LEGIONSRC/language/regent.py
alias regent=$REGENT

make -j 8
```


### Standalone Version

If you are building and running eri-regent in a standalone configuration follow these instructions:

```bash
export LEGION_SRC=<path to dir with Legion build>/legion
export REGENT=$LEGION_SRC/language/regent.py
alias regent=$REGENT
```
Run with Regent using `top_jfock.rg`, `top_kfock.rg`, and `top_kgrad.rg` which contain the top level tasks of each routine, the Regent equivalent of main functions in C/C++.
```bash
cd eri-regent/src
# Example of running the J matrix algorithm:
regent top_jfock.rg -L P -i tests/integ/h2o -v tests/h2o/jfock_output.dat
# Example of running the K matrix algorithm:
regent top_kfock.rg -fflow 0 -L S -i tests/integ/h2_S -v tests/integ/h2_S/kfock_output.dat
# Note: at this time, `-fflow 0` is necessary for compiling kfock
# Example of running the K matrix gradient algorithm:
regent top_kgrad.rg -fflow 0 -L S -i tests/integ/h2_S -v tests/integ/h2_S/kgrad_output.dat

# To partition tasks and run in parallel on 2 GPUs:
regent top_jfock.rg -L P -i tests/h2o -v tests/h2o/output.dat -p 2 -ll:gpu 2
```
Note that these commands will both compile and execute the Regent code.

#### Options
- `-L [S|P|D|F|G]` specifies the max angular momentum. Compiler will generate all kernels up to and including those containing `L`.
- `-i` specifies path to directory containing input files (see below)
- `-v` verify output with reference data in this file (see below)
- `-p` specifies the number of partitions of each integral task. Default is 1.
- `-ll:gpu` directive passed to Legion specifying the number of GPUs per node to parallelize across.
- `-ll:cpu` directive passed to Legion specifying the number of CPUs per node to parallelize across. See all Legion command-line flags here: https://legion.stanford.edu/starting/
- `-fflow 0` compiles kernels faster
- `-h` print usage (including these and more options) and exit

See Legion/Regent documentation for further details on `-ll` runtime options.

#### Tests
The `tests` directory contains 5 tests on different systems with different max angular momentum. See `INFO` file in each directory.

Each test has sample data generated from a TeraChem SCF iteration of an RHF calculation.
- `jfock_bras.dat` coordinates (x,y,z) and Gaussian basis information (eta, C) of the bra in the Hermite basis
- `jfock_kets.dat` coordinates (x,y,z) and Gaussian basis information (eta, C) of the ket in the Hermite basis  and corresponding density value(s)
- `jfock_output.dat` output data (J matrix in Hermite basis) generated from TeraChem used to verify the Regent calculation
- `parameters.dat` parameters from TeraChem input, e.g. Schwartz screening
- Files for jgrad, kfock, and kgrad are also included. To see how these files are generated, look at the `dump` routines in `/terachem/intbox/src/pairsorter.cpp`. (Note: this is also how the data is packed in the version of eri-regent that is integrated in TeraChem using a C++ interface, though of course using data structures and not printing to file.)

## Code structure
- `src`: contains the top level tasks (`top_jfock.rg`, `top_kfock.rg`, `top_kgrad.rg`), the driver for kernel generation and execution (`jfock.rg`, `kfock.rg`, and `kgrad.rg`), region specifications in `fields.rg`, and helper functions in `helper.rg`, and I/O parsing in `parse_files.rg`.
- `src/cpp`: C++ interface for TeraChem
- `src/mcmurchie`: code to compute intergrals using the McMurchie-Davidson algorithm
- `src/rys`: initial code to compute integrals using the Rys algorithm (incomplete)

## Notes on angular momentum and compilation time

Be sure to select the appropriate angular momentum using the `-L [S|P|D|F|G]` option. This will tell Lua to produce the correct number of Regent tasks. Higher angular momenta require more and larger kernels which can take a longer time to compile to CUDA code. The number of J kernels needed is <code>(2L-1)<sup>2</sup></code>.

| Max Angular Momentum | Number of J Kernels | Memory     | Compilation wall-time |
|:--------------------:|:-------------------:|:----------:|:---------------------:|
| S = 0                | 1                   | Negligible | < 1 Minute            |
| P = 1                | 9                   | 2 GB       | 2 Minutes             |
| D = 2                | 25                  | > 4 GB     | > 5 Minutes           |
| F = 3                | 49                  | > 7 GB     | > 7 Minutes           |
| G = 4                | 81                  | > 31 GB    | > 1 Hour              |

## Parallel compilation

The K kernels take far longer than the J kernels to compile. Regent code generation and compilation can be done in parallel. See `parallel_compilation` branch of `eri-regent` (and thank you to Elliott Slaughter for his help with this).

## Regent Vim settings
Normal text editor settings do not format Regent code. If you would like to enable these features in your Vim settings, please follow the instructions here: https://github.com/StanfordLegion/regent.vim
