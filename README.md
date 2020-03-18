# eri-regent
* stable [![Build Status](https://travis-ci.com/ellishg/eri-regent.svg?token=g46Mfub8GMWqdPYXVqEs&branch=stable)](https://travis-ci.com/ellishg/eri-regent)
* master [![Build Status](https://travis-ci.com/ellishg/eri-regent.svg?token=g46Mfub8GMWqdPYXVqEs&branch=master)](https://travis-ci.com/ellishg/eri-regent)

Calculates two-electron repulsion integrals with Regent

## Setup

Start by cloning Legion

```bash
git clone https://github.com/StanfordLegion/legion.git
export LEGION_DIR=$PWD/legion
cd $LEGION_DIR
git checkout master
```

When more than 1 GB of memory is needed, you must build Legion with `luajit2.1`.
Instructions for building on Ubuntu Linux:

```bash
cd $LEGION_DIR/language
./install.py --cmake --terra-url https://github.com/StanfordLegion/terra.git --terra-branch luajit2.1
make -C build install
```

Instructions for building on xstream:
```bash
cd $LEGION_DIR/language
module load GCC/4.9.2-binutils-2.25  
module load OpenMPI/1.8.5
module load Python/3.6.0
module load CMake/3.5.2
module load CUDA/8.0.61
module load LLVM/3.7.0
export CONDUIT=ibv
export CC=gcc
export CXX=g++
./scripts/setup_env.py --cmake  --terra-url https://github.com/StanfordLegion/terra.git --terra-branch luajit2.1
```

Define an alias to regent
```bash
alias regent="$LEGION_DIR/language/regent.py"
```

## Building

Use the Makefile to compile and run inside C++. This will generate a header file and a library for the eri tasks so they can be called within C++. The Makefile assumes the `RG_MAX_MOMENTUM` environment variable has been set. If you want to compile for a new `RG_MAX_MOMENTUM` then you need to run `make rg.clean` before the environment variable will affect the build.

```bash
cd eri-regent
export RG_MAX_MOMENTUM=P
make
```

## Running and Testing
Run with Regent using `top_jfock.rg` or `top_kfock.rg` inside `src/` for testing. Note that running eri-regent with this method does not require you to run `make`.

```bash
cd eri-regent/src
# To run JFock algorithm
regent top_jfock.rg -L P -i tests/integ/h2o -v tests/integ/h2o/output.dat
# To run KFock algorithm
regent top_kfock.rg -L S -i tests/integ/h2 -v tests/integ/h2/kfock_output.dat
# Use option `-fflow 0` to compile eri-regent faster
```

To test eri-regent with C++, compile the test program inside `src/tests/cpp` after building eri-regent.
```bash
cd eri-regent/src/tests/cpp
make
```

This will produce a binary inside `eri-regent/build`.
```bash
cd eri-regent
# To run JFock algorithm
build/eri_regent_test -i src/tests/integ/h2o -a jfock
# To run KFock algorithm
build/eri_regent_test -i src/tests/integ/h2o -a kfock
```

### Higher Angular Momentum Systems

Be sure to select the appropriate angular momentum using the `-L [S|P|D|F|G]` option. This will tell Lua to produce the correct number of Regent tasks. Higher angular momentums need more and larger kernels which can take a long time to compile to Cuda code. The number of J kernels needed is <code>(2L-1)<sup>2</sup></code> and the number of K kernels needed is <code>L<sup>2</sup> * (L<sup>2</sup> + 1) / 2</code>.

| Angular Momentum | Number of J Kernels | Number of K Kernels | Memory     | Wall-time   |
|:----------------:|:-------------------:|:-------------------:|:----------:|:-----------:|
| S = 1            | 1                   | 1                   | Negligible | < 1 Minute  |
| P = 2            | 9                   | 10                  | 2 GB       | 2 Minutes   |
| D = 3            | 25                  | 45                  | > 4 GB     | > 5 Minutes |
| F = 4            | 49                  | 136                 | > 7 GB     | > 7 Minutes |
| G = 5            | 81                  | 325                 | > 31 GB    | > 1 Hour    |

## Testing with Python3
First compile the test program in `eri-regent/src/tests/cpp`, then you can use `python3` to run the binary on all test inputs.
```bash
python scripts/test.py
python scripts/test_boys.py
```
