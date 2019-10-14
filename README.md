# eri-regent
[![Build Status](https://travis-ci.com/sparkasaurusRex/eri-regent.svg?token=g46Mfub8GMWqdPYXVqEs&branch=master)](https://travis-ci.com/sparkasaurusRex/eri-regent)

Calculates two-electron repulsion integrals with Regent

## Running
Run in Regent using `top.rg` inside `src/` for testing.

```
cd src
regent top.rg -L P -i data/h2o -v data/h2o/output.dat
# Use option `-fflow 0` to compile Regent faster
```

Use the Makefile in `src/cpp/` to compile and run inside C++. This will generate a header file and a library for the Coulomb tasks so they can be called within C++. The Makefile assumes the `$LG_RT_DIR` environment variable has been set.

```
cd src/cpp
export LG_RT_DIR=$PATH_TO_LEGION/runtime
make all
make run
```

### Higher Angular Momentum Systems

Be sure to select the appropriate angular momentum using the `-L [S|P|D|F|G]` option. This will tell Lua to produce the correct number of Regent tasks. Higher angular momentums need more and larger kernels which can take a long time to compile to Cuda code. The number of kernels needed is <code>(2L-1)<sup>2</sup></code>.

| Angular Momentum | Number of J Kernels | Memory     | Wall-time  |
|:----------------:|:-------------------:|:----------:|:----------:|
| S                | 1                   | Negligible | Negligible |
| P                | 9                   | Negligible | Negligible |
| D                | 25                  | 1.5 GB     | 1 Minute   |
| F                | 49                  | 7 GB       | 7 Minutes  |
| G                | 81                  | 31 GB      | 1 Hour     |


## Testing with Python3

```
python scripts/test.py
python scripts/test_boys.py
```
