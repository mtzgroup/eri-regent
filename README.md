# eri-regent
* stable [![Build Status](https://travis-ci.com/sparkasaurusRex/eri-regent.svg?token=g46Mfub8GMWqdPYXVqEs&branch=stable)](https://travis-ci.com/sparkasaurusRex/eri-regent)
* master [![Build Status](https://travis-ci.com/sparkasaurusRex/eri-regent.svg?token=g46Mfub8GMWqdPYXVqEs&branch=master)](https://travis-ci.com/sparkasaurusRex/eri-regent)

Calculates two-electron repulsion integrals with Regent

## Running
Run with Regent using `top.rg` inside `src/` for testing.

```bash
cd src
regent top.rg -L P -i data/h2o -v data/h2o/output.dat
# Use option `-fflow 0` to compile eri-regent faster
```

Use the Makefile to compile and run inside C++. This will generate a header file and a library for the eri tasks so they can be called within C++. The Makefile assumes the `$LG_RT_DIR` and `$RG_MAX_MOMENTUM` environment variables have been set. If you want to compile for a new `$RG_MAX_MOMENTUM` then you need to run `make rg.clean` before the environment variable affects the build.

```bash
export LG_RT_DIR="/path/to/legion/runtime"
export RG_MAX_MOMENTUM=P
make
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

When more than 1 GB of memory is needed, you must build Legion with `luajit2.1`.
```bash
./install.py --terra-url https://github.com/StanfordLegion/terra.git --terra-branch luajit2.1
```


## Testing with Python3

```bash
python scripts/test.py
python scripts/test_boys.py
```
