# eri-regent
[![Build Status](https://travis-ci.com/sparkasaurusRex/eri-regent.svg?token=g46Mfub8GMWqdPYXVqEs&branch=master)](https://travis-ci.com/sparkasaurusRex/eri-regent)

Calculates two-electron repulsion integrals with Regent

## Precomputed Boys Function Values
Use the python script in `scripts/` to generate a header file containing values of the Boys function.

```
python scripts/generate_boys_region.py src/mcmurchie/precomputedBoys.h
```

## Running
Run in Regent using `top.rg` inside `src/` for testing.

```
cd src
regent top.rg -i data/h2_6-311g.dat -v data/h2_6-311g_out.dat
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

The default maximum angular momentum is `2` for systems with P orbitals (the Bra PP has an angular momentum of `2`). To compute with higher systems, use the option `--max_momentum L` (must be the first option). Also, make sure there are enough Boys values generated with `scripts/generate_boys_region.py`. The table below lists the current compile times and memory usage for each `max_momentum`. Compile time is when Lua generates code before Regent starts executing it.

| Orbital | Angular Momentum | Memory     | Wall-time  |
|:-------:|:----------------:|:----------:|:----------:|
| P       | 2                | Negligible | Negligible |
| D       | 4                | 1.5 GB     | 1 Minute   |
| F       | 6                | 7 GB       | 7 Minutes  |
| G       | 8                | 31 GB      | 1 Hour     |


## Testing

```
python scripts/test.py
python scripts/test_boys.py
```
