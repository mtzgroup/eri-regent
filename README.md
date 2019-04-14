# eri-regent
[![Build Status]]
Calculates two-electron repulsion integrals with Regent

# Precomputed Boys Function Values
Use the python script in `scripts/` to generate a header file containing values of the Boys function.

```
python scripts/generate_boys_region.py src/precomputedBoys.h
```

# Running
Run in Regent using `top.rg` inside `src/` for testing.

```
cd src
regent top.rg -i h2_6-311g.dat -o output.dat
```

Use the Makefile in `src/cpp/` to compile and run inside C++. This will generate a header file and a library for the Coulomb tasks so they can be called within C++. The Makefile assumes the `$LG_RT_DIR` environment variable has been set.

```
cd src/cpp
export LG_RT_DIR=[PATH TO LEGION]/runtime
make all
make run
```

# Testing

```
python scripts/test.py
```
