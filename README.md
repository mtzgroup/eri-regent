# eri-regent
Calculates two-electron repulsion integrals with Regent

# Precomputed Boys Function Values
Use the python script to generate an HDF5 file containing values of the Boys function.

```
python gen_boys_region.py
```

# Running

```
regent coulomb.rg -i h2_6-311g.dat -o output.dat
```
