# eri-regent
Calculates two-electron repulsion integrals with Regent

# Precomputed Boys Function
Use the python script to generate a header file containing the lookup table of the Boys function.

```
# Requires python3.7 or higher
./gen_precomputed_header.py
```

# Testing
The easiest way to test is to use docker.

```
cd eri-regent
docker run -ti -v $PWD:/eri stanfordlegion/regent regent /eri/coulomb.rg -i /eri/h2_6-311g.dat -o /eri/output.dat
```
