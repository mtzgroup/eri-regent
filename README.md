# eri-regent
Calculates two-electron repulsion integrals with Regent

# Testing
The easiest way to test is to use docker

```
cd eri-regent
docker run -ti -v $PWD:/eri stanfordlegion/regent regent /eri/coulomb.rg -i /eri/coulomb.rg -i /eri/h2_6-311g.dat -o /eri/output.dat
```
