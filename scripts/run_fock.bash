#!/bin/sh
#SBATCH -t 0:10:00
#SBATCH -J tc-build
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition normal
#SBATCH --gres gpu:2

source ~/setup.bash

module load intel/2017u8
module load binutils/2.25

cd ~/work/production/regintbox/src/eri-regent/src

srun $REGENT top_jfock.rg -L P -i tests/integ/h2o -v tests/integ/h2o/output.dat 

srun $REGENT top_kfock.rg -L S -i tests/integ/h2 -v tests/integ/h2/kfock_output.dat 


