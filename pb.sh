#!/bin/sh
#PBS -N compression_test
#PBS -P col761.cs1200373
#PBS -m bea
#PBS -q standard
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=0:mem=16G
#PBS -l walltime=01:00:00

# Inputs and outputs generated locally and not on hpc.

echo "==============================="
module load compiler/gcc/9.1.0
echo $PBS_JOBID
cd $PBS_O_WORKDIR

./run.sh
