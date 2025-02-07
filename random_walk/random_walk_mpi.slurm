#!/bin/bash
#SBATCH --job-name=random_walk_mpi
#SBATCH --partition=nu
#SBATCH --output=stdout-%x_%j
#SBATCH --error=stderr-%x_%j
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --nodelist=nu-1c.cnca

module load gcc/11.1.0
module load mpich
module load vtk/9.4.1-mpi
module load cmake/3.26.0
module load kokkos/openmp

mpirun -np 4 ./build/random_walk 1000

