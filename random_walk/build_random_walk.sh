module --force purge
module load gcc/11.1.0
module load mpich
module load vtk/9.4.1-mpi
module load cmake/3.26.0
module load kokkos/openmp

cmake -B ./build -S ./ \
    -DCMAKE_CXX_COMPILER=/opt/compilers/gcc-11.1.0/bin/g++ \
    -DCMAKE_C_COMPILER=/opt/compilers/gcc-11.1.0/bin/gcc \

cmake --build ./build -j4 
cmake --install ./build