#include <iostream>
#include <math.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>

#define PI25DT 3.141592653589793238462643
using namespace std;

int main(int argc, char** argv){
	
	int N;
	if(argc != 3){
		cerr << "Usage: " << argv[0] << " -n [int N]" << endl;
		exit(1);
	} else {
		N = atoi(argv[2]);
	}

	int size, rank;
	MPI_Init(&argc, &argv);
	Kokkos::initialize();
	{
		double h, x, pi;
		//sum = 0.0;
		h = 1.0 / (double) N;

		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		double time1 = MPI_Wtime();

		int rest = N%size;
		int nlocal = N/size + ( (rest && rank < rest) ? 1 : 0 );
		int start = nlocal*rank + ( (rest && rank >= rest) ? rest : 0);
		int end = nlocal + start;
		
		auto sequential_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(Kokkos::DefaultHostExecutionSpace(), start, end);

		double t_sum = 0.0;
		Kokkos::parallel_reduce("Local sum", sequential_policy, KOKKOS_LAMBDA(const int i, double &sum){
			sum += (4/(1+( (i + 0.5)*h )*( (i+0.5)*h )) )*h;
		}, t_sum);
		
		MPI_Reduce(&t_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		double time2 = MPI_Wtime();
		double diff = time2 - time1;

		if(rank == 0){
			printf("Valor de pi: %.16f\nNúmero de procesos: %d\nError: %.16f\n", pi, size, fabs(pi - PI25DT));
		}
		printf("Tiempo transcurrido para rank=%d: %.10f\n", rank, diff);

	}
	Kokkos::finalize();
	MPI_Finalize();
	
	return 0;
}
