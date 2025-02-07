#include <math.h>
#include <random>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataWriter.h>

#define G 1
#define M 1000
#define PI 3.141592653589793238462643


void calculate_forces(int Np, Kokkos::View<double **>& particles, int nlocal, int start, int end)
{ 

  for(int i = start; i < end; i++)
  {
    double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
	 Kokkos::parallel_reduce("Force iteration", Np, KOKKOS_LAMBDA(const int j, double &fx, double &fy, double &fz){
      if (i != j)
		{
			double dx = particles(i,1) - particles(j,1);
			double dy = particles(i,2) - particles(j,2);
			double dz = particles(i,3) - particles(j,3);

			double r = sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
			double Fmag = G * (particles(i,0) * particles(j,0)/(r*r));

         fx += -Fmag * dx / r;
         fy += -Fmag * dy / r;
         fz += -Fmag * dz / r;
      }
    },fx_sum,fy_sum,fz_sum);

    double dx = particles(i,1);
    double dy = particles(i,2);
    double dz = particles(i,3);

    double r = sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
    double Fmag = G * (M * particles(i,0)) / (r * r);

    fx_sum += -Fmag * dx / r;
    fy_sum += -Fmag * dy / r;
    fz_sum += -Fmag * dz / r;

    particles(i,7) = fx_sum;
    particles(i,8) = fy_sum;
    particles(i,9) = fz_sum;
  }
}

void solve(int Np, int Nt, int N_write, double dt, int nlocal, int start, int end, int rank)
{

  // Generar el View que contendrá todos los datos de las partículas.
  Kokkos::View<double **> particles("particles", Np, 10);

  if(rank==0){
  		// Generador de números aleatorios.
  		Kokkos::Random_XorShift64_Pool<> rand(69420);

		Kokkos::parallel_for("fill_particles", Np, KOKKOS_LAMBDA(const int i){
			auto rand1 = rand.get_state();
			particles(i,0) = 0.5 + rand1.drand(); // m
			int r = 10 + rand1.drand()*40; // r
			int theta = 2*PI*rand1.drand();
			particles(i,1) = cos(theta)*r; // x
			particles(i,2) = sin(theta)*r; // y
			particles(i,3) = 10*rand1.drand(); // z
			// Velocidad tangencial.
			auto rand2 = rand.get_state();
			int eps = -0.01 + 0.02*rand2.drand();
			int vt = 2*sqrt(G*M/r)*(1+eps);
			particles(i,4) = vt*particles(i,2)/r;
			particles(i,5) = -vt*particles(i,1)/r;
			particles(i,6) = 0.0;
		});

  }
  
  for (int step = 0; step < Nt; step++)
  {
	 MPI_Bcast(&particles, Np*10, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     if(rank==0){
    // Visualize
    if (step % N_write == 0)
    {
      std::cout << "Step: " << step << std::endl;

      // VTK Routines for plotting
      vtkNew<vtkPoints> points;
      
      // Iterate over particles to create VTK structures
      for (int i = 0; i < Np; ++i) {
        points->InsertNextPoint(
            particles(i,1), 
            particles(i,2), 
            particles(i,3)
        );
      }

      vtkNew<vtkDoubleArray> velocities;
      vtkNew<vtkDoubleArray> forces;
      velocities->SetName("Velocity");
      velocities->SetNumberOfComponents(3);
      forces->SetName("Force");
      forces->SetNumberOfComponents(3);

      for (int i = 0; i < Np; i++) {
        velocities->InsertNextTuple3(particles(i,4), particles(i,5), particles(i,6));
        forces->InsertNextTuple3(particles(i,7), particles(i,8), particles(i,9));
      }

      std::ostringstream filename;
      filename << "nbody_" << step << ".vtp";

      vtkNew<vtkPolyData> polyData;

      polyData->GetPointData()->AddArray(velocities);
      polyData->GetPointData()->AddArray(forces);

      polyData->SetPoints(points);
      vtkNew<vtkXMLPolyDataWriter> writer;
      writer->SetFileName(filename.str().c_str());
      writer->SetInputData(polyData);
      writer->Write();
    }
    }

    calculate_forces(Np, particles, nlocal, start, end);

    
    for (int i = 0; i < Np; i++) {
      particles(i,4) += 0.5*dt*particles(i,7) / particles(i,0);
      particles(i,5) += 0.5*dt*particles(i,8) / particles(i,0);
      particles(i,6) += 0.5*dt*particles(i,9) / particles(i,0);

      particles(i,1) += particles(i,4)*dt; 
      particles(i,2) += particles(i,5)*dt;
      particles(i,3) += particles(i,6)*dt;
    } 

    calculate_forces(Np, particles, nlocal, start, end);

    for (int i = 0; i < Np; i++)
    {
      particles(i,4) += 0.5*dt*particles(i,7) / particles(i,0);
      particles(i,5) += 0.5*dt*particles(i,8) / particles(i,0);
      particles(i,6) += 0.5*dt*particles(i,9) / particles(i,0);
    }
  }
}

int main(int argc, char* argv[]){

	int size, rank;
   MPI_Init(&argc, &argv);
  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int Np = atoi(argv[1]);
	int Nt = atoi(argv[2]);
	int N_write = atoi(argv[3]);
	double dt = atof(argv[4]);
	int rest = Np%size;
	int nlocal = Np/size + ( (rest && rank < rest) ? 1 : 0 );
	int start = nlocal*rank + ( (rest && rank >= rest) ? rest : 0 );
	int end = nlocal + start;
  
  solve(Np, Nt, N_write, dt, nlocal, start, end, rank);
}
