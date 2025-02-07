#include <math.h>
#include <random>


#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataWriter.h>

#define G 1
#define M 1000
#define PI 3.141592653589793238462643


void calculate_forces(double **particles, int Np)
{ 
  for (int i = 0; i < Np; i++)
  {
    double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
    for (int j = 0; j < Np; j++)
    {
      if (i != j)
      {
        double dx = particles[i][1] - particles[j][1];   
        double dy = particles[i][2] - particles[j][2];   
        double dz = particles[i][3] - particles[j][3];   

        double r = sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
        double Fmag = G * (particles[i][0] * particles[j][0]) / (r * r);

        fx_sum += -Fmag * dx / r;
        fy_sum += -Fmag * dy / r;
        fz_sum += -Fmag * dz / r;
      }
    }
    double dx = particles[i][1];
    double dy = particles[i][2];
    double dz = particles[i][3];

    double r = sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
    double Fmag = G * (M * particles[i][0]) / (r * r);

    fx_sum += -Fmag * dx / r;
    fy_sum += -Fmag * dy / r;
    fz_sum += -Fmag * dz / r;

    particles[i][7] = fx_sum;
    particles[i][8] = fy_sum;
    particles[i][9] = fz_sum;
  }
}

void solve(int Np, int Nt, int N_write, double dt)
{
  
  // Initialize data structure to hold particles 
  double **particles = new double*[Np]; // each particle (mass,x,y,z,vx,vy,vz,ax,ay,az)
  for (int i = 0; i < Np; i++) {
    particles[i] = new double[10]; // 11 to store rank
  }

  // random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> udist(0.0, 1.0);

  // Initialize the starting point for all particles
  for (int i = 0; i < Np; i++)
  {

    double r = 10 + udist(gen)*40;
    double theta = 2*PI*udist(gen);
    double z = 10*udist(gen) - 5;
 
    particles[i][0] = 0.5 + udist(gen);

    particles[i][1] = r*cos(theta); // x
    particles[i][2] = r*sin(theta); // y
    particles[i][3] = z; // z

    double eps = -0.01 + 0.02*udist(gen);
    double vt = 2.*sqrt(G*M/r)*(1+eps);

    // calculate tangential velocity (r_hat x k_hat)

    particles[i][4] = vt*particles[i][2]/r;
    particles[i][5] = -vt*particles[i][1]/r;
    particles[i][6] = 0.0;

  }

  for (int step = 0; step < Nt; step++)
  {
    // Visualize
    if (step % N_write == 0)
    {
      std::cout << "Step: " << step << std::endl;

      // VTK Routines for plotting
      vtkNew<vtkPoints> points;
      
      // Iterate over particles to create VTK structures
      for (int i = 0; i < Np; ++i) {
        points->InsertNextPoint(
            particles[i][1], 
            particles[i][2], 
            particles[i][3]
        );
      }

      vtkNew<vtkDoubleArray> velocities;
      vtkNew<vtkDoubleArray> forces;
      velocities->SetName("Velocity");
      velocities->SetNumberOfComponents(3);
      forces->SetName("Force");
      forces->SetNumberOfComponents(3);

      for (int i = 0; i < Np; i++) {
        velocities->InsertNextTuple3(particles[i][4], particles[i][5], particles[i][6]);
        forces->InsertNextTuple3(particles[i][7], particles[i][8], particles[i][9]);
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

    calculate_forces(particles, Np);

    
    for (int i = 0; i < Np; i++) {
      particles[i][4] += 0.5*dt*particles[i][7] / particles[i][0];
      particles[i][5] += 0.5*dt*particles[i][8] / particles[i][0];
      particles[i][6] += 0.5*dt*particles[i][9] / particles[i][0];

      particles[i][1] += particles[i][4]*dt; 
      particles[i][2] += particles[i][5]*dt;
      particles[i][3] += particles[i][6]*dt;
    } 

    calculate_forces(particles, Np);

    for (int i = 0; i < Np; i++)
    {
      particles[i][4] += 0.5*dt*particles[i][7] / particles[i][0];
      particles[i][5] += 0.5*dt*particles[i][8] / particles[i][0];
      particles[i][6] += 0.5*dt*particles[i][9] / particles[i][0];
    }
  }

  for (int i = 0; i < Np; i++) {
    delete[] particles[i];
  }

  delete[] particles;
}

int main(int argc, char* argv[]){
  int Np = atoi(argv[1]);       // Number of particles in simulation
  int Nt = atoi(argv[2]);       // Number of timesteps for simulation
  int N_write = atoi(argv[3]);  // Write frequency for output
  double dt = atof(argv[4]);    // timestep for simulation
  solve(Np, Nt, N_write, dt);
}
