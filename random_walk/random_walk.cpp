#include <random>

#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h> //
#include <vtkPolyLine.h> //
#include <vtkXMLPolyDataWriter.h>

#include <cstdlib>
#include <ctime>

#define PI 3.141592653589793238462643

int main(int argv, char* argc[]) {

    // Parameters
    const int Np = 1000;   // Number of particles
    const int numSteps = 1000;       // Steps per particle
    const double R = 0.001;         // Step size
    const int n_write = 2;         // write frequency


    // Initialize data structure to hold particles 
    double **positions = new double*[Np]; // each particle x,y,z
    for (int i = 0; i < Np; i++) {
        positions[i] = new double[3]; // 11 to store rank
    }

    // Initialize the starting point for all particles
    for (int i = 0; i < Np; i++){
        positions[i][0] = 0.0; // x
        positions[i][1] = 0.0; // y
        positions[i][2] = 0.0; // z
    }

    // random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> udist(0.0, 1.0);


    for (int i = 0; i < numSteps; i++)
    {

      std::cout << "Step: " << i <<  std::endl; 

      if (i % n_write == 0)
      {
        // VTK Routines for plotting
        vtkNew<vtkPoints> points;

        // Iterate over particles to create VTK structures
        for (int j = 0; j < Np; ++j) {
          points->InsertNextPoint(
              positions[j][0], 
              positions[j][1], 
              positions[j][2]
          );
        }
        
        std::ostringstream filename;
        filename << "random_walk_" << i << ".vtp";

        vtkNew<vtkPolyData> polyData;
        polyData->SetPoints(points);
        vtkNew<vtkXMLPolyDataWriter> writer;
        writer->SetFileName(filename.str().c_str());
        writer->SetInputData(polyData);
        writer->Write();
      }

      // Generate random walks for all particles
      for (int j= 0; j < Np; j++){ 
        double theta = PI*udist(gen);
        double phi = 2*PI*udist(gen);

        double dx = R*sin(theta)*cos(phi);
        double dy = R*sin(theta)*sin(phi);
        double dz = R*cos(theta);

        // do not forget to release the state of the engine

        // Compute next position
        positions[i][0] += dx;
        positions[i][1] += dy;
        positions[i][2] += dz;
      }
    }

  return 0;
}
