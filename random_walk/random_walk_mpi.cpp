#include <Kokkos_Core.hpp>           // Incluye la biblioteca Kokkos para programación paralela en CPU/GPU
#include <Kokkos_Random.hpp>         // Incluye la biblioteca Kokkos para generación de números aleatorios
#include <vtkNew.h>                  // Incluye la biblioteca VTK para manejo de objetos (smart pointers)
#include <vtkPoints.h>               // Incluye la biblioteca VTK para manejo de puntos en 3D
#include <vtkPolyData.h>             // Incluye la biblioteca VTK para representación de datos poligonales
#include <vtkCellArray.h>            // Incluye la biblioteca VTK para manejo de arreglos de celdas
#include <vtkPolyLine.h>             // Incluye la biblioteca VTK para manejo de líneas poligonales
#include <vtkXMLPolyDataWriter.h>    // Incluye la biblioteca VTK para escritura de datos poligonales en formato XML
#include <mpi.h>                     // Incluye la biblioteca MPI para programación paralela distribuida
#include <cstdlib>                   // Incluye la biblioteca estándar para funciones generales (como rand)
#include <ctime>                     // Incluye la biblioteca estándar para manejo de tiempo
#include <sstream>                   // Incluye la biblioteca estándar para manejo de cadenas de texto

#define PI 3.141592653589793238462643 // Define el valor de PI para cálculos matemáticos

using namespace std; // Usa el espacio de nombres estándar para evitar escribir std::

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Inicializa el entorno MPI
    int rank, size; // Declara variables para el rango (ID del proceso) y el tamaño (número total de procesos)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtiene el rango del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtiene el número total de procesos

    Kokkos::initialize(argc, argv); // Inicializa el entorno Kokkos para programación paralela
    {
        // Parámetros de la simulación
        const int numParticles = 1000;   // Número total de partículas
        const int numSteps = 1000;       // Número de pasos por partícula
        const double R = 0.001;          // Tamaño del paso en el caminante aleatorio
        const int n_write = 2;           // Frecuencia de escritura de archivos VTK

        // Distribución de partículas entre todos los procesos MPI
        int localParticles = numParticles / size; // Número base de partículas por proceso
        int rest = numParticles % size; // Partículas restantes para distribuir entre los primeros procesos
        if (rank < rest) localParticles++; // Ajusta el número de partículas para los primeros procesos

        int start = rank * (numParticles / size) + min(rank, rest); // Índice de inicio de las partículas locales
        int end = start + localParticles; // Índice de fin de las partículas locales

        // Vista de Kokkos para almacenar las posiciones de las partículas locales
        Kokkos::View<double**> positions("positions", localParticles, 3);
        // Vista de Kokkos para almacenar todas las posiciones en el proceso 0
        Kokkos::View<double**> La_Gloriosa("La_Gloriosa", numParticles, 3);

        // Pool de generadores de números aleatorios en Kokkos
        Kokkos::Random_XorShift64_Pool<> random_pool(12345 + rank);

        // Inicialización de las posiciones de las partículas locales
        Kokkos::parallel_for("InitializeParticles", Kokkos::RangePolicy<>(0, localParticles), KOKKOS_LAMBDA(const int i) {
            positions(i, 0) = 0.0; // Inicializa la coordenada x
            positions(i, 1) = 0.0; // Inicializa la coordenada y
            positions(i, 2) = 0.0; // Inicializa la coordenada z
        });

        // Bucle principal de la simulación
        for (int step = 0; step < numSteps; step++) {
            // Actualiza las posiciones de las partículas locales usando un caminante aleatorio
            Kokkos::parallel_for("RandomWalk", Kokkos::RangePolicy<>(0, localParticles), KOKKOS_LAMBDA(const int i) {
                auto rand_gen = random_pool.get_state(); // Obtiene un generador de números aleatorios
                double theta = PI * rand_gen.drand(); // Genera un ángulo theta aleatorio
                double phi = 2 * PI * rand_gen.drand(); // Genera un ángulo phi aleatorio

                // Actualiza las coordenadas de la partícula
                positions(i, 0) += R * sin(theta) * cos(phi); // Actualiza la coordenada x
                positions(i, 1) += R * sin(theta) * sin(phi); // Actualiza la coordenada y
                positions(i, 2) += R * cos(theta); // Actualiza la coordenada z

                random_pool.free_state(rand_gen); // Libera el generador de números aleatorios
            });

            // Recolectar datos en el proceso 0 cada `n_write` pasos
            if (step % n_write == 0) {
                std::vector<int> counts(size); // Vector para almacenar el número de partículas por proceso
                std::vector<int> displs(size); // Vector para almacenar los desplazamientos en el arreglo global

                // Calcula el número de partículas y los desplazamientos para cada proceso
                for (int i = 0; i < size; i++) {
                    counts[i] = (numParticles / size + (i < rest ? 1 : 0)) * 3; // Número de partículas * 3 (coordenadas)
                    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1]; // Desplazamiento acumulado
                }

                // Recolecta las posiciones de todas las partículas en el proceso 0
                MPI_Gatherv(positions.data(), localParticles * 3, MPI_DOUBLE,
                            La_Gloriosa.data(), counts.data(), displs.data(), MPI_DOUBLE,
                            0, MPI_COMM_WORLD);

                // Solo el proceso 0 escribe los datos en un archivo VTK
                if (rank == 0) {
                    vtkNew<vtkPoints> points; // Crea un objeto para almacenar puntos en 3D
                    for (int i = 0; i < numParticles; ++i) {
                        // Añade las coordenadas de cada partícula al objeto points
                        points->InsertNextPoint(La_Gloriosa(i, 0),
                                               La_Gloriosa(i, 1),
                                               La_Gloriosa(i, 2));
                    }

                    std::ostringstream filename; // Crea un nombre de archivo basado en el paso actual
                    filename << "random_walk_step_" << step << ".vtp";
                    vtkNew<vtkPolyData> polyData; // Crea un objeto para almacenar datos poligonales
                    polyData->SetPoints(points); // Asigna los puntos al objeto polyData
                    vtkNew<vtkXMLPolyDataWriter> writer; // Crea un escritor de archivos VTK en formato XML
                    writer->SetFileName(filename.str().c_str()); // Asigna el nombre del archivo
                    writer->SetInputData(polyData); // Asigna los datos al escritor
                    writer->Write(); // Escribe el archivo VTK
                }
            }
        }
    }

    Kokkos::finalize(); // Finaliza el entorno Kokkos
    MPI_Finalize(); // Finaliza el entorno MPI
    return 0; // Termina el programa
}

// Créditos:
// Este código fue desarrollado por el grupo La Gloriosa y comentado por DeepSeek (Nueva IA China)
// Autor: La Gloriosa Costa Rica
// Fecha: Hoy