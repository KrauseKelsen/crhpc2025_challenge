#include <chrono>
#include <iostream>
#include <math.h>

#define PI25DT 3.141592653589793238462643

int main(int argc, char** argv){
    
    int N;

    if (argc < 2) {
      std::cerr << "Usage: " << argv[0]
              << " <N> \n";
      exit(-1);
    } else {
      N = std::atoi(argv[1]);
    }

    double pi, h, sum, x;

    h = 1.0 / (double) N;
    sum = 0.0; 

    std::chrono::duration<double> total_time;
    auto start{std::chrono::steady_clock::now()};

    for (int i = 0; i <= N; i++) {
        x = h*((double)i-0.5);
        sum += 4.0 / (1.0 + x*x);
    }
    pi = h*sum;
    
    total_time += std::chrono::steady_clock::now() - start;

    printf("Total time: %.5f seconds\n", total_time.count());    
    printf("pi is approximately %.16f, Error is %.16f\n",
      pi, fabs(pi - PI25DT));

    return 0;
}