#include <mpi.h>
#include <iostream>
#include <cmath>

double f(double x) {
    return std::sin(x);
}

double compute_integral(double a, double b, int n) {
    double h = (b - a) / n;
    double integral = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = a + i * h;
        integral += f(x) * h;
    }
    return integral;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 0.0, b = M_PI;
    int n = 10000;
    double local_a, local_b;
    int local_n = n / size;

    local_a = a + rank * (b - a) / size;
    local_b = a + (rank + 1) * (b - a) / size;

    double local_integral = compute_integral(local_a, local_b, local_n);

    double total_integral = 0.0;
    MPI_Reduce(&local_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Integral from " << a << " to " << b << " is approximately " << total_integral << std::endl;
    }

    MPI_Finalize();
    return 0;
}