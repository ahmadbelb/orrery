#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Send (REAL)");
    const size_t N = 1 << 20;
    char* d = nullptr;
    cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(N, 0);
    if (t_rank == 0) {
        double g = T_DELTA(orrery_copy_seconds, MPI_Send(d, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD));
        T_CHECK(t_near(g, leg), "device Send pays model staging (%.2f us)", g * 1e6);
        g = T_DELTA(orrery_copy_seconds, MPI_Send(h, N, MPI_BYTE, 1, 2, MPI_COMM_WORLD));
        T_CHECK(t_near(g, 0.0), "host Send pays nothing");
    } else {
        MPI_Recv(h, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h, N, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return t_finish();
}
