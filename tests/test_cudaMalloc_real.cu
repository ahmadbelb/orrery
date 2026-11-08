#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMalloc (REAL, managed)");
    const size_t N = 1 << 20;
    char* d = nullptr;
    T_CHECK(cudaMalloc((void**)&d, N) == cudaSuccess && d, "managed alloc");
    d[0] = 42;
    T_CHECK(d[0] == 42, "host-addressable");
    double leg = orrery::mpi_device_staging_seconds(N, 0);
    if (t_rank == 0) {
        double g = T_DELTA(orrery_copy_seconds, MPI_Send(d, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD));
        T_CHECK(t_near(g, leg), "MPI staging charge == model even in REAL mode (%.2f us)",
                g * 1e6);
    } else {
        char* h = (char*)malloc(N);
        MPI_Recv(h, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        free(h);
    }
    T_CHECK(cudaFree(d) == cudaSuccess, "free");
    return t_finish();
}
