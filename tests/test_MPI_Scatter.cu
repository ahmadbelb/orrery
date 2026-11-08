#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Scatter");
    const size_t N = 1 << 18;
    const int P = 2;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N * P); cudaMalloc((void**)&dr, N);
    double exp = t_rank == 0 ? orrery::mpi_device_staging_seconds(N * P, N)
                             : orrery::mpi_device_staging_seconds(0, N);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Scatter(ds, N, MPI_BYTE, dr, N, MPI_BYTE, 0, MPI_COMM_WORLD));
    T_CHECK(t_near(g, exp), "%s (%.2f us)", t_rank == 0 ? "root sources N*P" : "leaf lands N",
            g * 1e6);
    return t_finish();
}
