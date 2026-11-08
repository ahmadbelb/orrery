#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Gather");
    const size_t N = 1 << 18;
    const int P = 2;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N); cudaMalloc((void**)&dr, N * P);
    double exp = t_rank == 0 ? orrery::mpi_device_staging_seconds(N, N * P)
                             : orrery::mpi_device_staging_seconds(N, 0);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Gather(ds, N, MPI_BYTE, dr, N, MPI_BYTE, 0, MPI_COMM_WORLD));
    T_CHECK(t_near(g, exp), "%s (%.2f us)", t_rank == 0 ? "root lands N*P" : "leaf sends only",
            g * 1e6);
    return t_finish();
}
