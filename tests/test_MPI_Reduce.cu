#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Reduce");
    const size_t N = 1 << 18;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N); cudaMalloc((void**)&dr, N);
    double exp = t_rank == 0 ? orrery::mpi_device_staging_seconds(N, N)
                             : orrery::mpi_device_staging_seconds(N, 0);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Reduce(ds, dr, N, MPI_BYTE, MPI_MAX, 0, MPI_COMM_WORLD));
    T_CHECK(t_near(g, exp), "%s pays %s (%.2f us)", t_rank == 0 ? "root" : "leaf",
            t_rank == 0 ? "D2H+H2D" : "D2H only", g * 1e6);
    return t_finish();
}
