#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Bcast");
    const size_t N = 1 << 18;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double exp = t_rank == 0 ? orrery::mpi_device_staging_seconds(N, 0)
                             : orrery::mpi_device_staging_seconds(0, N);
    double g = T_DELTA(orrery_copy_seconds, MPI_Bcast(d, N, MPI_BYTE, 0, MPI_COMM_WORLD));
    T_CHECK(t_near(g, exp), "%s pays the %s leg (%.2f us)", t_rank == 0 ? "root" : "non-root",
            t_rank == 0 ? "D2H" : "H2D", g * 1e6);
    g = T_DELTA(orrery_copy_seconds, MPI_Bcast(h, N, MPI_BYTE, 0, MPI_COMM_WORLD));
    T_CHECK(t_near(g, 0.0), "host buffer pays nothing");
    return t_finish();
}
