#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Recv");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(0, N);
    if (t_rank == 1) {
        double g1 = T_DELTA(orrery_copy_seconds,
                            MPI_Recv(d, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        T_CHECK(t_near(g1, leg), "device Recv pays H2D leg (%.2f us)", g1 * 1e6);
        double g2 = T_DELTA(orrery_copy_seconds,
                            MPI_Recv(h, N, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        T_CHECK(t_near(g2, 0.0), "host Recv pays nothing");
    } else {
        MPI_Send(h, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
        MPI_Send(h, N, MPI_BYTE, 1, 2, MPI_COMM_WORLD);
    }
    return t_finish();
}
