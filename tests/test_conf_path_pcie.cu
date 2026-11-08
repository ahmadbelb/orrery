#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: mpi.device_path=pcie");
    const size_t N = 1 << 20;
    double leg = 4e-6 + (double)N / 2.5e9;
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(N, 0), leg), "leg == p2p_pcie cost");
    char* d = nullptr; cudaMalloc((void**)&d, N);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Sendrecv(d, N, MPI_BYTE, 1 - t_rank, 1, d, N, MPI_BYTE, 1 - t_rank, 1,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, 2 * leg), "charged on pcie path (%.2f us)", g * 1e6);
    return t_finish();
}
