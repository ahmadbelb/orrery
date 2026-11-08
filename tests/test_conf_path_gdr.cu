#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: mpi.device_path=gdr");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    double leg = 3e-6 + (double)N / 1e9;
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(N, 0), leg), "send leg == gdr alpha-beta");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(N, N), 2 * leg), "both legs == 2x");
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Sendrecv(d, N, MPI_BYTE, 1 - t_rank, 1, d, N, MPI_BYTE, 1 - t_rank, 1,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, 2 * leg), "charged end to end (%.2f us)", g * 1e6);
    return t_finish();
}
