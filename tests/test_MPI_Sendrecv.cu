#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Sendrecv");
    const size_t N = 1 << 20;
    const int peer = 1 - t_rank;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N); cudaMalloc((void**)&dr, N);
    char* h = (char*)malloc(N);
    char* h2 = (char*)malloc(N);
    double both = orrery::mpi_device_staging_seconds(N, N);
    double send_only = orrery::mpi_device_staging_seconds(N, 0);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Sendrecv(ds, N, MPI_BYTE, peer, 1, dr, N, MPI_BYTE, peer, 1,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, both), "dev/dev pays both legs (%.2f us)", g * 1e6);
    g = T_DELTA(orrery_copy_seconds,
                MPI_Sendrecv(ds, N, MPI_BYTE, peer, 2, h, N, MPI_BYTE, peer, 2, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, send_only), "dev-send/host-recv pays send leg only");
    g = T_DELTA(orrery_copy_seconds,
                MPI_Sendrecv(h, N, MPI_BYTE, peer, 3, h2, N, MPI_BYTE, peer, 3, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, 0.0), "host/host pays nothing");
    return t_finish();
}
