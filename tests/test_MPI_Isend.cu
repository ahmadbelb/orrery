#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Isend");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(N, 0);
    MPI_Request rq;
    if (t_rank == 0) {
        double g = T_DELTA(orrery_copy_seconds,
                           MPI_Isend(d, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD, &rq));
        T_CHECK(t_near(g, leg), "device Isend charges D2H at post (%.2f us)", g * 1e6);
        g = T_DELTA(orrery_copy_seconds, MPI_Wait(&rq, MPI_STATUS_IGNORE));
        T_CHECK(t_near(g, 0.0), "Wait after Isend charges nothing");
        g = T_DELTA(orrery_copy_seconds, MPI_Isend(h, N, MPI_BYTE, 1, 2, MPI_COMM_WORLD, &rq));
        T_CHECK(t_near(g, 0.0), "host Isend charges nothing");
        MPI_Wait(&rq, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(h, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h, N, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return t_finish();
}
