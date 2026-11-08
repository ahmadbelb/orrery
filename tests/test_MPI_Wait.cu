#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Wait");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(0, N);
    if (t_rank == 1) {
        MPI_Request rq;
        MPI_Irecv(d, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &rq);
        double g = T_DELTA(orrery_copy_seconds, MPI_Wait(&rq, MPI_STATUS_IGNORE));
        T_CHECK(t_near(g, leg), "Wait charges the H2D leg (%.2f us)", g * 1e6);
        MPI_Request nullr = MPI_REQUEST_NULL;
        g = T_DELTA(orrery_copy_seconds, MPI_Wait(&nullr, MPI_STATUS_IGNORE));
        T_CHECK(t_near(g, 0.0), "Wait on MPI_REQUEST_NULL charges nothing");
    } else {
        MPI_Send(h, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
    }
    return t_finish();
}
