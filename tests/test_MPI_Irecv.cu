#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Irecv");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(0, N);
    MPI_Request rq;
    if (t_rank == 1) {
        double gpost = T_DELTA(orrery_copy_seconds,
                               MPI_Irecv(d, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &rq));
        T_CHECK(t_near(gpost, 0.0), "device Irecv charges NOTHING at post");
        double gwait = T_DELTA(orrery_copy_seconds, MPI_Wait(&rq, MPI_STATUS_IGNORE));
        T_CHECK(t_near(gpost + gwait, leg), "post+completion total == one H2D leg");
        MPI_Irecv(h, N, MPI_BYTE, 0, 2, MPI_COMM_WORLD, &rq);
        double g = T_DELTA(orrery_copy_seconds, MPI_Wait(&rq, MPI_STATUS_IGNORE));
        T_CHECK(t_near(g, 0.0), "host Irecv charges nothing at completion either");
    } else {
        MPI_Send(h, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
        MPI_Send(h, N, MPI_BYTE, 1, 2, MPI_COMM_WORLD);
    }
    return t_finish();
}
