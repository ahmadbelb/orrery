#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Waitall");
    const size_t N = 1 << 20;
    char *d1 = nullptr, *d2 = nullptr;
    cudaMalloc((void**)&d1, N); cudaMalloc((void**)&d2, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(0, N);
    if (t_rank == 1) {
        MPI_Request rq[3];
        MPI_Irecv(d1, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &rq[0]);
        MPI_Irecv(h, N, MPI_BYTE, 0, 2, MPI_COMM_WORLD, &rq[1]);
        MPI_Irecv(d2, N, MPI_BYTE, 0, 3, MPI_COMM_WORLD, &rq[2]);
        double g = T_DELTA(orrery_copy_seconds, MPI_Waitall(3, rq, MPI_STATUSES_IGNORE));
        T_CHECK(t_near(g, 2 * leg), "2 device + 1 host recvs -> exactly 2 legs (%.2f us)",
                g * 1e6);
    } else {
        MPI_Send(h, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
        MPI_Send(h, N, MPI_BYTE, 1, 2, MPI_COMM_WORLD);
        MPI_Send(h, N, MPI_BYTE, 1, 3, MPI_COMM_WORLD);
    }
    return t_finish();
}
