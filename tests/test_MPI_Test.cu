#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Test");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double leg = orrery::mpi_device_staging_seconds(0, N);
    if (t_rank == 1) {
        MPI_Request rq;
        MPI_Irecv(d, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &rq);
        double c0 = orrery_copy_seconds();
        int flag = 0;
        int polls = 0;
        while (!flag) { MPI_Test(&rq, &flag, MPI_STATUS_IGNORE); ++polls; }
        MPI_Test(&rq, &flag, MPI_STATUS_IGNORE);
        MPI_Test(&rq, &flag, MPI_STATUS_IGNORE);
        double g = orrery_copy_seconds() - c0;
        T_CHECK(t_near(g, leg), "charged exactly once over %d polls + 2 extra Tests", polls);
    } else {
        MPI_Send(h, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
    }
    return t_finish();
}
