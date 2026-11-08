#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMallocHost");
    const size_t N = 1 << 20;
    char *hp = nullptr, *ha = nullptr;
    T_CHECK(cudaMallocHost((void**)&hp, N) == cudaSuccess && hp, "cudaMallocHost allocates");
    T_CHECK(cudaHostAlloc((void**)&ha, N, 0) == cudaSuccess && ha, "cudaHostAlloc allocates");
    hp[0] = 1; ha[0] = 2;
    double got;
    if (t_rank == 0)
        got = T_DELTA(orrery_copy_seconds, MPI_Send(hp, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD));
    else {
        got = -1; MPI_Recv(ha, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (t_rank == 0)
        T_CHECK(t_near(got, 0.0), "pinned host buffer pays NO staging (%.3f us)", got * 1e6);
    else
        T_CHECK(t_near(T_DELTA(orrery_copy_seconds, (void)0), 0.0), "receiver paid nothing either");
    T_CHECK(cudaFreeHost(hp) == cudaSuccess && cudaFreeHost(ha) == cudaSuccess, "freeHost ok");
    return t_finish();
}
