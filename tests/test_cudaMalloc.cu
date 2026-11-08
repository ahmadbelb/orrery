#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMalloc");
    const size_t N = 1 << 20;
    char* d = nullptr;
    T_CHECK(cudaMalloc((void**)&d, N) == cudaSuccess && d, "allocates");
    d[0] = 42; d[N - 1] = 43;
    T_CHECK(d[0] == 42 && d[N - 1] == 43, "memory usable end to end");
    double leg = orrery::mpi_device_staging_seconds(N, 0);
    double got;
    if (t_rank == 0)
        got = T_DELTA(orrery_copy_seconds, MPI_Send(d, N, MPI_BYTE, 1, 1, MPI_COMM_WORLD));
    else {
        char* h = (char*)malloc(N);
        got = -1; MPI_Recv(h, N, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); free(h);
    }
    if (t_rank == 0)
        T_CHECK(t_near(got, leg), "buffer is device-tracked: MPI pays staging (%.2f us)", got * 1e6);
    T_CHECK(cudaMalloc((void**)&d, 0) == cudaSuccess, "zero-byte alloc succeeds");
    char* d2 = nullptr;
    cudaMalloc((void**)&d2, N);
    T_CHECK(cudaFree(d2) == cudaSuccess, "free succeeds");
    T_CHECK(cudaFree(nullptr) == cudaSuccess, "free(nullptr) is a no-op success");
    return t_finish();
}
