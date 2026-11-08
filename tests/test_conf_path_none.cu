#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: mpi.device_path=none");
    const size_t N = 1 << 20;
    char* d = nullptr; cudaMalloc((void**)&d, N);
    char* h = (char*)malloc(N);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Sendrecv(d, N, MPI_BYTE, 1 - t_rank, 1, d, N, MPI_BYTE, 1 - t_rank, 1,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, 0.0), "device MPI pays no staging");
    g = T_DELTA(orrery_copy_seconds, cudaMemcpy(d, h, N, cudaMemcpyHostToDevice));
    T_CHECK(t_near(g, 10e-6 + N / 1e9), "explicit H2D still charged (lat 10us + N/1e9)");
    return t_finish();
}
