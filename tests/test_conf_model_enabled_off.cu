#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: model.enabled=false");
    T_CHECK(!orrery::model_enabled(), "model reports disabled");
    const size_t N = 1 << 20;
    char* h = (char*)malloc(N);
    char* d = nullptr; cudaMalloc((void**)&d, N);
    h[3] = 33;
    double g = T_DELTA(orrery_copy_seconds, cudaMemcpy(d, h, N, cudaMemcpyHostToDevice));
    T_CHECK(t_near(g, 0.0), "memcpy charges nothing");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(N, N), 0.0), "staging model returns 0");
    g = T_DELTA(orrery_copy_seconds,
                MPI_Sendrecv(d, N, MPI_BYTE, 1 - t_rank, 1, d, N, MPI_BYTE, 1 - t_rank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    T_CHECK(t_near(g, 0.0), "device MPI charges nothing");
    char* h2 = (char*)malloc(N);
    cudaMemcpy(h2, d, N, cudaMemcpyDeviceToHost);
    T_CHECK(h2[3] == 33, "data still moves");
    return t_finish();
}
