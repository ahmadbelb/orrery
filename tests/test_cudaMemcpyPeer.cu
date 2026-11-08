#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMemcpyPeer");
    const size_t N = 1 << 20;
    char *a = nullptr, *b = nullptr;
    cudaMalloc((void**)&a, N); cudaMalloc((void**)&b, N);
    a[5] = 55;
    double exp = orrery::transfer_seconds(orrery::TransferPath::P2P_NVLINK, N);
    double t0 = MPI_Wtime();
    T_CHECK(cudaMemcpyPeer(b, 0, a, 0, N) == cudaSuccess, "returns success");
    double dt = MPI_Wtime() - t0;
    T_CHECK(exp > 0 && t_near(dt, exp), "clock advances by peer.path cost (%.2f us)", dt * 1e6);
    T_CHECK(b[5] == 55, "data moved");
    return t_finish();
}
