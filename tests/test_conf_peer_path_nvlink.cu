#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: peer.path=nvlink");
    const size_t N = 1 << 20;
    char *a = nullptr, *b = nullptr;
    cudaMalloc((void**)&a, N); cudaMalloc((void**)&b, N);
    double exp = 2e-6 + (double)N / 5e9;
    double t0 = MPI_Wtime();
    cudaMemcpyPeer(b, 0, a, 0, N);
    double dt = MPI_Wtime() - t0;
    T_CHECK(t_near(dt, exp), "peer copy on nvlink ('banana' ignored): %.2f us", dt * 1e6);
    return t_finish();
}
