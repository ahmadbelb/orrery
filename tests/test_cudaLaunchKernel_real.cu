#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
__global__ void fill_ramp(char* p, char v) { p[blockIdx.x * blockDim.x + threadIdx.x] = v; }
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaLaunchKernel (REAL)");
    const int B = 4, T = 256;
    const size_t N = B * T;
    char* d = nullptr;
    cudaMalloc((void**)&d, N);
    cudaMemset(d, 0, N);
    double g = T_DELTA(orrery_compute_seconds, ({
                           fill_ramp<<<B, T>>>(d, 7);
                           cudaDeviceSynchronize();
                       }));
    T_CHECK(g > 0, "sync books real wall time to the compute bucket (%.1f us)", g * 1e6);
    char* h = (char*)malloc(N);
    cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost);
    int filled = 0;
    for (size_t i = 0; i < N; ++i)
        if (h[i] == 7) ++filled;
    T_CHECK(filled == (int)N, "kernel EXECUTED: %d/%zu bytes filled", filled, N);
    return t_finish();
}
