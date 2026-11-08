#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
__global__ void fill_ramp(char* p, char v) { p[blockIdx.x * blockDim.x + threadIdx.x] = v; }
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaLaunchKernel (skeleton)");
    const int B = 4, T = 256;
    const size_t N = B * T;
    char* d = nullptr;
    cudaMalloc((void**)&d, N);
    cudaMemset(d, 0, N);
    double exp = (double)(B * T) * 2.0 / 1e9;
    double t0 = MPI_Wtime();
    double got = T_DELTA(orrery_compute_seconds, ({
                             fill_ramp<<<B, T>>>(d, 7);
                             cudaDeviceSynchronize();
                         }));
    double dt = MPI_Wtime() - t0;
    T_CHECK(t_near(got, exp), "compute charge threads*fpt/rate (%.3f us)", got * 1e6);
    T_CHECK(t_near(dt, exp), "clock advances by the charge");
    char* h = (char*)malloc(N);
    cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost);
    int touched = 0;
    for (size_t i = 0; i < N; ++i)
        if (h[i] == 7) ++touched;
    T_CHECK(touched == 0, "kernel did NOT execute (%d bytes touched)", touched);
    return t_finish();
}
