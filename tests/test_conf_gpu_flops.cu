#include "orrery_test.h"
#include <cuda_runtime.h>
__global__ void k(char* p) { p[threadIdx.x] = 1; }
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: gpu flops model");
    char* d = nullptr; cudaMalloc((void**)&d, 1024);
    double exp = 1024.0 * 4.0 / 2e9;
    double g = T_DELTA(orrery_compute_seconds, ({ k<<<4, 256>>>(d); cudaDeviceSynchronize(); }));
    T_CHECK(t_near(g, exp), "1024 threads * 4 fpt / 2e9 rate = %.3f us (got %.3f)", exp * 1e6,
            g * 1e6);
    return t_finish();
}
