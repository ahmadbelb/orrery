#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: gpu.count=3, gpu.mem_bytes=1e9");
    int n = 0;
    T_CHECK(cudaGetDeviceCount(&n) == cudaSuccess && n == 3, "3 devices visible, got %d", n);
    T_CHECK(cudaSetDevice(2) == cudaSuccess, "device 2 valid");
    T_CHECK(cudaSetDevice(3) != cudaSuccess, "device 3 rejected");
    cudaDeviceProp p;
    T_CHECK(cudaGetDeviceProperties(&p, 2) == cudaSuccess && p.totalGlobalMem == (size_t)1e9,
            "props on device 2 report 1e9 memory");
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    T_CHECK(t == (size_t)1e9 && f <= t, "memGetInfo matches conf");
    return t_finish();
}
