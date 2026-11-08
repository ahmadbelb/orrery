#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstring>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaGetDeviceProperties (REAL)");
    cudaDeviceProp p;
    memset(&p, 0, sizeof(p));
    T_CHECK(cudaGetDeviceProperties(&p, 0) == cudaSuccess, "returns success");
    T_CHECK(p.name[0] != 0 && strcmp(p.name, "SimGrid Virtual GPU") != 0,
            "real card name: '%s'", p.name);
    T_CHECK(p.warpSize == 32, "warp size 32");
    T_CHECK(p.totalGlobalMem > 0, "real memory size %zu", p.totalGlobalMem);
    T_CHECK(p.multiProcessorCount > 0, "real SM count %d", p.multiProcessorCount);
    return t_finish();
}
