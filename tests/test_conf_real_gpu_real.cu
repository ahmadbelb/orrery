#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstring>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: real_gpu=true (file-driven)");
    cudaDeviceProp p;
    memset(&p, 0, sizeof(p));
    T_CHECK(cudaGetDeviceProperties(&p, 0) == cudaSuccess, "props query ok");
    T_CHECK(p.name[0] && strcmp(p.name, "SimGrid Virtual GPU") != 0,
            "REAL card visible ('%s') -> conf switched the mode", p.name);
    return t_finish();
}
