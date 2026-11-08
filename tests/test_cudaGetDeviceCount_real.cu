#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaGetDeviceCount (REAL)");
    int n = 0;
    T_CHECK(cudaGetDeviceCount(&n) == cudaSuccess && n >= 1, "real driver reports %d device(s)", n);
    return t_finish();
}
