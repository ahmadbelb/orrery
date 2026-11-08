#include "orrery_test.h"
#include <cuda_runtime.h>
extern "C" int cuDeviceGetCount(int*);
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaGetDeviceCount");
    int n = -1;
    T_CHECK(cudaGetDeviceCount(&n) == cudaSuccess, "returns success");
    T_CHECK(n == 1, "reports gpu.count from the conf (1), got %d", n);
    int dn = -1;
    T_CHECK(cuDeviceGetCount(&dn) == 0 && dn == n, "driver API agrees (%d)", dn);
    return t_finish();
}
