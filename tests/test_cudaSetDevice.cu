#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaSetDevice");
    T_CHECK(cudaSetDevice(0) == cudaSuccess, "device 0 accepted");
    T_CHECK(cudaSetDevice(7) != cudaSuccess, "device 7 rejected (1 device modeled)");
    T_CHECK(cudaSetDevice(-1) != cudaSuccess, "negative device rejected");
    int d = -1;
    T_CHECK(cudaGetDevice(&d) == cudaSuccess && d == 0, "cudaGetDevice reports 0");
    return t_finish();
}
