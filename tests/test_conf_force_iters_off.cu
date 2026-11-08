#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: force_iters=false");
    double* d = nullptr;
    cudaMalloc((void**)&d, 64);
    double xd = 7.5;
    cudaMemcpy(d, &xd, 8, cudaMemcpyHostToDevice);
    double out = 0;
    cudaMemcpy(&out, d, 8, cudaMemcpyDeviceToHost);
    T_CHECK(out == 7.5, "tiny D2H preserved (got %g)", out);
    return t_finish();
}
