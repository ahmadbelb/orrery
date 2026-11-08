#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "force_iters default: tiny D2H poked");
    double* d = nullptr;
    cudaMalloc((void**)&d, 64);
    double xd = 7.5;
    cudaMemcpy(d, &xd, 8, cudaMemcpyHostToDevice);
    double out = 0;
    cudaMemcpy(&out, d, 8, cudaMemcpyDeviceToHost);
    T_CHECK(out == 1.0, "8-byte D2H returns 1.0 (double poke), got %g", out);
    float out4 = 0;
    cudaMemcpy(&out4, d, 4, cudaMemcpyDeviceToHost);
    T_CHECK(out4 == 1.0f, "4-byte D2H returns 1.0f (float poke), got %g", (double)out4);
    char big[16];
    cudaMemcpy(big, d, 16, cudaMemcpyDeviceToHost);
    double asd; memcpy(&asd, big, 8);
    T_CHECK(asd == 7.5, "9+ byte D2H NOT poked (got %g)", asd);
    return t_finish();
}
