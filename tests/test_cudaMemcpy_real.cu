#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMemcpy (REAL)");
    const size_t N = 1 << 20;
    char* h = (char*)malloc(N);
    char* h2 = (char*)malloc(N);
    char* d = nullptr;
    cudaMalloc((void**)&d, N);
    for (size_t i = 0; i < N; ++i) h[i] = (char)(i * 7);
    double g = T_DELTA(orrery_copy_seconds, ({
                           cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
                           cudaMemcpy(h2, d, N, cudaMemcpyDeviceToHost);
                       }));
    T_CHECK(g > 0, "real wall time booked to the copy bucket (%.1f us)", g * 1e6);
    int bad = 0;
    for (size_t i = 0; i < N; ++i)
        if (h2[i] != (char)(i * 7)) ++bad;
    T_CHECK(bad == 0, "real copy round-trip intact (%d bad)", bad);
    return t_finish();
}
