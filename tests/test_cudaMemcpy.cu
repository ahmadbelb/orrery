#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMemcpy");
    const size_t N = 1 << 20;
    char* h = (char*)malloc(N); char* h2 = (char*)malloc(N);
    char *d1 = nullptr, *d2 = nullptr;
    cudaMalloc((void**)&d1, N); cudaMalloc((void**)&d2, N);
    struct { int kind; const char* nm; } K[] = {{1, "H2D"}, {2, "D2H"}, {3, "D2D"}};
    for (auto& k : K) {
        double exp = orrery::transfer_seconds_for_kind(k.kind, N);
        void* dst = k.kind == 1 ? (void*)d1 : k.kind == 2 ? (void*)h : (void*)d2;
        const void* src = k.kind == 1 ? (void*)h : (void*)d1;
        double t0 = MPI_Wtime();
        double got = T_DELTA(orrery_copy_seconds,
                             cudaMemcpy(dst, src, N, (cudaMemcpyKind)k.kind));
        double dt = MPI_Wtime() - t0;
        T_CHECK(exp > 0 && t_near(got, exp), "%s bucket charge == model (%.2f us)", k.nm, got * 1e6);
        T_CHECK(t_near(dt, exp), "%s advances sim clock by the charge", k.nm);
    }
    T_CHECK(t_near(T_DELTA(orrery_copy_seconds, cudaMemcpy(h2, h, N, cudaMemcpyHostToHost)), 0.0),
            "H2H charges nothing");
    T_CHECK(t_near(T_DELTA(orrery_copy_seconds, cudaMemcpy(d1, h, 0, cudaMemcpyHostToDevice)), 0.0),
            "zero bytes charges nothing");
    h[7] = 77; cudaMemcpy(d1, h, N, cudaMemcpyHostToDevice);
    cudaMemcpy(h2, d1, N, cudaMemcpyDeviceToHost);
    T_CHECK(h2[7] == 77, "data integrity H2D+D2H");
    return t_finish();
}
