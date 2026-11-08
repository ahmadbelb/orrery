#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: per-path alpha-beta");
    using orrery::TransferPath;
    struct { TransferPath p; double lat, bw; const char* nm; } C[] = {
        {TransferPath::H2D, 1e-6, 1e9, "h2d"},       {TransferPath::D2H, 2e-6, 2e9, "d2h"},
        {TransferPath::D2D, 3e-6, 3e9, "d2d"},       {TransferPath::P2P_NVLINK, 4e-6, 4e9, "nvl"},
        {TransferPath::P2P_PCIE, 5e-6, 5e9, "pcie"}, {TransferPath::GDR, 6e-6, 6e9, "gdr"}};
    for (auto& c : C) {
        orrery::LinkCost lc = orrery::get_cost(c.p);
        T_CHECK(t_near(lc.latency, c.lat) && t_near(lc.bandwidth, c.bw),
                "%s parsed: %.0fus / %.0fGB/s", c.nm, lc.latency * 1e6, lc.bandwidth / 1e9);
    }
    const size_t N = 1 << 20;
    char* h = (char*)malloc(N);
    char *d1 = nullptr, *d2 = nullptr;
    cudaMalloc((void**)&d1, N); cudaMalloc((void**)&d2, N);
    double g = T_DELTA(orrery_copy_seconds, cudaMemcpy(d1, h, N, cudaMemcpyHostToDevice));
    T_CHECK(t_near(g, 1e-6 + N / 1e9), "H2D charge from conf");
    g = T_DELTA(orrery_copy_seconds, cudaMemcpy(h, d1, N, cudaMemcpyDeviceToHost));
    T_CHECK(t_near(g, 2e-6 + N / 2e9), "D2H charge from conf");
    g = T_DELTA(orrery_copy_seconds, cudaMemcpy(d2, d1, N, cudaMemcpyDeviceToDevice));
    T_CHECK(t_near(g, 3e-6 + N / 3e9), "D2D charge from conf");
    return t_finish();
}
