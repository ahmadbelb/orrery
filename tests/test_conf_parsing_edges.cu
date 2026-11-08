#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: parsing edge cases");
    T_CHECK(orrery::model_enabled(), "'on' parsed as true through spacing + trailing comment");
    orrery::LinkCost h2d = orrery::get_cost(orrery::TransferPath::H2D);
    T_CHECK(t_near(h2d.latency, 7e-6) && t_near(h2d.bandwidth, 1e9),
            "indented key + values parsed");
    int n = 0;
    cudaGetDeviceCount(&n);
    T_CHECK(n == 1, "gpu.count=-5 clamped to 1 (got %d)", n);
    T_CHECK(t_near(orrery::transfer_seconds(orrery::TransferPath::D2H, 8), 6e-6 + 8.0),
            "bandwidth=0 falls back to the 1 B/s guard");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(0, 1 << 20), 7e-6 + (double)(1 << 20) / 1e9),
            "staging.share=0 ignored, share stays 1");
    cudaDeviceProp pr;
    cudaGetDeviceProperties(&pr, 0);
    T_CHECK(pr.totalGlobalMem == (size_t)(96.0 * 1024 * 1024 * 1024),
            "gpu.mem_bytes=-3 ignored, default 96 GiB kept");
    return t_finish();
}
