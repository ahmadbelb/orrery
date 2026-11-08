#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf vs env: env wins");
    orrery::LinkCost h2d = orrery::get_cost(orrery::TransferPath::H2D);
    T_CHECK(t_near(h2d.bandwidth, 4e9), "ORRERY_H2D_BW=4e9 overrode conf's 1e9");
    T_CHECK(t_near(h2d.latency, 10e-6), "conf latency kept where env is silent");
    const size_t N = 1 << 20;
    char* h = (char*)malloc(N);
    char* d = nullptr; cudaMalloc((void**)&d, N);
    double g = T_DELTA(orrery_copy_seconds, cudaMemcpy(d, h, N, cudaMemcpyHostToDevice));
    T_CHECK(t_near(g, 10e-6 + N / 4e9), "H2D charged with env bandwidth");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(N, N), 0.0),
            "ORRERY_MPI_DEVICE_PATH=none overrode conf's host_staged");
    return t_finish();
}
