#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: staging.bw banded");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(4096, 0), 20e-6 + 4096.0 / 8e9),
            "below first band: path bandwidth");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(16384, 0), 20e-6 + 16384.0 / 1e9),
            "band 1 from 16Ki (inclusive)");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 0), 20e-6 + (double)(1 << 20) / 4e9),
            "band 2 from 1Mi");
    const size_t N = 1 << 20;
    char* h = (char*)malloc(N);
    char* d = nullptr; cudaMalloc((void**)&d, N);
    double g = T_DELTA(orrery_copy_seconds, cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost));
    T_CHECK(t_near(g, 20e-6 + N / 8e9), "explicit D2H ignores staging.bw bands");
    return t_finish();
}
