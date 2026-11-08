#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: staging.share=4");
    const size_t N = 1 << 20;
    double d2h_leg = 20e-6 + (double)N * 4.0 / 2e9;
    double h2d_leg = 10e-6 + (double)N * 4.0 / 1e9;
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(N, 0), d2h_leg),
            "send leg: latency unscaled, bandwidth /4");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(0, N), h2d_leg),
            "recv leg: latency unscaled, bandwidth /4");
    return t_finish();
}
