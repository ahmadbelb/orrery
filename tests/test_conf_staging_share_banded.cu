#include "orrery_test.h"
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: staging.share banded");
    auto leg = [](size_t n, double share) { return 20e-6 + (double)n * share / 2e9; };
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(4096, 0), leg(4096, 1.0)),
            "below 64Ki: share 1");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(65536, 0), leg(65536, 3.0)),
            "boundary 64Ki inclusive: share 3");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 0), leg(1 << 20, 3.0)),
            "above: share 3");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(65535, 0), leg(65535, 1.0)),
            "one byte below boundary: share 1");
    return t_finish();
}
