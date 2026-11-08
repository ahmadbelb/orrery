#include "orrery_test.h"
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: malformed band tables");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(4096, 0), 20e-6 + 4096.0 / 8e9),
            "below both bands: path bandwidth, share 1");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(16384, 0), 20e-6 + 16384.0 / 1e9),
            "16Ki band applies despite junk neighbors");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 0),
                   20e-6 + (double)(1 << 20) * 2.0 / 4e9),
            "unsorted input sorted: 1Mi band + share band 2");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(100, 0), 20e-6 + 100.0 / 8e9),
            "zero-value band '99:0' was dropped, not applied");
    return t_finish();
}
