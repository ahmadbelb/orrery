#include "orrery_test.h"
int main(int argc, char** argv) {
    t_init(&argc, &argv, "env: ORRERY_TRANSFER_MODEL=0 wins over conf");
    T_CHECK(!orrery::model_enabled(), "model disabled by env");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 1 << 20), 0.0), "no staging");
    return t_finish();
}
