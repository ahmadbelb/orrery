#include "orrery_test.h"
#include <cstring>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: alias staged / truthy yes");
    T_CHECK(orrery::model_enabled(), "'yes' parsed as true");
    T_CHECK(strcmp(orrery::mpi_path_name(), "host_staged") == 0,
            "'staged' aliases host_staged (got %s)", orrery::mpi_path_name());
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 0), 20e-6 + (double)(1 << 20) / 2e9),
            "staging uses D2H leg as host_staged");
    return t_finish();
}
