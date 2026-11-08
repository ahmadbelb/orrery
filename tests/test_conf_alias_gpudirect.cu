#include "orrery_test.h"
#include <cstring>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: alias gpudirect + invalid ignored");
    T_CHECK(orrery::model_enabled(), "'1' parsed as true");
    T_CHECK(strcmp(orrery::mpi_path_name(), "gdr") == 0,
            "'gpudirect' aliases gdr; 'banana' ignored (got %s)", orrery::mpi_path_name());
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 0), 3e-6 + (double)(1 << 20) / 1e9),
            "staging on the gdr leg");
    return t_finish();
}
