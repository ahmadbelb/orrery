#include "orrery_test.h"
int main(int argc, char** argv) {
    t_init(&argc, &argv, "env: per-path latency + staging share");
    orrery::LinkCost d2h = orrery::get_cost(orrery::TransferPath::D2H);
    T_CHECK(t_near(d2h.latency, 55e-6), "ORRERY_D2H_LAT=55e-6 applied");
    T_CHECK(t_near(d2h.bandwidth, 2e9), "conf d2h bandwidth kept");
    T_CHECK(t_near(orrery::mpi_device_staging_seconds(1 << 20, 0),
                   55e-6 + (double)(1 << 20) * 2.0 / 2e9),
            "send leg uses env latency AND env share 2");
    return t_finish();
}
