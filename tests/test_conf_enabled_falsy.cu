#include "orrery_test.h"
int main(int argc, char** argv) {
    t_init(&argc, &argv, "conf: model.enabled=off is falsy");
    T_CHECK(!orrery::model_enabled(), "'off' disables the model");
    T_CHECK(t_near(orrery::transfer_seconds(orrery::TransferPath::H2D, 1 << 20), 0.0),
            "disabled model returns 0 cost");
    return t_finish();
}
