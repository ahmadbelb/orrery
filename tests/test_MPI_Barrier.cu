#include "orrery_test.h"
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Barrier");
    double g = T_DELTA(orrery_copy_seconds, MPI_Barrier(MPI_COMM_WORLD));
    T_CHECK(t_near(g, 0.0), "no staging charge");
    return t_finish();
}
