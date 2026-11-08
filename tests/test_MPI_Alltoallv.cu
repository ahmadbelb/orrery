#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Alltoallv");
    const size_t A = 1 << 18, B = 1 << 16;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, A + B); cudaMalloc((void**)&dr, A + B);
    int sc[2], sd[2] = {0, 0}, rc[2], rd[2] = {0, 0};
    sc[0] = t_rank == 0 ? (int)A : (int)B;
    sc[1] = t_rank == 0 ? (int)B : (int)A;
    rc[0] = t_rank == 0 ? (int)A : (int)B;
    rc[1] = t_rank == 0 ? (int)B : (int)A;
    sd[1] = sc[0]; rd[1] = rc[0];
    double exp = orrery::mpi_device_staging_seconds(A + B, A + B);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Alltoallv(ds, sc, sd, MPI_BYTE, dr, rc, rd, MPI_BYTE, MPI_COMM_WORLD));
    T_CHECK(t_near(g, exp), "summed asymmetric counts both ways (%.2f us)", g * 1e6);
    return t_finish();
}
