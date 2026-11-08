#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Alltoall");
    const size_t N = 1 << 18;
    const int P = 2;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N * P); cudaMalloc((void**)&dr, N * P);
    const size_t remote = N * (P - 1);
    double legs = orrery::mpi_device_staging_seconds(remote, remote);
    double self = orrery::transfer_seconds(orrery::TransferPath::D2D, N);
    double t0 = MPI_Wtime();
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Alltoall(ds, N, MPI_BYTE, dr, N, MPI_BYTE, MPI_COMM_WORLD));
    double dt = MPI_Wtime() - t0;
    T_CHECK(t_near(g, legs), "legs on the (P-1)/P that leaves the node, both ways (%.2f us)", g * 1e6);
    T_CHECK(dt >= legs + self - 1e-9, "clock >= legs + D2D self block (%.2f us)", dt * 1e6);
    return t_finish();
}
