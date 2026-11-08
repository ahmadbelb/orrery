#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Allgather");
    const size_t N = 1 << 18;
    const int P = 2;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N); cudaMalloc((void**)&dr, N * P);
    double exp = orrery::mpi_device_staging_seconds(N, N * P);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Allgather(ds, N, MPI_BYTE, dr, N, MPI_BYTE, MPI_COMM_WORLD));
    T_CHECK(t_near(g, exp), "send N, land N*P (%.2f us)", g * 1e6);
    return t_finish();
}
