#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPI_Allreduce");
    const size_t N = 1 << 18;
    char *ds = nullptr, *dr = nullptr;
    cudaMalloc((void**)&ds, N); cudaMalloc((void**)&dr, N);
    char* h = (char*)malloc(N);
    double both = orrery::mpi_device_staging_seconds(N, N);
    double g = T_DELTA(orrery_copy_seconds,
                       MPI_Allreduce(ds, dr, N, MPI_BYTE, MPI_MAX, MPI_COMM_WORLD));
    T_CHECK(t_near(g, both), "device: both legs on every rank (%.2f us)", g * 1e6);
    g = T_DELTA(orrery_copy_seconds,
                MPI_Allreduce(MPI_IN_PLACE, dr, N, MPI_BYTE, MPI_MAX, MPI_COMM_WORLD));
    T_CHECK(t_near(g, both), "MPI_IN_PLACE on device: both legs");
    g = T_DELTA(orrery_copy_seconds,
                MPI_Allreduce(MPI_IN_PLACE, h, N, MPI_BYTE, MPI_MAX, MPI_COMM_WORLD));
    T_CHECK(t_near(g, 0.0), "host buffers pay nothing");
    return t_finish();
}
