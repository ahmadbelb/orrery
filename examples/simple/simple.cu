#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <vector>

__global__ void scale(double* a, int n, double f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= f;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size, gpus = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cudaGetDeviceCount(&gpus);
    cudaSetDevice(rank % gpus);

    const int n = 1 << 20;
    const int halo = 1 << 13;
    const int block = 256;
    const int iterations = 100;
    const int next = (rank + 1) % size;
    const int prev = (rank + size - 1) % size;

    double *field, *ghost, *norm;
    cudaMalloc((void**)&field, n * sizeof(double));
    cudaMalloc((void**)&ghost, halo * sizeof(double));
    cudaMalloc((void**)&norm, sizeof(double));

    std::vector<double> initial(n, 1.0);
    cudaMemcpy(field, initial.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int it = 0; it < iterations; ++it) {
        scale<<<n / block, block>>>(field, n, 1.0);
        cudaDeviceSynchronize();
        MPI_Sendrecv(field, halo, MPI_DOUBLE, next, 0, ghost, halo, MPI_DOUBLE, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Allreduce(MPI_IN_PLACE, norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    if (rank == 0)
        printf("%d GPUs, %d iterations: simulated time %.6f s\n", size, iterations, elapsed);

    cudaFree(field);
    cudaFree(ghost);
    cudaFree(norm);
    MPI_Finalize();
    return 0;
}
