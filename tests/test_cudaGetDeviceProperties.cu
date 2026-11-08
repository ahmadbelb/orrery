#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstring>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaGetDeviceProperties");
    cudaDeviceProp p;
    memset(&p, 0x5a, sizeof(p));
    T_CHECK(cudaGetDeviceProperties(&p, 0) == cudaSuccess, "returns success");
    T_CHECK(strcmp(p.name, "SimGrid Virtual GPU") == 0, "name '%s'", p.name);
    T_CHECK(p.warpSize == 32, "warp size 32, got %d", p.warpSize);
    T_CHECK(p.major == 9 && p.minor == 0, "compute capability %d.%d", p.major, p.minor);
    T_CHECK(p.multiProcessorCount > 0 && p.multiProcessorCount < 1000, "SM count %d sane",
            p.multiProcessorCount);
    T_CHECK(p.maxThreadsPerBlock == 1024, "max threads/blk %d", p.maxThreadsPerBlock);
    T_CHECK(p.totalGlobalMem == (size_t)8e9, "totalGlobalMem == gpu.mem_bytes, got %zu",
            p.totalGlobalMem);
    T_CHECK(p.l2CacheSize > 0, "L2 %d > 0", p.l2CacheSize);
    T_CHECK(p.memoryBusWidth > 0 && p.memoryClockRate > 0, "memory bus/clock positive");
    T_CHECK(p.asyncEngineCount >= 0 && p.asyncEngineCount < 16, "async engines %d sane",
            p.asyncEngineCount);
    T_CHECK(p.sharedMemPerBlock > 0 && p.sharedMemPerMultiprocessor >= p.sharedMemPerBlock,
            "shared memory sizes coherent");
    cudaDeviceProp bad;
    T_CHECK(cudaGetDeviceProperties(&bad, 3) != cudaSuccess, "invalid device rejected");
    char peer_name[256], peer_uuid[16];
    MPI_Sendrecv(p.name, 256, MPI_CHAR, 1 - t_rank, 1, peer_name, 256, MPI_CHAR, 1 - t_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(p.uuid.bytes, 16, MPI_CHAR, 1 - t_rank, 2, peer_uuid, 16, MPI_CHAR, 1 - t_rank,
                 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    T_CHECK(strcmp(p.name, peer_name) == 0, "model name identical on both nodes");
    T_CHECK(memcmp(p.uuid.bytes, peer_uuid, 16) != 0, "UUID differs between nodes");
    cudaDeviceProp q;
    cudaGetDeviceProperties(&q, 0);
    T_CHECK(memcmp(p.uuid.bytes, q.uuid.bytes, 16) == 0, "UUID stable across queries");
    return t_finish();
}
