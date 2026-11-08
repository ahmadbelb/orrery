#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMemGetInfo");
    size_t freeb = 0, total = 0;
    T_CHECK(cudaMemGetInfo(&freeb, &total) == cudaSuccess, "returns success");
    T_CHECK(total == (size_t)8e9, "total == gpu.mem_bytes (8e9), got %zu", total);
    T_CHECK(freeb > 0 && freeb <= total, "0 < free (%zu) <= total", freeb);
    return t_finish();
}
