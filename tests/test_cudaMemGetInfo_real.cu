#include "orrery_test.h"
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaMemGetInfo (REAL)");
    size_t freeb = 0, total = 0;
    T_CHECK(cudaMemGetInfo(&freeb, &total) == cudaSuccess, "returns success");
    T_CHECK(total > 0 && freeb > 0 && freeb <= total, "real: free %zu <= total %zu", freeb, total);
    T_CHECK(total != (size_t)8e9, "NOT the synthetic 8e9 value (real card)");
    return t_finish();
}
