#include "orrery_test.h"
extern "C" int MPIX_Query_cuda_support(void);
int main(int argc, char** argv) {
    t_init(&argc, &argv, "MPIX_Query_cuda_support");
    T_CHECK(MPIX_Query_cuda_support() == 1, "reports CUDA-aware");
    return t_finish();
}
