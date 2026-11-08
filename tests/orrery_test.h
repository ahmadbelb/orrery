#ifndef ORRERY_TEST_H
#define ORRERY_TEST_H
#include "../src/transfer_model.hpp"
#include <cmath>
#include <cstdio>
#include <mpi.h>

extern "C" double orrery_copy_seconds(void);
extern "C" double orrery_net_seconds(void);
extern "C" double orrery_compute_seconds(void);

static int t_fails = 0;
static int t_rank = -1;

static inline bool t_near(double a, double b) {
    return std::fabs(a - b) <= 1e-9 + 1e-9 * std::fabs(b);
}

static inline bool t_close(double a, double b) {
    return std::fabs(a - b) <= 0.5e-6 + 1e-4 * std::fabs(b);
}

#define T_CHECK(cond, ...)                                                     \
    do {                                                                       \
        if (cond) {                                                            \
            printf("  [ok]   r%d ", t_rank);                                   \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        } else {                                                               \
            ++t_fails;                                                         \
            printf("  [FAIL] r%d ", t_rank);                                   \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
        fflush(stdout);                                                        \
    } while (0)

#define T_DELTA(bucket, stmt)                                                  \
    ([&]() -> double {                                                         \
        double c0_ = bucket();                                                 \
        stmt;                                                                  \
        return bucket() - c0_;                                                 \
    }())

static inline void t_init(int* argc, char*** argv, const char* name) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &t_rank);
    if (t_rank == 0) {
        printf("==== %s ====\n", name);
        fflush(stdout);
    }
}

static inline int t_finish(void) {
    int total = 0;
    MPI_Allreduce(&t_fails, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (t_rank == 0) {
        printf("RESULT: %s (%d failing checks)\n", total == 0 ? "PASS" : "FAIL", total);
        fflush(stdout);
    }
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
#endif
