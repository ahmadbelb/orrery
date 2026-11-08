#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstdlib>
static double WIRE(size_t n) { return 100e-6 + (double)n / 1e9; }
int main(int argc, char** argv) {
    t_init(&argc, &argv, "end-to-end wire timing between ranks");
    const int peer = 1 - t_rank;
    const size_t N = 1 << 20, S = 64;
    char* h = (char*)malloc(N);
    char* h2 = (char*)malloc(N);
    char* d = nullptr; cudaMalloc((void**)&d, N);
    double leg_s = orrery::mpi_device_staging_seconds(N, 0);
    double leg_r = orrery::mpi_device_staging_seconds(0, N);

    for (int pass = 0; pass < 2; ++pass) {
        size_t n = pass == 0 ? N : S;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        if (t_rank == 0) {
            MPI_Send(h, n, MPI_BYTE, peer, 1, MPI_COMM_WORLD);
            MPI_Recv(h, n, MPI_BYTE, peer, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double half = (MPI_Wtime() - t0) / 2.0;
            T_CHECK(t_close(half, WIRE(n)), "host ping-pong %zuB: RTT/2 == wire (%.2f us)", n,
                    half * 1e6);
        } else {
            MPI_Recv(h, n, MPI_BYTE, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(h, n, MPI_BYTE, peer, 2, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double c0 = orrery_copy_seconds(), n0 = orrery_net_seconds();
    double t0 = MPI_Wtime();
    if (t_rank == 0) {
        MPI_Send(d, N, MPI_BYTE, peer, 3, MPI_COMM_WORLD);
        MPI_Recv(d, N, MPI_BYTE, peer, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double rtt = MPI_Wtime() - t0;
        double expect = leg_s + WIRE(N) + leg_r;
        T_CHECK(t_close(rtt / 2.0, expect), "device ping-pong: RTT/2 == D2H+wire+H2D (%.2f us)",
                rtt / 2.0 * 1e6);
        double dcopy = orrery_copy_seconds() - c0, dnet = orrery_net_seconds() - n0;
        T_CHECK(t_close(dcopy, leg_s + leg_r), "copy bucket == the two staging legs");
        T_CHECK(t_close(dcopy + dnet, rtt), "copy + net buckets == the whole RTT");
    } else {
        MPI_Recv(d, N, MPI_BYTE, peer, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(d, N, MPI_BYTE, peer, 4, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    MPI_Sendrecv(h, N, MPI_BYTE, peer, 5, h2, N, MPI_BYTE, peer, 5, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    double dt = MPI_Wtime() - t0;
    T_CHECK(dt >= WIRE(N) - 1e-9 && dt <= 2 * WIRE(N) + 1e-9,
            "host Sendrecv within [1x, 2x] wire (%.2f us; link sharing dependent)", dt * 1e6);
    return t_finish();
}
