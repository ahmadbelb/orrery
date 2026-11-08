#pragma once

#include <cstddef>

namespace orrery {

enum class TransferPath {
    H2D = 0,
    D2H,
    D2D,
    P2P_NVLINK,
    P2P_PCIE,
    GDR,
    COUNT
};

struct LinkCost {
    double latency;
    double bandwidth;
};

double transfer_seconds(TransferPath path, std::size_t bytes);

void charge_transfer(TransferPath path, std::size_t bytes);

void charge_peer_transfer(std::size_t bytes);

double transfer_seconds_for_kind(int cuda_memcpy_kind, std::size_t bytes);
void charge_transfer_for_kind(int cuda_memcpy_kind, std::size_t bytes);

void charge_mpi_device_staging(std::size_t send_bytes, std::size_t recv_bytes);

double mpi_device_staging_seconds(std::size_t send_bytes, std::size_t recv_bytes);

double exchange_overhead_seconds(std::size_t bytes);
double host_reduce_seconds(std::size_t bytes, int rounds);
double host_selfcopy_seconds(std::size_t bytes);
double bcast_call_overhead_seconds(std::size_t bytes);
double allgather_call_overhead_seconds(std::size_t bytes);
double allreduce_call_overhead_seconds(bool device, std::size_t bytes);

bool op_collective_bounces_to_host();
double coll_cuda_bounce_seconds(std::size_t bytes);
double coll_cuda_reduce_bounce_seconds(std::size_t bytes, bool is_root);
double alltoall_call_overhead_seconds(bool device, std::size_t bytes);
double transfer_seconds_for_kind_hostmem(int kind, std::size_t bytes, bool pinned_host);

const char* mpi_path_name();

LinkCost get_cost(TransferPath path);
void set_cost(TransferPath path, LinkCost cost);
bool model_enabled();

void dump_config();

}
