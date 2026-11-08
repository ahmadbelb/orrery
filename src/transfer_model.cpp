#include "transfer_model.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cstdarg>
#include <cstdio>

extern "C" void smpi_execute_benched(double duration);

static void xtrace(const char* fmt, ...) {
    static int on = -1;
    if (on < 0)
        on = std::getenv("ORRERY_TRACE") ? 1 : 0;
    if (!on)
        return;
    std::fprintf(stderr, "[trace shim] ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

namespace orrery {
namespace {

enum class MpiDevicePath { HOST_STAGED, GDR, NVLINK, PCIE, NONE };

struct Model {
    bool enabled = true;
    std::array<LinkCost, static_cast<std::size_t>(TransferPath::COUNT)> cost;
    MpiDevicePath mpi_path = MpiDevicePath::HOST_STAGED;
    bool op_coll_host_bounce = false;
    TransferPath peer_path = TransferPath::P2P_NVLINK;
    double staging_share = 1.0;
    std::vector<std::pair<std::size_t, double>> staging_bw_segments;
    std::vector<std::pair<std::size_t, double>> staging_share_segments;
    std::vector<std::pair<std::size_t, double>> exchange_overhead_segments;
    std::vector<std::pair<std::size_t, double>> bcast_call_segments;
    std::vector<std::pair<std::size_t, double>> allgather_call_segments;
    std::vector<std::pair<std::size_t, double>> nvlink_bw_segments;
    double direct_intra_latency = -1.0;
    double direct_inter_latency = -1.0;
    double op_bw = 0.0;
    double memcpy_bw = 0.0;
    std::vector<std::pair<std::size_t, double>> ar_call_host_segments;
    std::vector<std::pair<std::size_t, double>> ar_call_dev_segments;
    std::vector<std::pair<std::size_t, double>> a2a_call_host_segments;
    std::vector<std::pair<std::size_t, double>> a2a_call_dev_segments;
    LinkCost pageable_h2d{0.0, 0.0};
    LinkCost pageable_d2h{0.0, 0.0};
    std::vector<std::pair<std::size_t, double>> pageable_h2d_bw_segments;
    std::vector<std::pair<std::size_t, double>> pageable_d2h_bw_segments;
    std::string source = "built-in defaults";

    Model() {
        cost[idx(TransferPath::H2D)] = {1.2e-6, 24.0e9};
        cost[idx(TransferPath::D2H)] = {1.2e-6, 24.0e9};
        cost[idx(TransferPath::D2D)] = {1.0e-6, 1500.0e9};
        cost[idx(TransferPath::P2P_NVLINK)] = {1.5e-6, 50.0e9};
        cost[idx(TransferPath::P2P_PCIE)] = {1.8e-6, 22.0e9};
        cost[idx(TransferPath::GDR)] = {1.0e-6, 1.0e12};
    }

    static std::size_t idx(TransferPath p) {
        return static_cast<std::size_t>(p);
    }
};

std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool path_from_name(const std::string& name, TransferPath& out) {
    if (name == "h2d") {
        out = TransferPath::H2D;
        return true;
    }
    if (name == "d2h") {
        out = TransferPath::D2H;
        return true;
    }
    if (name == "d2d") {
        out = TransferPath::D2D;
        return true;
    }
    if (name == "p2p_nvlink") {
        out = TransferPath::P2P_NVLINK;
        return true;
    }
    if (name == "p2p_pcie") {
        out = TransferPath::P2P_PCIE;
        return true;
    }
    if (name == "gdr") {
        out = TransferPath::GDR;
        return true;
    }
    return false;
}

bool mpi_path_from_str(const std::string& s, MpiDevicePath& out) {
    if (s == "host_staged" || s == "staged") {
        out = MpiDevicePath::HOST_STAGED;
        return true;
    }
    if (s == "gdr" || s == "gpudirect" || s == "direct") {
        out = MpiDevicePath::GDR;
        return true;
    }
    if (s == "nvlink") {
        out = MpiDevicePath::NVLINK;
        return true;
    }
    if (s == "pcie") {
        out = MpiDevicePath::PCIE;
        return true;
    }
    if (s == "none" || s == "off") {
        out = MpiDevicePath::NONE;
        return true;
    }
    return false;
}

bool truthy(const std::string& v) {
    std::string s = lower(v);
    return s == "true" || s == "1" || s == "on" || s == "yes";
}

void parse_segments(const std::string& v, std::vector<std::pair<std::size_t, double>>& out) {
    out.clear();
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ';')) {
        const std::size_t colon = tok.find(':');
        if (colon == std::string::npos)
            continue;
        const std::size_t thr = std::strtoull(tok.c_str(), nullptr, 10);
        const double val = std::strtod(tok.c_str() + colon + 1, nullptr);
        if (val > 0.0)
            out.emplace_back(thr, val);
    }
    std::sort(out.begin(), out.end());
}

void apply_kv(Model& m, const std::string& raw_key, const std::string& val) {
    const std::string key = lower(trim(raw_key));
    const std::string v = trim(val);
    if (key.empty() || v.empty())
        return;

    if (key == "model.enabled") {
        m.enabled = truthy(v);
        return;
    }
    if (key == "mpi.op_collective_path") {
        m.op_coll_host_bounce = (lower(v) == "coll_cuda" || lower(v) == "host_bounce");
        return;
    }
    if (key == "mpi.device_path") {
        mpi_path_from_str(lower(v), m.mpi_path);
        return;
    }
    if (key == "staging.share") {
        if (v.find(':') != std::string::npos) {
            parse_segments(v, m.staging_share_segments);
        } else {
            const double s = std::strtod(v.c_str(), nullptr);
            if (s > 0.0)
                m.staging_share = s;
        }
        return;
    }
    if (key == "staging.bw") {
        parse_segments(v, m.staging_bw_segments);
        return;
    }
    if (key == "bcast.call_overhead") {
        parse_segments(v, m.bcast_call_segments);
        return;
    }
    if (key == "allgather.call_overhead") {
        parse_segments(v, m.allgather_call_segments);
        return;
    }
    if (key == "p2p_nvlink.bw") {
        parse_segments(v, m.nvlink_bw_segments);
        return;
    }
    if (key == "direct.intra.latency") { m.direct_intra_latency = atof(v.c_str()); return; }
    if (key == "direct.inter.latency") { m.direct_inter_latency = atof(v.c_str()); return; }
    if (key == "exchange.overhead") {
        parse_segments(v, m.exchange_overhead_segments);
        return;
    }
    if (key == "op.bw") {
        const double d = std::strtod(v.c_str(), nullptr);
        if (d > 0.0)
            m.op_bw = d;
        return;
    }
    if (key == "memcpy.bw") {
        const double d = std::strtod(v.c_str(), nullptr);
        if (d > 0.0)
            m.memcpy_bw = d;
        return;
    }
    if (key == "allreduce.call_overhead") {
        if (v.find(':') != std::string::npos) {
            parse_segments(v, m.ar_call_host_segments);
        } else {
            const double d = std::strtod(v.c_str(), nullptr);
            if (d >= 0.0)
                m.ar_call_host_segments = {{0, d}};
        }
        return;
    }
    if (key == "allreduce.call_overhead.dev") {
        parse_segments(v, m.ar_call_dev_segments);
        return;
    }
    if (key == "alltoall.call_overhead") {
        parse_segments(v, m.a2a_call_host_segments);
        return;
    }
    if (key == "alltoall.call_overhead.dev") {
        parse_segments(v, m.a2a_call_dev_segments);
        return;
    }
    if (key == "h2d.pageable.latency")   { m.pageable_h2d.latency = std::strtod(v.c_str(), nullptr); return; }
    if (key == "h2d.pageable.bandwidth") { m.pageable_h2d.bandwidth = std::strtod(v.c_str(), nullptr); return; }
    if (key == "d2h.pageable.latency")   { m.pageable_d2h.latency = std::strtod(v.c_str(), nullptr); return; }
    if (key == "d2h.pageable.bandwidth") { m.pageable_d2h.bandwidth = std::strtod(v.c_str(), nullptr); return; }
    if (key == "h2d.pageable.bw") { parse_segments(v, m.pageable_h2d_bw_segments); return; }
    if (key == "d2h.pageable.bw") { parse_segments(v, m.pageable_d2h_bw_segments); return; }
    if (key == "peer.path") {
        TransferPath p;
        if (path_from_name("p2p_" + lower(v), p) || path_from_name(lower(v), p))
            m.peer_path = p;
        return;
    }

    const std::size_t dot = key.find('.');
    if (dot == std::string::npos)
        return;
    const std::string path_name = key.substr(0, dot);
    const std::string param = key.substr(dot + 1);
    TransferPath p;
    if (!path_from_name(path_name, p))
        return;
    const double num = std::strtod(v.c_str(), nullptr);
    if (param == "latency")
        m.cost[Model::idx(p)].latency = num;
    else if (param == "bandwidth")
        m.cost[Model::idx(p)].bandwidth = num;
}

void load_from_file(Model& m, const std::string& path) {
    std::ifstream f(path);
    if (!f)
        return;
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        apply_kv(m, line.substr(0, eq), line.substr(eq + 1));
    }
    m.source = path;
}

void apply_env(Model& m) {
    if (const char* e = std::getenv("ORRERY_TRANSFER_MODEL"))
        m.enabled = truthy(e);
    if (const char* e = std::getenv("ORRERY_MPI_DEVICE_PATH"))
        mpi_path_from_str(lower(trim(e)), m.mpi_path);
    if (const char* e = std::getenv("ORRERY_STAGING_SHARE"))
        apply_kv(m, "staging.share", e);
    if (const char* e = std::getenv("ORRERY_STAGING_BW"))
        apply_kv(m, "staging.bw", e);
    static const char* ENV[] = {"H2D", "D2H", "D2D", "P2P_NVLINK", "P2P_PCIE", "GDR"};
    static const char* KEY[] = {"h2d", "d2h", "d2d", "p2p_nvlink", "p2p_pcie", "gdr"};
    for (int i = 0; i < 6; ++i) {
        if (const char* e = std::getenv((std::string("ORRERY_") + ENV[i] + "_BW").c_str()))
            apply_kv(m, std::string(KEY[i]) + ".bandwidth", e);
        if (const char* e = std::getenv((std::string("ORRERY_") + ENV[i] + "_LAT").c_str()))
            apply_kv(m, std::string(KEY[i]) + ".latency", e);
    }
}

Model& model() {
    static Model m = [] {
        Model mm;
        std::string path;
        bool explicit_cfg = false;
        if (const char* cfg = std::getenv("ORRERY_TRANSFER_CONFIG")) {
            path = cfg;
            explicit_cfg = true;
        }
        if (path.empty()) {
            for (const char* cand : {"transfers.conf", "simgrid/transfers.conf"}) {
                std::ifstream t(cand);
                if (t) {
                    path = cand;
                    break;
                }
            }
        }
        if (!path.empty()) {
            std::ifstream probe(path);
            if (!probe) {
                std::cerr << "[orrery] FATAL: ORRERY_TRANSFER_CONFIG='" << path
                          << "' cannot be opened (cwd-relative path under smpirun?)\n";
                if (explicit_cfg)
                    std::abort();
            }
            load_from_file(mm, path);
        }
        apply_env(mm);
        std::cerr << "[orrery] model: source=" << (mm.source.empty() ? "(builtin defaults)" : mm.source)
                  << " enabled=" << (mm.enabled ? 1 : 0) << "\n";
        return mm;
    }();
    return m;
}

const char* path_label(TransferPath p) {
    switch (p) {
    case TransferPath::H2D: return "H2D";
    case TransferPath::D2H: return "D2H";
    case TransferPath::D2D: return "D2D";
    case TransferPath::P2P_NVLINK: return "P2P_NVLINK";
    case TransferPath::P2P_PCIE: return "P2P_PCIE";
    case TransferPath::GDR: return "GDR";
    default: return "?";
    }
}

const char* mpi_path_label(MpiDevicePath p) {
    switch (p) {
    case MpiDevicePath::HOST_STAGED: return "host_staged (D2H+H2D)";
    case MpiDevicePath::GDR: return "gdr";
    case MpiDevicePath::NVLINK: return "nvlink";
    case MpiDevicePath::PCIE: return "pcie";
    case MpiDevicePath::NONE: return "none (network only)";
    default: return "?";
    }
}

}

double transfer_seconds(TransferPath path, std::size_t bytes) {
    Model& m = model();
    if (!m.enabled || bytes == 0)
        return 0.0;
    const LinkCost& c = m.cost[Model::idx(path)];
    const double bw = c.bandwidth > 0.0 ? c.bandwidth : 1.0;
    return c.latency + static_cast<double>(bytes) / bw;
}

void charge_transfer(TransferPath path, std::size_t bytes) {
    const double t = transfer_seconds(path, bytes);
    if (t > 0.0)
        smpi_execute_benched(t);
    xtrace("cudaMemcpy %s %zuB -> charge %.2f us", path_label(path), bytes, t * 1e6);
}

void charge_peer_transfer(std::size_t bytes) {
    charge_transfer(model().peer_path, bytes);
}

static bool path_for_kind(int k, TransferPath& out) {
    switch (k) {
    case 1: out = TransferPath::H2D; return true;
    case 2: out = TransferPath::D2H; return true;
    case 3: out = TransferPath::D2D; return true;
    default: return false;
    }
}
double transfer_seconds_for_kind(int kind, std::size_t bytes) {
    TransferPath p;
    return path_for_kind(kind, p) ? transfer_seconds(p, bytes) : 0.0;
}

double transfer_seconds_for_kind_hostmem(int kind, std::size_t bytes, bool pinned_host) {
    if (pinned_host || (kind != 1 && kind != 2))
        return transfer_seconds_for_kind(kind, bytes);
    Model& m = model();
    if (!m.enabled || bytes == 0)
        return 0.0;
    const LinkCost& c = (kind == 1) ? m.pageable_h2d : m.pageable_d2h;
    if (c.bandwidth <= 0.0)
        return transfer_seconds_for_kind(kind, bytes);
    double bw = c.bandwidth;
    const auto& segs = (kind == 1) ? m.pageable_h2d_bw_segments : m.pageable_d2h_bw_segments;
    for (const auto& seg : segs)
        if (seg.first <= bytes)
            bw = seg.second;
    return c.latency + static_cast<double>(bytes) / bw;
}
void charge_transfer_for_kind(int kind, std::size_t bytes) {
    TransferPath p;
    if (path_for_kind(kind, p))
        charge_transfer(p, bytes);
}

static double staging_leg_seconds(TransferPath path, std::size_t bytes) {
    Model& m = model();
    if (bytes == 0)
        return 0.0;
    const LinkCost& c = m.cost[Model::idx(path)];
    double bw = c.bandwidth > 0.0 ? c.bandwidth : 1.0;
    double share = 1.0;
    const bool host_staging_leg =
        (path == TransferPath::D2H || path == TransferPath::H2D);
    if (host_staging_leg) {
        for (const auto& seg : m.staging_bw_segments)
            if (seg.first <= bytes)
                bw = seg.second;
        share = m.staging_share;
        for (const auto& seg : m.staging_share_segments)
            if (seg.first <= bytes)
                share = seg.second;
    } else if (path == TransferPath::P2P_NVLINK) {
        for (const auto& seg : m.nvlink_bw_segments)
            if (seg.first <= bytes)
                bw = seg.second;
    }
    return c.latency + static_cast<double>(bytes) * share / bw;
}

double mpi_device_staging_seconds(std::size_t send_bytes, std::size_t recv_bytes) {
    Model& m = model();
    if (!m.enabled)
        return 0.0;
    switch (m.mpi_path) {
    case MpiDevicePath::NONE: return 0.0;
    case MpiDevicePath::HOST_STAGED:
        return staging_leg_seconds(TransferPath::D2H, send_bytes) +
               staging_leg_seconds(TransferPath::H2D, recv_bytes);
    case MpiDevicePath::GDR:
        return staging_leg_seconds(TransferPath::GDR, send_bytes) +
               staging_leg_seconds(TransferPath::GDR, recv_bytes);
    case MpiDevicePath::NVLINK:
        return staging_leg_seconds(TransferPath::P2P_NVLINK, send_bytes) +
               staging_leg_seconds(TransferPath::P2P_NVLINK, recv_bytes);
    case MpiDevicePath::PCIE:
        return staging_leg_seconds(TransferPath::P2P_PCIE, send_bytes) +
               staging_leg_seconds(TransferPath::P2P_PCIE, recv_bytes);
    }
    return 0.0;
}

double exchange_overhead_seconds(std::size_t bytes) {
    Model& m = model();
    if (!m.enabled || m.exchange_overhead_segments.empty())
        return 0.0;
    double s = 0.0;
    for (const auto& seg : m.exchange_overhead_segments)
        if (seg.first <= bytes)
            s = seg.second;
    return s;
}

double host_reduce_seconds(std::size_t bytes, int rounds) {
    Model& m = model();
    if (!m.enabled || m.op_bw <= 0.0 || bytes == 0 || rounds <= 0)
        return 0.0;
    return static_cast<double>(bytes) * rounds / m.op_bw;
}

static double band_lookup(const std::vector<std::pair<std::size_t, double>>& segs,
                          std::size_t bytes) {
    double s = 0.0;
    for (const auto& seg : segs)
        if (seg.first <= bytes)
            s = seg.second;
    return s;
}
double bcast_call_overhead_seconds(std::size_t bytes) {
    Model& m = model();
    return m.enabled ? band_lookup(m.bcast_call_segments, bytes) : 0.0;
}
double allgather_call_overhead_seconds(std::size_t bytes) {
    Model& m = model();
    return m.enabled ? band_lookup(m.allgather_call_segments, bytes) : 0.0;
}

double host_selfcopy_seconds(std::size_t bytes) {
    Model& m = model();
    if (!m.enabled || m.memcpy_bw <= 0.0 || bytes == 0)
        return 0.0;
    return static_cast<double>(bytes) / m.memcpy_bw;
}

static double seg_lookup(const std::vector<std::pair<std::size_t, double>>& segs,
                         std::size_t bytes) {
    double s = 0.0;
    for (const auto& seg : segs)
        if (seg.first <= bytes)
            s = seg.second;
    return s;
}

bool op_collective_bounces_to_host() {
    Model& m = model();
    return m.enabled && m.op_coll_host_bounce;
}

double coll_cuda_reduce_bounce_seconds(std::size_t bytes, bool is_root) {
    Model& m = model();
    if (!m.enabled || bytes == 0)
        return 0.0;
    double s = transfer_seconds_for_kind_hostmem(2, bytes, false);
    if (is_root)
        s += transfer_seconds_for_kind_hostmem(2, bytes, false) +
             transfer_seconds_for_kind_hostmem(1, bytes, false);
    return s;
}

double coll_cuda_bounce_seconds(std::size_t bytes) {
    Model& m = model();
    if (!m.enabled || bytes == 0)
        return 0.0;
    return 2.0 * transfer_seconds_for_kind_hostmem(2, bytes, false) +
           1.0 * transfer_seconds_for_kind_hostmem(1, bytes, false);
}

double allreduce_call_overhead_seconds(bool device, std::size_t bytes) {
    Model& m = model();
    if (!m.enabled)
        return 0.0;
    return seg_lookup(device ? m.ar_call_dev_segments : m.ar_call_host_segments, bytes);
}

double alltoall_call_overhead_seconds(bool device, std::size_t bytes) {
    Model& m = model();
    if (!m.enabled)
        return 0.0;
    return seg_lookup(device ? m.a2a_call_dev_segments : m.a2a_call_host_segments, bytes);
}

void charge_mpi_device_staging(std::size_t send_bytes, std::size_t recv_bytes) {
    const double t = mpi_device_staging_seconds(send_bytes, recv_bytes);
    if (t > 0.0)
        smpi_execute_benched(t);
    xtrace("MPI staging send=%zuB recv=%zuB | path=%s -> %.2f us", send_bytes, recv_bytes,
           mpi_path_name(), t * 1e6);
}

LinkCost get_cost(TransferPath path) {
    return model().cost[Model::idx(path)];
}
void set_cost(TransferPath path, LinkCost cost) {
    model().cost[Model::idx(path)] = cost;
}
bool model_enabled() {
    return model().enabled;
}

const char* mpi_path_name() {
    switch (model().mpi_path) {
    case MpiDevicePath::HOST_STAGED: return "host_staged";
    case MpiDevicePath::GDR: return "gdr";
    case MpiDevicePath::NVLINK: return "nvlink";
    case MpiDevicePath::PCIE: return "pcie";
    case MpiDevicePath::NONE: return "none";
    }
    return "?";
}

void dump_config() {
    Model& m = model();
    std::ostringstream os;
    os << "# ---- Orrery SimGrid transfer cost model ----\n"
       << "# source        : " << m.source << "\n"
       << "# model.enabled  : " << (m.enabled ? "true" : "false") << "\n"
       << "# mpi.device_path: " << mpi_path_label(m.mpi_path) << "\n"
       << "# staging.share  : " << m.staging_share << "\n"
       << "# peer.path      : " << path_label(m.peer_path) << "\n"
       << "#   path          latency(us)     bandwidth(GB/s)\n";
    for (int i = 0; i < static_cast<int>(TransferPath::COUNT); ++i) {
        const LinkCost& c = m.cost[i];
        os << "#   " << path_label(static_cast<TransferPath>(i))
           << std::string(14 - std::string(path_label(static_cast<TransferPath>(i))).size(), ' ')
           << c.latency * 1e6 << "\t\t" << c.bandwidth / 1e9 << "\n";
    }
    std::cerr << os.str();
}

}
