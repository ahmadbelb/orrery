#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "transfer_model.hpp"
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <set>
#include <mpi.h>
#include <mutex>
#include <string>
#include <vector>

static double wall() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef int cudaError_t;
extern "C" void smpi_execute_benched(double seconds);

static double g_compute_s = 0, g_net_s = 0, g_copy_s = 0;

static int g_trace = -1;
static bool tracing() {
    if (g_trace < 0)
        g_trace = getenv("ORRERY_TRACE") ? 1 : 0;
    return g_trace;
}
static int myrank() {
    int ini = 0;
    PMPI_Initialized(&ini);
    if (!ini)
        return -1;
    int r = -1;
    PMPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}
static void tlog(const char* fmt, ...) {
    if (!tracing())
        return;
    double t = 0;
    int ini = 0;
    PMPI_Initialized(&ini);
    if (ini)
        t = PMPI_Wtime() * 1e6;
    fprintf(stderr, "[r%d @%9.2fus] ", myrank(), t);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
static const char* kindname(int k) {
    return k == 1 ? "H2D" : k == 2 ? "D2H" : k == 3 ? "D2D" : k == 0 ? "H2H" : "default";
}

namespace {
double envd(const char* k, double d) {
    const char* v = getenv(k);
    return v ? atof(v) : d;
}
bool envb(const char* k) {
    const char* v = getenv(k);
    return v && atoi(v) != 0;
}
bool tru(const std::string& v) {
    return v == "true" || v == "1" || v == "on" || v == "yes";
}
std::string trim(std::string s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
struct Model {
    bool real_gpu = false, force_iters = true;
    double fpt = 5.0, rate = 794e9;
    double launch_oh = 0.0, sync_oh = 0.0;
    int gpu_count = 1;
    double gpu_mem = 96.0 * 1024.0 * 1024.0 * 1024.0;
    std::vector<std::pair<std::string, double>> kernel_rates;
    void setkv(const std::string& k, const std::string& v) {
        if (k.rfind("kernel.", 0) == 0 && k.size() > 12 &&
            k.compare(k.size() - 5, 5, ".rate") == 0) {
            kernel_rates.emplace_back(k.substr(7, k.size() - 12), atof(v.c_str()));
            return;
        }
        if (k == "real_gpu") {
            real_gpu = tru(v);
            return;
        }
        if (k == "force_iters") {
            force_iters = tru(v);
            return;
        }
        double d = atof(v.c_str());
        if (k == "gpu.flops_rate")
            rate = d;
        else if (k == "gpu.flops_per_thread")
            fpt = d;
        else if (k == "gpu.launch_overhead")
            launch_oh = d >= 0 ? d : 0.0;
        else if (k == "gpu.sync_overhead")
            sync_oh = d >= 0 ? d : 0.0;
        else if (k == "gpu.count")
            gpu_count = (int)d > 0 ? (int)d : 1;
        else if (k == "gpu.mem_bytes")
            gpu_mem = d > 0 ? d : gpu_mem;
    }
    Model() {
        std::string path;
        if (const char* c = getenv("ORRERY_TRANSFER_CONFIG"))
            path = c;
        if (path.empty())
            for (const char* cand : {"transfers.conf", "simgrid/transfers.conf"}) {
                std::ifstream t(cand);
                if (t) {
                    path = cand;
                    break;
                }
            }
        if (!path.empty()) {
            std::ifstream f(path);
            std::string line;
            while (std::getline(f, line)) {
                size_t h = line.find('#');
                if (h != std::string::npos)
                    line = line.substr(0, h);
                size_t e = line.find('=');
                if (e == std::string::npos)
                    continue;
                std::string k = trim(line.substr(0, e)), v = trim(line.substr(e + 1));
                if (!k.empty() && !v.empty())
                    setkv(k, v);
            }
        }
        if (getenv("ORRERY_REAL_GPU"))
            real_gpu = envb("ORRERY_REAL_GPU");
        if (getenv("ORRERY_FORCE_ITERS"))
            force_iters = !(atoi(getenv("ORRERY_FORCE_ITERS")) == 0);
        fpt = envd("ORRERY_FLOPS_PER_THREAD", fpt);
        rate = envd("ORRERY_GPU_FLOPS_RATE", rate);
        launch_oh = envd("ORRERY_GPU_LAUNCH_OVERHEAD", launch_oh);
        sync_oh = envd("ORRERY_GPU_SYNC_OVERHEAD", sync_oh);
        if (getenv("ORRERY_GPU_COUNT"))
            gpu_count = atoi(getenv("ORRERY_GPU_COUNT")) > 0
                            ? atoi(getenv("ORRERY_GPU_COUNT")) : gpu_count;
        gpu_mem = envd("ORRERY_GPU_MEM", gpu_mem);
    }
};
const Model& M() {
    static Model m;
    return m;
}
template <class FP, class... A> auto real(const char* n, A... a) -> decltype(((FP) nullptr)(a...)) {
    return ((FP)dlsym(RTLD_NEXT, n))(a...);
}
double charge_copy(size_t n, int kind) {
    if (!n)
        return 0;
    double s = orrery::transfer_seconds_for_kind(kind, n);
    if (s > 0) {
        smpi_execute_benched(s);
        g_copy_s += s;
    }
    return s;
}

static void charge_compute(double seconds) {
    if (seconds > 0) {
        g_compute_s += seconds;
        smpi_execute_benched(seconds);
    }
}
double charge_mpi_staging(size_t send_bytes, size_t recv_bytes, MPI_Comm comm) {
    int n = 1;
    PMPI_Comm_size(comm, &n);
    if (n < 2)
        return 0;
    double s = orrery::mpi_device_staging_seconds(send_bytes, recv_bytes);
    if (s > 0) {
        smpi_execute_benched(s);
        g_copy_s += s;
    }
    return s;
}
double charge_model_extra(double s, double& bucket) {
    if (s > 0) {
        smpi_execute_benched(s);
        bucket += s;
    }
    return s;
}
int comm_rounds_log2(MPI_Comm comm) {
    int n = 1;
    PMPI_Comm_size(comm, &n);
    int r = 0;
    while ((1 << r) < n)
        ++r;
    return r;
}
size_t mpi_bytes(int count, MPI_Datatype dt) {
    int sz = 0;
    PMPI_Type_size(dt, &sz);
    return (size_t)count * (sz > 0 ? sz : 0);
}
void keep_converging(void* d, size_t n, int kind) {
    if (M().force_iters && kind == 2 && n > 0 && n <= 8) {
        if (n >= 8)
            *(double*)d = 1.0;
        else
            *(float*)d = 1.0f;
    }
}
}

struct dim3 {
    unsigned int x, y, z;
};

static double g_kernel_threads = 0.0;
static std::map<std::string, std::pair<double, long>> g_kernel_stats;

static const char* kernel_name(const void* f) {
    static std::map<const void*, std::string> cache;
    auto it = cache.find(f);
    if (it == cache.end()) {
        Dl_info info{};
        const char* n = (dladdr(const_cast<void*>(f), &info) && info.dli_sname)
                            ? info.dli_sname : "";
        it = cache.emplace(f, n).first;
    }
    return it->second.empty() ? nullptr : it->second.c_str();
}

extern "C" {
cudaError_t cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** a, size_t sh, void* st) {
    double threads = (double)g.x * g.y * g.z * (double)b.x * b.y * b.z;
    if (M().real_gpu) {
        tlog("cudaLaunchKernel %.0f threads -> REAL launch on GPU", threads);
        return real<cudaError_t (*)(const void*, dim3, dim3, void**, size_t, void*)>(
            "cudaLaunchKernel", f, g, b, a, sh, st);
    }
    double rate = M().rate;
    const char* kname = kernel_name(f);
    if (!M().kernel_rates.empty() && kname) {
        for (const auto& kr : M().kernel_rates)
            if (std::strstr(kname, kr.first.c_str())) {
                rate = kr.second;
                break;
            }
    }
    double t = M().launch_oh + ((rate > 0) ? (threads * M().fpt / rate) : 0.0);
    if (t > 0) {
        smpi_execute_benched(t);
        g_compute_s += t;
    }
    g_kernel_threads += threads;
    {
        auto& e = g_kernel_stats[kname ? kname : "?"];
        e.first += threads;
        e.second += 1;
    }
    tlog("cudaLaunchKernel %s grid=%ux%ux%u block=%ux%ux%u = %.0f threads -> model COMPUTE "
         "%.4f ms (rate=%.3g%s, fpt=%.0f, launch=%.2fus)",
         kname ? kname : "?", g.x, g.y, g.z, b.x, b.y, b.z, threads, t * 1e3, rate,
         (rate == M().rate ? "" : " [per-kernel]"), M().fpt, M().launch_oh * 1e6);
    return 0;
}
static std::mutex g_dev_mu;
static std::map<uintptr_t, size_t>& dev_ranges() {
    static std::map<uintptr_t, size_t> m;
    return m;
}
static void dev_track(void* p, size_t s) {
    if (!p || !s)
        return;
    std::lock_guard<std::mutex> lk(g_dev_mu);
    dev_ranges()[(uintptr_t)p] = s;
}
static void dev_untrack(void* p) {
    if (!p)
        return;
    std::lock_guard<std::mutex> lk(g_dev_mu);
    dev_ranges().erase((uintptr_t)p);
}
static bool is_dev_ptr(const void* p) {
    if (!p)
        return false;
    std::lock_guard<std::mutex> lk(g_dev_mu);
    auto& m = dev_ranges();
    auto it = m.upper_bound((uintptr_t)p);
    if (it == m.begin())
        return false;
    --it;
    return (uintptr_t)p < it->first + it->second;
}
static size_t dev_bytes(const void* p, size_t n) {
    return is_dev_ptr(p) ? n : 0;
}

extern "C" void* smpi_shared_malloc(size_t size, const char* file, int line);
extern "C" void smpi_shared_free(void* ptr);
std::set<void*> g_shared_ptrs;
bool shared_alloc_on() {
    static int v = -1;
    if (v < 0)
        v = getenv("ORRERY_SHARED_ALLOC") ? atoi(getenv("ORRERY_SHARED_ALLOC")) : 0;
    return v > 0;
}
void* skel_alloc(size_t s) {
    if (shared_alloc_on() && s >= (1u << 20)) {
        void* p = smpi_shared_malloc(s, "orrery_shared", 0);
        if (p) {
            g_shared_ptrs.insert(p);
            return p;
        }
    }
    return malloc(s);
}
void skel_free(void* p) {
    if (!p)
        return;
    auto it = g_shared_ptrs.find(p);
    if (it != g_shared_ptrs.end()) {
        g_shared_ptrs.erase(it);
        smpi_shared_free(p);
        return;
    }
    free(p);
}

cudaError_t cudaMalloc(void** p, size_t s) {
    if (M().real_gpu) {
        tlog("cudaMalloc %.3g MB -> REAL cudaMallocManaged (unified, host-addressable)", s / 1e6);
        cudaError_t r = real<cudaError_t (*)(void**, size_t, unsigned)>("cudaMallocManaged", p, s, 1u);
        if (r == 0)
            dev_track(*p, s);
        return r;
    }
    *p = skel_alloc(s);
    dev_track(*p, s);
    tlog("cudaMalloc %.3g MB -> skeleton malloc (host memory)", s / 1e6);
    return (*p || !s) ? 0 : 1;
}
cudaError_t cudaMallocManaged(void** p, size_t s, unsigned f) {
    if (M().real_gpu) {
        cudaError_t r = real<cudaError_t (*)(void**, size_t, unsigned)>("cudaMallocManaged", p, s, f);
        if (r == 0)
            dev_track(*p, s);
        return r;
    }
    *p = skel_alloc(s);
    dev_track(*p, s);
    return (*p || !s) ? 0 : 1;
}
static std::map<uintptr_t, size_t>& pin_ranges() {
    static std::map<uintptr_t, size_t> m;
    return m;
}
static void pin_track(void* p, size_t s) {
    if (p && s) {
        std::lock_guard<std::mutex> lk(g_dev_mu);
        pin_ranges()[(uintptr_t)p] = s;
    }
}
static bool pin_contains(const void* p) {
    std::lock_guard<std::mutex> lk(g_dev_mu);
    auto& m = pin_ranges();
    auto it = m.upper_bound((uintptr_t)p);
    if (it == m.begin())
        return false;
    --it;
    return (uintptr_t)p < it->first + it->second;
}
cudaError_t cudaMallocHost(void** p, size_t s) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void**, size_t)>("cudaMallocHost", p, s);
    *p = skel_alloc(s);
    pin_track(*p, s);
    return (*p || !s) ? 0 : 1;
}
cudaError_t cudaHostAlloc(void** p, size_t s, unsigned f) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void**, size_t, unsigned)>("cudaHostAlloc", p, s, f);
    *p = skel_alloc(s);
    pin_track(*p, s);
    return (*p || !s) ? 0 : 1;
}
cudaError_t cudaFree(void* p) {
    dev_untrack(p);
    if (M().real_gpu)
        return real<cudaError_t (*)(void*)>("cudaFree", p);
    skel_free(p);
    return 0;
}
cudaError_t cudaFreeHost(void* p) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*)>("cudaFreeHost", p);
    skel_free(p);
    return 0;
}
cudaError_t cudaHostRegister(void* p, size_t s, unsigned f) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, size_t, unsigned)>("cudaHostRegister", p, s, f);
    return 0;
}
cudaError_t cudaHostUnregister(void* p) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*)>("cudaHostUnregister", p);
    return 0;
}
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int k) {
    if (M().real_gpu) {
        double t0 = wall();
        auto r = real<cudaError_t (*)(void*, const void*, size_t, int)>("cudaMemcpy", d, s, n, k);
        double w = wall() - t0;
        g_copy_s += w;
        tlog("cudaMemcpy %s %.3g MB -> REAL copy, wall %.2f us", kindname(k), n / 1e6, w * 1e6);
        return r;
    }
    memcpy(d, s, n);
    bool pinned = true;
    if (k == 1)
        pinned = pin_contains(s);
    else if (k == 2)
        pinned = pin_contains(d);
    double cs = orrery::transfer_seconds_for_kind_hostmem(k, n, pinned);
    if (cs > 0) {
        smpi_execute_benched(cs);
        g_copy_s += cs;
    }
    keep_converging(d, n, k);
    tlog("cudaMemcpy %s %.3g MB (%s host) -> model TRANSFER charge %.2f us", kindname(k),
         n / 1e6, pinned ? "pinned" : "pageable", cs * 1e6);
    return 0;
}
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, int k, void* st) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, const void*, size_t, int, void*)>("cudaMemcpyAsync", d,
                                                                             s, n, k, st);
    memcpy(d, s, n);
    charge_copy(n, k);
    keep_converging(d, n, k);
    return 0;
}
cudaError_t cudaMemset(void* p, int v, size_t n) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, int, size_t)>("cudaMemset", p, v, n);
    memset(p, v, n);
    return 0;
}
cudaError_t cudaMemsetAsync(void* p, int v, size_t n, void* st) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, int, size_t, void*)>("cudaMemsetAsync", p, v, n, st);
    memset(p, v, n);
    return 0;
}
cudaError_t cudaGetDeviceCount(int* c) {
    if (M().real_gpu)
        return real<cudaError_t (*)(int*)>("cudaGetDeviceCount", c);
    if (c)
        *c = M().gpu_count;
    return 0;
}
cudaError_t cudaSetDevice(int d) {
    if (M().real_gpu)
        return real<cudaError_t (*)(int)>("cudaSetDevice", d);
    return (d >= 0 && d < M().gpu_count) ? 0 : 101;
}
cudaError_t cudaGetDevice(int* d) {
    if (M().real_gpu)
        return real<cudaError_t (*)(int*)>("cudaGetDevice", d);
    if (d)
        *d = 0;
    return 0;
}
namespace fakeprop {
enum : size_t { SIZE = 1032, NAME = 0, UUID = 256 };
static void i32(char* p, size_t off, int v) { memcpy(p + off, &v, 4); }
static void sz(char* p, size_t off, size_t v) { memcpy(p + off, &v, sizeof(size_t)); }
}
cudaError_t cudaGetDeviceProperties_v2(void* prop, int dev) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, int)>("cudaGetDeviceProperties_v2", prop, dev);
    if (dev < 0 || dev >= M().gpu_count)
        return 101;
    if (prop) {
        using namespace fakeprop;
        char* p = (char*)prop;
        memset(p, 0, SIZE);
        snprintf(p + NAME, 256, "SimGrid Virtual GPU");
        char host[256] = "no-mpi";
        int ini = 0, hl = 0;
        PMPI_Initialized(&ini);
        if (ini)
            PMPI_Get_processor_name(host, &hl);
        unsigned long long h = 1469598103934665603ULL;
        for (const char* c = host; *c; ++c) h = (h ^ (unsigned char)*c) * 1099511628211ULL;
        h = (h ^ (unsigned)dev) * 1099511628211ULL;
        memcpy(p + UUID, &h, 8);
        h = (h ^ 0x5bd1e995u) * 1099511628211ULL;
        memcpy(p + UUID + 8, &h, 8);
        sz(p, 288, (size_t)M().gpu_mem);
        sz(p, 296, 49152);
        i32(p, 304, 65536);
        i32(p, 308, 32);
        i32(p, 320, 1024);
        i32(p, 348, 1980000);
        i32(p, 360, 9);
        i32(p, 364, 0);
        i32(p, 388, 132);
        i32(p, 404, 0);
        i32(p, 576, 1);
        i32(p, 580, 1);
        i32(p, 600, 3);
        i32(p, 604, 1);
        i32(p, 608, 2619000);
        i32(p, 612, 5120);
        i32(p, 616, 52428800);
        i32(p, 624, 2048);
        sz(p, 640, 233472);
        i32(p, 652, 1);
        i32(p, 672, 1);
    }
    return 0;
}
cudaError_t cudaGetDeviceProperties(void* prop, int dev) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, int)>("cudaGetDeviceProperties", prop, dev);
    return cudaGetDeviceProperties_v2(prop, dev);
}
cudaError_t cudaMemGetInfo(size_t* freeb, size_t* total) {
    if (M().real_gpu)
        return real<cudaError_t (*)(size_t*, size_t*)>("cudaMemGetInfo", freeb, total);
    if (total)
        *total = (size_t)M().gpu_mem;
    if (freeb)
        *freeb = (size_t)M().gpu_mem;
    return 0;
}
cudaError_t cudaDeviceGetPCIBusId(char* id, int len, int dev) {
    if (M().real_gpu)
        return real<cudaError_t (*)(char*, int, int)>("cudaDeviceGetPCIBusId", id, len, dev);
    if (id && len > 0)
        snprintf(id, (size_t)len, "0009:%02x:00.0", 1 + dev);
    return 0;
}
extern "C" int MPIX_Query_cuda_support(void) { return 1; }
cudaError_t cudaMemcpyPeer(void* d, int ddev, const void* s, int sdev, size_t n) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, int, const void*, int, size_t)>("cudaMemcpyPeer", d,
                                                                           ddev, s, sdev, n);
    memcpy(d, s, n);
    orrery::charge_peer_transfer(n);
    tlog("cudaMemcpyPeer %.3g MB dev%d->dev%d -> peer path charge", n / 1e6, sdev, ddev);
    return 0;
}
cudaError_t cudaDeviceReset() {
    if (M().real_gpu)
        return real<cudaError_t (*)()>("cudaDeviceReset");
    return 0;
}
cudaError_t cudaMemPrefetchAsync(const void* p, size_t n, int dst, void* st) {
    if (M().real_gpu)
        return real<cudaError_t (*)(const void*, size_t, int, void*)>("cudaMemPrefetchAsync", p, n,
                                                                      dst, st);
    return 0;
}
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
CUresult cuInit(unsigned int f) {
    if (M().real_gpu)
        return real<CUresult (*)(unsigned)>("cuInit", f);
    return 0;
}
CUresult cuDeviceGet(CUdevice* d, int ord) {
    if (d)
        *d = ord;
    return 0;
}
CUresult cuDeviceGetCount(int* c) {
    if (c)
        *c = M().gpu_count;
    return 0;
}
CUresult cuDevicePrimaryCtxRetain(CUcontext* c, CUdevice d) {
    (void)d;
    if (c)
        *c = (CUcontext)1;
    return 0;
}
CUresult cuCtxGetCurrent(CUcontext* c) {
    if (c)
        *c = (CUcontext)1;
    return 0;
}
CUresult cuCtxSetCurrent(CUcontext c) {
    (void)c;
    return 0;
}
cudaError_t cudaDeviceSynchronize() {
    if (M().real_gpu) {
        double t0 = wall();
        auto r = real<cudaError_t (*)()>("cudaDeviceSynchronize");
        charge_compute(wall() - t0);
        return r;
    }
    charge_compute(M().sync_oh);
    return 0;
}
cudaError_t cudaStreamCreate(void** s) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void**)>("cudaStreamCreate", s);
    if (s)
        *s = 0;
    return 0;
}
cudaError_t cudaStreamDestroy(void* s) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*)>("cudaStreamDestroy", s);
    return 0;
}
cudaError_t cudaStreamSynchronize(void* s) {
    if (M().real_gpu) {
        double t0 = wall();
        auto r = real<cudaError_t (*)(void*)>("cudaStreamSynchronize", s);
        g_compute_s += wall() - t0;
        charge_compute(wall() - t0);
        return r;
    }
    charge_compute(M().sync_oh);
    return 0;
}
cudaError_t cudaStreamWaitEvent(void* s, void* e, unsigned f) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, void*, unsigned)>("cudaStreamWaitEvent", s, e, f);
    return 0;
}
cudaError_t cudaEventCreate(void** e) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void**)>("cudaEventCreate", e);
    if (e)
        *e = 0;
    return 0;
}
cudaError_t cudaEventCreateWithFlags(void** e, unsigned f) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void**, unsigned)>("cudaEventCreateWithFlags", e, f);
    if (e)
        *e = 0;
    return 0;
}
cudaError_t cudaEventRecord(void* e, void* s) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*, void*)>("cudaEventRecord", e, s);
    return 0;
}
cudaError_t cudaEventSynchronize(void* e) {
    if (M().real_gpu) {
        double t0 = wall();
        auto r = real<cudaError_t (*)(void*)>("cudaEventSynchronize", e);
        g_compute_s += wall() - t0;
        charge_compute(wall() - t0);
        return r;
    }
    charge_compute(M().sync_oh);
    return 0;
}
cudaError_t cudaEventElapsedTime(float* ms, void* a, void* b) {
    if (M().real_gpu)
        return real<cudaError_t (*)(float*, void*, void*)>("cudaEventElapsedTime", ms, a, b);
    if (ms)
        *ms = 0.f;
    return 0;
}
cudaError_t cudaEventDestroy(void* e) {
    if (M().real_gpu)
        return real<cudaError_t (*)(void*)>("cudaEventDestroy", e);
    return 0;
}
cudaError_t cudaDeviceCanAccessPeer(int* c, int a, int b) {
    if (M().real_gpu)
        return real<cudaError_t (*)(int*, int, int)>("cudaDeviceCanAccessPeer", c, a, b);
    if (c)
        *c = 1;
    return 0;
}
cudaError_t cudaDeviceEnablePeerAccess(int d, unsigned f) {
    if (M().real_gpu)
        return real<cudaError_t (*)(int, unsigned)>("cudaDeviceEnablePeerAccess", d, f);
    return 0;
}
cudaError_t cudaGetLastError() {
    if (M().real_gpu)
        return real<cudaError_t (*)()>("cudaGetLastError");
    return 0;
}
cudaError_t cudaPeekAtLastError() {
    if (M().real_gpu)
        return real<cudaError_t (*)()>("cudaPeekAtLastError");
    return 0;
}
const char* cudaGetErrorString(cudaError_t e) {
    if (M().real_gpu)
        return real<const char* (*)(cudaError_t)>("cudaGetErrorString", e);
    return "cuda-skeleton: no error";
}

int MPI_Init(int* argc, char*** argv) {
    int r = PMPI_Init(argc, argv);
    if (tracing()) {
        int rk = -1;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rk);
        if (rk == 0)
            fprintf(stderr,
                    "[trace] CONFIG  mode=%s  device_path=%s  flops_rate=%.3g  fpt=%.0f  (set "
                    "ORRERY_* to change)\n",
                    M().real_gpu ? "REAL-GPU" : "skeleton", orrery::mpi_path_name(), M().rate,
                    M().fpt);
    }
    return r;
}
int MPI_Init_thread(int* argc, char*** argv, int req, int* prov) {
    int r = PMPI_Init_thread(argc, argv, req, prov);
    if (tracing()) {
        int rk = -1;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rk);
        if (rk == 0)
            fprintf(stderr, "[trace] CONFIG  mode=%s  device_path=%s  flops_rate=%.3g  fpt=%.0f\n",
                    M().real_gpu ? "REAL-GPU" : "skeleton", orrery::mpi_path_name(), M().rate,
                    M().fpt);
    }
    return r;
}

int MPI_Sendrecv(const void* sb, int sc, MPI_Datatype st, int d, int stg, void* rb, int rc,
                 MPI_Datatype rt, int src, int rtg, MPI_Comm comm, MPI_Status* s) {
    size_t sbN = mpi_bytes(sc, st), rbN = mpi_bytes(rc, rt);
    double stage = charge_mpi_staging(dev_bytes(sb, sbN), dev_bytes(rb, rbN), comm);
    if (comm_rounds_log2(comm) > 0 && (dev_bytes(sb, sbN) || dev_bytes(rb, rbN)))
        charge_model_extra(orrery::exchange_overhead_seconds(sbN > rbN ? sbN : rbN), g_net_s);
    double t0 = PMPI_Wtime();
    int r = PMPI_Sendrecv(sb, sc, st, d, stg, rb, rc, rt, src, rtg, comm, s);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Sendrecv  send->r%d (%zuB)  recv<-r%d (%zuB) | path=%s device_leg=%.2fus | "
         "net(SIMULATED)=%.2fus",
         d, sbN, src, rbN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Send(const void* b, int c, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    size_t bN = mpi_bytes(c, dt);
    double stage = charge_mpi_staging(dev_bytes(b, bN), 0, comm);
    double t0 = PMPI_Wtime();
    int r = PMPI_Send(b, c, dt, dest, tag, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Send  ->r%d (%zuB) | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus", dest, bN,
         orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Recv(void* b, int c, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status* s) {
    size_t bN = mpi_bytes(c, dt);
    double t0 = PMPI_Wtime();
    int r = PMPI_Recv(b, c, dt, src, tag, comm, s);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    double stage = charge_mpi_staging(0, dev_bytes(b, bN), comm);
    tlog("MPI_Recv  <-r%d (%zuB) | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus", src, bN,
         orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Allreduce(const void* sb, void* rb, int c, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
    size_t bN = mpi_bytes(c, dt);
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    int rounds = comm_rounds_log2(comm);
    const bool devbuf = dev_bytes(se, bN) || dev_bytes(rb, bN);
    const bool bounce = devbuf && orrery::op_collective_bounces_to_host();
    double stage = bounce ? 0.0
                          : charge_mpi_staging(dev_bytes(se, bN), dev_bytes(rb, bN), comm);
    double t_bounce = 0.0, t_reduce = 0.0, t_band = 0.0;
    if (rounds > 0) {
        if (bounce) {
            int psize = 1;
            PMPI_Comm_size(comm, &psize);
            double passes = psize > 1 ? (double)(psize - 1) / psize : 0.0;
            t_bounce = orrery::coll_cuda_bounce_seconds(bN);
            t_reduce = orrery::host_reduce_seconds(bN, 1) * passes;
            charge_model_extra(t_bounce + t_reduce, g_copy_s);
        } else if (dev_bytes(se, bN) || dev_bytes(rb, bN)) {
            t_band = rounds * orrery::exchange_overhead_seconds(bN) +
                     orrery::allreduce_call_overhead_seconds(true, bN);
            charge_model_extra(t_band, g_net_s);
        } else {
            int psize_h = 1;
            PMPI_Comm_size(comm, &psize_h);
            double passes_h = psize_h > 1 ? (double)(psize_h - 1) / psize_h : 0.0;
            t_band = orrery::allreduce_call_overhead_seconds(false, bN);
            t_reduce = orrery::host_reduce_seconds(bN, 1) * passes_h;
            charge_model_extra(t_band + t_reduce, g_copy_s);
        }
    }
    double t0 = PMPI_Wtime();
    int r = PMPI_Allreduce(sb, rb, c, dt, op, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Allreduce  bytes=%zuB | path=%s device_leg=%.2fus | "
         "bounce=%.3fus reduce=%.3fus band=%.3fus | net(SIMULATED)=%.2fus",
         bN, bounce ? "coll_cuda_bounce" : orrery::mpi_path_name(), stage * 1e6,
         t_bounce * 1e6, t_reduce * 1e6, t_band * 1e6, net * 1e6);
    return r;
}
int MPI_Alltoall(const void* sb, int sc, MPI_Datatype st, void* rb, int rc, MPI_Datatype rt,
                 MPI_Comm comm) {
    int P = 1;
    PMPI_Comm_size(comm, &P);
    size_t sbN = mpi_bytes(sc, st) * P, rbN = mpi_bytes(rc, rt) * P;
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    size_t sb_remote = sbN - (P > 0 ? sbN / (size_t)P : 0);
    size_t rb_remote = rbN - (P > 0 ? rbN / (size_t)P : 0);
    double stage = charge_mpi_staging(dev_bytes(se, sb_remote), dev_bytes(rb, rb_remote), comm);
    double selfcopy = 0.0;
    if (P >= 2) {
        size_t per_pair = sbN / (size_t)P;
        if (dev_bytes(se, sbN) || dev_bytes(rb, rbN)) {
            orrery::charge_transfer(orrery::TransferPath::D2D, per_pair);
            charge_model_extra((P - 1) * orrery::exchange_overhead_seconds(per_pair) +
                                   orrery::alltoall_call_overhead_seconds(true, per_pair),
                               g_net_s);
        } else {
            charge_model_extra(orrery::alltoall_call_overhead_seconds(false, per_pair),
                               g_copy_s);
            selfcopy = orrery::host_selfcopy_seconds(per_pair);
        }
    }
    double t0 = PMPI_Wtime();
    int r = PMPI_Alltoall(sb, sc, st, rb, rc, rt, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    if (selfcopy > net)
        charge_model_extra(selfcopy - net, g_copy_s);
    tlog("MPI_Alltoall  total send=%zuB recv=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus",
         sbN, rbN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Barrier(MPI_Comm comm) {
    double t0 = PMPI_Wtime();
    int r = PMPI_Barrier(comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Barrier | net(SIMULATED)=%.2fus", net * 1e6);
    return r;
}

int MPI_Isend(const void* b, int c, MPI_Datatype dt, int dest, int tag, MPI_Comm comm,
              MPI_Request* req) {
    size_t bN = mpi_bytes(c, dt);
    double stage = charge_mpi_staging(dev_bytes(b, bN), 0, comm);
    int r = PMPI_Isend(b, c, dt, dest, tag, comm, req);
    tlog("MPI_Isend ->r%d (%zuB) | path=%s device_leg=%.2fus", dest, bN,
         orrery::mpi_path_name(), stage * 1e6);
    return r;
}
static std::mutex g_req_mu;
static std::map<MPI_Request, size_t>& irecv_pending() {
    static std::map<MPI_Request, size_t> m;
    return m;
}
static void finish_req(MPI_Request h) {
    if (h == MPI_REQUEST_NULL)
        return;
    size_t bN = 0;
    {
        std::lock_guard<std::mutex> lk(g_req_mu);
        auto it = irecv_pending().find(h);
        if (it == irecv_pending().end())
            return;
        bN = it->second;
        irecv_pending().erase(it);
    }
    double s = orrery::mpi_device_staging_seconds(0, bN);
    if (s > 0) {
        smpi_execute_benched(s);
        g_copy_s += s;
    }
    tlog("MPI_Irecv complete (%zuB) | path=%s device_leg=%.2fus", bN,
         orrery::mpi_path_name(), s * 1e6);
}
int MPI_Irecv(void* b, int c, MPI_Datatype dt, int src, int tag, MPI_Comm comm,
              MPI_Request* req) {
    size_t bN = mpi_bytes(c, dt);
    int P = 1;
    PMPI_Comm_size(comm, &P);
    int r = PMPI_Irecv(b, c, dt, src, tag, comm, req);
    if (r == MPI_SUCCESS && req && P > 1) {
        size_t chg = dev_bytes(b, bN);
        if (chg) {
            std::lock_guard<std::mutex> lk(g_req_mu);
            irecv_pending()[*req] = chg;
        }
    }
    tlog("MPI_Irecv <-r%d (%zuB) posted", src, bN);
    return r;
}
int MPI_Wait(MPI_Request* req, MPI_Status* st) {
    MPI_Request h = req ? *req : MPI_REQUEST_NULL;
    double t0 = PMPI_Wtime();
    int r = PMPI_Wait(req, st);
    g_net_s += PMPI_Wtime() - t0;
    if (r == MPI_SUCCESS)
        finish_req(h);
    return r;
}
int MPI_Waitall(int n, MPI_Request reqs[], MPI_Status sts[]) {
    std::vector<MPI_Request> saved(reqs, reqs + (n > 0 ? n : 0));
    double t0 = PMPI_Wtime();
    int r = PMPI_Waitall(n, reqs, sts);
    g_net_s += PMPI_Wtime() - t0;
    if (r == MPI_SUCCESS)
        for (MPI_Request h : saved)
            finish_req(h);
    return r;
}
int MPI_Test(MPI_Request* req, int* flag, MPI_Status* st) {
    MPI_Request h = req ? *req : MPI_REQUEST_NULL;
    int r = PMPI_Test(req, flag, st);
    if (r == MPI_SUCCESS && flag && *flag)
        finish_req(h);
    return r;
}
int MPI_Bcast(void* b, int c, MPI_Datatype dt, int root, MPI_Comm comm) {
    int rk = 0;
    PMPI_Comm_rank(comm, &rk);
    size_t bN = mpi_bytes(c, dt), chg = dev_bytes(b, bN);
    double stage = charge_mpi_staging(rk == root ? chg : 0, rk == root ? 0 : chg, comm);
    if (chg)
        charge_model_extra(orrery::bcast_call_overhead_seconds(bN), g_net_s);
    double t0 = PMPI_Wtime();
    int r = PMPI_Bcast(b, c, dt, root, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Bcast  root=%d bytes=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus", root,
         bN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Reduce(const void* sb, void* rb, int c, MPI_Datatype dt, MPI_Op op, int root,
               MPI_Comm comm) {
    int rk = 0, P = 1;
    PMPI_Comm_rank(comm, &rk);
    PMPI_Comm_size(comm, &P);
    size_t bN = mpi_bytes(c, dt);
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    const bool devbuf = dev_bytes(se, bN) || (rk == root && dev_bytes(rb, bN));
    const bool bounce = devbuf && orrery::op_collective_bounces_to_host();
    double stage = 0.0;
    if (bounce && P > 1) {
        double passes = (double)(P - 1) / P;
        charge_model_extra(orrery::coll_cuda_reduce_bounce_seconds(bN, rk == root) +
                               orrery::host_reduce_seconds(bN, 1) * passes,
                           g_copy_s);
    } else {
        stage = charge_mpi_staging(dev_bytes(se, bN),
                                   rk == root ? dev_bytes(rb, bN) : 0, comm);
    }
    double t0 = PMPI_Wtime();
    int r = PMPI_Reduce(sb, rb, c, dt, op, root, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Reduce  root=%d bytes=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus", root,
         bN, bounce ? "coll_cuda_bounce" : orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Reduce_scatter(const void* sb, void* rb, const int rc[], MPI_Datatype dt, MPI_Op op,
                       MPI_Comm comm) {
    int rk = 0, P = 1;
    PMPI_Comm_rank(comm, &rk);
    PMPI_Comm_size(comm, &P);
    size_t total = 0, mine = 0;
    for (int i = 0; i < P; i++) {
        size_t b = mpi_bytes(rc[i], dt);
        total += b;
        if (i == rk) mine = b;
    }
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    double stage = charge_mpi_staging(dev_bytes(se, total), dev_bytes(rb, mine), comm);
    double t0 = PMPI_Wtime();
    int r = PMPI_Reduce_scatter(sb, rb, rc, dt, op, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Reduce_scatter  bytes=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus",
         total, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Allgather(const void* sb, int sc, MPI_Datatype st, void* rb, int rc, MPI_Datatype rt,
                  MPI_Comm comm) {
    int P = 1;
    PMPI_Comm_size(comm, &P);
    size_t sN = (sb == MPI_IN_PLACE) ? mpi_bytes(rc, rt) : mpi_bytes(sc, st);
    size_t rN = mpi_bytes(rc, rt) * P;
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    double stage = charge_mpi_staging(dev_bytes(se, sN), dev_bytes(rb, rN), comm);
        if (dev_bytes(sb == MPI_IN_PLACE ? rb : sb, 1) || dev_bytes(rb, 1))
        charge_model_extra(orrery::allgather_call_overhead_seconds(
                               mpi_bytes(rc, rt)), g_net_s);
double t0 = PMPI_Wtime();
    int r = PMPI_Allgather(sb, sc, st, rb, rc, rt, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Allgather  send=%zuB recv=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus",
         sN, rN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Gather(const void* sb, int sc, MPI_Datatype st, void* rb, int rc, MPI_Datatype rt,
               int root, MPI_Comm comm) {
    int rk = 0, P = 1;
    PMPI_Comm_rank(comm, &rk);
    PMPI_Comm_size(comm, &P);
    size_t sN = mpi_bytes(sc, st), rN = mpi_bytes(rc, rt) * P;
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    double stage = charge_mpi_staging(dev_bytes(se, sN),
                                      rk == root ? dev_bytes(rb, rN) : 0, comm);
    double t0 = PMPI_Wtime();
    int r = PMPI_Gather(sb, sc, st, rb, rc, rt, root, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Gather  root=%d send=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus", root,
         sN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Scatter(const void* sb, int sc, MPI_Datatype st, void* rb, int rc, MPI_Datatype rt,
                int root, MPI_Comm comm) {
    int rk = 0, P = 1;
    PMPI_Comm_rank(comm, &rk);
    PMPI_Comm_size(comm, &P);
    size_t sN = mpi_bytes(sc, st) * P, rN = mpi_bytes(rc, rt);
    void* re = (rb == MPI_IN_PLACE) ? (void*)sb : rb;
    double stage = charge_mpi_staging(rk == root ? dev_bytes(sb, sN) : 0,
                                      dev_bytes(re, rN), comm);
    double t0 = PMPI_Wtime();
    int r = PMPI_Scatter(sb, sc, st, rb, rc, rt, root, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Scatter  root=%d recv=%zuB | path=%s device_leg=%.2fus | net(SIMULATED)=%.2fus", root,
         rN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
int MPI_Alltoallv(const void* sb, const int sc[], const int sd[], MPI_Datatype st, void* rb,
                  const int rc[], const int rd[], MPI_Datatype rt, MPI_Comm comm) {
    int P = 1, ssz = 0, rsz = 0;
    PMPI_Comm_size(comm, &P);
    PMPI_Type_size(st, &ssz);
    PMPI_Type_size(rt, &rsz);
    size_t sN = 0, rN = 0;
    for (int i = 0; i < P; ++i) {
        if (sc)
            sN += (size_t)sc[i] * ssz;
        if (rc)
            rN += (size_t)rc[i] * rsz;
    }
    const void* se = (sb == MPI_IN_PLACE) ? rb : sb;
    double stage = charge_mpi_staging(dev_bytes(se, sN), dev_bytes(rb, rN), comm);
    double t0 = PMPI_Wtime();
    int r = PMPI_Alltoallv(sb, sc, sd, st, rb, rc, rd, rt, comm);
    double net = PMPI_Wtime() - t0;
    g_net_s += net;
    tlog("MPI_Alltoallv  total send=%zuB recv=%zuB | path=%s device_leg=%.2fus | "
         "net(SIMULATED)=%.2fus",
         sN, rN, orrery::mpi_path_name(), stage * 1e6, net * 1e6);
    return r;
}
double orrery_copy_seconds(void) { return g_copy_s; }
double orrery_net_seconds(void) { return g_net_s; }
double orrery_compute_seconds(void) { return g_compute_s; }

int MPI_Finalize(void) {
    int rank = 0;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        fprintf(stderr, "[breakdown rank0]  compute=%.4f s   network=%.4f s   H2D/D2H=%.4f s%s\n",
                g_compute_s, g_net_s, g_copy_s,
                M().real_gpu ? "   (REAL GPU: compute=real wall-time in GPU sync)" : "");
        fprintf(stderr, "[breakdown rank0]  kernel_threads=%.6g  fpt=%.0f\n",
                g_kernel_threads, M().fpt);
        for (const auto& kv : g_kernel_stats)
            fprintf(stderr, "[kernel] %-60s launches=%-8ld threads=%.6g\n",
                    kv.first.c_str(), kv.second.second, kv.second.first);
    return PMPI_Finalize();
}
}
