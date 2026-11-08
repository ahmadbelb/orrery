#include "orrery_test.h"
#include <cuda_runtime.h>
#include <cstring>
int main(int argc, char** argv) {
    t_init(&argc, &argv, "cudaDeviceGetPCIBusId");
    char pci[32] = {0};
    T_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0) == cudaSuccess, "returns success");
    T_CHECK(strncmp(pci, "0009:", 5) == 0, "PCI domain: '%s'", pci);
    unsigned dom, bus, devfn;
    T_CHECK(sscanf(pci, "%x:%x:%x.0", &dom, &bus, &devfn) == 3, "well-formed id");
    return t_finish();
}
