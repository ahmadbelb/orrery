# Orrery

Orrery runs unmodified CUDA-aware MPI applications inside the
[SimGrid](https://simgrid.org) simulator. It intercepts GPU runtime and MPI calls
at link level and charges each one a duration from a small cost model. No GPU is
needed.

<p align="center">
  <img src="docs/figures/orrery_overview.png" width="520"
       alt="Orrery overview: a GPU-aware MPI application issues GPU runtime calls and MPI calls; the Orrery interposer handles memory, copies, kernels and MPI (PMPI) using a machine description, and hands the resulting times to SimGrid">
</p>

## Quick start with Docker

You only need Docker. Give it about 15 GB of free disk and 8 GB of memory.

```bash
docker build -t orrery .
docker run --rm orrery ctest --test-dir build
docker run --rm orrery cmake --build build --target example
```

The example simulates 4 GPUs and prints:

```
4 GPUs, 100 iterations: simulated time 0.005223 s
```

## Install without Docker

On Linux, install SimGrid:

```bash
sudo apt install build-essential cmake git python3 libboost-dev libboost-context-dev
git clone https://framagit.org/simgrid/simgrid.git
cd simgrid
git checkout 1ea158bafc
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -Denable_smpi=ON -Denable_documentation=OFF -Denable_sthread=OFF .
make -j"$(nproc)"
sudo make install
sudo ldconfig
cd ..
```

Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (tested
with 12.4; no driver or GPU needed) and a `g++` it accepts (13 or older for
CUDA 12). Then, in this repository:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
cmake --build build --target example
```

## Run your own application

```bash
./run_cuda_mpi.sh my_app.cu -np 4 -- <application arguments>
```

In Docker, start a shell with `docker run --rm -it -v "$PWD":/work orrery` and use
`./run_cuda_mpi.sh /work/my_app.cu -np 4`.

- `MPI_Wtime()` returns simulated time; wall-clock timers measure your computer.
- Kernels are not executed, only charged, so results are not computed. To run them
  on a local GPU, use `GPU_ARCH=sm_90 ORRERY_REAL_GPU=1` with your GPU's architecture.
- Several source files: `EXTRA_SRCS="a.cpp b.cpp" ./run_cuda_mpi.sh main.cu -np 4`.

## Configuration

[machines/example](machines/example) is a sample machine with round numbers, 2
nodes with 4 GPUs each (up to `-np 8`). It is used by default:

- `transfers.conf`: device costs (copies, GPUDirect, kernels, host reduction)
- `platform.xml`, `hostfile`: the network, one simulated host per GPU
- `smpi_flags.txt`: SMPI's latency and bandwidth factors

To describe your own machine, copy the folder, change the values and pass it with
`-machine <folder>`.

## Limits

Orrery models Open MPI-style CUDA-aware MPI for applications that use the CUDA
runtime API. Device work is charged synchronously, so overlap with communication is
not modelled. CUDA calls it does not intercept go to the real runtime and fail
without a GPU.

## Citing

See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
