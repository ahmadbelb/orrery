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

### Or use the ready image

To skip the build, pull the published image and use its name wherever this page
says `orrery`:

```bash
docker pull ghcr.io/ahmadbelb/orrery:latest
docker run --rm ghcr.io/ahmadbelb/orrery:latest ctest --test-dir build
docker run --rm ghcr.io/ahmadbelb/orrery:latest cmake --build build --target example
```

**On a Mac with Apple Silicon this is not the best option.** The image is built for
`linux/amd64`, so it runs under emulation and is slower. The simulated times stay
the same. To run it anyway, add `--platform linux/amd64`:

```bash
docker run --rm --platform linux/amd64 ghcr.io/ahmadbelb/orrery:latest ctest --test-dir build
```

Building the image yourself with `docker build` avoids the emulation.

## Install without Docker

This needs no root access. You need a C++ compiler, CMake, git, Python 3, Boost and
the [CUDA 12 toolkit](https://developer.nvidia.com/cuda-toolkit-archive), any 12.x
(tested with 12.4 and 12.9; no driver or GPU needed; CUDA 13 does not work yet)
with a `g++` it accepts (13 or older; Orrery uses `g++-13` by itself when it is
installed). On a shared machine these usually come from `module load`.

Build SimGrid into your home folder:

```bash
git clone https://framagit.org/simgrid/simgrid.git
cd simgrid
git checkout 1ea158bafc
cmake -DCMAKE_INSTALL_PREFIX=$HOME/opt/simgrid -DCMAKE_INSTALL_LIBDIR=lib -Denable_smpi=ON -Denable_documentation=OFF -Denable_sthread=OFF .
make -j"$(nproc)" install
cd ..
```

Tell your shell where SimGrid and CUDA 12 are. Change `/usr/local/cuda-12.4` if
your CUDA 12 is somewhere else, and add these lines to `~/.bashrc` to keep them:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$HOME/opt/simgrid/bin:$CUDA_HOME/bin:$PATH
```

Then, in this repository:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
cmake --build build --target example
```

CMake remembers what it found the first time, so after changing any of this start
again with `rm -rf build`. If it still picks the wrong CUDA, compiler or SimGrid,
name them yourself:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DSIMGRID_PREFIX=$HOME/opt/simgrid
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

## Tests

```bash
ctest --test-dir build
ctest --test-dir build -R conf_
ctest --test-dir build -R '^test_cudaMemcpy$' -V
```

The first command runs all 61 tests (about 15 seconds). The second runs only the
tests with `conf_` in their name. The third runs one test and prints every check.
In Docker, put `docker run --rm orrery` in front.

You don't need to change anything to run them. The tests have their own settings in
`tests/transfers_test.conf`, separate from your machine folder. To see how a
parameter behaves, edit it there and rerun the test that uses it:

| Test | What it checks | Parameters it uses |
|---|---|---|
| `test_cudaMemcpy` | Time of a copy between host and GPU, or inside a GPU | `h2d.latency` and `h2d.bandwidth` for host to GPU, `d2h.*` for GPU to host, `d2d.*` for inside a GPU |
| `test_cudaMemcpyPeer` | Time of a copy between two GPUs | `peer.path`, the link to use: `nvlink` or `pcie` |
| `test_cudaLaunchKernel` | Time of a kernel | `gpu.flops_rate`, the GPU's speed, and `gpu.flops_per_thread`, the work per GPU thread |
| `test_cudaGetDeviceCount`, `test_cudaMemGetInfo`, `test_cudaGetDeviceProperties` | The GPU count and memory size an application sees | `gpu.count`, `gpu.mem_bytes` |
| `test_MPI_*` | Time of each MPI call on a GPU buffer | `mpi.device_path`, how MPI moves GPU data; `h2d.*` and `d2h.*`; `staging.share`, the slowdown when all GPUs on a node copy at once |
| `test_e2e_wire_timing` | Time of a message across the network | The link in `tests/platform_test.xml` |
| `test_conf_*` | That each setting and `ORRERY_*` variable is read correctly; `test_conf_path_gdr` checks `gdr.*`, for example | The test's own file in `tests/conf/` |
| `*_real` | The same calls on a real GPU; skipped if there is none | Same as the test without `_real` |
| `smoke` | The whole tool: the example must give 0.005223 s | `machines/example`, the only test that uses it |

The other `test_cuda*` tests check calls like `cudaMalloc` and have no parameter of
their own.

`test_cudaMemcpy` and `test_MPI_*` work out the expected time from the same
settings, so they keep passing with any value; add `-V` to see the times change.
Most of the other tests expect fixed numbers, so they fail if you change a value
they use. That is expected.

## Limits

Orrery models Open MPI-style CUDA-aware MPI for applications that use the CUDA
runtime API. Device work is charged synchronously, so overlap with communication is
not modelled. CUDA calls it does not intercept go to the real runtime and fail
without a GPU.

## Publications

Orrery has been accepted at PMBS 2026, the International Workshop on Performance
Modeling, Benchmarking and Simulation of High Performance Computer Systems, held
with SC26. The paper is also available on
[arXiv](https://arxiv.org/abs/2610.18565). If you use Orrery, please cite it:

```bibtex
@misc{belbeisi2026orrerymodelingsimulationgpudirect,
      title={Orrery: Modeling and Simulation of GPU-direct MPI Communication with SimGrid}, 
      author={Ahmad Moh'd Saleh A. Belbeisi and Amir Raoofy and Josef Weidendorfer and Majid Salimi Beni},
      year={2026},
      eprint={2610.18565},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2610.18565}, 
}
```

## License

[MIT](LICENSE)
