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
the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (tested with 12.4;
no driver or GPU needed) with a `g++` it accepts (13 or older for CUDA 12). On a
shared machine these usually come from `module load`.

Build SimGrid into your home folder:

```bash
git clone https://framagit.org/simgrid/simgrid.git
cd simgrid
git checkout 1ea158bafc
cmake -DCMAKE_INSTALL_PREFIX=$HOME/opt/simgrid -Denable_smpi=ON -Denable_documentation=OFF -Denable_sthread=OFF .
make -j"$(nproc)" install
cd ..
```

Tell your shell where it is. Add these lines to `~/.bashrc` to keep them:

```bash
export SIMGRID_PREFIX=$HOME/opt/simgrid
export PATH=$SIMGRID_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$SIMGRID_PREFIX/lib:$LD_LIBRARY_PATH
```

Then, in this repository:

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

## Tests

```bash
ctest --test-dir build
ctest --test-dir build -R conf_
ctest --test-dir build -R '^test_cudaMemcpy$' -V
```

These run all 61 tests (about 15 seconds), one group, and one test showing each
check. In Docker, put `docker run --rm orrery` in front.

You don't need to adjust anything to run them. They use their own values from
`tests/`, not your machine folder. To try a parameter, change it where the table
says:

| Tests | What they check | Where the values are |
|---|---|---|
| `test_cuda*` | CUDA calls | `tests/transfers_test.conf`: `h2d.*`, `d2h.*`, `d2d.*` (`cudaMemcpy`), `peer.path` (`cudaMemcpyPeer`), `gpu.flops_*` (`cudaLaunchKernel`), `gpu.count`, `gpu.mem_bytes` (device queries) |
| `test_MPI_*` | MPI calls on GPU buffers | `tests/transfers_test.conf`: `mpi.device_path`, `h2d.*`, `d2h.*`, `staging.share` |
| `test_conf_*` | Config keys and `ORRERY_*` variables, named after the key: `test_conf_path_gdr` checks `gdr.*` | `tests/conf/<test>.conf` and `.env` |
| `test_e2e_wire_timing` | Simulated network time | `tests/platform_test.xml` |
| `*_real` | The same calls with a real GPU; skipped without one | As for the groups above |
| `smoke` | The example, against 0.005223 s | `machines/example` |

`test_MPI_*` and `test_cudaMemcpy` compare against the model, so they pass with any
value; use `-V` to see the new times. Most of the others expect fixed numbers, so a
failure after you change a value they use is normal.

## Limits

Orrery models Open MPI-style CUDA-aware MPI for applications that use the CUDA
runtime API. Device work is charged synchronously, so overlap with communication is
not modelled. CUDA calls it does not intercept go to the real runtime and fail
without a GPU.

## Citing

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
