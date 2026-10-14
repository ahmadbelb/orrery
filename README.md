# Orrery

Orrery runs **unmodified CUDA-aware MPI applications** inside the
[SimGrid](https://simgrid.org) simulator. It is a thin interposition layer
that sits between the application and SimGrid's MPI implementation (SMPI):

<p align="center">
  <img src="docs/figures/orrery_overview.png" width="520"
       alt="Orrery overview: a GPU-aware MPI application issues GPU runtime calls and MPI calls; the Orrery interposer handles memory, copies, kernels and MPI (PMPI) using a machine description, and hands the resulting times to SimGrid">
</p>


## Citing

See [CITATION.cff](CITATION.cff).
