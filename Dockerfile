ARG UBUNTU=22.04

FROM ubuntu:${UBUNTU} AS simgrid
ARG SIMGRID_COMMIT=1ea158bafce54c909c1b26a8e85f4af1fc6ed6db
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates python3 libboost-dev libboost-context-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://framagit.org/simgrid/simgrid.git /src/simgrid \
 && cd /src/simgrid && git checkout ${SIMGRID_COMMIT} \
 && cmake -DCMAKE_INSTALL_PREFIX=/opt/simgrid -DCMAKE_BUILD_TYPE=Release \
          -Denable_smpi=ON -Denable_documentation=OFF -Denable_model-checking=OFF \
          -Denable_fortran=OFF -Denable_lto=OFF -Denable_sthread=OFF . \
 && make -j"$(awk -v n="$(nproc)" '/^MemAvailable:/ {j=int($2/1572864); if (j<1) j=1; if (j>n) j=n; print j}' /proc/meminfo)" \
 && make install && rm -rf /src/simgrid

FROM ubuntu:${UBUNTU}
ARG CUDA=12-4
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN case "${TARGETARCH:-amd64}" in arm64) repo=sbsa ;; *) repo=x86_64 ;; esac \
 && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${repo}/cuda-keyring_1.1-1_all.deb" \
         -o /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
 && apt-get update && apt-get install -y --no-install-recommends \
      cuda-compiler-${CUDA} cuda-cudart-dev-${CUDA} \
      build-essential cmake libboost-context-dev \
 && rm -rf /var/lib/apt/lists/*
COPY --from=simgrid /opt/simgrid /opt/simgrid
ENV SIMGRID_PREFIX=/opt/simgrid \
    PATH=/opt/simgrid/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/simgrid/lib:/usr/local/cuda/lib64

WORKDIR /opt/orrery
COPY . .
RUN find . -name '*.sh' -exec chmod +x {} + \
 && cmake -S . -B build \
 && cmake --build build -j"$(awk -v n="$(nproc)" '/^MemAvailable:/ {j=int($2/1572864); if (j<1) j=1; if (j>n) j=n; print j}' /proc/meminfo)" \
 && ctest --test-dir build --output-on-failure
CMD ["/bin/bash"]
