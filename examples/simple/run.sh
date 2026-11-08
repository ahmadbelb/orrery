#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; root="$(cd "$here/../.." && pwd)"
NP="${1:-4}"; M="$root/machines/example"
if [[ -n "${SIMPLE_SO:-}" ]]; then
  [[ -f "$SIMPLE_SO" ]] || { echo "SIMPLE_SO not found: $SIMPLE_SO" >&2; exit 1; }
  export ORRERY_TRANSFER_CONFIG="$M/transfers.conf"
  mapfile -t F < "$M/smpi_flags.txt"
  exec smpirun -np "$NP" --cfg=smpi/host-speed:1Gf --cfg=smpi/simulate-computation:no \
    "${F[@]}" --cfg=smpi/coll-selector:ompi \
    -platform "$M/platform.xml" -hostfile "$M/hostfile" "$SIMPLE_SO"
fi
SMPI_HOST_SPEED="${SMPI_HOST_SPEED:-1Gf}" bash "$root/run_cuda_mpi.sh" "$here/simple.cu" -np "$NP"
