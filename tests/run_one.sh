#!/usr/bin/env bash
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; so="$2"
[[ -f "$so" ]] || { echo "no such object: $so" >&2; exit 1; }

unset ORRERY_REAL_GPU
real=()
if [[ "$name" == *_real ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q GPU; then
    echo "SKIP: $name needs a local GPU"; exit 77
  fi
  [[ "$name" != *conf_real_gpu* ]] && real=(ORRERY_REAL_GPU=1)
fi
conf="$here/transfers_test.conf"
[[ -f "$here/conf/$name.conf" ]] && conf="$here/conf/$name.conf"
if [[ -f "$here/conf/$name.env" ]]; then
  set -a; . "$here/conf/$name.env"; set +a
fi
out="$(env ORRERY_TRANSFER_CONFIG="$conf" ${real[@]+"${real[@]}"} \
  smpirun -np 2 --cfg=smpi/host-speed:100Gf \
    --cfg=smpi/lat-factor:0:1.0 --cfg=smpi/bw-factor:0:1.0 \
    --cfg=smpi/simulate-computation:no --cfg=smpi/wtime:0 --cfg=smpi/coll-selector:ompi \
    -platform "$here/platform_test.xml" -hostfile "$here/hostfile_test" "$so" 2>&1)"
status=$?
echo "$out" | grep -vE "xbt_cfg/INFO|smpi_bench/INFO|smpi_coll/INFO|smpi_colls/INFO|breakdown rank0|^\[kernel\]"
grep -q "RESULT: PASS" <<< "$out" && exit 0
echo "smpirun exit status $status" >&2; exit 1
