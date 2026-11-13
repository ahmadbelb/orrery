#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP=""; NP=2; MACHINE="$here/machines/example"; PLAT=""; HF=""; FLAGSFILE=""
OUTSO=""; BUILD_ONLY=0; APPARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -np) NP="$2"; shift 2;;
    -machine) MACHINE="$2"; shift 2;;
    -platform) PLAT="$2"; shift 2;;
    -hostfile) HF="$2"; shift 2;;
    -flags) FLAGSFILE="$2"; shift 2;;
    -o) OUTSO="$2"; shift 2;;
    -build-only) BUILD_ONLY=1; shift;;
    --) shift; APPARGS=("$@"); break;;
    *) if [[ -z "$APP" ]]; then APP="$1"; shift; else APPARGS+=("$1"); shift; fi;;
  esac
done
[[ -n "$APP" && -f "$APP" ]] || { echo "usage: $0 <app.cu> [-np N] [-machine dir] [-platform p.xml] [-hostfile hf] [-flags file] [-o out.so] [-build-only] [-- args]"; exit 1; }
for p in "$here" "$APP" "$MACHINE" "$PLAT" "$HF" "$OUTSO" "${ORRERY_BUILD_DIR:-}"; do
  [[ "$p" != *[[:space:]]* ]] || { echo "path contains whitespace, which smpirun cannot handle: $p"; exit 1; }
done

if [[ -z "$PLAT" && -z "$HF" && -z "$FLAGSFILE" ]]; then
  PLAT="$MACHINE/platform.xml"; HF="$MACHINE/hostfile"
  FLAGSFILE="$MACHINE/smpi_flags.txt"
fi
if [[ -z "${ORRERY_TRANSFER_CONFIG:-}" && -f "$MACHINE/transfers.conf" ]]; then
  export ORRERY_TRANSFER_CONFIG="$MACHINE/transfers.conf"
fi
if [[ "$BUILD_ONLY" == 0 && -f "$HF" ]]; then
  slots="$(awk -F: 'NF { n += (NF > 1 ? $2 : 1) } END { print n + 0 }' "$HF")"
  if [[ "$NP" -gt "$slots" ]]; then
    printf '\033[31mERROR: You requested to use %s ranks, but there is only %s processes in your hostfile...\033[0m\n' "$NP" "$slots" >&2
    exit 1
  fi
fi

SMPI_FLAGS=()
if [[ -n "$FLAGSFILE" ]]; then
  [[ -f "$FLAGSFILE" ]] || { echo "flags file not found: $FLAGSFILE"; exit 1; }
  mapfile -t SMPI_FLAGS < "$FLAGSFILE"
else
  SMPI_FLAGS+=( --cfg=smpi/bw-factor:"0:1.0" --cfg=smpi/lat-factor:"0:1.0" )
fi
printf '%s\n' "${SMPI_FLAGS[@]}" | grep -q "smpi/coll-selector" \
  || SMPI_FLAGS+=( --cfg=smpi/coll-selector:ompi )
SIMCOMP=()
printf '%s\n' "${SMPI_FLAGS[@]}" | grep -q "simulate-computation" \
  || SIMCOMP=( --cfg=smpi/simulate-computation:no )

GPU_ARCH="${GPU_ARCH:-sm_80}"
if [[ -z "${HOST_CXX:-}" ]]; then
  if command -v g++-13 >/dev/null 2>&1; then HOST_CXX=g++-13; else HOST_CXX=g++; fi
fi
if [[ -z "${SIMGRID_PREFIX:-}" ]] && command -v smpirun > /dev/null; then
  SIMGRID_PREFIX="$(dirname "$(dirname "$(command -v smpirun)")")"
fi
SG="${SIMGRID_PREFIX:-/usr/local}"
bdir="${ORRERY_BUILD_DIR:-$here/build}"; mkdir -p "$bdir"
out="${OUTSO:-$bdir/$(basename "${APP%.*}")_smpi.so}"
interp="$here/src/orrery_interpose.cpp"
interp_o="$bdir/orrery_interpose.o"
xfer="$here/src/transfer_model.cpp"
xfer_o="$bdir/transfer_model.o"

echo ">> building interposer + shared transfer model"
g++ -c -fPIC -I"$SG/include" -I"$SG/include/smpi" "$interp" -o "$interp_o"
g++ -c -fPIC -I"$SG/include" -I"$SG/include/smpi" "$xfer"   -o "$xfer_o"
echo ">> building $APP -> $out  (nvcc kernels + $HOST_CXX host + SimGrid SMPI)"
rm -f "$out"
nvcc -ccbin "$HOST_CXX" -arch="$GPU_ARCH" -DSMPI_NO_OVERRIDE_MALLOC \
  ${EXTRA_DEFS:-} \
  -Xcompiler -fPIC -shared \
  -DOPEN_MPI=1 -DMPIX_CUDA_AWARE_SUPPORT=1 -I"$here/include" \
  --pre-include "$here/include/mpi-ext.h" \
  -I"$SG/include" -I"$SG/include/smpi" \
  "$APP" ${EXTRA_SRCS:-} "$interp_o" "$xfer_o" -Xlinker -Bsymbolic \
  -L"$SG/lib" -lsimgrid -lcudart -lm -ldl -o "$out"
[[ -f "$out" ]] || { echo "build failed"; exit 1; }
if [[ "$BUILD_ONLY" == 1 ]]; then echo ">> built $out"; exit 0; fi

mode="skeleton: kernels modelled, no GPU used"
conf_found=""
for cand in "${ORRERY_TRANSFER_CONFIG:-}" transfers.conf simgrid/transfers.conf; do
  [[ -n "$cand" && -f "$cand" ]] && { conf_found="$cand"; break; }
done
if [[ -n "${ORRERY_REAL_GPU:-}" ]]; then
  case "$ORRERY_REAL_GPU" in
    [1-9]*|-[1-9]*) mode="REAL GPU kernels";;
  esac
elif [[ -n "$conf_found" ]] && grep -Eq '^[[:space:]]*real_gpu[[:space:]]*=[[:space:]]*(true|1|on|yes)[[:space:]]*(#.*)?$' "$conf_found"; then
  mode="REAL GPU kernels"
fi
echo ">> machine description: ${conf_found:-<none found: built-in defaults>}"
echo ">> running under SimGrid ($mode; simulated MPI)"
exec smpirun -np "$NP" \
  --cfg=smpi/host-speed:"${SMPI_HOST_SPEED:-auto}" \
  ${SIMCOMP[@]+"${SIMCOMP[@]}"} \
  "${SMPI_FLAGS[@]}" \
  -platform "$PLAT" -hostfile "$HF" "$out" "${APPARGS[@]}"
