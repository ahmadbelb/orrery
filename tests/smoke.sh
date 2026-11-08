#!/usr/bin/env bash
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ge 1 ]]; then
  export SIMPLE_SO="$1"; so="$1"
else
  export ORRERY_BUILD_DIR="$root/build/tests"; so="$ORRERY_BUILD_DIR/simple_smpi.so"
fi
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fail=0
ok()  { echo "  ok    $1"; }
bad() { echo "  FAIL  $1"; fail=$(( fail + 1 )); }

echo "== 1/3 run the example on 4 simulated GPUs"
if bash "$root/examples/simple/run.sh" > "$tmp/run.log" 2>&1; then ok "run completed"
else bad "run failed; last lines:"; tail -15 "$tmp/run.log"; fi

echo "== 2/3 simulated time"
REF=0.005223
got="$(sed -n 's/.*simulated time \([0-9.]*\) s.*/\1/p' "$tmp/run.log" | head -1)"
if [[ -n "$got" ]] && awk -v g="$got" -v r="$REF" 'BEGIN{d=g-r; if(d<0)d=-d; exit !(d<0.00005)}'; then
  ok "simulated time $got s (reference $REF s)"
else bad "simulated time '${got:-none}' differs from reference $REF s"; fi

echo "== 3/3 interposition symbols"
if [[ -f "$so" ]]; then
  syms="$(nm -D "$so")"
  c="$(grep -c ' T cuda' <<< "$syms")"; m="$(grep -c ' W MPI_' <<< "$syms")"
  [[ "$c" == 39 ]] && ok "39 CUDA runtime wrappers defined" || bad "CUDA wrappers: $c (expected 39)"
  [[ "$m" == 21 ]] && ok "21 MPI wrappers (weak)"           || bad "MPI wrappers: $m (expected 21)"
  grep -q ' U smpi_execute_benched' <<< "$syms" \
    && ok "smpi_execute_benched left undefined (resolved by SimGrid)" \
    || bad "smpi_execute_benched is not an undefined symbol"
else bad "no object at $so"; fi

echo
if [[ "$fail" == 0 ]]; then echo "all checks passed"; else echo "$fail check(s) failed"; fi
exit "$fail"
