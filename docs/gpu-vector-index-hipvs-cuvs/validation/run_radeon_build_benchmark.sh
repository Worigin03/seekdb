#!/usr/bin/env bash
set -Eeuo pipefail

benchmark=$(realpath "${1:?path to benchmark_hnsw_build.py required}")
jobs=${JOBS:-8}
port=${PORT:-2981}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath "${BENCH_SOURCE_DIR:-$script_dir/../../..}")
hipvs_root=${HIPVS_ROOT:-/opt/hipvs}
rocm_root=${ROCM_ROOT:-/opt/rocm}
hipcc=${HIPCC:-$rocm_root/bin/hipcc}
mysql_bin=${MYSQL:-mysql}

if [[ ! -f "$src/build.sh" || ! -f "$src/tools/hipvs_bridge/build.sh" ]]; then
  echo "Seekdb source not found at $src. Set BENCH_SOURCE_DIR to the checkout to benchmark." >&2
  exit 2
fi
if [[ ! -f "$benchmark" || ! -x "$hipcc" || ! -f "$hipvs_root/lib/libcuvs_c.so" ]]; then
  echo "Missing benchmark, hipcc, or hipVS library. Check $benchmark, $rocm_root and $hipvs_root." >&2
  exit 2
fi
if ! command -v "$mysql_bin" >/dev/null; then
  echo "MySQL client not found: $mysql_bin" >&2
  exit 2
fi
if ! python3 - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
then
  echo "Choose a free SQL port with PORT=<port>." >&2
  exit 2
fi

if [[ -n "${BENCH_WORK:-}" ]]; then
  work=$(realpath "$BENCH_WORK")
  if [[ ! -d "$work" || "${work##*/}" != seekdb-hipvs-build.* ]]; then
    echo "BENCH_WORK must point to an existing seekdb-hipvs-build.* directory." >&2
    exit 2
  fi
else
  bench_root=${BENCH_ROOT:-${TMPDIR:-/tmp}}
  if [[ -z "${BENCH_ROOT:-}" && -d /workspace && -w /workspace ]]; then
    bench_root=/workspace
  fi
  work=$(mktemp -d "$bench_root/seekdb-hipvs-build.XXXXXX")
fi
echo "Artifacts: $work"
if [[ -e "$work/hnsw-build-results.json" ]]; then
  echo "This workspace already has results. Unset BENCH_WORK to start a fresh run." >&2
  exit 2
fi

if [[ -e "$src/.git" ]]; then
  revision=$(git -C "$src" rev-parse HEAD)
elif [[ -f "$src/.radeon-benchmark/source-revision.txt" ]]; then
  revision=$(cat "$src/.radeon-benchmark/source-revision.txt")
else
  revision=unknown
fi
echo "Using seekdb source $revision at $src"
printf '%s\n' "$revision" >"$work/source-revision.txt"

bridge=$work/bridge/libseekdb_cuvs_bridge.so
bridge_source=$src/tools/hipvs_bridge
if [[ -n "${BENCH_CAGRA_PARAMS:-}" ]]; then
  bridge_source=$work/bridge-source
  mkdir -p "$bridge_source"
  cp "$src/tools/hipvs_bridge/"{build.sh,seekdb_cuvs_bridge.h,seekdb_cuvs_bridge.c} "$bridge_source/"
  python3 - "$bridge_source/seekdb_cuvs_bridge.c" "$BENCH_CAGRA_PARAMS" <<'PY'
import re
import sys
from pathlib import Path

fields = ("graph_degree", "intermediate_graph_degree", "nn_descent_niter")
values = sys.argv[2].split(",")
if len(values) != len(fields) or any(not re.fullmatch(r"[1-9][0-9]{0,2}", v) for v in values):
    sys.exit("BENCH_CAGRA_PARAMS requires graph degree, intermediate degree, and iterations")
degree, intermediate, iterations = map(int, values)
if not (degree <= intermediate <= 256 and iterations <= 100):
    sys.exit("CAGRA tuning requires graph degree <= intermediate degree <= 256 and iterations <= 100")
path = Path(sys.argv[1])
source = path.read_text()
for field, value in zip(fields, values):
    source, count = re.subn(rf"(h->ip->{field} = )\d+;", rf"\g<1>{value};", source)
    if count != 1:
        sys.exit(f"expected exactly one assignment for {field}, found {count}")
path.write_text(source)
print(f"CAGRA parameters: graph={degree}, intermediate={intermediate}, iterations={iterations}")
PY
  printf '%s\n' "$BENCH_CAGRA_PARAMS" >"$work/cagra-params.txt"
fi
HIPVS_ROOT="$hipvs_root" ROCM_ROOT="$rocm_root" HIPCC="$hipcc" \
  bash "$bridge_source/build.sh" "$bridge"

observer=$src/build_release/src/observer/seekdb
if [[ -n "${BENCH_PREBUILT_OBSERVER:-}" ]]; then
  observer=$(realpath "$BENCH_PREBUILT_OBSERVER")
  echo "Using prebuilt seekdb: $observer"
else
  echo "Building seekdb $revision. Follow progress in $work/build.log"
  (
    cd "$src"
    bash ./build.sh release --init -DOB_BUILD_CUVS=ON -DCUVS_BRIDGE_LIB="$bridge" --make -j"$jobs"
  ) >"$work/build.log" 2>&1 &
  build_pid=$!
  while kill -0 "$build_pid" 2>/dev/null; do
    sleep 20
    if kill -0 "$build_pid" 2>/dev/null; then
      printf 'Build running (%ss). Latest log line:\n' "$SECONDS"
      tail -1 "$work/build.log"
    fi
  done
  if ! wait "$build_pid"; then
    tail -40 "$work/build.log" >&2
    exit 1
  fi
fi
if [[ ! -x "$observer" ]] || ! grep -aqF 'serialized GPU CAGRA snapshot' "$observer"; then
  echo "Build did not produce a seekdb binary with the GPU snapshot marker: $observer" >&2
  exit 1
fi
run=$work/run
mkdir -p "$run/store"
export LD_LIBRARY_PATH="$(dirname "$bridge"):$hipvs_root/lib:$rocm_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
loaded_bridge=$(ldd "$observer" | awk '$1 == "libseekdb_cuvs_bridge.so" && $2 == "=>" { print $3 }')
if [[ "$loaded_bridge" != "$bridge" ]]; then
  echo "Expected to load bridge $bridge, loader reported: ${loaded_bridge:-missing}" >&2
  exit 1
fi
mysql_cmd=("$mysql_bin" --protocol=TCP -h 127.0.0.1 -P "$port" -uroot --connect-timeout=2)

(
  cd "$run"
  exec "$observer" --nodaemon --port "$port" --base-dir "$run" --data-dir "$run/store" \
    --log-level INFO --parameter local_ip=127.0.0.1 --parameter cpu_count="$jobs" \
    --parameter __min_full_resource_pool_memory=2147483648 \
    --parameter memory_budget=4G --parameter datafile_size=20G
) >"$work/observer.out" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

echo "Starting candidate on SQL port $port"
ready=false
for _ in $(seq 1 120); do
  kill -0 "$pid" 2>/dev/null || break
  if "${mysql_cmd[@]}" -e 'select 1' >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  tail -40 "$work/observer.out" >&2
  exit 1
fi

"${mysql_cmd[@]}" -e 'alter system set ob_vector_memory_limit_percentage=30;'
python3 "$benchmark" --mysql "$mysql_bin" --port "$port" --log "$run/log/seekdb.log" \
  --output "$work/hnsw-build-results.json" "${@:2}"
