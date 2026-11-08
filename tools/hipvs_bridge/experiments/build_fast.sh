#!/usr/bin/env bash
set -Eeuo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=${1:?specify output library}
hipvs=${HIPVS_ROOT:-/opt/hipvs}
rocm=${ROCM_ROOT:-/opt/rocm}
hipcc=${HIPCC:-$rocm/bin/hipcc}

[[ -x "$hipcc" ]] || { echo "Missing hipcc: $hipcc" >&2; exit 2; }
[[ -f "$hipvs/include/cuvs/neighbors/cagra.h" ]] || {
  echo "Missing hipVS CAGRA header: $hipvs/include" >&2
  exit 2
}
mkdir -p "$(dirname "$output")"
bridge_object=$(mktemp "$(dirname "$output")/seekdb-bridge.XXXXXXXX.o")
trap 'rm -f -- "$bridge_object"' EXIT
includes=(-I"$hipvs/include" -I"$rocm/include" -D__HIP_PLATFORM_AMD__)
if [[ -f "$hipvs/include/cuvs/neighbors/ivf_flat.h" ]]; then
  includes+=(-DSEEKDB_IVF_FLAT_BUILD)
fi
# Include the evaluation GPU even when the build worker has no GPU attached.
offload=(--offload-arch=gfx1100)
if [[ -n "${HIPVS_GPU_ARCH:-}" ]]; then
  offload=("--offload-arch=$HIPVS_GPU_ARCH")
elif [[ -x "$rocm/bin/rocm_agent_enumerator" ]]; then
  while IFS= read -r arch; do
    if [[ "$arch" =~ ^gfx[0-9a-f]+$ && "$arch" != gfx000 && "$arch" != gfx1100 ]]; then
      offload+=("--offload-arch=$arch")
    fi
  done < <("$rocm/bin/rocm_agent_enumerator" -t GPU 2>/dev/null | sort -u)
fi
"$hipcc" -fPIC -O2 "${includes[@]}" -DSEEKDB_EXPERIMENTAL_GPU_PRUNE \
  -c "$here/../seekdb_cuvs_bridge.c" -o "$bridge_object"
"$hipcc" -shared -fPIC -O3 -std=c++17 -pthread "${includes[@]}" "${offload[@]}" \
  "$bridge_object" "$here/cagra_detours.hip" "$here/cagra_prune_probe.cpp" \
  "$here/cagra_fast_build.cpp" -L"$hipvs/lib" \
  -Wl,-z,defs -Wl,-soname,libseekdb_cuvs_bridge.so -Wl,-rpath,"$hipvs/lib:$rocm/lib" \
  -lcuvs_c -o "$output"
