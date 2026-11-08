"""Run stock CAGRA timings and compare two-hop detour counters on Radeon."""

import argparse
from collections import defaultdict
import csv
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


DEGREE = 256
ITERATIONS = 48


def proxy_base(rows, dimension):
    # Same base vectors as the earlier Radeon bridge benchmark.
    rng = np.random.default_rng(8643 + dimension)
    centers = rng.normal(0, 0.30, size=(192, dimension)).astype(np.float32)
    scales = np.geomspace(1, 0.18, dimension, dtype=np.float32)
    vectors = rng.standard_normal((rows, dimension), dtype=np.float32)
    for start in range(0, rows, 4096):
        batch = vectors[start:start + 4096]
        batch *= scales
        batch += centers[rng.integers(0, len(centers), len(batch))]
    return vectors


class DeviceMemory:
    def __init__(self):
        self.counters = sorted(Path("/sys/class/drm").glob("card[0-9]*/device/mem_info_vram_used"))
        self.stop = threading.Event()
        self.start = self.sample()
        self.peak = self.start.copy()

    def sample(self):
        result = {}
        for counter in self.counters:
            try:
                result[counter.parent.parent.name] = int(counter.read_text())
            except (OSError, ValueError):
                pass
        return result

    def monitor(self):
        while not self.stop.wait(0.05):
            for card, used in self.sample().items():
                self.peak[card] = max(used, self.peak.get(card, 0))


def read_function(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


def stock_build(work, base):
    library = ct.CDLL(str(work / "libseekdb_cuvs_bridge.so"))
    build = read_function(library, "seekdb_cuvs_build", ct.c_void_p, ct.c_void_p, ct.c_long, ct.c_long)
    free = read_function(library, "seekdb_cuvs_free", None, ct.c_void_p)
    serialize = read_function(library, "seekdb_cuvs_serialize", ct.c_int, ct.c_void_p, ct.c_char_p)
    search = read_function(library, "seekdb_cuvs_search", ct.c_int, ct.c_void_p, ct.c_void_p,
                           ct.c_long, ct.c_long, ct.c_void_p, ct.c_void_p)
    snapshot = work / "stock.snapshot"
    start = time.perf_counter()
    handle = build(base.ctypes.data, *base.shape)
    built = time.perf_counter()
    if not handle:
        raise RuntimeError("Stock CAGRA build failed.")
    try:
        if serialize(handle, os.fsencode(snapshot)):
            raise RuntimeError("Stock CAGRA serialization failed.")
        saved = time.perf_counter()
        ids = np.empty(10, np.uint32)
        distances = np.empty(10, np.float32)
        if search(handle, base[0].ctypes.data, 1, 10, ids.ctypes.data, distances.ctypes.data):
            raise RuntimeError("Stock CAGRA first query failed.")
        if len(np.unique(ids)) != len(ids) or np.any(ids >= len(base)):
            raise RuntimeError("Stock CAGRA first query returned invalid IDs.")
        return {"stock_build_s": built - start, "stock_snapshot_s": saved - built,
                "stock_build_snapshot_first_s": time.perf_counter() - start}
    finally:
        free(handle)
        snapshot.unlink(missing_ok=True)


def oracle_detours(graph):
    rows, degree = graph.shape
    output = np.zeros(graph.shape, np.uint8)
    for row, direct in enumerate(graph):
        counts = [0] * degree
        for rank, via in enumerate(direct[:-1]):
            if via < rows:
                for target in graph[via]:
                    for destination_rank in range(rank + 1, degree):
                        if direct[destination_rank] == target:
                            counts[destination_rank] += 1
                            break
        output[row] = [min(count, 255) for count in counts]
    return output


def probe_detours(work, base):
    library = ct.CDLL(str(work / "libcagra_prune_probe.so"))
    pointer, uint = ct.c_void_p, ct.c_uint
    build = read_function(library, "seekdb_probe_nn_descent", ct.c_int,
                          pointer, uint, uint, uint, uint, pointer, ct.POINTER(ct.c_double))
    compare = read_function(library, "seekdb_probe_detours", ct.c_int,
                            pointer, uint, uint, pointer, pointer, ct.POINTER(ct.c_float))

    def check(graph):
        reference = np.empty(graph.shape, np.uint8)
        optimized = np.empty_like(reference)
        times = (ct.c_float * 3)()
        status = compare(graph.ctypes.data, *graph.shape, reference.ctypes.data,
                         optimized.ctypes.data, times)
        if status:
            raise RuntimeError(f"Detour comparison exited with status {status}.")
        return reference, list(times)

    # Exercise duplicate targets, invalid IDs, and rank-dependent matching.
    malformed = np.array([[1, 2, 1, 9], [2, 0, 4, 3], [3, 1, 2, 0], [0, 1, 2, 3]], np.uint32)
    small, _ = check(malformed)
    if not np.array_equal(small, oracle_detours(malformed)):
        raise RuntimeError("GPU reference disagrees with the independent CPU oracle.")
    seconds = ct.c_double()
    graph = np.empty((len(base), DEGREE), np.uint32)
    if build(base.ctypes.data, *base.shape, DEGREE, ITERATIONS, graph.ctypes.data, ct.byref(seconds)):
        raise RuntimeError("Could not export the intermediate NN-Descent graph.")
    _, times = check(graph)
    return {"nn_descent_s": seconds.value, "exact_counts_match": True,
            "compared_edges": int(graph.size),
            "linear_kernel_ms": times[0], "hash_one_wave_ms": times[1],
            "hash_128_threads_ms": times[2]}


def profile_summary(path):
    counts = defaultdict(int)
    durations = defaultdict(float)
    copy_counts = defaultdict(int)
    copy_durations = defaultdict(float)
    reverse_ranges = []
    markers = []
    for source in sorted(path.rglob("*.csv")):
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("Kernel_Name", "")
                try:
                    start = int(row["Start_Timestamp"])
                    end = int(row["End_Timestamp"])
                except (KeyError, ValueError, TypeError):
                    continue
                duration = (end - start) / 1e6
                if "kern_prune" in name:
                    phase = "stock_prune_gpu"
                elif "kern_make_rev_graph" in name:
                    phase = "stock_reverse_gpu"
                    reverse_ranges.append((start, end))
                elif "Kernel_Name" in row:
                    phase = "other_gpu"
                else:
                    direction = row.get("Direction")
                    if direction:
                        copy_counts[direction] += 1
                        copy_durations[direction] += duration
                    label = row.get("Message") or row.get("Name") or row.get("Function") or ""
                    if "cagra::graph::optimize" in label and duration > 0:
                        markers.append({"label": label, "wall_ms": duration})
                    continue
                counts[phase] += 1
                durations[phase] += duration
    result = {"launches": dict(counts), "kernel_ms": dict(durations), "optimize_markers": markers,
              "copy_counts": dict(copy_counts), "copy_ms": dict(copy_durations)}
    if reverse_ranges:
        # Includes stalls between dispatches, but excludes work before the first
        # reverse kernel and after the last one.
        result["reverse_kernel_span_ms"] = (
            max(end for _, end in reverse_ranges) - min(start for start, _ in reverse_ranges)
        ) / 1e6
    return result


def run(work, rows, hipvs_root, rocm_root):
    hipcc = Path(os.environ.get("HIPCC", str(rocm_root / "bin" / "hipcc")))
    profiler = shutil.which("rocprofv3") or str(rocm_root / "bin" / "rocprofv3")
    env = dict(os.environ, HIPVS_ROOT=str(hipvs_root), ROCM_ROOT=str(rocm_root), HIPCC=str(hipcc),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "24"))
    library_dirs = [str(hipvs_root / "lib"), str(rocm_root / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        library_dirs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(library_dirs)
    if not Path("/dev/kfd").exists() or not hipcc.is_file():
        raise RuntimeError("需要在有 Radeon GPU 和 hipcc 的 Linux 工作区运行。")
    root = Path(__file__).resolve().parent
    report = {"rows": rows, "intermediate_degree": DEGREE, "iterations": ITERATIONS, "runs": {}}
    result = work / "hipvs-pruning-profile.json"

    def save():
        pending = result.with_suffix(".tmp")
        pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending.replace(result)

    def execute(name, command):
        log = work / (name + ".log")
        with log.open("w", encoding="utf-8") as output:
            try:
                status = subprocess.run(command, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=240).returncode
            except subprocess.TimeoutExpired:
                status = 124
        if status:
            tail = log.read_text(encoding="utf-8", errors="replace")[-3500:]
            print(f"{name} 失败 ({status})\n{tail}", flush=True)
            return {"exit_code": status, "log_tail": tail}
        return {"exit_code": 0}

    save()
    print(f"诊断目录 {work}", flush=True)
    report["stock_compile"] = execute("stock_compile", ["bash", str(root.parent / "build.sh"),
                                                       str(work / "libseekdb_cuvs_bridge.so"), "--stock"])
    report["prune_compile"] = execute("prune_compile", [
        str(hipcc), "-shared", "-fPIC", "-O3", "-std=c++17", "-D__HIP_PLATFORM_AMD__",
        f"-I{hipvs_root / 'include'}", f"-I{rocm_root / 'include'}",
        str(root / "cagra_detours.hip"), str(root / "cagra_prune_probe.cpp"),
        f"-L{hipvs_root / 'lib'}", f"-Wl,-rpath,{':'.join(library_dirs[:2])}",
        "-lcuvs_c", "-o", str(work / "libcagra_prune_probe.so"),
    ])
    save()
    for dimension in (128, 256):
        base = proxy_base(rows, dimension)
        base_file = work / f"base{dimension}.npy"
        np.save(base_file, base)
        del base
        record = report["runs"][str(dimension)] = {}
        try:
            for mode, compile_label in (("stock", "stock_compile"), ("detours", "prune_compile"),
                                        ("profile", "stock_compile")):
                if report[compile_label]["exit_code"]:
                    continue
                target = work / f"{dimension}-{mode}.json"
                child_mode = "stock" if mode == "profile" else mode
                command = [sys.executable, __file__, "--child", child_mode, "--work", str(work),
                           "--base", str(base_file), "--result", str(target)]
                if mode == "profile":
                    if not Path(profiler).is_file():
                        record[mode] = {"unavailable": "rocprofv3 is not installed"}
                        continue
                    trace = work / f"trace{dimension}"
                    trace.mkdir()
                    command = [profiler, "--kernel-trace", "--marker-trace", "--memory-copy-trace",
                               "--output-format", "csv", "--output-directory", str(trace), "--", *command]
                print(f"{dimension}D {mode}", flush=True)
                stage = record[mode] = execute(f"{dimension}-{mode}", command)
                if target.is_file():
                    stage.update(json.loads(target.read_text()))
                elif not stage["exit_code"]:
                    stage["error"] = "Process exited without producing timing data."
                if mode == "profile" and not stage["exit_code"]:
                    stage.update(profile_summary(trace))
                save()
        finally:
            base_file.unlink(missing_ok=True)
    save()
    print(f"JSON: {result}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--hipvs-root", type=Path, default=os.environ.get("HIPVS_ROOT", "/opt/hipvs"))
    parser.add_argument("--rocm-root", type=Path, default=os.environ.get("ROCM_ROOT", "/opt/rocm"))
    parser.add_argument("--child", choices=("stock", "detours"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    if args.child:
        data = np.load(args.base, mmap_mode="r")
        memory = DeviceMemory()
        monitor = threading.Thread(target=memory.monitor, daemon=True)
        monitor.start()
        try:
            outcome = stock_build(args.work, data) if args.child == "stock" else probe_detours(args.work, data)
        finally:
            memory.stop.set()
            monitor.join()
        outcome["sampled_vram_start_bytes"] = memory.start
        outcome["sampled_vram_peak_bytes"] = memory.peak
        args.result.write_text(json.dumps(outcome), encoding="utf-8")
        return
    if args.rows <= DEGREE * 3 // 2:
        parser.error("row count must exceed NN-Descent's internal degree")
    work = Path(tempfile.mkdtemp(prefix="hipvs-pruning-", dir=Path.cwd()))
    run(work, args.rows, args.hipvs_root, args.rocm_root)


if __name__ == "__main__":
    main()
