#!/usr/bin/env python3

import os
import sys

import argparse
import logging
import itertools

import inspect
import types
import subprocess
from pathlib import Path
from typing import Any, Optional
import time
import datetime

# With profiler the resolution is higher, requiring fewer n_vecs
GTX_1060_TOTAL_SM = 10
GTX_1060_WARPS_PER_SM = 64
GTX_1060_TOTAL_WARPS = GTX_1060_TOTAL_SM * GTX_1060_WARPS_PER_SM
TARGET_WAVES = 20
MAX_CONCURRENT_VECTORS_PER_WARP = 8
DATA_N_VECS = GTX_1060_TOTAL_WARPS * MAX_CONCURRENT_VECTORS_PER_WARP * TARGET_WAVES

MICROBENCHMARK_EXECUTABLE = "./bin/micro-benchmarks"

FLS_TYPES = ["u32", "u64"]
ALP_TYPES = ["f32", "f64"]
KERNELS = ["decompress", "query"]
UNPACK_N_VECS = [1, 4]
UNPACK_N_VALS = [1]
UNPACKERS = [
    "dummy",
    "old-fls",
    #"SwitchCase",
    #"Stateless",
    #"StatelessBranchless",
    #"StatefulCache",
    #"StatefulLocal1",
    #"StatefulLocal2",
    #"StatefulLocal4",
    #"StatefulShared1",
    #"StatefulShared2",
    #"StatefulShared4",
    #"StatefulRegister1",
    #"StatefulRegister2",
    #"StatefulRegister4",
    #"StatefulRegisterBranchless1",
    #"StatefulRegisterBranchless2",
    #"StatefulRegisterBranchless4",
    "stateful-branchless",
]
MULTI_COLUMN_UNPACKERS = [
    UNPACKERS[1],
    UNPACKERS[2],
]

PATCHERS = [
    "none",
    "dummy",
    "stateless",
    "stateful",
    "naive",
    "naive-branchless",
    "prefetch-all",
    "prefetch-all-branchless",
]
MULTI_COLUMN_PATCHERS = [
    PATCHERS[0],
    PATCHERS[3],
    PATCHERS[4],
    PATCHERS[5],
    PATCHERS[6],
    PATCHERS[7],
]

# ===========
# Compressors
# ===========
COMPRESSORS_EXECUTABLE = "./bin/compressors-benchmarks"
COMPARISON_KERNELS = ["decompression", "decompression_query"]
COMPRESSORS = [
    "Thrust",
    "ALP",
    "GALP",
    "Bitcomp",
    "BitcompSparse",
    "LZ4",
    "zstd",
    "Deflate",
    "GDeflate",
    "Snappy",
]
DECOMPRESSION_COMPRESSORS = COMPRESSORS[1:]
DECOMPRESSION_QUERY_COMPRESSORS = COMPRESSORS[0:]

NVVP_PATH = "/usr/local/cuda-12.5/bin/nvprof"
NVVP_METRICS = {
    # Warps / SM
    "sm_efficiency": "The percentage of time at least one warp is active on a specific multiprocessor",
    "eligible_warps_per_cycle": "Average number of warps that are eligible to issue per active cycle",
    "unique_warps_launched": "Number of warps launched. Value is unaffected by compute preemption.",
    "warp_execution_efficiency": "Ratio of the average active threads per warp to the maximum number of threads per warp supported on a multiprocessor",
    "warp_nonpred_execution_efficiency": "Ratio of the average active threads per warp executing non-predicated instructions to the maximum number of threads per warp supported on a multiprocessor",
    "achieved_occupancy": "Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor",
    "branch_efficiency": "Ratio of non-divergent branches to total branches expressed as percentage",
    # Instructions
    "ldst_executed": "Number of executed local, global, shared and texture memory load and store instructions",
    "ldst_issued": "Number of issued local, global, shared and texture memory load and store instructions",
    "inst_compute_ld_st": "Number of compute load/store instructions executed by non-predicated threads",
    "inst_control": "Number of control-flow instructions executed by non-predicated threads (jump, branch, etc.)",
    "inst_executed_global_loads": "Warp level instructions for global loads",
    "inst_executed_global_stores": "Warp level instructions for global stores",
    "inst_executed_local_loads": "Warp level instructions for local loads",
    "inst_executed_local_stores": "Warp level instructions for local stores",
    "inst_executed": "The number of instructions executed",
    "inst_fp_32": "Number of single-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_fp_64": "Number of double-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)",
    "inst_integer": "Number of integer instructions executed by non-predicated threads",
    "inst_issued": "The number of instructions issued",
    "inst_per_warp": "Average number of instructions executed by each warp",
    "cf_executed": "Number of executed control-flow instructions",
    "cf_fu_utilization": "The utilization level of the multiprocessor function units that execute control-flow instructions on a scale of 0 to 10",
    "cf_issued": "Number of issued control-flow instructions",
    "special_fu_utilization": "The utilization level of the multiprocessor function units that execute sin, cos, ex2, popc, flo, and similar instructions on a scale of 0 to 10",
    "inst_bit_convert": "Number of bit-conversion instructions executed by non-predicated threads",
    # Issued instructions
    "ipc": "Instructions executed per cycle",
    "issued_ipc": "Instructions issued per cycle",
    "issue_slots": "The number of issue slots used",
    "issue_slot_utilization": "Percentage of issue slots that issued at least one instruction, averaged across all cycles",
    # Stalls
    "stall_constant_memory_dependency": "Percentage of stalls occurring because of immediate constant cache miss",
    "stall_exec_dependency": "Percentage of stalls occurring because an input required by the instruction is not yet available",
    "stall_inst_fetch": "Percentage of stalls occurring because the next assembly instruction has not yet been fetched",
    "stall_memory_dependency": "Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding",
    "stall_memory_throttle": "Percentage of stalls occurring because of memory throttle",
    "stall_not_selected": "Percentage of stalls occurring because warp was not selected",
    "stall_other": "Percentage of stalls occurring due to miscellaneous reasons",
    "stall_pipe_busy": "Percentage of stalls occurring because a compute operation cannot be performed because the compute pipeline is busy",
    "stall_sync": "Percentage of stalls occurring because the warp is blocked at a __syncthreads() call",
    "stall_texture": "Percentage of stalls occurring because the texture sub-system is fully utilized or has too many outstanding requests",
    # Global memory
    "gld_efficiency": "Ratio of requested global memory load throughput to required global memory load throughput expressed as percentage.",
    "gld_requested_throughput": "Requested global memory load throughput",
    "gld_throughput": "Global memory load throughput",
    "gld_transactions": "Number of global memory load transactions",
    "gld_transactions_per_request": "Average number of global memory load transactions performed for each global memory load.",
    "global_hit_rate": "Hit rate for global loads in unified l1/tex cache. Metric value maybe wrong if malloc is used in kernel.",
    "global_load_requests": "Total number of global load requests from Multiprocessor",
    "global_store_requests": "Total number of global store requests from Multiprocessor. This does not include atomic requests.",
    "gst_efficiency": "Ratio of requested global memory store throughput to required global memory store throughput expressed as percentage.",
    "gst_requested_throughput": "Requested global memory store throughput",
    "gst_throughput": "Global memory store throughput",
    "gst_transactions": "Number of global memory store transactions",
    "gst_transactions_per_request": "Average number of global memory store transactions performed for each global memory store",
    # Local memory
    "local_hit_rate": "Hit rate for local loads and stores",
    "local_load_requests": "Total number of local load requests from Multiprocessor",
    "local_load_throughput": "Local memory load throughput",
    "local_load_transactions": "Number of local memory load transactions",
    "local_load_transactions_per_request": "Average number of local memory load transactions performed for each local memory load",
    "local_memory_overhead": "Ratio of local memory traffic to total memory traffic between the L1 and L2 caches expressed as percentage",
    "local_store_requests": "Total number of local store requests from Multiprocessor",
    "local_store_throughput": "Local memory store throughput",
    "local_store_transactions": "Number of local memory store transactions",
    "local_store_transactions_per_request": "Average number of local memory store transactions performed for each local memory store",
    # Unified / tex cache (L1 Cache)
    "tex_cache_hit_rate": "Unified cache hit rate",
    "tex_cache_throughput": "Unified cache throughput",
    "tex_cache_transactions": "Unified cache read transactions",
    "tex_fu_utilization": "The utilization level of the multiprocessor function units that execute global, local and texture memory instructions on a scale of 0 to 10",
    "tex_utilization": "The utilization level of the unified cache relative to the peak utilization on a scale of 0 to 10",
    # L2
    "l2_global_load_bytes": "Bytes read from L2 for misses in Unified Cache for global loads",
    "l2_global_reduction_bytes": "Bytes written to L2 from Unified cache for global reductions",
    "l2_local_global_store_bytes": "Bytes written to L2 from Unified Cache for local and global stores. This does not include global atomics.",
    "l2_local_load_bytes": "Bytes read from L2 for misses in Unified Cache for local loads",
    "l2_read_throughput": "Memory read throughput seen at L2 cache for all read requests",
    "l2_read_transactions": "Memory read transactions seen at L2 cache for all read requests",
    "l2_surface_load_bytes": "Bytes read from L2 for misses in Unified Cache for surface loads",
    "l2_surface_reduction_bytes": "Bytes written to L2 from Unified Cache for surface reductions",
    "l2_surface_store_bytes": "Bytes written to L2 from Unified Cache for surface stores. This does not include surface atomics.",
    "l2_tex_hit_rate": "Hit rate at L2 cache for all requests from texture cache",
    "l2_tex_read_hit_rate": "Hit rate at L2 cache for all read requests from texture cache",
    "l2_tex_read_throughput": "Memory read throughput seen at L2 cache for read requests from the texture cache",
    "l2_tex_read_transactions": "Memory read transactions seen at L2 cache for read requests from the texture cache",
    "l2_tex_write_hit_rate": "Hit Rate at L2 cache for all write requests from texture cache",
    "l2_tex_write_throughput": "Memory write throughput seen at L2 cache for write requests from the texture cache",
    "l2_tex_write_transactions": "Memory write transactions seen at L2 cache for write requests from the texture cache",
    "l2_utilization": "The utilization level of the L2 cache relative to the peak utilization on a scale of 0 to 10",
    "l2_write_throughput": "Memory write throughput seen at L2 cache for all write requests",
    "l2_write_transactions": "Memory write transactions seen at L2 cache for all write requests",
    # DRAM
    "dram_read_bytes": "Total bytes read from DRAM to L2 cache",
    "dram_read_throughput": "Device memory read throughput",
    "dram_read_transactions": "Device memory read transactions",
    "dram_utilization": "The utilization level of the device memory relative to the peak utilization on a scale of 0 to 10",
    "dram_write_bytes": "Total bytes written from L2 cache to DRAM",
    "dram_write_throughput": "Device memory write throughput",
    "dram_write_transactions": "Device memory write transactions",
    # Sysmem
    "sysmem_read_bytes": "Number of bytes read from system memory",
    "sysmem_read_throughput": "System memory read throughput",
    "sysmem_read_transactions": "Number of system memory read transactions",
    "sysmem_read_utilization": "The read utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "sysmem_utilization": "The utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
    "sysmem_write_bytes": "Number of bytes written to system memory",
    "sysmem_write_throughput": "System memory write throughput",
    "sysmem_write_transactions": "Number of system memory write transactions",
    "sysmem_write_utilization": "The write utilization level of the system memory relative to the peak utilization on a scale of 0 to 10",
}


class Stopwatch:
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self):
        assert self._start is None, "Stopwatch already started"
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self._start is not None, "Stopwatch not started"
        assert self._end is None, "Stopwatch already stopped"
        self._end = time.perf_counter()
        return self._end - self._start


def has_root_privileges() -> bool:
    return os.geteuid() == 0


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_all_files_in_dir(dir: str) -> list[str]:
    file_paths = []

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths


def get_benchmarking_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    benchmarking_function_prefix = "bench_"
    benchmarking_functions = filter(
        lambda x: x[0].startswith(benchmarking_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(benchmarking_function_prefix, ""), x[1]),
        benchmarking_functions,
    )
    return list(stripped_prefixes_from_name)


def does_file_exist(file_path: str) -> bool:
    return os.path.isfile(file_path)


def execute_command(
    command: str, out: str, save_stderr: bool = False, append_signal: bool = False
):
    if args.only_new_runs and does_file_exist(out):
        logging.debug(f"Skipping command, output file exists already: {out}")
        return

    if args.dry_run:
        print(command, file=sys.stderr)
        return

    stopwatch: Stopwatch = Stopwatch().start()
    logging.debug(f"Started @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logging.info(f"Executed in {stopwatch.stop():.03f}s: {command}")

    if result.returncode != 0:
        logging.critical(f"STDOUT: {result.stdout}")
        logging.critical(f"STDERR: {result.stderr}")
        logging.critical(f"Exited with code {result.returncode}: {command}")

        if args.exit_non_zero_exit_code:
            exit(0)
        return

    if out:
        to_write = result.stderr if save_stderr else result.stdout
        if append_signal:
            # signal is a negative return code
            to_write += f"\nRETURN_CODE,{abs(result.returncode)}"
        with open(out, "w") as f:
            f.write(to_write)


class NoProfiler:
    def benchmark_command(
        self,
        command: str,
        out: str,
        save_stderr: bool = False,
        append_signal: bool = False,
    ) -> None:
        execute_command(
            command,
            out,
            save_stderr=save_stderr,
            append_signal=append_signal,
        )


class NCUProfiler:
    def benchmark_command(
        self,
        command: str,
        out: str,
        save_stderr: bool = False,
        append_signal: bool = False,
    ) -> None:
        assert has_root_privileges(), "Reading GPU performance counters requires sudo, rerun program with sudo rights"
        execute_command(
            f"ncu --csv {command}",
            out,
            save_stderr=save_stderr,
            append_signal=append_signal,
        )


class NVVPProfiler:
    def _create_metrics_parameter(self, metrics: list[str]) -> str:
        assert all([m in NVVP_METRICS for m in metrics])
        return "--metrics " + ",".join(metrics) + " "

    def benchmark_command(
        self,
        command: str,
        out: str,
        save_stderr: bool = True,
        append_signal: bool = False,
    ) -> None:
        assert has_root_privileges(), "Reading GPU performance counters requires sudo, rerun program with sudo rights"
        command = f"{NVVP_PATH} --print-gpu-trace {command}"
        metrics = None
        if metrics is not None:
            execute_command(
                "sudo " + command + self._create_metrics_parameter(metrics),
                out,
                save_stderr,
                append_signal=append_signal,
            )
        else:
            execute_command(command, out, save_stderr, append_signal=append_signal)


def get_profiler(args):
    if args.profiler == "ncu":
        return NCUProfiler() 
    if args.profiler == "nvvp":
        return NVVPProfiler() 
    if args.profiler == "none":
        return NoProfiler() 
    raise Exception("Did not recognize profiler")


def format_file_parameter(parameter: str) -> str:
    return parameter.replace("-", "_").replace(".", "_")


def convert_list_to_str(values: list[Any]) -> list[str]:
    return list(map(str, values))


def bench_ffor(output_dir: str, profiler):
    for data_type, kernel, n_vecs, n_vals, unpacker in itertools.product(
        FLS_TYPES, KERNELS, UNPACK_N_VECS, UNPACK_N_VALS, UNPACKERS
    ):
        # For switch cace only single vec and single value are supported
        if unpacker == UNPACKERS[2] and (n_vecs != 1 or n_vals != 1):
            continue
        # For OldFls only single vec and 32 values are supported
        if unpacker == UNPACKERS[1]:
            if n_vecs != 1 or n_vals != 1 or "32" not in data_type:
                continue
            n_vals = 32

        parameters = convert_list_to_str([data_type, kernel, n_vecs, n_vals, unpacker])

        vbw_start = 0
        vbw_end = data_type[1:]

        # If the unpacker is of type dummy, only max bit width is supported
        if unpacker == "dummy":
            vbw_start = data_type[1:]

        out = os.path.join(
            output_dir,
            "ffor-"
            + "-".join(map(format_file_parameter, parameters))
            + f"-{vbw_start}-{vbw_end}-{DATA_N_VECS}-{args.number_sampling_runs}",
        )
        profiler.benchmark_command(
            MICROBENCHMARK_EXECUTABLE
            + " "
            + " ".join(parameters)
            + " "
            + f"none {vbw_start} {vbw_end} 0 0 {DATA_N_VECS} {args.number_sampling_runs} 0",
            out=out,
        )


def bench_alp_ec(output_dir: str, profiler):
    for data_type, kernel, n_vecs, n_vals, unpacker, patcher in itertools.product(
        ALP_TYPES,
        KERNELS,
        UNPACK_N_VECS,
        UNPACK_N_VALS,
        UNPACKERS[1:],
        PATCHERS[1:],
    ):
        # For OldFls only single vec and 32 values are supported
        if unpacker == UNPACKERS[1]:
            if n_vecs != 1 or n_vals != 1 or "32" not in data_type:
                continue
            n_vals = 32

        parameters = convert_list_to_str(
            [data_type, kernel, n_vecs, n_vals, unpacker, patcher]
        )

        vbw_start = 3
        vbw_end = 8
        ec_start = 0
        ec_end = 50

        # If the patcher is of type dummy only 0 exceptions is supported
        if patcher == "dummy":
            ec_end = 0

        out = os.path.join(
            output_dir,
            "alp-ec-"
            + "-".join(map(format_file_parameter, parameters))
            + f"-{vbw_start}-{vbw_end}-{ec_start}-{ec_end}-{DATA_N_VECS}-{args.number_sampling_runs}",
        )
        profiler.benchmark_command(
            MICROBENCHMARK_EXECUTABLE
            + " "
            + " ".join(parameters)
            + " "
            + f"{vbw_start} {vbw_end} {ec_start} {ec_end} {DATA_N_VECS} {args.number_sampling_runs} 0",
            out=out,
        )


def bench_multi_column(output_dir: str, profiler):
    kernel = "query-multi-column"

    # Test ffor multicolumn
    for data_type, n_vecs, n_vals, unpacker in itertools.product(
        FLS_TYPES, UNPACK_N_VECS, UNPACK_N_VALS, MULTI_COLUMN_UNPACKERS
    ):
        # For OldFls only single vec and 32 values are supported
        if unpacker == UNPACKERS[1]:
            if n_vecs != 1 or n_vals != 1 or "32" not in data_type:
                continue
            n_vals = 32

        patcher = PATCHERS[0]
        parameters = convert_list_to_str(
            [data_type, kernel, n_vecs, n_vals, unpacker, patcher]
        )

        vbw_start = 0
        vbw_end = data_type[1:]

        out = os.path.join(
            output_dir,
            "multi-column-"
            + "-".join(map(format_file_parameter, parameters))
            + f"-{vbw_start}-{vbw_end}-0-0-{DATA_N_VECS}-{args.number_sampling_runs}",
        )
        profiler.benchmark_command(
            MICROBENCHMARK_EXECUTABLE
            + " "
            + " ".join(parameters)
            + " "
            + f"{vbw_start} {vbw_end} 0 0 {DATA_N_VECS} {args.number_sampling_runs} 0",
            out=out,
        )

    # Test ALP multicolumn
    for data_type, n_vecs, n_vals, unpacker, patcher in itertools.product(
        ALP_TYPES,
        UNPACK_N_VECS,
        UNPACK_N_VALS,
        MULTI_COLUMN_UNPACKERS,
        MULTI_COLUMN_PATCHERS[1:],
    ):
        # For OldFls only single vec and 32 values are supported
        if unpacker == UNPACKERS[1]:
            if n_vecs != 1 or n_vals != 1 or "32" not in data_type:
                continue
            n_vals = 32

        parameters = convert_list_to_str(
            [data_type, kernel, n_vecs, n_vals, unpacker, patcher]
        )

        vbw_start = 3
        vbw_end = 8
        ec_start = 20
        ec_end = 20

        # If the patcher is of type dummy only 0 exceptions is supported
        if patcher == "dummy":
            ec_start = 0
            ec_end = 0

        out = os.path.join(
            output_dir,
            "multi-column-"
            + "-".join(map(format_file_parameter, parameters))
            + f"-{vbw_start}-{vbw_end}-{ec_start}-{ec_end}-{DATA_N_VECS}-{args.number_sampling_runs}",
        )
        profiler.benchmark_command(
            MICROBENCHMARK_EXECUTABLE
            + " "
            + " ".join(parameters)
            + " "
            + f"{vbw_start} {vbw_end} {ec_start} {ec_end} {DATA_N_VECS} {args.number_sampling_runs} 0",
            out=out,
        )


def bench_compressors(output_dir: str, profiler):
    compressor_n_vecs = 25600
    float_files = get_all_files_in_dir("../../data/float")
    double_files = get_all_files_in_dir("../../data/double")

    for data_type, files in zip(ALP_TYPES, [float_files, double_files]):
        for kernel, compressors in [
            [
                COMPARISON_KERNELS[0],
                DECOMPRESSION_COMPRESSORS,
            ],
            [COMPARISON_KERNELS[1], DECOMPRESSION_QUERY_COMPRESSORS],
        ]:
            for file, compressor in itertools.product(files, compressors):
                for i in range(1, args.number_sampling_runs + 1):
                    out_parameters = convert_list_to_str(
                        [
                            data_type,
                            kernel,
                            compressor,
                            os.path.basename(file),
                            compressor_n_vecs,
                        ]
                    )
                    out = os.path.join(
                        output_dir,
                        "compressors-"
                        + "-".join(map(format_file_parameter, out_parameters))
                        + f"-{i}",
                    )
                    bin_parameters = convert_list_to_str(
                        [data_type, kernel, compressor, file, compressor_n_vecs]
                    )
                    profiler.benchmark_command(
                        COMPRESSORS_EXECUTABLE + " " + " ".join(bin_parameters),
                        out=out,
                        append_signal=True,
                    )


def main(args):
    assert directory_exists(args.output_dir)
    args.benchmarking_function(args.output_dir, get_profiler(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    benchmarking_functions = {func[0]: func[1] for func in get_benchmarking_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "benchmarking_function",
        type=str,
        choices=list(benchmarking_functions.keys()) + ["all"],
        help="function to execute",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory_to_write_results_to",
    )
    parser.add_argument(
        "-p",
        "--profiler",
        type=str,
        default="ncu",
        choices=["ncu", "nvvp", "none"],
    )
    parser.add_argument(
        "-nsr",
        "--number-sampling-runs",
        type=int,
        default=5,
        help="Executes commands multiple times",
    )
    parser.add_argument(
        "-onr",
        "--only-new-runs",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only runs new runs. If the filename already exists in the output, the run is skipped",
    )
    parser.add_argument(
        "-enzec",
        "--exit-non-zero-exit-code",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Exit when a command return a non zero statuscode",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Dry run",
    )

    parser.add_argument(
        "-ll",
        "--logging-level",
        type=int,
        default=logging.INFO,
        choices=[logging.CRITICAL, logging.ERROR, logging.INFO, logging.DEBUG],
        help=f"logging level to use: {logging.CRITICAL}=CRITICAL, {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
        + f"{logging.DEBUG}=DEBUG, higher number means less output",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )

    if args.benchmarking_function == "all":
        args.benchmarking_function = lambda out_dir, profiler: list(
            func(out_dir, profiler) for func in benchmarking_functions.values()
        )
    else:
        args.benchmarking_function = benchmarking_functions[args.benchmarking_function]
    main(args)
