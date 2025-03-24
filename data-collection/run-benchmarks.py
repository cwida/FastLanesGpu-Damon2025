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

REAL_DATA_EXECUTABLE = "./bin/benchmark-single-compressor"
HWC_GENERATED_DATA_EXECUTABLE = "./bin/benchmark-all-compressors"
MICROBENCHMARK_EXECUTABLE = "./bin/micro-benchmarks"
MULTICOLUMN_EXECUTABLE = "./bin/multi-column-benchmarks"

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
        lambda x: (x[0].replace(benchmarking_function_prefix, ""), x[1]), benchmarking_functions
    )
    return list(stripped_prefixes_from_name)


def execute_command(command: str, out: str|None=None) -> str:
    if args.dry_run:
        print(command, file=sys.stderr)
        return ""

    logging.info(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logging.critical(f"Exited with code {result.returncode}: {command}")
        logging.critical(f"STDOUT: {result.stdout}")
        logging.critical(f"STDERR: {result.stderr}")
        exit(0)

    if out:
        with open(out, 'w') as f:
            f.write(result.stdout)

    return result.stdout


def benchmark_command(command: str, out: str) -> str:
    execute_command(f"ncu --csv {command}", out)


def bench_bp_micros(output_dir: str, n_vecs: int):
    decompressors=["bp-1v","bp-4v"]
    vbw=(0, 32)

    for i, decompressor in enumerate(decompressors):
        benchmark_command(f"{MICROBENCHMARK_EXECUTABLE} 0 "
                        f"{i} {vbw[0]} {vbw[1]} {n_vecs}",
                        out=f"{os.path.join(output_dir, f"bp-micro-{decompressor}-{vbw[0]}-{vbw[1]}-{n_vecs}")}")


def bench_alp_micros(output_dir: str, n_vecs: int):
    decompressors=["alp-1v","galp-branchless-1v","galp-branchy-1v","galp-branchy-4v", "galp-prefetch-branchy-1v","galp-prefetch-branchy-4v",]
    vbw = 8
    ec = (0, 50)

    for i, decompressor in enumerate(decompressors):
        benchmark_command(f"{MICROBENCHMARK_EXECUTABLE} 1 "
                        f"{i + 2} {vbw} {ec[0]} {ec[1]} {n_vecs}",
                        out=f"{os.path.join(output_dir, f"alp-micro-{decompressor}-{vbw}-{ec[0]}-{ec[1]}-{n_vecs}")}")

def bench_multicolumn(output_dir: str, n_vecs: int):
    ec = 10
    vbw = (0, 8)
    n_cols = 10

    for i in range(1, n_cols + 1):
        benchmark_command(f"{MULTICOLUMN_EXECUTABLE} "
                        f"{i} {vbw[0]} {vbw[1]} {ec} {n_vecs} ",
                        out=f"{os.path.join(output_dir, f"multicolumn-{i}-{vbw[0]}-{vbw[1]}-{ec}-{n_vecs}")}")


def bench_real_data(output_dir:str, n_vecs:int):
    dataset_dir = "./binary-columns"
    output = ""

    for kernel in range(0, 1 + 1):
        for file in get_all_files_in_dir(dataset_dir):
            for compressor in range(0 if kernel == 1 else 1, 9 + 1):
                output += execute_command(f"{REAL_DATA_EXECUTABLE} "
                            f"{kernel} {compressor} {file} {n_vecs}")
                        
    path = os.path.join(output_dir, "real-data")
    with open(path, "w") as f:
        f.write(output)


def main(args):
    assert has_root_privileges()
    assert(directory_exists(args.output_dir))
    args.benchmarking_function(args.output_dir, args.n_vecs)


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
        "-nv",
        "--n-vecs",
        type=int,
        default=125 * 1000, # 500 MB
        help="N-vecs",
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
        args.benchmarking_function = lambda out_dir, n_vecs: list(
            func(out_dir, n_vecs) for func in benchmarking_functions.values()
        )
    else:
        args.benchmarking_function = benchmarking_functions[args.benchmarking_function]
    main(args)
