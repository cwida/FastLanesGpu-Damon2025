#!/usr/bin/env python3

from io import StringIO
import os
import sys

import argparse
import logging
import itertools

import inspect
import types
import subprocess
from pathlib import Path

import polars as pl

def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_processing_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    processing_function_prefix = "process_"
    processing_functions = filter(
        lambda x: x[0].startswith(processing_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(processing_function_prefix, ""), x[1]), processing_functions
    )
    return list(stripped_prefixes_from_name)


def get_all_files_in_dir(dir: str) -> list[str]: 
    file_paths = []

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths

def get_all_files_with_prefix_in_dir(dir: str, prefix: str) -> list[str]:
    return list(filter(lambda x: x.split('/')[-1].startswith(prefix), get_all_files_in_dir(dir)))

def read_profiler_output_as_df(file: str) -> pl.DataFrame:
    with open(file, "r") as f:
        content = f.readlines()
        csv = content[2:]
        str_buffer = StringIO("".join(csv))
        df = pl.read_csv(str_buffer)
        return df

def clean_up_raw_df(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = [
        "Process ID",
        "Process Name",
        "Host Name",
        "Context",
        "Stream",
        "Device",
        "CC",
        "Section Name",
        "Rule Name",
        "Rule Type",
        "Rule Description"
    ]
    df = df.drop(drop_columns)

    redundant_v100_cols = [
        'Estimated Speedup Type', 
        'Estimated Speedup'
    ]
    df = df.drop(redundant_v100_cols, strict=False)

    rotate_columns = [
        "Metric Name",
        "Metric Unit",
        "Metric Value",
    ]

    rotated_rows = []

    for label, group in df.group_by("ID", maintain_order=True):
        new_row = group.drop(rotate_columns).unique(maintain_order=True)

        for name, unit, data in zip(
                group.get_column("Metric Name"), 
                group.get_column("Metric Unit"), 
                group.get_column("Metric Value")):
            new_row = new_row.with_columns(pl.Series(name=f"{name} ({unit})", values=[data]))

        rotated_rows.append(new_row)

    return pl.concat(rotated_rows)


def process_bp_micro(input_dir: str) -> tuple[str, pl.DataFrame]:
    bp_micro_files = get_all_files_with_prefix_in_dir(input_dir, "bp-micro")

    df_list = []
    for file in sorted(bp_micro_files):
        split_file_name = file.split("-")
        decompressor = "-".join(split_file_name[2:-3])
        vbw_start = int(split_file_name[-3])
        vbw_end = int(split_file_name[-2])
        n_vecs = int(split_file_name[-1])

        df = read_profiler_output_as_df(file)
        df = clean_up_raw_df(df)

        value_bit_widths = list(range(vbw_start, vbw_end + 1))
        df = df.with_columns([
                pl.lit(decompressor).alias("alias"),
                pl.Series(name="vbw", values=value_bit_widths),
                pl.lit(n_vecs).alias("n_vecs"),
                ]
        ) 

        df_list.append(df)

    combined_df = pl.concat(df_list)
    combined_df = combined_df.select([
        'ID',
        'alias',
        'vbw',
        'n_vecs',
        'Elapsed Cycles (cycle)',
        'Kernel Name',
        'Block Size',
        'Grid Size',
        'Memory Throughput (%)',
        'DRAM Throughput (%)',
        'L1/TEX Cache Throughput (%)',
        'L2 Cache Throughput (%)',
        'SM Active Cycles (cycle)',
        'Compute (SM) Throughput (%)',
        'Registers Per Thread (register/thread)',
        'Threads (thread)',
        'Waves Per SM ()',
        'Block Limit SM (block)',
        'Block Limit Registers (block)',
        'Block Limit Shared Mem (block)',
        'Block Limit Warps (block)',
        'Theoretical Active Warps per SM (warp)',
        'Theoretical Occupancy (%)',
         'Achieved Occupancy (%)',
         'Achieved Active Warps Per SM (warp)',
    ])

    return "bp-micro.csv", combined_df 

def process_alp_micro(input_dir: str) -> tuple[str, pl.DataFrame]:
    alp_micro_files = get_all_files_with_prefix_in_dir(input_dir, "alp-micro")

    df_list = []
    for file in sorted(alp_micro_files):
        split_file_name = file.split("-")
        decompressor = "-".join(split_file_name[2:-4])
        vbw = int(split_file_name[-4])
        ec_start = int(split_file_name[-3])
        ec_end = int(split_file_name[-2])
        n_vecs = int(split_file_name[-1])

        df = read_profiler_output_as_df(file)
        df = clean_up_raw_df(df)

        exception_counts = list(range(ec_start, ec_end + 1))
        df = df.with_columns([
                pl.lit(decompressor).alias("alias"),
                pl.lit(vbw).alias("vbw"),
                pl.Series(name="ec", values=exception_counts),
                pl.lit(n_vecs).alias("n_vecs"),
                ]
        ) 

        df_list.append(df)

    combined_df = pl.concat(df_list)
    combined_df = combined_df.select([
        'ID',
        'alias',
        'vbw',
        'ec',
        'n_vecs',
        'Elapsed Cycles (cycle)',
        'Kernel Name',
        'Block Size',
        'Grid Size',
        'Memory Throughput (%)',
        'DRAM Throughput (%)',
        'L1/TEX Cache Throughput (%)',
        'L2 Cache Throughput (%)',
        'Compute (SM) Throughput (%)',
        'Registers Per Thread (register/thread)',
        'Shared Memory Configuration Size (byte)',
        'Waves Per SM ()',
        'Block Limit SM (block)',
        'Block Limit Registers (block)',
        'Block Limit Shared Mem (block)',
        'Block Limit Warps (block)',
        'Theoretical Active Warps per SM (warp)',
        'Theoretical Occupancy (%)',
         'Achieved Occupancy (%)',
         'Achieved Active Warps Per SM (warp)',
    ])

    return "alp-micro.csv", combined_df


def process_multicolumn(input_dir: str) -> tuple[str, pl.DataFrame]:
    multicolumn_files = get_all_files_with_prefix_in_dir(input_dir, "multicolumn")

    df_list = []
    for file in sorted(multicolumn_files):
        split_file_name = file.split("-")
        file_n_cols = int(split_file_name[1])
        vbw_start = int(split_file_name[2])
        vbw_end = int(split_file_name[3])
        ec = int(split_file_name[4])
        n_vecs = int(split_file_name[5])

        df = read_profiler_output_as_df(file)
        df = clean_up_raw_df(df)

        len_single_kernel = vbw_end - vbw_start + 1
        keys = ["ALP-32", "ALP-1","GALP-32", "GALP-1"]
        values = keys * len_single_kernel
        value_bit_widths = list(itertools.chain.from_iterable([[x] * len(keys) for x in range(vbw_start, vbw_end + 1)]))
        df = df.with_columns([
                pl.lit(file_n_cols).alias("n_cols"),
                pl.Series(name="alias", values=values),
                pl.Series(name="vbw", values=value_bit_widths),
                pl.lit(ec).alias("ec"),
                pl.lit(n_vecs).alias("n_vecs"),
            ]
        ) 

        df_list.append(df)

    combined_df = pl.concat(df_list)

    combined_df = combined_df.select([
        'ID',
        'n_cols',
        'alias',
        'vbw',
        'ec',
        'n_vecs',
        'Elapsed Cycles (cycle)',
        'Theoretical Occupancy (%)',
        'Achieved Occupancy (%)',
        'Kernel Name',
        'Block Size',
        'Grid Size',
        'Registers Per Thread (register/thread)',
        'Memory Throughput (%)',
        'DRAM Throughput (%)',
        'L1/TEX Cache Throughput (%)',
        'L2 Cache Throughput (%)',
        'SM Active Cycles (cycle)',
        'Compute (SM) Throughput (%)',
        'Threads (thread)',
        'Waves Per SM ()',
        'Block Limit SM (block)',
        'Block Limit Registers (block)',
        'Block Limit Shared Mem (block)',
        'Block Limit Warps (block)',
        'Theoretical Active Warps per SM (warp)',
        'Achieved Active Warps Per SM (warp)',
    ])

    return "multicolumn.csv", combined_df 


def process_real_data(input_dir: str) -> tuple[str, pl.DataFrame]:
    file = os.path.join(input_dir, "real-data")
    df = pl.read_csv(file, has_header=False,new_columns=[
        'kernel',
        'compressor',
        'file',
        'result',
        'bytes',
        'execution_time_ms',
        'compression_ratio',
    ])

    return "real-data.csv", df 

def main(args):
    assert(directory_exists(args.input_dir))
    assert(directory_exists(args.output_dir))

    for default_name, df in args.processing_function(args.input_dir):
        df.write_csv(os.path.join(args.output_dir, default_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    processing_functions = {func[0]: func[1] for func in get_processing_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "processing_function",
        type=str,
        choices=list(processing_functions.keys()) + ["all"],
        help="function to execute",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="directory_to_read_results_from",
    )
    parser.add_argument(
        "output_dir",
        default=None,
        type=str,
        help="directory_to_write_results_to",
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

    if args.processing_function == "all":
        args.processing_function = lambda in_dir: list(
            func(in_dir) for func in processing_functions.values()
        )
    else:
        string_value = args.processing_function
        args.processing_function = lambda in_dir: [processing_functions[string_value](in_dir)]
    main(args)
