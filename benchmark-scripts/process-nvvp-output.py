#!/usr/bin/env python3

import os
import sys

import argparse
import logging
import itertools

import inspect
import types
from pathlib import Path
from typing import Iterable

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
        lambda x: (x[0].replace(processing_function_prefix, ""), x[1]),
        processing_functions,
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
    return list(
        filter(lambda x: x.split("/")[-1].startswith(prefix), get_all_files_in_dir(dir))
    )


def parse_duration(line: str) -> int:
    """
    Parses the time, returns as nanoseconds
    """
    multiplier = None

    duration = line.split()[1]
    unit: str = duration[-2]
    if unit == "u":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1000 * 1000
    else:
        if duration[-2].isnumeric() and duration[-1] == "s":
            multiplier = 1000 * 1000 * 1000
        else:
            raise Exception(f"Could not parse {duration}, unknown unit: {unit}.")

    value_float: float = float(duration[:-2])

    return int(value_float * multiplier)


def read_profiler_output_as_df(file: str) -> pl.DataFrame:
    lines = []

    with open(file, "r") as f:
        lines = f.readlines()

    kernel_lines = filter(lambda x: "void" in x, lines)
    durations = list(map(parse_duration, kernel_lines))

    return pl.DataFrame(
        {"kernel_index": list(range(0, len(durations))), "duration_ns": durations}
    )


def read_compressors_output_as_df(file: str) -> pl.DataFrame:
    lines = []

    with open(file, "r") as f:
        lines = f.readlines()

    return_code = int(lines[-1].split(",")[-1])
    avg_bits_per_value = ""
    avg_exceptions_per_vector = ""
    line_counter = 0
    if "ALP_COMPRESSION_PARAMETERS" in lines[0]:
        (
            _,
            _,
            _,
            _,
            avg_bits_per_value,
            avg_exceptions_per_vector,
        ) = lines[
            0
        ].split(",")
        line_counter += 2 # One warmup run

    kernel, compressor, file, _, n_bytes, duration_ms, compression_ratio = lines[
        line_counter
    ].split(",")

    return pl.DataFrame(
        {
            "return_code": [return_code],
            "avg_bits_per_value": [avg_bits_per_value],
            "avg_exceptions_per_vector": [avg_exceptions_per_vector.strip()],
            "kernel": [kernel],
            "compressor": [compressor],
            "file": [file],
            "n_bytes": [n_bytes],
            "duration_ms": [duration_ms],
            "compression_ratio": [compression_ratio.strip()],
        }
    )


def duplicate_each(collection: Iterable, repeat_n_times: int) -> list:
    return [x for x in collection for _ in range(repeat_n_times)]


def add_sample_run_column(df: pl.DataFrame, n_samples: int) -> pl.DataFrame:
    assert df.height % n_samples == 0, "Number of samples is not divisible by n_samples"
    n_unique_kernels = df.height // n_samples
    return df.with_columns(
        pl.Series("sample_run", list(range(1, n_samples + 1)) * n_unique_kernels),
    )


def convert_ffor_file_to_df(file: str) -> pl.DataFrame:
    params = os.path.basename(file).split("-")
    n_samples = int(params[9])

    df = read_profiler_output_as_df(file)
    df = add_sample_run_column(df, n_samples)
    df = df.with_columns(
        pl.lit(params[0]).alias("encoding"),
        pl.lit(params[1]).alias("data_type"),
        pl.lit(params[2]).alias("kernel"),
        pl.lit(params[3]).alias("unpack_n_vectors"),
        pl.lit(params[4]).alias("unpack_n_values"),
        pl.lit(params[5]).alias("unpacker"),
        pl.Series(
            "vbw", duplicate_each(range(int(params[6]), int(params[7]) + 1), n_samples)
        ),
        pl.lit(params[8]).alias("n_vecs"),
    )

    return df


def convert_alp_ec_file_to_df(file: str) -> pl.DataFrame:
    params = os.path.basename(file).split("-")
    n_samples = int(params[13])

    df = read_profiler_output_as_df(file)
    df = add_sample_run_column(df, n_samples)
    df = df.with_columns(
        pl.lit(params[0]).alias("encoding"),
        pl.lit(params[2]).alias("data_type"),
        pl.lit(params[3]).alias("kernel"),
        pl.lit(params[4]).alias("unpack_n_vectors"),
        pl.lit(params[5]).alias("unpack_n_values"),
        pl.lit(params[6]).alias("unpacker"),
        pl.lit(params[7]).alias("patcher"),
        pl.lit(params[8]).alias("start_vbw"),
        pl.lit(params[9]).alias("end_vbw"),
        pl.Series(
            "ec", duplicate_each(range(int(params[10]), int(params[11]) + 1), n_samples)
        ),
        pl.lit(params[12]).alias("n_vecs"),
    )

    return df


def convert_multi_column_file_to_df(file: str) -> pl.DataFrame:
    n_cols = 10
    params = os.path.basename(file).split("-")
    n_samples = int(params[13])

    df = read_profiler_output_as_df(file)
    df = add_sample_run_column(df, n_samples)
    df = df.with_columns(
        pl.lit("FFOR" if "u" in params[2] else "ALP").alias("encoding"),
        pl.lit(params[2]).alias("data_type"),
        pl.lit(params[3]).alias("kernel"),
        pl.lit(params[4]).alias("unpack_n_vectors"),
        pl.lit(params[5]).alias("unpack_n_values"),
        pl.lit(params[6]).alias("unpacker"),
        pl.lit(params[7]).alias("patcher"),
        pl.lit(params[8]).alias("start_vbw"),
        pl.lit(params[9]).alias("end_vbw"),
        pl.lit(params[10]).alias("start_ec"),
        pl.lit(params[11]).alias("end_ec"),
        pl.Series("n_cols", duplicate_each(range(1, n_cols + 1), n_samples)),
        pl.lit(params[12]).alias("n_vecs"),
    )

    return df


def convert_compressors_file_to_df(file: str) -> pl.DataFrame:
    params = os.path.basename(file).split("-")

    df = read_compressors_output_as_df(file)
    df = df.with_columns(
        pl.lit(params[1]).alias("data_type"),
        pl.lit(params[-2]).alias("n_vecs"),
        pl.lit(params[-1]).alias("sample_run"),
    )

    return df


def convert_heterogeneous_pipelines_experiment_file_to_df(file: str) -> pl.DataFrame:
    params = os.path.basename(file).split("-")
    ARITHMETIC_32 = "ARITHMETIC_32"
    ARITHMETIC_64 = "ARITHMETIC_64"
    MEMORY = "MEMORY"
    NONE = "NONE"
    df = read_profiler_output_as_df(file).with_columns(
        pl.Series(
            "first_kernel",
            [
                ARITHMETIC_32,
                ARITHMETIC_64,
                MEMORY,
                ARITHMETIC_32,
                ARITHMETIC_32,
                MEMORY,
                MEMORY,
            ],
        ),
        pl.Series(
            "second_kernel",
            [
                NONE,
                NONE,
                NONE,
                ARITHMETIC_32,
                MEMORY,
                ARITHMETIC_32,
                MEMORY,
            ],
        ),
        pl.lit(params[-1]).alias("sample_run"),
    )

    return df


def collect_files_into_df(
    input_dir: str, prefix: str, convertor_lambda
) -> pl.DataFrame:
    files = get_all_files_with_prefix_in_dir(input_dir, prefix)
    return pl.concat(map(convertor_lambda, files))


def process_ffor(input_dir: str) -> tuple[str, pl.DataFrame]:
    return "ffor.csv", collect_files_into_df(input_dir, "ffor", convert_ffor_file_to_df)


def process_alp_ec(input_dir: str) -> tuple[str, pl.DataFrame]:
    return "alp-ec.csv", collect_files_into_df(
        input_dir, "alp-ec", convert_alp_ec_file_to_df
    )


def process_multi_column(input_dir: str) -> tuple[str, pl.DataFrame]:
    return "multi-column.csv", collect_files_into_df(
        input_dir, "multi-column", convert_multi_column_file_to_df
    )


def process_compressors(input_dir: str) -> tuple[str, pl.DataFrame]:
    return "compressors.csv", collect_files_into_df(
        input_dir, "compressors", convert_compressors_file_to_df
    )


def main(args):
    assert directory_exists(args.input_dir)
    assert directory_exists(args.output_dir)

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
        args.processing_function = lambda in_dir: [
            processing_functions[string_value](in_dir)
        ]
    main(args)
