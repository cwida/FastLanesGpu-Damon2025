#!/usr/bin/env python3

import os
import sys

import argparse
import logging
import itertools

import inspect
import types
from typing import Any
import subprocess
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

FLOAT_BYTES = 4
GB_BYTES = 1024 * 1024 * 1024
MS_IN_S = 1000

VECTOR_SIZE = 1024

COLOR_SET = [
        "tab:blue",
        "#66bf5c", # Light green
        "#177314", # Darker green
        "tab:orange",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "tab:green",
]


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_plotting_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    plotting_function_prefix = "plot_"
    plotting_functions = filter(
        lambda x: x[0].startswith(plotting_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(plotting_function_prefix, ""), x[1]), plotting_functions
    )
    return list(stripped_prefixes_from_name)


@dataclass
class DataSource:
    label: str
    color: int
    x_data: list[Any]
    y_data: list[Any]


def create_scatter_graph(
    data_sources: list[DataSource],
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int]|None=None,
    octal_grid: bool=False,
    figsize: tuple[int,int]=(5,5),
    legend_pos: str='best',
        ):
    fig, ax = plt.subplots(figsize=figsize)
    x_min = min(data_sources[0].x_data)
    x_max = max(data_sources[0].x_data)
    for source in data_sources:
        x_min = min(source.x_data)
        x_max = max(source.x_data)
        ax.scatter(
            source.x_data,
            source.y_data,
            s=26,
            linewidths=0,
            c=COLOR_SET[source.color],
            label=source.label,
        )

    ax.set_xticks(range(math.floor(x_min), math.ceil(x_max)), minor=True)
    ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    if octal_grid:
        ax.set_xticks(range(math.floor(x_min), math.ceil(x_max), 8))

    ax.grid(which="major", linestyle="--", linewidth=0.1)


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)

    ax.legend(loc=legend_pos)

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def plot_bp(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "bp-micro.csv"))

    df = df.with_columns(
        ((pl.col("n_vecs") * VECTOR_SIZE) / pl.col("Elapsed Cycles (cycle)")).alias("throughput")
    )

    create_scatter_graph(
        [
            DataSource(
                str(label_data[0][0]), 
                i, 
                label_data[1].get_column("vbw").to_list(), 
                label_data[1].get_column("throughput").to_list()
            ) for i, label_data in enumerate(df.group_by("alias"))
        ],
        "Bits/value",
        "Throughput (values/cycle)",
        os.path.join(output_dir, "bp-ilp.eps"),
        octal_grid=True,
        figsize=(6,6),
    )


def plot_alp(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "alp-micro.csv"), schema_overrides={"Theoretical Occupancy (%)": pl.Float64})
    df = df.with_columns(
        ((pl.col("n_vecs") * VECTOR_SIZE) / pl.col("Elapsed Cycles (cycle)")).alias("throughput")
    )
    
    decompressor_sources = { 
        str(label_data[0][0]):  DataSource(
            str(label_data[0][0]), 
            i, 
            label_data[1].get_column("ec").to_list(), 
            label_data[1].get_column("throughput").to_list()
        ) for i, label_data in enumerate(df.group_by("alias", maintain_order=True))
     }
    
    decompressor_sources["alp-1v"].label = "ALP"
    decompressor_sources["galp-branchless-1v"].label = "GALP-naive"
    decompressor_sources["galp-prefetch-branchy-1v"].label = "GALP-prefetch"

    y_lim = max([max(data.y_data) for data in decompressor_sources.values()]) * 1.1

    create_scatter_graph(
        [
            decompressor_sources["alp-1v"],
            decompressor_sources["galp-branchless-1v"],
            decompressor_sources["galp-prefetch-branchy-1v"],
        ],
        "Exception count",
        "Throughput (values/cycle)",
        os.path.join(output_dir, "summary.eps"),
        y_lim=(0,y_lim),
        legend_pos='lower left',
    )


def create_multi_bar_graph(
    data_per_bar: dict[str, list[Any]],
    bargroup_labels: list[str]|None,
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int]|None=None,
    colors: list[int]|None=None,
):
    n_bars = len(data_per_bar)
    n_groups = len(list(data_per_bar.values())[0])
    assert all(len(x) == n_groups for x in data_per_bar.values())

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.8 / n_bars
    indices = np.arange(n_groups)
    for i, bar_label_data in enumerate(data_per_bar.items()):
        bar_label = bar_label_data[0]
        bar_data = bar_label_data[1]

        positions = indices + (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(
            positions, bar_data, bar_width, label=bar_label, color=COLOR_SET[colors[i]] if colors else COLOR_SET[i],
        )

    if bargroup_labels:
        ax.set_xticks(indices)
        ax.set_xticklabels(bargroup_labels)
    else:
        plt.tick_params(
            axis='x',          
            which='both',     
            bottom=False,    
            top=False,      
            labelbottom=False) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    ax.legend()

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def plot_multicolumn(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "multicolumn.csv"))
    df = df.with_columns(
        ((pl.col("n_cols") * pl.col("n_vecs") * 1024)/ pl.col("Elapsed Cycles (cycle)")).alias("throughput")
    )

    df_avg = df.sort("n_cols", maintain_order=True).group_by("n_cols", "alias", maintain_order=True).agg(pl.col(pl.NUMERIC_DTYPES).mean())
    
    data = {str(label[0]): data.get_column("throughput").to_list() for label, data in df_avg.group_by("alias", maintain_order=True)}
    create_multi_bar_graph(
        data,
        sorted([int(x) for x in df.get_column("n_cols").unique()]),
        "Number of columns scanned in parallel",
        "Throughput (values/cycle)",
        os.path.join(output_dir, "multicolumn_throughput_comparison.eps"),
    )
    data = {str(label[0]): data.get_column("Achieved Occupancy (%)").to_list() for label, data in df_avg.group_by("alias", maintain_order=True)}
    create_multi_bar_graph(
        data,
        sorted([int(x) for x in df.get_column("n_cols").unique()]),
        "Number of columns scanned in parallel",
        "Achieved Occupancy (%)",
        os.path.join(output_dir, "multicolumn_occupancy_comparison.eps"),
    )


def plot_real_data(input_dir: str, output_dir: str):
    complete_df = pl.read_csv(os.path.join(input_dir, "real-data.csv"))
    complete_df = complete_df.with_columns(
        (pl.col("bytes") * MS_IN_S / GB_BYTES / pl.col("execution_time_ms")).alias("throughput")
    )

    max_throughput = complete_df.get_column("throughput").max()

    for label, df in complete_df.group_by("kernel", maintain_order=True):
        label = label[0]
        n_compressors = len(df.get_column("compressor").unique())

        create_scatter_graph(
            [
                DataSource(
                    label=str(label_data[0][0]),
                    color=i if label == "decompression_query" else i + 1,
                    x_data=label_data[1].get_column("compression_ratio"),
                    y_data=label_data[1].get_column("throughput"),
                )
                for i, label_data in enumerate(df.group_by("compressor", maintain_order=True))
            ],
            "Compression Ratio",
            "Throughput (GB/s)",
            os.path.join(output_dir, f"{label}_throughput_vs_compression_ratio.eps"),
            figsize=(8,8),
        )

        data = {str(label[0]): [data.get_column("throughput").mean()] for label, data in df.group_by("compressor", maintain_order=True)}
        create_multi_bar_graph(
            data,
            None,
            "Compressors",
            "Throughput (GB/s)",
            os.path.join(output_dir, f"{label}_throughput_comparison.eps"),
            y_lim=(0,max_throughput),
            colors=[i + (0 if label == "decompression_query" else 1) for i in range(n_compressors)],
        )
        data = {str(label[0]): [data.get_column("compression_ratio").mean()] for label, data in df.group_by("compressor", maintain_order=True)}
        create_multi_bar_graph(
            data,
            None,
            "Compressors",
            "Compression ratio",
            os.path.join(output_dir, f"{label}_compression_ratio_comparison.eps"),
            colors=[i + (0 if label == "decompression_query" else 1) for i in range(n_compressors)],
        )


def main(args):
    assert directory_exists(args.input_dir)
    assert directory_exists(args.output_dir)

    _ = args.plotting_function(args.input_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    plotting_functions = {func[0]: func[1] for func in get_plotting_functions()}
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "plotting_function",
        type=str,
        choices=list(plotting_functions.keys()) + ["all"],
        help="function to execute",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="directory_to_write_results_to",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory_to_write_results_to",
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

    if args.plotting_function == "all":
        args.plotting_function = lambda in_dir, out_dir: list(
            func(in_dir, out_dir) for func in plotting_functions.values()
        )
    else:
        args.plotting_function = plotting_functions[args.plotting_function]
    main(args)
