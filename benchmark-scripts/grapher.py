#!/usr/bin/env python3

from collections.abc import Callable
import os
import sys

import itertools
import argparse
import logging
import copy

import inspect
import types
from typing import Any, Iterable, Optional
from pathlib import Path

import statistics
import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

FLOAT_BYTES = 4
GB_BYTES = 1024 * 1024 * 1024
MS_IN_S = 1000

VECTOR_SIZE = 1024

COLOR_SET = [
    "tab:blue",
    "tab:orange",
    "tab:red",
    "tab:green",
    "tab:pink",
    "tab:purple",
    "tab:brown",
    "tab:olive",
    "tab:gray",
    "tab:cyan",
    "tab:cyan",
    "tab:cyan",
    "tab:cyan",
    "tab:cyan",
    "tab:cyan",
    "tab:cyan",
]

UNPACKERS_ORDER_AND_NAMES = {
    "old_fls": "FastLanesOnGPU",
    "switch_case": "Single Value Switch",
    "stateless": "Static",
    "stateful_cache": "Streaming No Buffer",
}
STATEFUL_STORAGE_TYPES = {
    "local": "Local",
    "shared": "Shared",
    "register": "Switch Registers",
    "register_branchless": "Rotate Registers",
}
for s, b in itertools.product(STATEFUL_STORAGE_TYPES.items(), [1, 2, 4]):
    UNPACKERS_ORDER_AND_NAMES[f"stateful_{s[0]}_{b}"] = f"Streaming {s[1]} Buffer"

UNPACKERS_ORDER_AND_NAMES["stateful_branchless"] = "Branchless Streaming"

PATCHERS_ORDER_AND_NAMES = {
    "stateless": "ALP Static",
    "stateful": "ALP",
    "naive": "GALP",
    "naive_branchless": "GALP Branchless ",
    "prefetch_all": "GALP Buffer",
    "prefetch_all_branchless": "GALP Branchless Buffer",
}

COMPRESSORS_ORDER_AND_NAMES = {
    "ALP": "ALP",
    "GALP": "G-ALP",
    "Thrust": "Thrust",
    "zstd": "nv-zstd",
    "LZ4": "nv-LZ4",
    "Snappy": "nv-Snappy",
    "Deflate": "nv-Deflate",
    "GDeflate": "nv-GDeflate",
    "Bitcomp": "Bitcomp",
    "BitcompSparse": "BitcompSparse",
    "ndzip": "ndzip",
    "single-fpcompressor-ratio": "DP-ratio",
    "single-fpcompressor-speed": "DP-speed",
    "double-fpcompressor-ratio": "DP-ratio",
    "double-fpcompressor-speed": "DP-speed",
}

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


@dataclass
class GroupedDataSource:
    group_by_column_values: tuple
    x_data: list[Any]
    y_data: list[Any]
    color: int = 0
    label: str | None = None

    def set_label(self, label: str):
        self.label = label
        return self


def create_scatter_graph(
    data_sources: list[DataSource | GroupedDataSource],
    x_label: str,
    y_label: str,
    out: str,
    x_lim: tuple[int | float, int | float] | None = None,
    y_lim: tuple[int | float, int | float] | None = None,
    y_axis_log_scale: bool = False,
    octal_grid: bool = False,
    figsize: tuple[int, int] = (5, 5),
    legend_pos: str = "best",
    add_lines: bool = False,
    x_axis_percentage: bool = False,
    title: Optional[str] = None,
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
        if add_lines:
            ax.plot(
                source.x_data,
                source.y_data,
                c=COLOR_SET[source.color],
            )

    if y_axis_log_scale:
        ax.set_yscale('log')

    if x_axis_percentage:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks(range(math.floor(x_min), math.ceil(x_max)), minor=True)
    ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    if octal_grid:
        ax.set_xticks(range(math.floor(x_min), math.ceil(x_max) + 1, 8))

    ax.grid(which="major", linestyle="--", linewidth=0.1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if title:
        ax.set_title(title)

    ax.legend(loc=legend_pos)

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def create_multi_bar_graph(
    data_sources: list[DataSource | GroupedDataSource],
    bargroup_labels: list[str] | None,
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int] | None = None,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 7),
):
    n_bars = len(data_sources)
    n_groups = len(data_sources[0].x_data)
    assert all(len(x.x_data) == n_groups for x in data_sources)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.8 / n_bars
    indices = np.arange(n_groups)
    for i, source in enumerate(data_sources):
        bar_label = source.label
        bar_data = source.y_data

        positions = indices + (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(
            positions,
            bar_data,
            bar_width,
            label=bar_label,
            color=COLOR_SET[source.color],
        )

    if bargroup_labels:
        ax.set_xticks(indices)
        ax.set_xticklabels(bargroup_labels)
    else:
        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    ax.grid(axis="y", which="both", linestyle="--", linewidth=0.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    if title:
        ax.set_title(title)
    ax.legend()

    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def create_boxplot_graph(
    data_sources: list[DataSource | GroupedDataSource],
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int] | None = None,
    title: Optional[str] = None,
    x_label_rotation: int = 0,
    show_means: bool = False,
):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(
        x=list(map(lambda s: s.y_data, data_sources)),
        labels=list(map(lambda x: x.label, data_sources)),
        showmeans=show_means,
    )

    for label in ax.get_xticklabels():
        label.set_rotation(x_label_rotation)
        label.set_horizontalalignment("right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)

    ax.set_title(title)
    fig.savefig(out, dpi=300, format="eps", bbox_inches="tight")
    plt.close(fig)


def format_row_colors(
    values: list[Any],
    format_value: Callable[[Any], str],
    lower_is_better: bool = False,
) -> list[str]:
    formatted_values = list(map(format_value, values))
    top_indices = sorted(
        range(len(values)), key=lambda i: values[i], reverse=not lower_is_better
    )

    first = top_indices[0]
    second = top_indices[1]

    formatted_values[first] = (
        f"\\cellcolor{{green!18}}\\textbf{{{format_value(values[first])}}}"
    )
    formatted_values[second] = (
        f"\\cellcolor{{yellow!10}}\\textbf{{{format_value(values[second])}}}"
    )

    return formatted_values


def create_latex_table(
    columns: list[list[int]],
    column_labels: list[str],
    row_labels: list[str],
    out: str,
    format_row: Callable[[list[Any]], list[str]],
    add_averages: bool = False,
    add_medians: bool = False,
    vertical_column_names: bool = False,
) -> None:
    return
    assert len(columns) == len(column_labels), "Each column must have a label"
    assert all(
        len(col) == len(row_labels) for col in columns
    ), "All columns must have the same number of rows"

    num_columns = len(columns)
    num_rows = len(row_labels)

    if vertical_column_names:
        column_labels = [f"\\rotatebox{{90}}{{{label}}}" for label in column_labels]

    average_formatter = lambda x: f"\\textbf{{{x}}}"
    label_formatter = lambda x: x.replace("_", r"\_")
    column_label_formatter = lambda x: f"\\textbf{{{label_formatter(x)}}}"
    row_label_formatter = lambda x: label_formatter(x)

    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "c" * num_columns + "}\n")
        f.write("\\toprule\n")

        header = (
            " & ".join([""] + [column_label_formatter(l) for l in column_labels])
            + " \\\\\n"
        )
        f.write(header)
        f.write("\\midrule\n")

        if add_averages:
            averages = list(map(statistics.mean, columns))
            row = [r"\textbf{Average}"] + [
                average_formatter(a) for a in format_row(averages)
            ]
            f.write(" & ".join(row) + " \\\\\n")
        if add_medians:
            averages = list(map(statistics.median, columns))
            row = [r"\textbf{Median}"] + [
                average_formatter(a) for a in format_row(averages)
            ]
            f.write(" & ".join(row) + " \\\\\n")
        if add_averages or add_medians:
            f.write("\\midrule\n")

        for i in range(num_rows):
            row = [row_label_formatter(row_labels[i])] + format_row(
                [columns[j][i] for j in range(num_columns)]
            )
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def reorder_columns(df: pl.DataFrame, reference_columns: list[str]) -> pl.DataFrame:
    ordered_columns = [col for col in reference_columns if col in df.columns]
    return df.select(ordered_columns)


def average_samples(df: pl.DataFrame, columns_to_avg: list[str]) -> pl.DataFrame:
    dropped_columns = ["kernel_index", "sample_run"]
    reference_columns = [col for col in df.columns if col not in dropped_columns]
    group_by_cols = [
        col
        for col in df.columns
        if col not in columns_to_avg and col not in dropped_columns
    ]

    df = df.group_by(group_by_cols, maintain_order=True).agg(
        [pl.col(col).mean().alias(col) for col in columns_to_avg]
    )

    df = reorder_columns(df, reference_columns)
    return df


def create_grouped_data_sources(
    df: pl.DataFrame, group_by_columns: list[str], x_column: str, y_column: str
) -> list[GroupedDataSource]:
    return [
        GroupedDataSource(
            group_by_column_values,
            data.get_column(x_column).to_list(),
            data.get_column(y_column).to_list(),
        )
        for group_by_column_values, data in df.group_by(
            group_by_columns, maintain_order=True
        )
    ]


def define_graph(
    sources: list[GroupedDataSource],
    filters: list[list[Any] | Any],
    label_func: Callable[[tuple], str],
) -> list[GroupedDataSource]:
    assert len(sources[0].group_by_column_values) == len(
        filters
    ), "Mismatch filter length and group by length"
    filter_lists = list(map(lambda x: x if isinstance(x, list) else [x], filters))

    filtered = filter(
        lambda source: all(
            [
                source.group_by_column_values[i] in filter or len(filter) == 0
                for i, filter in enumerate(filter_lists)
            ]
        ),
        sources,
    )

    labelled = list(
        map(
            lambda x: copy.copy(x.set_label(label_func(x.group_by_column_values))),
            filtered,
        )
    )

    return sorted(labelled, key=lambda x: x.label)


@dataclass
class SourceSet:
    file_name: str
    sources: list[GroupedDataSource]
    title: Optional[str] = None
    colors: Optional[Iterable[int]] = None


def calculate_common_y_lim(
    sources: Iterable[DataSource | GroupedDataSource], top_n: int = 1
) -> float:
    all_y_data = list(itertools.chain.from_iterable(map(lambda x: x.y_data, sources)))
    return sorted(all_y_data)[-min(len(all_y_data), top_n)] * 1.1


def assign_colors(
    sources: Iterable[DataSource | GroupedDataSource],
    colors: Optional[Iterable[int]] = None,
) -> Iterable[DataSource | GroupedDataSource]:
    if colors:
        for c, source in zip(colors, sources):
            source.color = c
    else:
        for c, source in enumerate(sources):
            source.color = c

    return sources


def format_str_to_label(label: str) -> str:
    return label.replace("_", " ").title()


def replace_label(label: str, name_and_order_map: dict[str, str]) -> str:
    # Sort on lenght of substring to avoid short substring
    # replacing long ones first
    for key, value in sorted(
        name_and_order_map.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if key in label:
            return label.replace(key, value)
    return label


def reorder_and_relabel(
    sources: list[DataSource | GroupedDataSource],
    name_and_order_map: dict[str, str],
) -> list[DataSource | GroupedDataSource]:
    def get_index(label: str) -> int:
        for i, key in enumerate(name_and_order_map.keys()):
            if key in label:
                return i
        raise Exception("No key found in source label")

    # First sort on label to reorder "unpacker 1v" and "unpacker 4v"
    reordered_on_label_sources = sorted(sources, key=lambda x: x.label)
    reordered_sources = sorted(
        reordered_on_label_sources, key=lambda x: get_index(x.label)
    )

    reordered_and_relabeled_sources = list(
        map(
            lambda x: x.set_label(
                replace_label(
                    x.label,
                    name_and_order_map,
                )
            ),
            reordered_sources,
        )
    )

    return reordered_and_relabeled_sources


def format_concurrent_vectors(n_vec) -> str:
    return "Single vector" if n_vec == 1 else "Multivector (4)"


def plot_ffor(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "ffor.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (pl.col("n_vecs") / (pl.col("duration_ns") / 1000)).alias("throughput"),
        (pl.col("duration_ns") / 1000).alias("duration_us"),
    )

    for measurement, label in zip(["throughput", "duration_us",], ["Throughput (vecs/us)", "Execution time (us)",]):
        sources = create_grouped_data_sources(
            df,
            ["data_type", "kernel", "unpack_n_vectors", "unpacker"],
            "vbw",
            measurement,
        )

        stateful_storage_types = [
            "cache",
            "local",
            "shared",
            "register",
            "register_branchless",
        ]
        source_sets = (
            [
                SourceSet(
                    f"32-switch-vs-1-switch-{'duration-' if measurement == 'duration_us' else ''}v1-u32",
                    define_graph(
                        sources,
                        [
                            "u32",
                            "query",
                            1,
                            [
                                "old_fls",
                                "switch_case",
                            ],
                        ],
                        lambda x: x[3],
                    ),
                    title=f"u32, {format_concurrent_vectors(1)}",
                    colors=range(0, 2),
                )
            ]
            + [
                SourceSet(
                    f"old-fls-vs-stateless-{'duration-' if measurement == 'duration_us' else ''}{kernel}-v{n_vec}-{data_type}",
                    define_graph(
                        sources,
                        [
                            data_type,
                            kernel,
                            n_vec,
                            [
                                "old_fls" if n_vec == 1 and data_type == "u32" else "",
                                "switch_case" if n_vec == 1 and data_type == "u32" else "",
                                "stateless",
                            ],
                        ],
                        lambda x: x[3],
                    ),
                    title=f"{data_type}, {format_concurrent_vectors(n_vec)}",
                    colors=range(0 if n_vec == 1 and data_type == "u32" else 2, 3),
                )
                for kernel, n_vec, data_type in itertools.product(
                    ["query", "decompress"], [1, 4], ["u32", "u64"]
                )
            ]
            + [
                SourceSet(
                    f"stateful-{'duration-' if measurement == 'duration_us' else ''}b{buffer_size}-v{n_vec}-{data_type}",
                    define_graph(
                        sources,
                        [
                            data_type,
                            "query",
                            n_vec,
                            [
                                (
                                    (f"stateful_{storage_type}" if buffer_size == 1 else "")
                                    if storage_type == "cache"
                                    else f"stateful_{storage_type}_{buffer_size}"
                                )
                                for storage_type in stateful_storage_types
                            ],
                        ],
                        lambda x: x[3],
                    ),
                    title=f"{data_type}, {format_concurrent_vectors(n_vec)}, Buffer Size: {buffer_size}",
                    colors=range(0 if buffer_size == 1 else 1, len(stateful_storage_types)),
                )
                for buffer_size, n_vec, data_type in itertools.product(
                    [1, 2, 4],
                    [1, 4],
                    ["u32", "u64"],
                )
            ]
            + [
                SourceSet(
                    f"all-{kernel}-{'duration-' if measurement == 'duration_us' else ''}v{n_vec}-{data_type}",
                    define_graph(
                        sources,
                        [
                            data_type,
                            kernel,
                            n_vec,
                            [
                                "old_fls" if n_vec == 1 and data_type == "u32" else "",
                                "stateless",
                                "stateful_register_branchless_2",
                                "stateful_branchless",
                            ],
                        ],
                        lambda x: x[3],
                    ),
                    title=f"{data_type}, {format_concurrent_vectors(n_vec)}",
                    colors=range(0 if n_vec == 1 and data_type == "u32" else 1, 5),
                )
                for kernel, n_vec, data_type in itertools.product(
                    ["query", "decompress"], [1, 4], ["u32", "u64"]
                )
            ]
        )

        y_lim = calculate_common_y_lim(
            itertools.chain.from_iterable(map(lambda x: x.sources, source_sets))
        )

        for source_set in source_sets:
            sources = assign_colors(source_set.sources, source_set.colors)
            sources = reorder_and_relabel(sources, UNPACKERS_ORDER_AND_NAMES)

            create_scatter_graph(
                sources,
                "Value bit width",
                label,
                os.path.join(output_dir, f"ffor-{source_set.file_name}.eps"),
                y_lim=(0, y_lim if measurement == "throughput" else calculate_common_y_lim(sources)),
                octal_grid=True,
                title=source_set.title,
            )


def plot_alp_ec(input_dir: str, output_dir: str):
    df = pl.read_csv(os.path.join(input_dir, "alp-ec.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (pl.col("duration_ns") / 1000).alias("duration_us"),
        (pl.col("n_vecs") / (pl.col("duration_ns") / 1000)).alias("throughput"),
    )

    sources = create_grouped_data_sources(
        df,
        ["data_type", "kernel", "unpack_n_vectors", "unpacker", "patcher"],
        "ec",
        "throughput",
    )

    alp_unpacker = "stateful_branchless"
    source_sets = (
        [
            SourceSet(
                f"alp-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        alp_unpacker,
                        [
                            "stateless",
                            "stateful",
                        ],
                    ],
                    lambda x: x[4],
                ),
                title=f"{data_type}, {format_concurrent_vectors(n_vec)}",
                colors=range(0, 2),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query"], [1, 4], ["f32", "f64"]
            )
        ]
        + [
            SourceSet(
                f"galp-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        alp_unpacker,
                        [
                            "naive",
                            "naive_branchless",
                        ],
                    ],
                    lambda x: x[4],
                ),
                title=f"{data_type}, {format_concurrent_vectors(n_vec)}",
                colors=range(2, 4),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query"], [1, 4], ["f32", "f64"]
            )
        ]
        + [
            SourceSet(
                f"galp-buffers-{kernel}-v{n_vec}-{data_type}",
                define_graph(
                    sources,
                    [
                        data_type,
                        kernel,
                        n_vec,
                        alp_unpacker,
                        [
                            "prefetch_all",
                            "prefetch_all_branchless",
                        ],
                    ],
                    lambda x: x[4],
                ),
                title=f"{data_type}, {format_concurrent_vectors(n_vec)}",
                colors=range(4, 6),
            )
            for kernel, n_vec, data_type in itertools.product(
                ["query"], [1, 4], ["f32", "f64"]
            )
        ]
    )

    y_lim = calculate_common_y_lim(
        itertools.chain.from_iterable(map(lambda x: x.sources, source_sets))
    )

    for source_set in source_sets:
        sources = assign_colors(source_set.sources, source_set.colors)
        sources = reorder_and_relabel(sources, PATCHERS_ORDER_AND_NAMES)

        create_scatter_graph(
            sources,
            "Exception count",
            "Throughput (vecs/us)",
            os.path.join(output_dir, f"alp-ec-{source_set.file_name}.eps"),
            y_lim=(0, y_lim),
            title=source_set.title,
        )


def plot_multi_column(input_dir: str, output_dir: str):
    n_cols = 10
    df = pl.read_csv(os.path.join(input_dir, "multi-column.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (
            (pl.col("n_vecs") * pl.col("n_cols")) / (pl.col("duration_ns") / 1000)
        ).alias("throughput"),
    )

    sources = create_grouped_data_sources(
        df,
        ["encoding", "data_type", "unpack_n_vectors", "unpacker", "patcher"],
        "n_cols",
        "throughput",
    )

    ffor_source_sets = [
        SourceSet(
            f"FFOR-{data_type}",
            define_graph(
                sources,
                [
                    "FFOR",
                    data_type,
                    [1, 4],
                    [
                        "old_fls",
                        "stateful_register_branchless_2",
                        "stateful_branchless",
                    ],
                    "none",
                ],
                lambda x: f"{x[3]} {x[2]}v",
            ),
            title=f"{data_type}",
            colors=range(0 if data_type == "u32" else 1, 6),
        )
        for data_type in [
            "u32",
            "u64",
        ]
    ]

    alp_source_sets = [
        SourceSet(
            f"ALP-{unpacker}-{data_type}",
            define_graph(
                sources,
                [
                    "ALP",
                    data_type,
                    [1, 4],
                    unpacker,
                    [
                        "stateful",
                        "prefetch_all",
                        "prefetch_all_branchless",
                    ],
                ],
                lambda x: f"{x[4]} {x[2]}v",
            ),
            title=f"{replace_label(unpacker, UNPACKERS_ORDER_AND_NAMES)} Decoder, {data_type}",
            colors=(
                range(0, 6)
                if unpacker != "old_fls"
                else [
                    0,
                    2,
                    4,
                ]
            ),
        )
        for unpacker, data_type in itertools.product(
            [
                "old_fls",
                "stateful_register_branchless_2",
                "stateful_branchless",
            ],
            [
                "f32",
                "f64",
            ],
        )
        if not (unpacker == "old_fls" and data_type == "f64")
    ]

    ffor_y_lim = calculate_common_y_lim(
        itertools.chain.from_iterable(map(lambda x: x.sources, ffor_source_sets))
    )
    alp_y_lim = calculate_common_y_lim(
        itertools.chain.from_iterable(map(lambda x: x.sources, alp_source_sets))
    )

    for source_sets, y_lim, name_and_order_map in zip(
        [
            ffor_source_sets,
            alp_source_sets,
        ],
        [
            ffor_y_lim,
            alp_y_lim,
        ],
        [
            UNPACKERS_ORDER_AND_NAMES,
            PATCHERS_ORDER_AND_NAMES,
        ],
    ):
        for source_set in source_sets:
            sources = reorder_and_relabel(source_set.sources, name_and_order_map)
            sources = assign_colors(sources, source_set.colors)

            create_multi_bar_graph(
                sources,
                list(map(str, range(1, n_cols + 1))),
                "Number of columns",
                "Throughput (vectors/column/us)",
                os.path.join(output_dir, f"multi-column-{source_set.file_name}.eps"),
                y_lim=(0, y_lim),
                title=source_set.title,
                figsize=(12, 6),
            )


def plot_compressors(input_dir: str, output_dir: str):
    compression_ratio_axis_limit = 12
    df = pl.read_csv(
        os.path.join(input_dir, "compressors.csv"),
        schema_overrides={
            "avg_bits_per_value": pl.Float64,
            "avg_exceptions_per_vector": pl.Float64,
        },
    ).sort("file")
    df = df.with_columns(
        (
            ((pl.col("n_bytes") / pl.col("compression_ratio")) * 8)
            / (pl.col("n_vecs") * VECTOR_SIZE)
        ).alias("paper_bits_per_value")
    )
    df = average_samples(
        df,
        [
            "duration_ms",
            "avg_bits_per_value",
            "paper_bits_per_value",
            "avg_exceptions_per_vector",
            "compression_ratio",
        ],
    )

    # Create avg bits/value exception count table
    for data_type in ["f32", "f64"]:
        alp_df = df.filter(
            (pl.col("compressor") == "ALP")
            & (pl.col("kernel") == "decompression")
            & (pl.col("data_type") == data_type)
        )
        galp_df = df.filter(
            (pl.col("compressor") == "GALP")
            & (pl.col("kernel") == "decompression")
            & (pl.col("data_type") == data_type)
        )
        create_latex_table(
            [
                alp_df.get_column("avg_bits_per_value").to_list(),
                alp_df.get_column("avg_exceptions_per_vector").to_list(),
                alp_df.get_column("paper_bits_per_value").to_list(),
                galp_df.get_column("paper_bits_per_value").to_list(),
                alp_df.get_column("compression_ratio").to_list(),
                galp_df.get_column("compression_ratio").to_list(),
            ],
            [
                "Avg. value bit width",
                "Avg. exceptions / vector",
                "Avg. bits / value ALP",
                "Avg. bits / value GALP",
                "Compression ratio ALP",
                "Compression ratio GALP",
            ],
            alp_df.get_column("file").to_list(),
            os.path.join(
                output_dir,
                f"compressors-table-alp-compression-parameters-{data_type}.tex",
            ),
            lambda x: [f"{v:.2f}" for v in x],
            vertical_column_names=True,
            add_averages=True,
            add_medians=True,
        )

    df = df.with_columns(
        (
            (pl.col("n_bytes") / (1024 * 1024 * 1024)) / (pl.col("duration_ms") / 1000)
        ).alias("throughput")
    )

    sources = create_grouped_data_sources(
        df,
        ["kernel", "compressor", "data_type"],
        "compression_ratio",
        "throughput",
    )

    source_sets = [
        SourceSet(
            f"scatter-{kernel}-{data_type}",
            define_graph(
                sources,
                [
                    kernel,
                    [],
                    data_type,
                ],
                lambda x: x[1],
            ),
        )
        for kernel, data_type in itertools.product(
            [
                "decompression",
                "decompression_query",
            ],
            [
                "f32",
                "f64",
            ],
        )
    ]
    y_lim = calculate_common_y_lim(
        itertools.chain.from_iterable(map(lambda x: x.sources, source_sets))
    )

    for source_set in source_sets:
        sources = reorder_and_relabel(source_set.sources, COMPRESSORS_ORDER_AND_NAMES)
        sources = assign_colors(sources, source_set.colors)

        create_scatter_graph(
            sources,
            "Compression ratio",
            "Throughput (GB/s) on Log scale",
            os.path.join(output_dir, f"compressors-{source_set.file_name}.eps"),
            y_lim=(0, y_lim),
            legend_pos="best",
            figsize=(9, 9),
            x_lim=(0, compression_ratio_axis_limit),
            y_axis_log_scale=True,
        )

    for label, measurement in zip(
        [
            "Compression Ratio",
            "Throughput (GB/s)",
        ],
        [
            "compression_ratio",
            "throughput",
        ],
    ):
        sources = create_grouped_data_sources(
            df,
            ["kernel", "compressor", "data_type"],
            "file",
            measurement,
        )

        source_sets = [
            SourceSet(
                f"boxplot-{measurement}-{kernel}-{data_type}",
                define_graph(
                    sources,
                    [
                        kernel,
                        [],
                        data_type,
                    ],
                    lambda x: x[1],
                ),
            )
            for kernel, data_type in itertools.product(
                [
                    "decompression",
                    "decompression_query",
                ],
                [
                    "f32",
                    "f64",
                ],
            )
        ]

        y_lim = calculate_common_y_lim(
            itertools.chain.from_iterable(map(lambda x: x.sources, source_sets))
        )

        for source_set in source_sets:
            sources = reorder_and_relabel(source_set.sources, COMPRESSORS_ORDER_AND_NAMES)
            sources = assign_colors(sources, source_set.colors)

            create_boxplot_graph(
                sources,
                "Compressor",
                label,
                os.path.join(output_dir, f"compressors-{source_set.file_name}.eps"),
                y_lim=(
                    (0, compression_ratio_axis_limit)
                    if measurement == "compression_ratio"
                    else (0, y_lim)
                ),
                x_label_rotation=45,
                title=f"{'Compression ratio' if measurement == 'compression_ratio' else 'Throughput'} per decompressor for {'single' if sources[0].group_by_column_values[2] == 'f32' else 'double'} precision floating-point datasets.",
            )

            create_latex_table(
                [s.y_data for s in sources],
                [s.label for s in sources],
                sources[0].x_data,
                os.path.join(
                    output_dir, f"compressors-table-{source_set.file_name}.tex"
                ),
                lambda x: format_row_colors(
                    x,
                    lambda y: f"{y:.2f}",
                    lower_is_better=measurement == "paper_bits_per_value",
                ),
                vertical_column_names=True,
                add_averages=True,
                add_medians=True,
            )


def plot_ilp_experiment(input_dir: str, output_dir: str):
    MAX_THREAD_BLOCK_SIZE = 1024
    TOTAL_PTRS_CHASED = 100 * 10 * 1024 * 1024
    df = pl.read_csv(os.path.join(input_dir, "ilp-experiment.csv"))
    df = average_samples(df, ["duration_ns"])
    df = df.with_columns(
        (TOTAL_PTRS_CHASED / pl.col("duration_ns")).alias("throughput"),
        (pl.col("duration_ns") / 1000).alias("duration_us"),
        (pl.col("threads/block") / MAX_THREAD_BLOCK_SIZE).alias("occupancy"),
    )

    sources = [
        DataSource(
            f"{label_data[0][0]} concurrent ptrs/warp",
            i,
            label_data[1].get_column("occupancy").to_list(),
            label_data[1].get_column("duration_us").to_list(),
        )
        for i, label_data in enumerate(df.group_by(["ilp"], maintain_order=True))
    ]

    create_scatter_graph(
        sources,
        "Occupancy (%)",
        "Duration (us)",
        os.path.join(output_dir, "ilp-experiment-duration.eps"),
        legend_pos="upper right",
        add_lines=True,
        x_axis_percentage=True,
    )

    sources = [
        DataSource(
            f"{label_data[0][0]} concurrent ptrs/warp",
            i,
            label_data[1].get_column("occupancy").to_list(),
            label_data[1].get_column("throughput").to_list(),
        )
        for i, label_data in enumerate(df.group_by(["ilp"], maintain_order=True))
    ]
    create_scatter_graph(
        sources,
        "Occupancy (%)",
        "Throughput (ptrs/ns)",
        os.path.join(output_dir, "ilp-experiment-throughput.eps"),
        legend_pos="lower right",
        add_lines=True,
        x_axis_percentage=True,
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
