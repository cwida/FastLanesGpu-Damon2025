#!/usr/bin/env python3


import os
from typing import Any
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

base_dir = "export-results"
graphs_dir = "graphs"

test = pd.read_csv(f"{base_dir}/rtx4070-alp-microbenchmark.csv")
comp_nor_data_tx = pd.read_csv(f"{base_dir}/rtx4070-compressors-normal-data.csv")
rtx_multicol = pl.read_csv(f"{base_dir}/rtx4070-multicolumn-microbenchmark.csv")
v100_multicol = pl.read_csv(f"{base_dir}/v100-multicolumn-microbenchmark.csv")

pd.set_option("display.max_columns", None)
v100_multicol.head(10)

rtx_float = rtx_multicol.filter(pl.col("data_type") == "f32")
rtx_double = rtx_multicol.filter(pl.col("data_type") == "f64")
v100_float = v100_multicol.filter((pl.col("data_type") == "f32")
                                  &(pl.col("encoding") == "ALP")
                                  &(pl.col("unpack_n_vectors") == 1)

                                  )
v100_double = v100_multicol.filter(pl.col("data_type") == "f64")

valid_combinations = [
    ("old_fls", "stateful", "ALP - 32"),
    ("old_fls", "prefetch_all", "G-ALP - 32"),
    ("stateful_branchless", "stateful", "ALP - 1"),
    ("stateful_branchless", "prefetch_all", "G-ALP - 1"),
]

valid_combinations_df = pl.from_records(
    valid_combinations, schema=["unpacker", "patcher", "alias"], orient="row"
)

rtx_filtered_float = rtx_float.join(
    valid_combinations_df, on=["unpacker", "patcher"], how="inner"
)

rtx_filtered_double = rtx_double.join(
    valid_combinations_df, on=["unpacker", "patcher"], how="inner"
)

v100_filtered_float = v100_float.join(
    valid_combinations_df, on=["unpacker", "patcher"], how="inner"
)

v100_filtered_double = v100_double.join(
    valid_combinations_df, on=["unpacker", "patcher"], how="inner"
)

rtx_filtered = rtx_multicol.join(
    valid_combinations_df, on=["unpacker", "patcher"], how="inner"
)

v100_filtered = v100_multicol.join(
    valid_combinations_df, on=["unpacker", "patcher"], how="inner"
)


def create_multi_bar_graph(
    data_per_bar: dict[str, list[Any]],
    bargroup_x_labels: list[str] | None,
    x_label: str,
    y_label: str,
    out: str,
    y_lim: tuple[int, int] | None = None,
    colors: list[int] | None = None,
    show: bool =False,
    save: bool = False,
    show_legend: bool = True,
    is_percent: bool = False,
):
    n_bars = len(data_per_bar)
    n_groups = len(list(data_per_bar.values())[0])
    assert all(len(x) == n_groups for x in data_per_bar.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.8 / n_bars
    indices = np.arange(n_groups)

    # for i, bar_label_data in enumerate(data_per_bar.items()):

    #     bar_label = bar_label_data[0]
    #     bar_data = bar_label_data[1]

    #     positions = indices + (i - n_bars / 2 + 0.5) * bar_width
    #     ax.bar(
    #         positions, bar_data, bar_width, label=bar_label, color=colors[i] if colors else COLOR_SET[i], edgecolor="black",
    # linewidth=1
    #     )

    for i, label in enumerate(bargroup_labels):
        bar_data = data_per_bar[label]
        positions = indices + (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(
            positions,
            bar_data,
            bar_width,
            label=label,
            color=colors[i] if colors else COLOR_SET[i],
            edgecolor="black",
            linewidth=1,
        )

    # Remove the borders (spines)
    ax = plt.gca()  # Get the current axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Optionally, you can also remove the ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    font_family = "Georgia"
    # Improve labels & title
    plt.xlabel("Number of columns scanned in parallel", fontsize=20, family=font_family)
    plt.ylabel("Throughput (32-bit values / cycle)", fontsize=20, family=font_family)
    # plt.title("Multicolumn Throughput Comparison", fontsize=24, fontweight="bold")

    if is_percent:
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_ylim((0, 100))

    plt.xticks(fontsize=18, family=font_family)
    plt.yticks(fontsize=18, family=font_family)

    if bargroup_labels:
        bargroup_x_labels = list(range(1, n_groups + 1))
        ax.set_xticks(indices)
        ax.set_xticklabels(bargroup_x_labels)
    else:
        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    legend = ax.legend(fontsize=12)
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_boxstyle('square')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)

    if not show_legend:
        legend = plt.gca().get_legend()
        if legend:
            legend.remove()

    if save:
        fig.savefig(out, dpi=300, format="pdf", bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


from polars import selectors as cs


def plot_multicolumn(input_df, input_name, color_list):
    df = input_df
    df = df.with_columns(
        (
            (pl.col("n_cols") * pl.col("n_vecs") * 1024)
            / pl.col("Elapsed Cycles (cycle)")
        ).alias("throughput")
    )

    df_avg = (
        df.sort("n_cols", maintain_order=True)
        .group_by("n_cols", "alias", maintain_order=True)
        .agg(cs.numeric().mean())
    )

    bargroup_labels = [
        "Previous decoder + G-ALP",
        "Previous decoder + ALP",
        "New decoder + G-ALP",
        "New decoder + ALP",
    ]

    data = {
        str(label[0]): data.get_column("throughput").to_list()
        for label, data in df_avg.group_by("alias", maintain_order=True)
    }
    create_multi_bar_graph(
        data,
        bargroup_labels,
        "Number of columns scanned in parallel",
        "Throughput (values/cycle)",
        os.path.join(f"{graphs_dir}/{input_name}_multicolumn_throughput_comparison.pdf"),
        colors=color_list,
        save=True,
    )
    
    metrics = [
        ("Compute (SM) Throughput (%)", "Compute Throughput (%)", "compute_throughput"),
        ("Achieved Occupancy (%)", "Achieved Occupancy (%)", "occupancy"),
    ]

    for metric, label, tag in metrics:
        data = {
            str(label[0]): data.get_column(metric).to_list()
            for label, data in df_avg.group_by("alias", maintain_order=True)
        }
        create_multi_bar_graph(
            data,
            bargroup_labels,
            "Number of columns scanned in parallel",
            label,
            os.path.join(f"{graphs_dir}/{input_name}_multicolumn_{tag}_comparison.pdf"),
            colors=color_list,
            save=True,
            show_legend=False,
            is_percent=(tag=="occupancy"),
        )


color_palette = {
    "light_blue": "#87bdff",
    "dark_blue": "#2881ed",
    "light_yellow": "#fce77c",
    "dark_yellow": "#fad51b",
    "light_green": "#a8e6a1",
    "dark_green": "#2e7d32",
}

color_map = {
    "ALP - 32": color_palette["light_yellow"],
    "G-ALP - 32": color_palette["dark_yellow"],
    "ALP - 1": color_palette["light_green"],
    "G-ALP - 1": color_palette["dark_green"],
}

bargroup_labels = [
    "ALP - 32",
    "G-ALP - 32",
    "ALP - 1",
    "G-ALP - 1",
]

color_list = [color_map[label] for label in bargroup_labels]


# (unpacker, patcher): (old_fls, stateful) (old_fls, prefetch_all) (stateful_branchless, stateful) (stateful_branchless, prefetch_all)
#print("rtx_floats")
#plot_multicolumn(rtx_filtered_float, "rtx_floats", color_list)
print("v100_floats")
plot_multicolumn(v100_filtered_float, "v100_floats", color_list)
#print("v100_doubles")
#plot_multicolumn(v100_filtered_double, "v100_double", color_list)
