#!/usr/bin/env python3

import itertools
import os
import polars as pl
import matplotlib.pyplot as plt

VECTOR_SIZE = 1024
BASE_DIR = "export-results"
GRAPHS_DIR = "graphs"
FONT_FAMILY = "Georgia"


def calculate_throughput(df):
    return df.with_columns(
        ((pl.col("n_vecs") * VECTOR_SIZE) / pl.col("Elapsed Cycles (cycle)")).alias(
            "throughput"
        )
    )


def create_scatter_graph(data_dict, x_label, y_label, output_path, legend_pos, y_lim=(0, None), 
                         show: bool=False, save: bool=False, show_legend:bool=False):
    plt.figure(figsize=(8, 6))

    colors = ["#440154", "#3F6F70", "#39D98C"]
    markers = ["^", "p", "o"]

    for i, (label, data) in enumerate(data_dict.items()):
        plt.scatter(
            data["x"],
            data["y"],
            label=data["display_label"],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=1,
            edgecolor="black",
            s=80,
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    plt.xlabel(x_label, fontsize=20, family=FONT_FAMILY)
    plt.ylabel(y_label, fontsize=20, family=FONT_FAMILY)
    plt.xticks(fontsize=18, family=FONT_FAMILY)
    plt.yticks(fontsize=20, family=FONT_FAMILY)
    plt.ylim(y_lim)
    legend = plt.legend(loc=legend_pos, fontsize=16)
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_boxstyle('square')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    plt.grid(True, linestyle="--", linewidth=0.5)

    if not show_legend:
        legend = plt.gca().get_legend()
        if legend:
            legend.remove()
    
    if save:
        plt.savefig(output_path)
    if show:
        plt.show()


def main():
    gpus = [
        "v100",
        #"rtx4070",
    ]

    data_types = [
        "f32",
        "f64",
    ]

    kernels = [
        #"decompress",
        "query",
    ]

    for gpu, data_type, kernel in itertools.product(
        gpus,
        data_types,
        kernels,
    ):
        df = pl.read_csv(f"{BASE_DIR}/{gpu}-alp-microbenchmark.csv", ignore_errors=True)

        df = df.filter(
            (pl.col("unpacker") == "stateful_branchless")
            & (pl.col("data_type") == data_type)
            & (pl.col("kernel") == kernel)
            & (pl.col("patcher").is_in(["stateful", "naive", "prefetch_all"]))
        )

        df = (
            df.group_by(["ec", "patcher"])
            .agg(
                [
                    pl.col("unpacker").first(),
                    pl.col("data_type").first(),
                    pl.col("sample_run").count().alias("sample_count"),
                    pl.col("Elapsed Cycles (cycle)").mean(),
                    pl.col("Achieved Occupancy (%)").mean(),
                    pl.col("n_vecs").mean(),
                ]
            )
        )

        df = calculate_throughput(df)

        data_dict = {}
        label_mapping = {
            "stateful": "ALP",
            "naive": "G-ALP without buffer",
            "prefetch_all": "G-ALP",
        }
        for patcher, group_df in df.group_by(
            "patcher", maintain_order=True
        ):
            patcher = patcher[0]
            data_dict[patcher] = {
                "x": group_df.get_column("ec").to_list(),
                "y": group_df.get_column("throughput").to_list(),
                "display_label": label_mapping[patcher],
            }

        # Order for plotting
        ordered_data = {}
        for key in ["prefetch_all", "naive", "stateful"]:
            if key in data_dict:
                ordered_data[key] = data_dict[key]
        data_dict = ordered_data

        name = f"micro-{gpu}-{data_type}-{kernel}"
        print(name)
        create_scatter_graph(
            data_dict,
            "Exception count",
            "Throughput (values / cycle)",
            os.path.join(GRAPHS_DIR, f"{name}.pdf"),
            legend_pos="lower left",
            y_lim=(0, 300),
            save=True,
            show_legend=(data_type == "f32"),
        )


if __name__ == "__main__":
    main()
