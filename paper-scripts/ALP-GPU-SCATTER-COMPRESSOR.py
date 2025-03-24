#!/usr/bin/env python3

import itertools
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BASE_DIR = "export-results"
GRAPHS_DIR = "graphs"
FONT_FAMILY = "Georgia"

COMPRESSORS_ORDER_AND_NAMES = {
    "ALP": "ALP",
    "GALP": "G-ALP",
    "Thrust": "Thrust",
    "zstd": "nv-zstd",
    "LZ4": "nv-LZ4",
    "Snappy": "nv-Snappy",
    "Deflate": "nv-Deflate",
    "GDeflate": "nv-GDeflate",
    "Bitcomp": "nv-Bitcomp",
    "BitcompSparse": "nv-BitcompSparse",
    "ndzip": "ndzip",
    "single-fpcompressor-speed": "SPspeed",
    "single-fpcompressor-ratio": "SPratio",
    "double-fpcompressor-speed": "DPspeed",
    "double-fpcompressor-ratio": "DPratio",
}

"""
COLOR_PALETTE = {
    "light_blue": "#87bdff",
    "dark_blue": "#2881ed",
    "light_yellow": "#fce77c",
    "light_green": "#a8e6a1",
    "dark_green": "#2e7d32",
    "light_red": "#ff9999",
    "dark_red": "#d32f2f",
    "light_pink": "#ffb3d1",
    "dark_pink": "#e91e63",
    "light_teal": "#80cbc4",
    "dark_teal": "#00695c",
}

COMPRESSORS_COLORS = {
    "ALP": COLOR_PALETTE["light_green"],
    "GALP": COLOR_PALETTE["dark_green"],
    "Thrust": COLOR_PALETTE["light_blue"],
    "zstd": COLOR_PALETTE["dark_yellow"],
    "LZ4": COLOR_PALETTE["dark_pink"],
    "Snappy": COLOR_PALETTE["light_teal"],
    "Deflate": COLOR_PALETTE["light_red"],
    "GDeflate": COLOR_PALETTE["dark_red"],
    "Bitcomp": COLOR_PALETTE["dark_orange"],
    "BitcompSparse": COLOR_PALETTE["light_orange"],
    "ndzip": COLOR_PALETTE["dark_yellow"],
    "single-fpcompressor-speed": COLOR_PALETTE["light_purple"],
    "single-fpcompressor-ratio": COLOR_PALETTE["dark_purple"],
    "double-fpcompressor-speed": COLOR_PALETTE["light_purple"],
    "double-fpcompressor-ratio": COLOR_PALETTE["dark_purple"],
}
"""

COLOR_PALETTE = {
    "light_green": "#90EE90",
    "dark_green": "#006400", 
    "dark_yellow": "#fad51b",
    "light_orange": "#ffb366",
    "dark_orange": "#f57c00",
    "blue_hue_1": "#87CEEB",  
    "blue_hue_2": "#5DADE2",  
    "blue_hue_3": "#3498DB",  
    "blue_hue_4": "#2E86C1",  
    "blue_hue_5": "#1F618D",  
    "blue_hue_6": "#154360",  
    "blue_hue_7": "#0B2F47",  
    "dark_pink": "#e91e63",
    "light_purple": "#c593ff",
    "dark_purple": "#7b1fa2",
}
COMPRESSORS_COLORS = {
    "ALP": COLOR_PALETTE["light_green"],
    "GALP": COLOR_PALETTE["dark_green"],
    "Thrust": COLOR_PALETTE["dark_yellow"],
    "zstd": COLOR_PALETTE["blue_hue_1"],
    "LZ4": COLOR_PALETTE["blue_hue_2"],
    "Snappy": COLOR_PALETTE["blue_hue_3"],
    "Deflate": COLOR_PALETTE["blue_hue_4"],
    "GDeflate": COLOR_PALETTE["blue_hue_5"],
    "Bitcomp": COLOR_PALETTE["blue_hue_6"],
    "BitcompSparse": COLOR_PALETTE["blue_hue_7"],
    "ndzip": COLOR_PALETTE["dark_pink"],
    "single-fpcompressor-speed": COLOR_PALETTE["light_purple"],
    "single-fpcompressor-ratio": COLOR_PALETTE["dark_purple"],
    "double-fpcompressor-speed": COLOR_PALETTE["light_purple"],
    "double-fpcompressor-ratio": COLOR_PALETTE["dark_purple"],
}


def filter_data(df: pl.DataFrame, data_type: str, kernel: str) -> pl.DataFrame:
    assert data_type in [
        "f32",
        "f64",
    ]
    assert kernel in [
        "decompression",
        "decompression_query",
    ]

    return df.filter((pl.col("data_type") == data_type) & (pl.col("kernel") == kernel))


def compute_throughput(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col("n_bytes") / (1024 * 1024 * 1024)) / (pl.col("duration_ms") / 1000)
        ).alias("GB_per_sec")
    )

def sort_using_dict(df: pl.DataFrame, category: str, ordered_dict: dict) -> pl.DataFrame:
    order_map = {k: i for i, k in enumerate(ordered_dict.keys())}

    return (
        df.with_columns(
            pl.col(category).replace(order_map).alias("__sort_key")
        )
        .sort("__sort_key")
        .drop("__sort_key")
    )

def alias_compressor_category(df: pl.DataFrame, sort: bool=True) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("compressor").replace(COMPRESSORS_ORDER_AND_NAMES).alias("compressor_alias")
    )
    if sort:
        df = sort_using_dict(df, "compressor", COMPRESSORS_ORDER_AND_NAMES)
    return df


def compute_bits_per_value(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (
            (
                (pl.col("n_bytes") / pl.col("compression_ratio")) * 8
            )  # Compressed size in bits
            / (
                pl.col("n_bytes")
                / (pl.when(pl.col("data_type") == "f32").then(4).otherwise(8))
            )  # Number of values
        ).alias("compressed_bits_per_value")
    )

    return df


def aggregate_by_compressor(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(["compressor", "data_type"]).agg(
        [
            pl.col("compression_ratio").log().mean().exp(),  # Geometric mean
            pl.col("compressed_bits_per_value").mean(),
            pl.col("GB_per_sec").mean(),
            pl.col("n_bytes").mean(),
            pl.col("duration_ms").mean(),
        ]
    )

    df = df.with_columns(
        (
            (pl.when(pl.col("data_type") == "f32").then(32).otherwise(64))
            / pl.col("compressed_bits_per_value")
        ).alias("inverse_compression_ratio")
    )

    return df


def plot_scatter_graph(
    df: pl.DataFrame,
    input_name: str,
    show: bool = False,
    save: bool = False,
    label_position_mapping: dict[str, tuple[int, int]] = {},
) -> None:
    plt.figure(figsize=(10, 6))
    # Convert to pandas for seaborn compatibility
    df_pandas = df.to_pandas()
    ax = sns.scatterplot(
        data=df_pandas,
        # x="compression_ratio",
        x="inverse_compression_ratio",
        y="GB_per_sec",
        hue="compressor",
        palette=COMPRESSORS_COLORS,
        s=100,
    )

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.yscale("log")
    plt.xlabel("Compression Ratio", fontsize=20, family=FONT_FAMILY)
    plt.ylabel("Throughput GB/s (log scale)", fontsize=20, family=FONT_FAMILY)
    # plt.title("Compressor Throughput vs. Compression Ratio", family=FONT_FAMILY)

    legend = plt.gca().get_legend()
    if legend:
        legend.remove()
    plt.tight_layout()

    for _, row in df_pandas.iterrows():
        compressor = row["compressor"]
        ax.annotate(
            COMPRESSORS_ORDER_AND_NAMES[compressor],
            (row["inverse_compression_ratio"], row["GB_per_sec"]),
            xytext=label_position_mapping.get(
                compressor, (10, 10)
            ),  # offset from the point
            textcoords="offset points",
            fontsize=18,
            fontfamily=FONT_FAMILY,
            # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )

    if save:
        output_path = f"{GRAPHS_DIR}/{input_name}_compressor_scatter.pdf"
        plt.savefig(output_path, format="pdf", dpi=300)

    if show:
        plt.show()

def create_compression_scatter_diagram() -> None:
    gpus = [
        "v100",
        # "rtx4070",
    ]

    data_sources = [
        "normal",
        # "repeated",
    ]

    data_types = [
        "f32",
        "f64",
    ]

    kernels = [
        "decompression",
        # "decompression_query",
    ]

    label_position_mapping = {
        "f32": {
            "zstd": (-50, 10),
            "GALP": (10, 0),
            "Bitcomp": (10, -10),
        },
        "f64": {
            "Bitcomp": (-10, -35),
            "Snappy": (10, -20),
            "ALP": (-80, -25),
            "GALP": (-65, 0),
        },
    }

    for gpu, data_source, data_type, kernel in itertools.product(
        gpus,
        data_sources,
        data_types,
        kernels,
    ):
        df = pl.read_csv(f"{BASE_DIR}/{gpu}-compressors-{data_source}-data.csv")
        df = filter_data(df, data_type, kernel)
        df = compute_throughput(df)
        df = compute_bits_per_value(df)
        df = aggregate_by_compressor(df)

        name = f"{gpu}-{data_source}-{data_type}-{kernel}"
        print(name)
        plot_scatter_graph(
            df,
            name,
            show=False,
            save=True,
            label_position_mapping=label_position_mapping[data_type],
        )

def create_table() -> None:
    gpus = [
        "v100",
        "rtx4070",
    ]

    data_sources = [
        #"normal",
        "repeated",
    ]

    data_types = [
        "f32",
        "f64",
    ]

    kernels = [
        "decompression",
        "decompression_query",
    ]

    dfs = []
    for gpu, data_source, data_type, kernel in itertools.product(
        gpus,
        data_sources,
        data_types,
        kernels,
    ):
        df = pl.read_csv(f"{BASE_DIR}/{gpu}-compressors-{data_source}-data.csv")
        df = filter_data(df, data_type, kernel)
        df = compute_throughput(df)
        df = compute_bits_per_value(df)
        df = aggregate_by_compressor(df)
        df = df.with_columns(
            (pl.lit(gpu)).alias("gpu"),
            (pl.lit(data_source)).alias("data_source"),
            (pl.lit(data_type)).alias("data_type"),
            (pl.lit(kernel)).alias("kernel"),
        )

        dfs.append(df)

    df = pl.concat(dfs)
    df = df.select([
        "compressor", 
        "gpu", 
        "data_type", 
        "kernel", 
        "inverse_compression_ratio", 
        "GB_per_sec", 
    ])

    df = sort_using_dict(df, "compressor", COMPRESSORS_ORDER_AND_NAMES)

    def select_value(
        df: pl.DataFrame,
        compressor: str,
        value: str,
        data_type: str,
        gpu: str,
        kernel: str|None=None,
    ):
        df = df.filter(
            (pl.col("data_type") == data_type)
            & (pl.col("gpu") == gpu)
            & (pl.col("compressor") == compressor)
        )

        if kernel is not None:
            df = df.filter(
                (pl.col("kernel") == kernel)
            )

        result = df[value]

        if len(result) == 0:
            return "-"
        return result[0]

    def format_compression_ratio(cr: float|None) -> str:
        if isinstance(cr, float):
            return f"{cr:.02f}"
        else: 
            return cr

    
    for compressor in df.select("compressor").unique(maintain_order=True).iter_rows():
        compressor = compressor[0]
        row_def = {
            "label": COMPRESSORS_ORDER_AND_NAMES[compressor],
            "float_cr": format_compression_ratio(select_value(df,compressor, "inverse_compression_ratio", "f32", "rtx4070")),
            "float_rtx4070_filter": select_value(df,compressor,  "GB_per_sec", "f32", "rtx4070", "decompression_query"),
            "float_rtx4070_decomp": select_value(df,compressor,  "GB_per_sec", "f32", "rtx4070", "decompression"),
            "float_v100_filter": select_value(df,compressor,  "GB_per_sec", "f32", "v100", "decompression_query"),
            "float_v100_decomp": select_value(df,compressor,  "GB_per_sec", "f32", "v100", "decompression"),
            "double_cr": format_compression_ratio(select_value(df,compressor,  "inverse_compression_ratio", "f64", "rtx4070")),
            "double_rtx4070_filter": select_value(df,compressor,  "GB_per_sec", "f64", "rtx4070", "decompression_query"),
            "double_rtx4070_decomp": select_value(df,compressor,  "GB_per_sec", "f64", "rtx4070", "decompression"),
            "double_v100_filter": select_value(df,compressor,  "GB_per_sec", "f64", "v100", "decompression_query"),
            "double_v100_decomp": select_value(df,compressor,  "GB_per_sec", "f64", "v100", "decompression"),
        }

        latex = "&".join([f"{x:.01f}" if isinstance(x, float) else x for x in row_def.values()]) + " \\\\\n\\hline"

        print(latex)

    





def plot_grouped_bar_chart(
    df: pl.DataFrame,
    input_name: str,
    show: bool = False,
    save: bool = False,
):
    plt.figure(figsize=(4, 3))
    df = df.to_pandas()

    ax = sns.barplot(
        data=df,
        x="GPU",
        y="throughput_in_ram",
        hue="data_type",
        palette=["#1f77b4", "#ff7f0e"],
        edgecolor='black',
        width=0.6,
    )

    ax.set_xlabel("GPU", fontsize=8, fontweight="bold", fontfamily=FONT_FAMILY)
    ax.set_ylabel(
        "Filter Throughput as % of RAM bandwidth",
        fontsize=8,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
    )
    ax.set_ylim((0, 2.5))

    # Format Y-axis as percentage
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.tick_params(axis="both", which="major", labelsize=9)

    legend = plt.gca().get_legend()
    if legend:
        legend.remove()
    plt.tight_layout()

    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()

    labels = [
        ["float", "float"],
        ["double", "double"],
    ]
    for i, container in enumerate(ax.containers):
        ax.bar_label(
            container, labels=labels[i], fontsize=8, padding=2, fontfamily=FONT_FAMILY
        )

    if save:
        output_path = f"{GRAPHS_DIR}/{input_name}.pdf"
        plt.savefig(output_path, format="pdf", dpi=300)

    if show:
        plt.show()


def create_hardware_real_data_comparison() -> None:
    # data_source = "normal"
    data_source = "repeated"
    kernel = "decompression_query"

    # V100
    # (1752 Mbps/1000)  * 4096 bit bus / 8 = 897 GB/s
    # NVIDIA GeForce RTX 4070 Ti SUPER
    # 21 Gbps  * 256 bit bus / 8 = 672 GB/s
    gpus = [  # gpu, RAM throughput GB/s
        ("v100", 897),
        ("rtx4070", 672),
    ]

    dfs = []
    for gpu, ram_throughput in gpus:
        df = pl.read_csv(f"{BASE_DIR}/{gpu}-compressors-{data_source}-data.csv")
        df = df.filter((pl.col("kernel") == kernel) & (pl.col("compressor") == "GALP"))
        df = compute_throughput(df)
        df = compute_bits_per_value(df)
        df = aggregate_by_compressor(df)
        df = df.with_columns(
            (pl.col("GB_per_sec") / ram_throughput).alias("throughput_in_ram"),
            pl.lit(gpu.upper()).alias("GPU"),
        )
        dfs.append(df)

    df = pl.concat(dfs).sort("GPU").sort("data_type", maintain_order=True)

    name = f"hardware-comparison-real-data"
    print(name)
    plot_grouped_bar_chart(
        df,
        name,
        show=False,
        save=True,
    )


def create_hardware_micro_benchmark_comparison() -> None:
    # V100
    # (1752 Mbps/1000)  * 4096 bit bus / 8 = 897 GB/s
    # NVIDIA GeForce RTX 4070 Ti SUPER
    # 21 Gbps  * 256 bit bus / 8 = 672 GB/s
    gpus = [  # gpu, RAM throughput GB/s
        ("v100", 897),
        ("rtx4070", 672),
    ]

    dfs = []
    for gpu, ram_throughput in gpus:
        print(gpu, ram_throughput)
        df = pl.read_csv(f"{BASE_DIR}/{gpu}-alp-microbenchmark.csv", ignore_errors=True)

        df = df.filter(
            (pl.col("kernel") == "query")
            & (pl.col("ec") == 10)
            & (pl.col("unpacker") == "stateful_branchless")
            & (pl.col("patcher") == "prefetch_all")
            & (pl.col("unpack_n_values") == 1)
            & (pl.col("unpack_n_vectors") == 1)
        )
        df = df.with_columns(
            [
                (
                    pl.col("n_vecs")
                    * 1024
                    * (pl.when(pl.col("data_type") == "f32").then(4).otherwise(8))
                ).alias("n_bytes"),
                (
                    (
                        pl.col(
                            "Duration (ns)" if gpu == "v100" else "Duration (nsecond)"
                        )
                        / (1000 * 1000)
                    ).alias("duration_ms")
                ),
            ]
        )
        df = compute_throughput(df)

        df = df.group_by(["data_type"]).agg(
            [
                pl.col("GB_per_sec").mean(),
            ]
        )
        df = df.with_columns(
            (pl.col("GB_per_sec") / ram_throughput).alias("throughput_in_ram"),
            pl.lit(gpu.upper()).alias("GPU"),
        )
        dfs.append(df)

    df = pl.concat(dfs).sort("GPU").sort("data_type", maintain_order=True)

    name = f"hardware-comparison-micro-benchmark"
    plot_grouped_bar_chart(
        df,
        name,
        show=False,
        save=True,
    )

def plot_bar_chart(df: pl.DataFrame, input_name: str, show: bool = False, save: bool = False):
    plt.style.use('default')
    
    sns.set_palette(list(COLOR_PALETTE.values()))
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    df_pd = df.to_pandas()

    bars = sns.barplot(
        data=df_pd,
        x='compressor_alias',
        y='GB_per_sec',
        ax=ax,
        order=list(COMPRESSORS_ORDER_AND_NAMES.values())[:11],
        palette=list(COMPRESSORS_COLORS.values()),
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_yscale('log')
    
    #ax.set_title('Performance Comparison', fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Compressor', fontsize=16, fontweight='bold', fontfamily=FONT_FAMILY)
    ax.set_ylabel('Filter Throughput GB/s (log scale)', fontsize=16, fontweight='bold', fontfamily=FONT_FAMILY)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.grid(True, axis='y', which='minor', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, axis='both', which='major', alpha=0.3, linestyle='-', linewidth=0.5)

    ax.set_axisbelow(True)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontname=FONT_FAMILY)   
    from matplotlib.ticker import LogFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatter(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(0.1, 1, 0.1)))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{GRAPHS_DIR}/{input_name}.pdf", dpi=300, bbox_inches='tight', 
                   format="pdf", edgecolor='none')
    
    if show:
        plt.show()



def create_filter_bar_diagram() -> None:
    gpus = [
        "v100",
        # "rtx4070",
    ]

    data_sources = [
        "normal",
        #"repeated",
    ]

    data_types = [
        #"f32",
        "f64",
    ]

    kernels = [
        #"decompression",
        "decompression_query",
    ]

    for gpu, data_source, data_type, kernel in itertools.product(
        gpus,
        data_sources,
        data_types,
        kernels,
    ):
        df = pl.read_csv(f"{BASE_DIR}/{gpu}-compressors-{data_source}-data.csv")
        df = df.filter(
            (~pl.col("compressor").str.contains("fpcompressor"))
            & (pl.col("data_type") == data_type)
            & (pl.col("kernel") == kernel)
        )
        df = compute_throughput(df)
        df = compute_bits_per_value(df)
        df = aggregate_by_compressor(df)
        df = alias_compressor_category(df)

        name = f"filter-throughput-bar-chart"
        print(name)
        plot_bar_chart(
            df,
            name,
            show=False,
            save=True,
        )

if __name__ == "__main__":
    create_compression_scatter_diagram()
    #create_table()
    create_hardware_real_data_comparison()
    create_hardware_micro_benchmark_comparison()
    create_filter_bar_diagram()
