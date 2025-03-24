#!/usr/bin/env python3

import os
import sys

import argparse
import logging

FILE_HEADER = """
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "./alp/alp-bindings.hpp"
#include "./datageneration.hpp"
#include "./gpu/alp.cuh"
#include "./gpu/host-alp-utils.cuh"
#include "./gpu/host-utils.cuh"
#include "./kernel.cuh"
#include "./nvcomp-compressors.cuh"
#include "./benchmark-compressors.cuh"

#ifndef MULTICOLUMNS_CUH
#define MULTICOLUMNS_CUH
"""

FILE_FOOTER = """
#endif // MULTICOLUMNS_CUH
"""


def generate_function_selector(n_cols: int, extended: bool):
    suffix = "_extended" if extended else ""
    compression_ratio_func = (
        "get_galp_compression_ratio" if extended else "get_alp_compression_ratio"
    )
    column_t = "AlpExtendedColumn<T>" if extended else "AlpColumn<T>"
    pass_column = lambda i: "".join([f"column_{x}," for x in range(i)])
    transfer_column = (
        lambda i: column_t
        + " column_"
        + str(i)
        + " = transfer::copy_alp"
        + suffix
        + "_column_to_gpu<T>(alp_data_ptr);"
    )
    destroy_column = lambda i: "transfer::destroy_alp_column<T>(column_" + str(i) + ");"

    return "\n".join(
        [
            "template <typename T, typename DecompressorT,  unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>",
            "__host__ bool scan_columns"
            + suffix
            + "_selector(const int32_t n_columns, const alp::AlpCompressionData<T>* alp_data_ptr) {",
            """
  constant_memory::load_alp_constants<float>();

  GPUArray<bool> d_out(1);
  T value = 123.456789;

  const ThreadblockMapping<float> mapping(
      utils::get_n_vecs_from_size(alp_data_ptr->size), N_VECS_CONCURRENTLY);
  CudaStopwatch stopwatch = CudaStopwatch();
  float execution_time_ms;
""",
            f"double compression_ratio = alp_data_ptr->{compression_ratio_func}();",
            "switch (n_columns) {",
            "\n".join(
                [
                    f"case {i}:"
                    + "{"
                    + "\n".join([transfer_column(x) for x in range(i)])
                    + "stopwatch.start();"
                    + f"scan_columns<T, DecompressorT, {column_t}, N_VECS_CONCURRENTLY, UNPACK_N_VALUES>"
                    + "<<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>("
                    + pass_column(i)
                    + "value, d_out.get());"
                    + "execution_time_ms = stopwatch.stop();"
                    + "\n".join([destroy_column(x) for x in range(i)])
                    + "}break;"
                    for i in range(1, n_cols + 1)
                ]
            ),
            "}",
            """
bool result;
d_out.copy_to_host(&result);
return result;
""",
            "}",
        ]
    )


def generate_global_function(n_cols: int):
    col_range = range(n_cols)
    return "\n".join(
        [
            "template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>",
            "__global__ void scan_columns("
            + ",".join([f"const ColumnT column_{i}" for i in col_range])
            + ", const T value, bool *out) {",
            f"constexpr int32_t N_COLS = {n_cols};",
            "const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();",
            "const int32_t vector_index = mapping.get_vector_index();",
            "const lane_t lane = mapping.get_lane();",
            f"T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * {n_cols}];",
            "bool all_columns_equal = true;",
            "\n".join(
                [
                    f"DecompressorT decompressor_{i} = DecompressorT(column_{i}, vector_index, lane);"
                    for i in col_range
                ]
            ),
            "for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {",
            "\n".join(
                [
                    f"decompressor_{i}.unpack_next_into(registers + {i} * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));"
                    for i in col_range
                ]
            ),
            "#pragma unroll",
            "for (int c{1}; c < N_COLS; ++c) {",
            "#pragma unroll",
            "for (int va{0}; va < UNPACK_N_VALUES; ++va) {",
            "#pragma unroll",
            "for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {",
            "all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];",
            "}",
            "}",
            "}",
            "#pragma unroll",
            "for (int va{0}; va < UNPACK_N_VALUES; ++va) {",
            "#pragma unroll",
            "for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {",
            "all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;",
            "}",
            "}",
            "}",
            "",
            "if (!all_columns_equal) {",
            "*out = true;",
            "}}",
        ]
    )


def main(args):
    n_cols = 10
    generated_str = "\n\n".join(
        [
            FILE_HEADER,
            *[generate_global_function(n) for n in range(1, n_cols + 1)],
            generate_function_selector(n_cols, False),
            generate_function_selector(n_cols, True),
            FILE_FOOTER,
        ]
    )

    args.out_file.write(generated_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "--out-file",
        default=sys.stdout,
        type=argparse.FileType("w"),
        help="Output file",
    )
    parser.add_argument(
        "-o", "--optional", type=int, help="This is an optional argument"
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
    main(args)
