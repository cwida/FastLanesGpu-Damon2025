#!/usr/bin/python3

from io import FileIO
import os
import sys

import argparse
import logging
import numpy as np

def read_csv(path: str, dtype) -> np.ndarray:
    with open(path, "r") as file:
        return np.array([float(x) for x in file.readlines()], dtype=dtype)

def read_bin(path: str, dtype) -> np.ndarray:
    with open(path, "r") as file:
        return np.fromfile(file, dtype=dtype)

def main(args):
    open_as_csv = args.file_type == "csv" 
    dtype = np.float64 if args.dtype == "double" else np.float32

    floats = read_csv(args.in_file, dtype) if open_as_csv else read_bin(args.in_file, dtype)

    if args.out is None or args.inspect:
        size = floats.size
        for x in range(5):
            print(x, floats[x])
        print("...")
        for x in reversed(range(5)):
            print(size - 1 - x, floats[- x - 1])
        print("Number of floats: ", size)
    else:
        floats.tofile(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument("file_type", type=str, choices=["csv", "bin"], help="Specify type of file")
    parser.add_argument("in_file", type=str, help="Input file")
    parser.add_argument("-o", "--out", type=str, help="Output file")
    parser.add_argument("-i", "--inspect", action=argparse.BooleanOptionalAction, default=False, help="Show first 10 values and value count")
    parser.add_argument("-dt", "--dtype", type=str, choices=["double", "float"], default="double")

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
