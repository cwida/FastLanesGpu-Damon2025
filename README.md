# FastLanesGpu-Damon2025

This repository contains the source code and benchmark results for the paper "G-ALP: Rethinking Light-weight Encodings for GPUs". The benchmarks can be repeated on a machine with an NVIDIA GPU using the benchmarking script provided in the repo.

## Compilation

Software requirements:

- nvcc 
- nvCOMP  
- clang++-14 (to compile ALP)
- g++-12 (to compile nvCOMP)

To compile all executables, run:

```sh
make all -j 8
```

The full compilation takes a while, `-j 8` adds multiprocessing to compilation. 

To only compile the compressor benchmarks for real data benchmarking:
```sh
make compressors-benchmark
```

## Benchmarks

Requires [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). To test on real datasets, place the binary files of single precision float arrays into the folder `binary-columns`.
NCU requires sudo to read performance counters.

Compile the code, and run all benchmarks:

```sh
make benchmark-all
```

To run only the benchmarks:

```sh
make benchmark-compressors
```
