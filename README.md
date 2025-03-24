# FastLanesGpu-Damon2025

This repository contains the source code and benchmark results for the paper "G-ALP: Rethinking Light-weight Encodings for GPUs". The benchmarks can be repeated on a machine with an NVIDIA GPU using the benchmarking script provided in the repo.

## Compilation

Software requirements:

- nvcc 
- nvCOMP  
- g++-12

To compile the code, run:

```sh
make all
```

## Benchmarks

Requires [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). To test on real datasets, place the binary files of single precision float arrays into the folder `binary-columns`.

Compile the code, and run:

```sh
sudo ./data-collection/run-benchmarks.py all output/
```
