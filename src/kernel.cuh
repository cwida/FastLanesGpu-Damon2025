#include "./nvcomp-compressors.cuh"
#include "gpu/alp.cuh"
#include "gpu/device-utils.cuh"

#ifndef KERNEL_CUH
#define KERNEL_CUH

// Using alias for readability
// This template metaprogramming to compose the ALPDecompressor is necessary
// for main repo to try all configs, not intended for actual API usage

template <typename T> struct BPColumn {
  size_t count;
  vbw_t vbw;
  T *buffer;
};

template <typename T, typename DecompressorT, typename FunctorT,
          typename ColumnT>
struct BPUnpacker {
  DecompressorT decompressor;
  __device__ __forceinline__ BPUnpacker(const ColumnT column,
                                        const vi_t vector_index,
                                        const lane_t lane)
      : decompressor(DecompressorT(column.buffer +
                                       consts::VALUES_PER_VECTOR * vector_index,
                                   lane, column.vbw, FunctorT())) {}

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) {
    decompressor.unpack_next_into(out);
  }
};

template <typename T>
using BP1VDecompressor =
    BPUnpacker<T, BitUnpackerStateful<T, 1, 1, BPFunctor<T>, CacheLoader<T, 1>>,
               BPFunctor<T>, BPColumn<T>>;

template <typename T>
using BP4VDecompressor =
    BPUnpacker<T, BitUnpackerStateful<T, 4, 1, BPFunctor<T>, CacheLoader<T, 4>>,
               BPFunctor<T>, BPColumn<T>>;

template <typename T>
using ALP1VDecompressor =
    AlpUnpacker<T, 1, 1,
                BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>,
                                              consts::VALUES_PER_VECTOR>,
                StatefulALPExceptionPatcher<T, 1, 1>, AlpColumn<T>>;

template <typename T>
using ALP4VDecompressor =
    AlpUnpacker<T, 4, 1,
                BitUnpackerStatefulBranchless<T, 4, 1, ALPFunctor<T, 4>,
                                              consts::VALUES_PER_VECTOR>,
                StatefulALPExceptionPatcher<T, 4, 1>, AlpColumn<T>>;

template <typename T>
using GALPBranchless1VDecompressor =
    AlpUnpacker<T, 1, 1,
                BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>,
                                              consts::VALUES_PER_VECTOR>,
                NaiveBranchlessALPExceptionPatcher<T, 1, 1>,
                AlpExtendedColumn<T>>;

template <typename T>
using GALPBranchy1VDecompressor =
    AlpUnpacker<T, 1, 1,
                BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>,
                                              consts::VALUES_PER_VECTOR>,
                NaiveALPExceptionPatcher<T, 1, 1>, AlpExtendedColumn<T>>;

template <typename T>
using GALPBranchy4VDecompressor =
    AlpUnpacker<T, 4, 1,
                BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>,
                                    RegisterBranchlessLoader<T, 4, 2>,
                                    consts::VALUES_PER_VECTOR>,
                NaiveALPExceptionPatcher<T, 4, 1>, AlpExtendedColumn<T>>;

template <typename T>
using GALPPrefetchBranchy1VDecompressor =
    AlpUnpacker<T, 1, 1,
                BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>,
                                              consts::VALUES_PER_VECTOR>,
                PrefetchAllALPExceptionPatcher<T, 1, 1>, AlpExtendedColumn<T>>;

template <typename T>
using GALPPrefetchBranchy4VDecompressor = 
    AlpUnpacker<T, 4, 1,
                BitUnpackerStateful<T, 4, 1, ALPFunctor<T, 4>,
                                    RegisterBranchlessLoader<T, 4, 2>,
                                    consts::VALUES_PER_VECTOR>,
                PrefetchAllALPExceptionPatcher<T, 4, 1>, AlpExtendedColumn<T>>;

template <typename T, typename DecompressorT, typename ColumnT,
          unsigned N_VECS_CONCURRENTLY>
__global__ void decompress_column(const ColumnT column, const T value, T *out) {
  // Figure out which vectors this warp should unpack,
  // and which lane this thread should unpack
  const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
  const int32_t vector_index = mapping.get_vector_index();
  const lane_t lane = mapping.get_lane();

  // Initialize some storage vars for intermediate results
  T registers[N_VECS_CONCURRENTLY];
  out += vector_index * consts::VALUES_PER_VECTOR;

  // Create decompressor
  DecompressorT decompressor = DecompressorT(column, vector_index, lane);

  // Execute query
  for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; ++i) {
    // Unpack 1 value from each of the 4 vectors
    decompressor.unpack_next_into(registers);

    write_registers_to_global<T, N_VECS_CONCURRENTLY, 1, mapping.N_LANES>(
        lane, i, registers, out);
  }
}

// Checks whether a certain value is in the column. If the value is in the
// column, it writes 1 to out. Otherwise nothing happens, as out is initialized
// with 0
template <typename T, typename DecompressorT, typename ColumnT,
          unsigned N_VECS_CONCURRENTLY>
__global__ void scan_column(const ColumnT column, const T value, bool *out) {
  // Figure out which vectors this warp should unpack,
  // and which lane this thread should unpack
  const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
  const int32_t vector_index = mapping.get_vector_index();
  const lane_t lane = mapping.get_lane();

  // Initialize some storage vars for intermediate results
  T registers[N_VECS_CONCURRENTLY];
  bool value_not_yet_found = true;

  // Create decompressor
  DecompressorT decompressor = DecompressorT(column, vector_index, lane);

  // Execute query
  for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; ++i) {
    // Unpack 1 value from each of the 4 vectors
    decompressor.unpack_next_into(registers);

    // Check if any of those values is the value we are looking for
#pragma unroll
    for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
      value_not_yet_found &= registers[v] != value;
    }
  }

  // If we found the value, write it to out
  if (!value_not_yet_found) {
    *out = true;
  }
}

#endif // KERNEL_CUH
