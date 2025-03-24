#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

#include "./common/consts.hpp"
#include "./common/utils.hpp"
#include "./gpu/device-types.cuh"
#include "./gpu/host-utils.cuh"

#ifndef NVCOMP_COMPRESSORS_H
#define NVCOMP_COMPRESSORS_H

#include "nvcomp.h"
#include "nvcomp.hpp"
#include "nvcomp/deflate.h"
#include "nvcomp/deflate.hpp"
#include "nvcomp/gdeflate.h"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManager.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#include "nvcomp/zstd.h"
#include "nvcomp/zstd.hpp"

namespace hwc {
using nvcompCompressionManager = nvcomp::PimplManager;

nvcompCompressionManager *
get_compressor_manager(const enums::CompressionType compression_type,
                       const nvcompType_t data_type = NVCOMP_TYPE_CHAR,
                       const size_t chunk_size = 1 << 16);

struct CompressedBuffer {
  nvcomp::CompressionConfig compression_config;
  uint8_t *compressed_buffer;
  size_t decompressed_size;
  size_t compressed_size;

  CompressedBuffer(nvcomp::CompressionConfig compression_config)
      : compression_config(compression_config) {

    CUDA_SAFE_CALL(cudaMalloc(&compressed_buffer,
                              compression_config.max_compressed_buffer_size));
  }

  double get_compression_ratio() {
    return static_cast<double>(decompressed_size) /
           static_cast<double>(compressed_size);
  }

  double get_allocation_compression_ratio() {
    return static_cast<double>(compression_config.max_compressed_buffer_size) /
           static_cast<double>(compression_config.uncompressed_buffer_size);
  }

  // Manual resource freeing
  void free() { CUDA_SAFE_CALL(cudaFree(compressed_buffer)); }
};

struct Compressor {
  nvcompCompressionManager *manager;

  Compressor(const enums::CompressionType compression_type) {
    manager = get_compressor_manager(compression_type);
  }

  CompressedBuffer compress(uint8_t *input_buffer,
                            const size_t input_buffer_len) {
    CompressedBuffer compressed_buffer(
        manager->configure_compression(input_buffer_len));
    manager->compress(input_buffer, compressed_buffer.compressed_buffer,
                      compressed_buffer.compression_config);
    compressed_buffer.decompressed_size = input_buffer_len;
    compressed_buffer.compressed_size = manager->get_compressed_output_size(
        compressed_buffer.compressed_buffer);
    return compressed_buffer;
  }

  uint8_t *decompress(const CompressedBuffer compressed_buffer) {
    nvcomp::DecompressionConfig decomp_config =
        manager->configure_decompression(compressed_buffer.compressed_buffer);
    uint8_t *decompressed_buffer;
    CUDA_SAFE_CALL(
        cudaMalloc(&decompressed_buffer, decomp_config.decomp_data_size));

    manager->decompress(decompressed_buffer,
                        compressed_buffer.compressed_buffer, decomp_config);

    return decompressed_buffer;
  }

  // Manual resource freeing
  void free() { delete manager; }
};

__global__ void check_buffer_equality(const uint8_t *buffer_a,
                                      const uint8_t *buffer_b,
                                      const size_t length, bool *out);

bool compare_d_buffers(const uint8_t *buffer_a, const uint8_t *buffer_b,
                       const size_t length);

template <typename T> struct DecompressedColumn {
  T *buffer;
  size_t size;
};

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
struct Loader {
  T *in;

  __device__ Loader(const DecompressedColumn<T> column, const vi_t vector_index,
                    const lane_t lane)
      : in(column.buffer + lane + consts::VALUES_PER_VECTOR * vector_index) {};
  __device__ __forceinline__ void unpack_next_into(T *__restrict out) {
    constexpr int32_t N_LANES = utils::get_n_lanes<T>();

#pragma unroll
    for (int v = 0; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
      for (int j = 0; j < UNPACK_N_VALUES; ++j) {
        out[v * UNPACK_N_VALUES + j] =
            in[v * consts::VALUES_PER_VECTOR + j * N_LANES];
      }
    }

    in += UNPACK_N_VALUES * N_LANES;
  }
};

} // namespace hwc

#endif // NVCOMP_COMPRESSORS_H
