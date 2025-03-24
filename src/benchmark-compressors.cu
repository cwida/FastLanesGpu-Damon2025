#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

#include "./alp/alp-bindings.hpp"
#include "./datageneration.hpp"
#include "./gpu/alp.cuh"
#include "./gpu/host-alp-utils.cuh"
#include "./gpu/host-utils.cuh"
#include "./kernel.cuh"
#include "./nvcomp-compressors.cuh"
#include "./benchmark-compressors.cuh"

BenchmarkResult benchmark_thrust(const float *input, const size_t value_count,
                                 const float value_to_search_for) {
  GPUArray<bool> d_result(1);

  CudaStopwatch stopwatch = CudaStopwatch();
  float execution_time_ms;

  thrust::device_vector<float> d_vec(input, input + value_count);
  stopwatch.start();
  bool result = thrust::any_of(thrust::device, d_vec.begin(), d_vec.end(),
                               is_equal_to(value_to_search_for));
  execution_time_ms = stopwatch.stop();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  return BenchmarkResult{result, execution_time_ms, 1.0};
}

BenchmarkResult benchmark_alp(const enums::ComparisonType comparison_type,
                              const enums::CompressionType decompressor_enum,
															const float* input,
															const alp::AlpCompressionData<float>* alp_data_ptr,
                              const float value_to_search_for) {
  constant_memory::load_alp_constants<float>();

  GPUArray<bool> d_query_result(1);
  GPUArray<float> d_decompression_result(alp_data_ptr->size);

	constexpr int32_t N_VECS_CONCURRENTLY = 4;
  const ThreadblockMapping<float> mapping(
      utils::get_n_vecs_from_size(alp_data_ptr->size), N_VECS_CONCURRENTLY);
  CudaStopwatch stopwatch = CudaStopwatch();
  float execution_time_ms;
  double compression_ratio;

  switch (decompressor_enum) {
  case enums::ALP: {
    AlpColumn<float> column = transfer::copy_alp_column_to_gpu(alp_data_ptr);
    stopwatch.start();
    if (comparison_type == enums::ComparisonType::DECOMPRESSION) {
      decompress_column<float, ALP4VDecompressor<float>, AlpColumn<float>,
                        N_VECS_CONCURRENTLY>
          <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
              column, value_to_search_for, d_decompression_result.get());
    } else {
      scan_column<float, ALP4VDecompressor<float>, AlpColumn<float>,
                  N_VECS_CONCURRENTLY>
          <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
              column, value_to_search_for, d_query_result.get());
    }
    execution_time_ms = stopwatch.stop();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    transfer::destroy_alp_column(column);
    compression_ratio = alp_data_ptr->get_alp_compression_ratio();
  } break;

  case enums::GALP: {
    AlpExtendedColumn<float> column =
        transfer::copy_alp_extended_column_to_gpu(alp_data_ptr);
    stopwatch.start();
    if (comparison_type == enums::ComparisonType::DECOMPRESSION) {
      decompress_column<float, GALPPrefetchBranchy4VDecompressor<float>,
                        AlpExtendedColumn<float>, N_VECS_CONCURRENTLY>
          <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
              column, value_to_search_for, d_decompression_result.get());
    } else {
      scan_column<float, GALPPrefetchBranchy4VDecompressor<float>,
                  AlpExtendedColumn<float>, N_VECS_CONCURRENTLY>
          <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
              column, value_to_search_for, d_query_result.get());
    }
    execution_time_ms = stopwatch.stop();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    transfer::destroy_alp_column(column);
    compression_ratio = alp_data_ptr->get_galp_compression_ratio();
  } break;
  default:
    throw std::invalid_argument("Could not parse decompressor enum for alp\n");
  }

  bool kernel_successful = false;
  if (comparison_type == enums::ComparisonType::DECOMPRESSION_QUERY) {
    d_query_result.copy_to_host(&kernel_successful);
  } else {
    GPUArray<float> d_input(alp_data_ptr->size, input);

    kernel_successful = check_if_device_buffers_are_equal<float>( d_decompression_result.get(), d_input.get(), alp_data_ptr->size);
  }

  return BenchmarkResult{kernel_successful, execution_time_ms,
                         compression_ratio};
}

BenchmarkResult benchmark_alp(const enums::ComparisonType comparison_type,
                              const enums::CompressionType decompressor_enum,
                              const float *input, const size_t value_count,
                              const float value_to_search_for) {
  alp::AlpCompressionData<float> *alp_data_ptr =
      new alp::AlpCompressionData<float>(value_count);
  alp::int_encode(input, value_count, alp_data_ptr);

	auto result = benchmark_alp(comparison_type, decompressor_enum, input, alp_data_ptr, value_to_search_for);

	delete alp_data_ptr;

	return result;
}

BenchmarkResult benchmark_hwc(const enums::ComparisonType comparison_type,
                              const enums::CompressionType compression_type,
                              const float *input, const size_t value_count,
                              const float value_to_search_for) {
  size_t size_in_bytes = value_count * sizeof(float);
  GPUArray<uint8_t> d_input_buffer(size_in_bytes,
                                   reinterpret_cast<const uint8_t *>(input));
  hwc::Compressor compressor(compression_type);
  hwc::CompressedBuffer d_compressed_buffer =
      compressor.compress(d_input_buffer.get(), size_in_bytes);

  double compression_ratio = d_compressed_buffer.get_compression_ratio();
  GPUArray<bool> d_query_result(1);

	constexpr int32_t N_VECS_CONCURRENTLY = 4;
  const ThreadblockMapping<float> mapping(
      utils::get_n_vecs_from_size(value_count), N_VECS_CONCURRENTLY);

  CudaStopwatch stopwatch = CudaStopwatch();
  stopwatch.start();
  uint8_t *d_output_buffer = compressor.decompress(d_compressed_buffer);

  if (comparison_type == enums::ComparisonType::DECOMPRESSION_QUERY) {
    hwc::DecompressedColumn<float> column{
        reinterpret_cast<float *>(d_output_buffer), value_count};
    scan_column<float, hwc::Loader<float, N_VECS_CONCURRENTLY, 1>,
                hwc::DecompressedColumn<float>, N_VECS_CONCURRENTLY>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_query_result.get());
  }

  float execution_time_ms = stopwatch.stop();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  bool kernel_successful = false;
  if (comparison_type == enums::ComparisonType::DECOMPRESSION_QUERY) {
    d_query_result.copy_to_host(&kernel_successful);
  } else {
    kernel_successful = check_if_device_buffers_are_equal<uint8_t>(
        d_output_buffer, d_input_buffer.get(), size_in_bytes);
  }

  d_compressed_buffer.free();
  compressor.free();

  return BenchmarkResult{kernel_successful, execution_time_ms,
                         compression_ratio};
}
