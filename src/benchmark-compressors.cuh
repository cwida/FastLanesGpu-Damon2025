#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
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

#ifndef BENCHMARK_COMPRESSORS_CUH
#define BENCHMARK_COMPRESSORS_CUH

template <typename T>
bool check_if_device_buffers_are_equal(const T *a, const T *b,
                                       const size_t n_values) {
  // Convert to uint8_t as we don't want to compare floats (-nan == -nan =>
  // false)
  thrust::device_ptr<uint8_t> d_a(
      reinterpret_cast<uint8_t *>(const_cast<T *>(a)));
  thrust::device_ptr<uint8_t> d_b(
      reinterpret_cast<uint8_t *>(const_cast<T *>(b)));

  return thrust::equal(d_a, d_a + n_values, d_b);
}

struct BenchmarkResult {
  bool found_value;
  float execution_time_ms;
  double compression_ratio;

  void log_result(const enums::ComparisonType comparison_type,
                  const enums::CompressionType compression_type,
									const size_t n_bytes,
                  const std::string data_name) const {
    printf("%s,%s,%s,%d,%lu,%f,%f\n",
           enums::get_name_for_comparison_type(comparison_type).c_str(),
           enums::get_name_for_compression_type(compression_type).c_str(),
           data_name.c_str(), found_value, n_bytes, execution_time_ms,
           compression_ratio);
  }
};

template<typename T>
struct is_equal_to {
  T value;
  is_equal_to(T value) : value(value) {}
  __host__ __device__ bool operator()(T x) { return x == value; }
};

BenchmarkResult benchmark_thrust(const float *input, const size_t value_count,
                                 const float value_to_search_for);

BenchmarkResult benchmark_alp(const enums::ComparisonType comparison_type,
                              const enums::CompressionType decompressor_enum,
                              const float *input,
                              const alp::AlpCompressionData<float> *data,
                              const float value_to_search_for);

BenchmarkResult benchmark_alp(const enums::ComparisonType comparison_type,
                              const enums::CompressionType decompressor_enum,
                              const float *input, const size_t value_count,
                              const float value_to_search_for);

BenchmarkResult benchmark_hwc(const enums::ComparisonType comparison_type,
                              const enums::CompressionType compression_type,
                              const float *input, const size_t value_count,
                              const float value_to_search_for);

#endif // BENCHMARK_COMPRESSORS_CUH
