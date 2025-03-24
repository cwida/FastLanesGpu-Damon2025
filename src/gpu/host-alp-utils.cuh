#include <cstddef>
#include <cstdint>
#include <tuple>

#include "../alp/alp-bindings.hpp"
#include "../common/consts.hpp"
#include "../common/utils.hpp"
#include "alp.cuh"
#include "host-utils.cuh"
#include "src/alp/config.hpp"

#ifndef HOST_ALP_UTILS_CUH
#define HOST_ALP_UTILS_CUH

namespace transfer {

template <typename T>
AlpColumn<T> copy_alp_column_to_gpu(const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);

  const size_t branchless_extra_access_buffer =
      sizeof(T) * utils::get_n_lanes<T>() * 4;
  GPUArray<UINT_T> d_ffor_array(count, branchless_extra_access_buffer,
                                data->ffor.array);

  GPUArray<UINT_T> d_ffor_bases(n_vecs, data->ffor.bases);
  GPUArray<uint8_t> d_bit_widths(n_vecs, data->ffor.bit_widths);
  GPUArray<uint8_t> d_exponents(n_vecs, data->exponents);
  GPUArray<uint8_t> d_factors(n_vecs, data->factors);

  GPUArray<T> d_exceptions(count, data->exceptions.exceptions);
  GPUArray<uint16_t> d_exception_positions(count, data->exceptions.positions);
  GPUArray<uint16_t> d_exception_counts(n_vecs, data->exceptions.counts);

  return AlpColumn<T>{d_ffor_array.release(),
                      d_ffor_bases.release(),
                      d_bit_widths.release(),
                      d_exponents.release(),
                      d_factors.release(),
                      d_exceptions.release(),
                      d_exception_positions.release(),
                      d_exception_counts.release()};
}

template <typename T> void destroy_alp_column(AlpColumn<T> &column) {
  free_device_pointer(column.ffor_array);
  free_device_pointer(column.ffor_bases);
  free_device_pointer(column.bit_widths);
  free_device_pointer(column.exponents);
  free_device_pointer(column.factors);
  free_device_pointer(column.exceptions);
  free_device_pointer(column.positions);
  free_device_pointer(column.counts);
}

template <typename T>
std::tuple<T *, uint16_t *, uint16_t *>
convert_exceptions_to_lane_divided_format(
    const alp::AlpExceptions<T> *original_exceptions, const size_t count,
    const size_t n_vecs) {
  constexpr auto N_LANES = utils::get_n_lanes<T>();
  constexpr auto VALUES_PER_LANE = utils::get_values_per_lane<T>();

  // INFO Yes this allocation is far too big, should not use count.
  T *out_exceptions = reinterpret_cast<T *>(malloc(sizeof(T) * count));
  uint16_t *out_positions =
      reinterpret_cast<uint16_t *>(malloc(sizeof(uint16_t) * count));
  uint16_t *out_offsets_counts =
      reinterpret_cast<uint16_t *>(malloc(sizeof(uint16_t) * n_vecs * N_LANES));

  T vec_exceptions[consts::VALUES_PER_VECTOR];
  T vec_exceptions_positions[consts::VALUES_PER_VECTOR];
  uint16_t lane_counts[N_LANES];

  for (size_t vec_index{0}; vec_index < n_vecs; ++vec_index) {
    uint32_t vec_exception_count = original_exceptions->counts[vec_index];

    // Reset counts
    for (size_t j{0}; j < N_LANES; ++j) {
      lane_counts[j] = 0;
    }

    // Split all exceptions into lanes
    for (size_t exception_index{0}; exception_index < vec_exception_count;
         ++exception_index) {
      T exception = original_exceptions
                        ->exceptions[vec_index * consts::VALUES_PER_VECTOR +
                                     exception_index];
      uint16_t position =
          original_exceptions->positions[vec_index * consts::VALUES_PER_VECTOR +
                                         exception_index];

      uint32_t lane = position % N_LANES;
      uint32_t lane_exception_count = lane_counts[lane];
      ++lane_counts[lane];
      vec_exceptions[lane * VALUES_PER_LANE + lane_exception_count] = exception;
      vec_exceptions_positions[lane * VALUES_PER_LANE + lane_exception_count] =
          position;
    }

    // Merge and concatenate all exceptions per lane into single contiguous
    // array
    uint32_t vec_exceptions_counter = 0;
    for (size_t lane{0}; lane < N_LANES; ++lane) {
      uint32_t exc_in_lane_count = lane_counts[lane];
      for (size_t exc_in_lane{0}; exc_in_lane < exc_in_lane_count;
           ++exc_in_lane) {

        out_exceptions[vec_index * consts::VALUES_PER_VECTOR +
                       vec_exceptions_counter] =
            vec_exceptions[lane * VALUES_PER_LANE + exc_in_lane];
        out_positions[vec_index * consts::VALUES_PER_VECTOR +
                      vec_exceptions_counter] =
            vec_exceptions_positions[lane * VALUES_PER_LANE + exc_in_lane];
        ++vec_exceptions_counter;
      }

      out_offsets_counts[vec_index * N_LANES + lane] =
          (exc_in_lane_count << 10) |
          (vec_exceptions_counter - exc_in_lane_count);
    }
  }

  return std::make_tuple(out_exceptions, out_positions, out_offsets_counts);
}

template <typename T>
AlpExtendedColumn<T>
copy_alp_extended_column_to_gpu(const alp::AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const auto count = data->size;
  const auto n_vecs = utils::get_n_vecs_from_size(count);
  constexpr auto N_LANES = utils::get_n_lanes<T>();

  const size_t branchless_extra_access_buffer =
      sizeof(T) * utils::get_n_lanes<T>() * 4;
  GPUArray<UINT_T> d_ffor_array(count, branchless_extra_access_buffer,
                                data->ffor.array);

  GPUArray<UINT_T> d_ffor_bases(n_vecs, data->ffor.bases);
  GPUArray<uint8_t> d_bit_widths(n_vecs, data->ffor.bit_widths);
  GPUArray<uint8_t> d_exponents(n_vecs, data->exponents);
  GPUArray<uint8_t> d_factors(n_vecs, data->factors);

  auto [exceptions, positions, offsets_counts] =
      convert_exceptions_to_lane_divided_format(&data->exceptions, count,
                                                n_vecs);

  GPUArray<T> d_exceptions(count, exceptions);
  GPUArray<uint16_t> d_exception_positions(count, positions);
  GPUArray<uint16_t> d_exception_offsets_counts(n_vecs * N_LANES,
                                                offsets_counts);

  free(exceptions);
  free(positions);
  free(offsets_counts);

  return AlpExtendedColumn<T>{d_ffor_array.release(),
                              d_ffor_bases.release(),
                              d_bit_widths.release(),
                              d_exponents.release(),
                              d_factors.release(),
                              d_exceptions.release(),
                              d_exception_positions.release(),
                              d_exception_offsets_counts.release()};
}

template <typename T> void destroy_alp_column(AlpExtendedColumn<T> &column) {
  free_device_pointer(column.ffor_array);
  free_device_pointer(column.ffor_bases);
  free_device_pointer(column.bit_widths);
  free_device_pointer(column.exponents);
  free_device_pointer(column.factors);
  free_device_pointer(column.exceptions);
  free_device_pointer(column.positions);
  free_device_pointer(column.offsets_counts);
}
} // namespace transfer

#endif // HOST_ALP_UTILS_CUH
