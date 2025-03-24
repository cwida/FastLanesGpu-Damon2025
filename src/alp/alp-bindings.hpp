#include <cstddef>
#include <cstdint>
#include <exception>
#include <type_traits>

#include "../common/utils.hpp"
#include "constants.hpp"
// WARNING The original ALP repo contains code that triggers warnings if all
// warnings are turned off. To make sure these warnings do not show up when the
// alp directory itself is not recompiled, I added this pragma to show it as a
// system header. So be carefult, warnings from the alp/* files do not show up
// when compiling
#pragma clang system_header

#ifndef ALP_BINDINGS_HPP
#define ALP_BINDINGS_HPP

#include "config.hpp"

namespace alp {

class EncodingException : public std::exception {
public:
  using std::exception::what;
  const char *what() { return "Could not encode data with desired encoding."; }
};

template <typename T> size_t get_bytes_overhead_size_per_alp_vector() {
  return sizeof(uint8_t) + // bit_width
         sizeof(uint8_t) + // factor-idx
         sizeof(uint8_t) + // exponent-idx
         (sizeof(T) * 1);  // ffor base
  // + 32; // Overhead of offset to find the bitpacked array
};

template <typename T> size_t get_bytes_overhead_size_per_galp_vector() {
  return get_bytes_overhead_size_per_alp_vector<T>() +
         sizeof(uint16_t) * utils::get_n_lanes<T>(); // pos + offset per lane
                                                     // data parallel_format;
  // + 32; // Overhead of offset to find the pos + offset array
};

template <typename T>
size_t get_bytes_vector_compressed_size_without_overhead(
    const uint8_t bit_width, const uint16_t exceptions_count) {
  constexpr size_t line_size = utils::get_n_lanes<T>() * sizeof(T);
  constexpr size_t exception_value_size = sizeof(T);
  constexpr size_t exception_position_size = sizeof(uint16_t);

  return bit_width * line_size +
         exceptions_count * (exception_value_size + exception_position_size);
}


template <typename T> struct AlpFFORVecHeader {
  T *array;
  T *base;
  uint8_t *bit_width;
};

template <typename T> struct AlpFFORArray {
  size_t count;

  T *array;
  T *bases;
  uint8_t *bit_widths;

  AlpFFORArray<T>(const size_t count) {
    const size_t n_vecs = utils::get_n_vecs_from_size(count);

    // TODO: This should not allocate the maximum amount of space
    array = new T[count];
    bases = new T[n_vecs];
    bit_widths = new uint8_t[n_vecs];
  }

  AlpFFORVecHeader<T> get_ffor_header_for_vec(const int64_t vec_index) const {
    return AlpFFORVecHeader<T>{
        array + vec_index * 1024,
        bases + vec_index,
        bit_widths + vec_index,
    };
  }

  ~AlpFFORArray<T>() {
    delete[] array;
    delete[] bases;
    delete[] bit_widths;
  }
};

template <typename T> struct AlpVecExceptions {
  T *exceptions;
  uint16_t *positions;
  uint16_t *count;
};

template <typename T> struct AlpExceptions {
  T *exceptions;
  uint16_t *counts;
  uint16_t *positions;

  AlpExceptions<T>(const size_t size) {
    const size_t n_vecs = utils::get_n_vecs_from_size(size);
    // TODO: This should not allocate the maximum number of exceptions
    exceptions = new T[size];
    positions = new uint16_t[size];
    counts = new uint16_t[n_vecs];
  }

  AlpVecExceptions<T> get_exceptions_for_vec(const int64_t vec_index) const {
    return AlpVecExceptions<T>{
        exceptions + vec_index * 1024,
        positions + vec_index * 1024,
        counts + vec_index,
    };
  }

  ~AlpExceptions<T>() {
    delete[] exceptions;
    delete[] positions;
    delete[] counts;
  }
};

template <typename T> struct AlpCompressionData {
  using UINT_T = typename utils::same_width_uint<T>::type;

  size_t rowgroup_offset = 0;
  size_t size; // n_compressed_values

  size_t compressed_alp_bytes_size;
  size_t compressed_galp_bytes_size;

  AlpFFORArray<UINT_T> ffor;
  uint8_t *exponents;
  uint8_t *factors;

  AlpExceptions<T> exceptions;

  AlpCompressionData<T>(const size_t size_a)
      : size(size_a), ffor(size_a), exceptions(AlpExceptions<T>(size_a)) {
    const size_t n_vecs = utils::get_n_vecs_from_size(size);
    exponents = new uint8_t[n_vecs];
    factors = new uint8_t[n_vecs];
  }

  ~AlpCompressionData<T>() {
    delete[] exponents;
    delete[] factors;
  }

  double get_alp_compression_ratio() const {
    return static_cast<double>(size * sizeof(T)) /
           static_cast<double>(compressed_alp_bytes_size);
  }

  double get_galp_compression_ratio() const {
    return static_cast<double>(size * sizeof(T)) /
           static_cast<double>(compressed_galp_bytes_size);
  }
};

// Test if data can be decoded in specified type
template <typename T>
bool is_encoding_possible(const T *input_array, const size_t count,
                          Scheme scheme);

// Default ALP encoding
template <typename T>
void int_encode(const T *input_array, const size_t count,
                AlpCompressionData<T> *data);

// Default ALP decoding
template <typename T>
void int_decode(T *output_array, const AlpCompressionData<T> *data);

} // namespace alp

#endif // ALP_BINDINGS_HPP
