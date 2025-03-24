#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <time.h>
#include <type_traits>
#include <utility>
#include <vector>

#include "./alp/alp-bindings.hpp"
#include "./alp/constants.hpp"
#include "./common/consts.hpp"
#include "./common/utils.hpp"

#ifndef DATAGENERATION_HPP
#define DATAGENERATION_HPP

namespace data {

template <typename T>
std::function<T()> get_random_number_generator(const T min, const T max) {
  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  std::uniform_int_distribution<T> uniform_dist(min, max);

  return std::bind(uniform_dist, random_engine);
}

template <typename T>
void fill_array_with_constant(T *array, const size_t count, const T value) {
  for (size_t i{0}; i < count; ++i) {
    array[i] = value;
  }
}

template <typename T>
void fill_array_with_random_bytes(T *array, const size_t count,
                                  const unsigned repeat = 1) {
  // WARNING Does not check if count % REPEAT == 0
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *out = reinterpret_cast<UINT_T *>(array);
  auto generator = get_random_number_generator<UINT_T>(
      std::numeric_limits<UINT_T>::min(), std::numeric_limits<UINT_T>::max());

  for (size_t i{0}; i < count; i += repeat) {
    UINT_T value = generator();

    for (size_t r{0}; r < repeat; ++r) {
      out[i + r] = value;
    }
  }
}

template <typename T>
void fill_array_with_sequence(T *array, const size_t count,
                              const unsigned repeat, T start) {
  // WARNING Does not check if count % REPEAT == 0
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *out = reinterpret_cast<UINT_T *>(array);

  for (size_t i{0}; i < count; i += repeat) {
    for (size_t r{0}; r < repeat; ++r) {
      out[i + r] = start;
    }

    ++start;
  }
}

template <typename T>
void fill_array_with_random_data(T *array, const size_t count,
                                 const unsigned repeat = 1,
                                 const T min = std::numeric_limits<T>::min(),
                                 const T max = std::numeric_limits<T>::max()) {
  // WARNING Does not check if count % REPEAT == 0
  using UINT_T = typename utils::same_width_uint<T>::type;
  UINT_T *out = reinterpret_cast<UINT_T *>(array);
  auto generator = get_random_number_generator<UINT_T>(min, max);

  for (size_t i{0}; i < count; i += repeat) {
    UINT_T value = generator();

    for (size_t r{0}; r < repeat; ++r) {
      out[i + r] = value;
    }
  }
}

template <typename T> std::vector<T> generate_indices() {
  std::vector<T> indices(consts::VALUES_PER_VECTOR);
  for (T i{0}; i < consts::VALUES_PER_VECTOR; ++i) {
    indices[i] = i;
  }
  return indices;
}

template <typename T>
alp::AlpCompressionData<T> *generate_alp_datastructure(
    const size_t count, const int32_t exceptions_per_vec = -1,
    const int32_t value_bit_width = -1, const unsigned repeat = 1) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  static_assert(std::is_floating_point<T>::value,
                "T should be a floating point type.");
  auto data = new alp::AlpCompressionData<T>(count);
  const size_t n_vecs = utils::get_n_vecs_from_size(count);

  const size_t vec_overhead =
      n_vecs * alp::get_bytes_vector_compressed_size_without_overhead<T>(
                   value_bit_width, exceptions_per_vec);
  data->compressed_alp_bytes_size =
      vec_overhead + alp::get_bytes_overhead_size_per_alp_vector<T>() * n_vecs;
  data->compressed_galp_bytes_size =
      vec_overhead + alp::get_bytes_overhead_size_per_galp_vector<T>() * n_vecs;

  int32_t frac_arr_size = sizeof(T) == 4 ? 11 : 21;
  int32_t fact_arr_size = sizeof(T) == 4 ? 10 : 19;
  int32_t max_bit_width = sizeof(T) == 4 ? 32 : 64;

  // Note we halve the frac and fact because otherwise you
  // will have integer overflow in the decoding for some combinations
  fill_array_with_random_data<uint8_t>(data->exponents, n_vecs, 1, 0,
                                       static_cast<uint8_t>(frac_arr_size / 2));
  fill_array_with_random_data<uint8_t>(data->factors, n_vecs, 1, 0,
                                       static_cast<uint8_t>(fact_arr_size / 2));

  fill_array_with_random_bytes<UINT_T>(data->ffor.array, count, 1);
  fill_array_with_random_data<UINT_T>(data->ffor.bases, n_vecs, 1, 2, 20);

  // Note we halve the bitwidth because otherwise you will have integer overflow
  // and a high bitwidth is not realistic anyway for alp encoding.
  // It can be parametrized though via the function args if needed.
  if (value_bit_width == -1) {
    fill_array_with_random_data<uint8_t>(
        data->ffor.bit_widths, n_vecs, repeat, 0,
        static_cast<uint8_t>(max_bit_width / 2));
  } else {
    fill_array_with_constant<uint8_t>(data->ffor.bit_widths, n_vecs,
                                      static_cast<uint8_t>(value_bit_width));
  }

  if (exceptions_per_vec == -1) {
    // fill_array_with_random_data<uint16_t>(data->exceptions.counts, n_vecs, 0,
    // 20);
    fill_array_with_constant<uint16_t>(data->exceptions.counts, n_vecs, 20);
  } else {
    fill_array_with_constant<uint16_t>(
        data->exceptions.counts, n_vecs,
        static_cast<uint16_t>(exceptions_per_vec));
  }

  fill_array_with_constant(data->exceptions.exceptions, count,
                           -std::numeric_limits<T>::infinity());

  // Create a vector with all indices
  auto indices = generate_indices<uint16_t>();

  // Shuffle them and copy to the positions array
  uint16_t *positions = data->exceptions.positions;
  std::random_device random_device;
  auto rng = std::default_random_engine(random_device());
  for (size_t i{0}; i < n_vecs; ++i) {
    std::shuffle(std::begin(indices), std::end(indices), rng);
    // We copy the entire shuffled indices to the array, as we
    // can then change the exception count without needing to
    // regenerate more exceptions
    // See modify_alp_exception_count
    std::memcpy(positions, indices.data(),
                sizeof(uint16_t) * consts::VALUES_PER_VECTOR);
    std::sort(positions, positions + data->exceptions.counts[i]);
    positions += consts::VALUES_PER_VECTOR;
  }

  return data;
}

template <typename T>
alp::AlpCompressionData<T> *
modify_alp_exception_count(const size_t count, const int32_t exceptions_per_vec,
                           alp::AlpCompressionData<T> *data) {
  const size_t n_vecs = utils::get_n_vecs_from_size(count);

  uint16_t *positions = data->exceptions.positions;
  std::random_device random_device;
  auto rng = std::default_random_engine(random_device());
  for (size_t i{0}; i < n_vecs; ++i) {
    std::shuffle(positions, positions + data->exceptions.counts[i], rng);
    std::sort(positions, positions + exceptions_per_vec);
    positions += consts::VALUES_PER_VECTOR;
  }

  fill_array_with_constant<uint16_t>(data->exceptions.counts, n_vecs,
                                     static_cast<uint16_t>(exceptions_per_vec));

  return data;
}

template <typename T>
alp::AlpCompressionData<T> *
modify_alp_value_bit_width(const size_t count, const int32_t value_bit_width,
                           alp::AlpCompressionData<T> *data) {
  const size_t n_vecs = utils::get_n_vecs_from_size(count);
  fill_array_with_constant<uint8_t>(data->ffor.bit_widths, n_vecs,
                                    static_cast<uint8_t>(value_bit_width));

  return data;
}

template <typename T>
std::pair<T *, size_t> read_file_as(size_t input_count, std::string path) {

  // Open file
  std::ifstream inputFile(path, std::ios::binary | std::ios::ate);
  if (!inputFile) {
    throw std::invalid_argument("Could not open the specified file.");
  }
  // Get file size
  const std::streamsize file_size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  // Check file size to contain right type of data
  bool file_size_is_multiple_of_T_size =
      static_cast<size_t>(file_size) % static_cast<size_t>(sizeof(T)) != 0;
  if (file_size_is_multiple_of_T_size) {
    throw std::invalid_argument(
        "File size is incorrect, it is not a multiple of the type's size.");
  }

  const size_t values_in_file = static_cast<size_t>(file_size) / sizeof(T);
  size_t count = input_count == 0 ? values_in_file : input_count;
  count = count - (count % consts::VALUES_PER_VECTOR);
  auto column = new T[count];

  // Read either the file size, or the total number of values needed,
  // whichever is smaller
  const std::streamsize read_size =
      std::min(file_size, static_cast<std::streamsize>((count * sizeof(T))));
  if (!inputFile.read(reinterpret_cast<char *>(column), read_size)) {
    throw std::invalid_argument("Failed to read file into column");
  }

  inputFile.close();

  // Copy paste the values in file until the column is filled
  if (values_in_file < count) {
    size_t n_filled_values = values_in_file;
    size_t n_empty_values_column = count - n_filled_values;
    while (n_empty_values_column > 0) {
      std::memcpy(column + n_filled_values, column,
                  std::min(n_empty_values_column, values_in_file));
      n_filled_values += values_in_file;

      if (n_empty_values_column < values_in_file) {
        break;
      }
      n_empty_values_column -= values_in_file;
    }
  }

  return std::make_pair(column, count);
}

} // namespace data

#endif // DATAGENERATION_HPP
