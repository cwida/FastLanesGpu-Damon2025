#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "../common/utils.hpp"
#include "../fls/compression.hpp"
#include "alp-bindings.hpp"
#include "config.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "falp.hpp"
#include "rd.hpp"

namespace alp {
constexpr int MAX_ATTEMPTS_TO_ENCODE = 10000;

template <typename T>
bool is_encoding_possible(const T *input_array, const size_t count,
                          alp::Scheme scheme) {
  T *sample_array = new T[count];
  state<T> alpstate;

  bool is_possible = false;
  for (int32_t attempts = 0; attempts < MAX_ATTEMPTS_TO_ENCODE; ++attempts) {
    alp::encoder<T>::init(input_array, 0, count, sample_array, alpstate);

    if ((is_possible = alpstate.scheme == scheme)) {
      break;
    }
  }

  delete[] sample_array;
  return is_possible;
}

template <typename T>
void int_encode(const T *input_array, const size_t count,
                AlpCompressionData<T> *data) {
  using INT_T = typename utils::same_width_int<T>::type;
  using UINT_T = typename utils::same_width_uint<T>::type;

  const size_t n_vecs = utils::get_n_vecs_from_size(count);

  T *sample_array = new T[count];
  state<T> alpstate;

  bool successful_encoding = false;
  for (int32_t attempts = 0; attempts < MAX_ATTEMPTS_TO_ENCODE; ++attempts) {
    alp::encoder<T>::init(input_array, 0, count, sample_array, alpstate);

    if ((successful_encoding = alpstate.scheme == Scheme::ALP)) {
      break;
    }
  }
  if (!successful_encoding) {
    throw alp::EncodingException();
  }
  delete[] sample_array;

  size_t compressed_vector_sizes = 0;
  INT_T *encoded_array = new INT_T[count];
  for (size_t i{0}; i < n_vecs; i++) {
    AlpVecExceptions<T> exceptions = data->exceptions.get_exceptions_for_vec(i);
    alp::encoder<T>::encode(input_array, exceptions.exceptions,
                            exceptions.positions, exceptions.count,
                            encoded_array, alpstate);
    data->exponents[i] = alpstate.exp;
    data->factors[i] = alpstate.fac;

    alp::encoder<T>::analyze_ffor(
        encoded_array, data->ffor.bit_widths[i],
        reinterpret_cast<INT_T *>(&data->ffor.bases[i]));

    fls::ffor(reinterpret_cast<UINT_T *>(encoded_array), data->ffor.array,
              data->ffor.bit_widths[i], &data->ffor.bases[i]);

    encoded_array += consts::VALUES_PER_VECTOR;
    data->ffor.array += consts::VALUES_PER_VECTOR;
    input_array += consts::VALUES_PER_VECTOR;

    compressed_vector_sizes +=
        alp::get_bytes_vector_compressed_size_without_overhead<T>(
            data->ffor.bit_widths[i], *exceptions.count);
  }

  data->ffor.array -= consts::VALUES_PER_VECTOR * n_vecs;
  encoded_array -= consts::VALUES_PER_VECTOR * n_vecs;
  data->compressed_alp_bytes_size =
      compressed_vector_sizes +
      n_vecs * alp::get_bytes_overhead_size_per_alp_vector<T>();
  data->compressed_galp_bytes_size =
      compressed_vector_sizes +
      n_vecs * alp::get_bytes_overhead_size_per_galp_vector<T>();
  delete[] encoded_array;
}

template <typename T>
void int_decode(T *output_array, const AlpCompressionData<T> *data) {
  using UINT_T = typename utils::same_width_uint<T>::type;
  const size_t n_vecs = utils::get_n_vecs_from_size(data->size);

  for (size_t i{0}; i < n_vecs; ++i) {
    AlpFFORVecHeader<UINT_T> ffor = data->ffor.get_ffor_header_for_vec(i);

    generated::falp::fallback::scalar::falp(
        ffor.array, output_array, *ffor.bit_width, ffor.base, data->factors[i],
        data->exponents[i]);

    AlpVecExceptions<T> exceptions = data->exceptions.get_exceptions_for_vec(i);
    alp::decoder<T>::patch_exceptions(output_array, exceptions.exceptions,
                                      exceptions.positions, exceptions.count);

    output_array += consts::VALUES_PER_VECTOR;
  }
}

} // namespace alp

template bool alp::is_encoding_possible(const float *input_array,
                                        const size_t count, alp::Scheme scheme);
template bool alp::is_encoding_possible(const double *input_array,
                                        const size_t count, alp::Scheme scheme);

template void alp::int_encode<float>(const float *input_array,
                                     const size_t count,
                                     alp::AlpCompressionData<float> *data);
template void
alp::int_decode<float>(float *output_array,
                       const alp::AlpCompressionData<float> *data);

template void alp::int_encode<double>(const double *input_array,
                                      const size_t count,
                                      alp::AlpCompressionData<double> *data);
template void
alp::int_decode<double>(double *output_array,
                        const alp::AlpCompressionData<double> *data);
