#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "./alp/alp-bindings.hpp"
#include "./datageneration.hpp"
#include "./gpu/alp.cuh"
#include "./gpu/device-types.cuh"
#include "./gpu/host-alp-utils.cuh"
#include "./gpu/host-utils.cuh"
#include "./gpu/old-fls.cuh"
#include "./kernel.cuh"
#include "./multicolumns.cuh"
#include "./nvcomp-compressors.cuh"

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES,
          typename OutputProcessor>
struct OldFLSAdjusted : BitUnpackerBase<T> {
  using UINT_T = typename utils::same_width_uint<T>::type;

  const UINT_T *in;
  const vbw_t value_bit_width;
  OutputProcessor processor;

  __device__ __forceinline__ OldFLSAdjusted(const UINT_T *__restrict a_in,
                                            const lane_t lane,
                                            const vbw_t a_value_bit_width,
                                            OutputProcessor processor)
      : in(a_in + lane), value_bit_width(a_value_bit_width),
        processor(processor) {
    static_assert(UNPACK_N_VECTORS == 1, "Old FLS can only unpack 1 at a time");
    static_assert(UNPACK_N_VALUES == utils::get_values_per_lane<T>(),
                  "Old FLS can only unpack entire lanes");
  };

  __device__ __forceinline__ void unpack_next_into(T *__restrict out) override {
    UINT_T registers[UNPACK_N_VALUES];
    oldfls::adjusted::unpack(reinterpret_cast<UINT_T *>(out), registers,
                             value_bit_width);

#pragma unroll
    for (int32_t i{0}; i < UNPACK_N_VALUES; ++i) {
      out[i] = processor(registers[i], 0);
    }
  }
};

struct MultiColumnCLIArgs {
  int n_columns;
  int value_bit_width_start;
  int value_bit_width_end;
  int exception_count;
  size_t value_count;
};

MultiColumnCLIArgs parse_multicolumn_cli_args(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << "<n_cols:int>"
                 "<value_bit_width_start:int> "
                 "<value_bit_width_end:int> "
                 "<exception_count:int>"
                 "<vector_count:int>\n";
    throw std::invalid_argument("Could not parse the args as multicolumn");
  }

  MultiColumnCLIArgs args;

  int32_t argcounter = 0;
  args.n_columns = std::atoi(argv[++argcounter]);
  args.value_bit_width_start = std::atoi(argv[++argcounter]);
  args.value_bit_width_end = std::atoi(argv[++argcounter]);
  args.exception_count = std::atoi(argv[++argcounter]);
  args.value_count = std::atoi(argv[++argcounter]) * consts::VALUES_PER_VECTOR;

  return args;
}

bool execute_multicolumn(MultiColumnCLIArgs args) {
  using T = float;

  alp::AlpCompressionData<T> *alp_data_ptr =
      data::generate_alp_datastructure<T>(args.value_count,
                                          args.exception_count,
                                          args.value_bit_width_start, 4);
  using ALP32Values1Vec =
      AlpUnpacker<T, 1, 32,
                  BitUnpackerStatefulBranchless<T, 1, 32, ALPFunctor<T, 1>,
                                                consts::VALUES_PER_VECTOR>,
                  StatefulALPExceptionPatcher<T, 1, 32>, AlpColumn<T>>;
  using ALP1Values1Vec =
      AlpUnpacker<T, 1, 1,
                  BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>,
                                                consts::VALUES_PER_VECTOR>,
                  StatefulALPExceptionPatcher<T, 1, 1>, AlpColumn<T>>;
  using GALP32Values1Vec =
      AlpUnpacker<T, 1, 32,
                  BitUnpackerStatefulBranchless<T, 1, 32, ALPFunctor<T, 1>,
                                                consts::VALUES_PER_VECTOR>,
                  PrefetchAllALPExceptionPatcher<T, 1, 32>, AlpExtendedColumn<T>>;
  using GALP1Values1Vec =
      AlpUnpacker<T, 1, 1,
                  BitUnpackerStatefulBranchless<T, 1, 1, ALPFunctor<T, 1>,
                                                consts::VALUES_PER_VECTOR>,
                  PrefetchAllALPExceptionPatcher<T, 1, 1>,
                  AlpExtendedColumn<T>>;

  bool result;
  for (int32_t vbw{args.value_bit_width_start}; vbw <= args.value_bit_width_end;
       ++vbw) {
    data::modify_alp_value_bit_width(args.value_count, vbw, alp_data_ptr);
    result = scan_columns_selector<T, ALP32Values1Vec, 1, 32>(args.n_columns,
                                                          alp_data_ptr);
    result = scan_columns_selector<T, ALP1Values1Vec, 1, 1>(args.n_columns,
                                                         alp_data_ptr);
    result = scan_columns_extended_selector<T, GALP32Values1Vec, 1, 32>(
        args.n_columns, alp_data_ptr);
    result = scan_columns_extended_selector<T, GALP1Values1Vec, 1, 1>(
        args.n_columns, alp_data_ptr);
  }

  delete alp_data_ptr;
  return result;
}

int main(int argc, char *argv[]) {

  MultiColumnCLIArgs args = parse_multicolumn_cli_args(argc, argv);
  execute_multicolumn(args);

  return 0;
}
