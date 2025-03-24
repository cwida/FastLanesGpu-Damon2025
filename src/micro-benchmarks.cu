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
#include "./kernel.cuh"
#include "./nvcomp-compressors.cuh"

enum DecompressorType {
  BP_1V,
  BP_4V,
  ALP_1V,
  GALP_BRANCHLESS_1V,
  GALP_BRANCHY_1V,
  GALP_BRANCHY_4V,
  GALP_PREFETCH_BRANCHY_1V,
  GALP_PREFETCH_BRANCHY_4V,
};

struct MicroBenchmarkResult {
  bool found_value;
};

template <typename T>
MicroBenchmarkResult benchmark_bp(const DecompressorType decompressor_type,
                                  const size_t count, const vbw_t vbw) {

  constant_memory::load_alp_constants<T>();

  GPUArray<bool> d_result(1);
  bool found_value_in_column = false;

  T value_to_search_for = 123456789;

  int32_t n_vecs_concurrently = 1;
  if (decompressor_type == DecompressorType::BP_4V) {
    n_vecs_concurrently = 4;
  }

  ThreadblockMapping<T> mapping = ThreadblockMapping<T>(
      utils::get_n_vecs_from_size(count), n_vecs_concurrently);

  size_t extra_buffer_for_branchless_unpackers = 4 * utils::get_n_lanes<T>();
  GPUArray<T> buffer(count + extra_buffer_for_branchless_unpackers);
  BPColumn<T> column{count, vbw, buffer.get()};

  switch (decompressor_type) {
  case BP_1V: {
    scan_column<T, BP1VDecompressor<T>, BPColumn<T>, 1>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
  } break;
  case BP_4V: {
    scan_column<T, BP4VDecompressor<T>, BPColumn<T>, 4>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
  } break;
  default:
    throw std::invalid_argument("Invalid bp decompressor type\n");
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_result.copy_to_host(&found_value_in_column);

  return MicroBenchmarkResult{found_value_in_column};
}

template <typename T>
MicroBenchmarkResult
benchmark_alp(const DecompressorType decompressor_type,
              const alp::AlpCompressionData<T> *alp_compressed_data) {

  constant_memory::load_alp_constants<T>();

  GPUArray<bool> d_result(1);
  bool found_value_in_column = false;

  T value_to_search_for = 1.23456789;

  int32_t n_vecs_concurrently = 1;
  if (decompressor_type == DecompressorType::GALP_BRANCHY_4V || 
			decompressor_type == DecompressorType::GALP_PREFETCH_BRANCHY_4V) {
    n_vecs_concurrently = 4;
  }

  ThreadblockMapping<T> mapping = ThreadblockMapping<T>(
      utils::get_n_vecs_from_size(alp_compressed_data->size),
      n_vecs_concurrently);

  switch (decompressor_type) {
  case ALP_1V: {
    AlpColumn<T> column = transfer::copy_alp_column_to_gpu(alp_compressed_data);
    scan_column<T, ALP1VDecompressor<T>, AlpColumn<T>, 1>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
    transfer::destroy_alp_column(column);
  } break;
  case GALP_BRANCHLESS_1V: {
    AlpExtendedColumn<T> column =
        transfer::copy_alp_extended_column_to_gpu(alp_compressed_data);
    scan_column<T, GALPBranchless1VDecompressor<T>, AlpExtendedColumn<T>, 1>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
    transfer::destroy_alp_column(column);
  } break;
  case GALP_BRANCHY_1V: {
    AlpExtendedColumn<T> column =
        transfer::copy_alp_extended_column_to_gpu(alp_compressed_data);
    scan_column<T, GALPBranchy1VDecompressor<T>, AlpExtendedColumn<T>, 1>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
    transfer::destroy_alp_column(column);
  } break;
  case GALP_BRANCHY_4V: {
    AlpExtendedColumn<T> column =
        transfer::copy_alp_extended_column_to_gpu(alp_compressed_data);
    scan_column<T, GALPBranchy4VDecompressor<T>, AlpExtendedColumn<T>, 4>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
    transfer::destroy_alp_column(column);
  } break;
  case GALP_PREFETCH_BRANCHY_1V: {
    AlpExtendedColumn<T> column =
        transfer::copy_alp_extended_column_to_gpu(alp_compressed_data);
    scan_column<T, GALPPrefetchBranchy1VDecompressor<T>, AlpExtendedColumn<T>, 1>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
    transfer::destroy_alp_column(column);
  } break;
  case GALP_PREFETCH_BRANCHY_4V: {
    AlpExtendedColumn<T> column =
        transfer::copy_alp_extended_column_to_gpu(alp_compressed_data);
    scan_column<T, GALPPrefetchBranchy4VDecompressor<T>, AlpExtendedColumn<T>, 4>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
            column, value_to_search_for, d_result.get());
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
    transfer::destroy_alp_column(column);
  } break;
  default:
    throw std::invalid_argument("Invalid alp decompressor type\n");
  }

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  d_result.copy_to_host(&found_value_in_column);

  return MicroBenchmarkResult{found_value_in_column};
}

struct BPCLIArgs {
  DecompressorType decompressor_enum;
  int value_bit_width_start;
  int value_bit_width_end;
  size_t value_count;
};

BPCLIArgs parse_bp_cli_args(int argc, char *argv[]) {
  if (argc != 6) { // Expecting program name + 4 arguments
    std::cerr
        << "Usage: " << argv[0]
        << "<microbenchmark:int> <decompressor_enum:int> <value_bit_width_start:int>"
           "<value_bit_width_end:int> "
           "<vector_count:int>\n";
    throw std::invalid_argument("Could not parse cli args as bp");
  }

  BPCLIArgs args;

  int32_t argcounter = 1;
  args.decompressor_enum =
      static_cast<DecompressorType>(std::atoi(argv[++argcounter]));
  args.value_bit_width_start = std::atoi(argv[++argcounter]);
  args.value_bit_width_end = std::atoi(argv[++argcounter]);
  args.value_count = std::atoi(argv[++argcounter]) * consts::VALUES_PER_VECTOR;

  if (args.decompressor_enum != DecompressorType::BP_1V &&
      args.decompressor_enum != DecompressorType::BP_4V) {
    throw std::invalid_argument("Incorrect BP decompressor");
  }

  return args;
}

struct ALPCLIArgs {
  DecompressorType decompressor_enum;
  int value_bit_width;
  int exception_count_start;
  int exception_count_end;
  size_t value_count;
};

ALPCLIArgs parse_alp_cli_args(int argc, char *argv[]) {
  if (argc != 7) { // Expecting program name + 4 arguments
    std::cerr
        << "Usage: " << argv[0]
        << "<microbenchmark:int> <decompressor_enum:int> <value_bit_width:int>"
           "<exception_count_start:int> <exception_count_end:int>"
           "<vector_count:int>\n";
    throw std::invalid_argument("Could not parse the args as alp");
  }

  ALPCLIArgs args;

  int32_t argcounter = 1;
  args.decompressor_enum =
      static_cast<DecompressorType>(std::atoi(argv[++argcounter]));
  args.value_bit_width = std::atoi(argv[++argcounter]);
  args.exception_count_start = std::atoi(argv[++argcounter]);
  args.exception_count_end = std::atoi(argv[++argcounter]);
  args.value_count = std::atoi(argv[++argcounter]) * consts::VALUES_PER_VECTOR;

  switch (args.decompressor_enum) {
  case DecompressorType::ALP_1V:
  case DecompressorType::GALP_BRANCHLESS_1V:
  case DecompressorType::GALP_BRANCHY_1V:
  case DecompressorType::GALP_BRANCHY_4V:
  case DecompressorType::GALP_PREFETCH_BRANCHY_1V:
  case DecompressorType::GALP_PREFETCH_BRANCHY_4V:
    break;
  default:
    throw std::invalid_argument("Could not parse decompressor arg");
  }

  return args;
}

enum MicroBenchmark {
  BP,
  ALP,
  MULTICOLUMN,
};

MicroBenchmark parse_decompressor(int argc, char *argv[]) {
  if (argc <= 1) {
    throw std::invalid_argument("Should at least specify microbenchmark");
  }
  return static_cast<MicroBenchmark>(std::atoi(argv[1]));
}

void execute_bp(BPCLIArgs args) {
  using T = uint32_t;

  MicroBenchmarkResult result;
  for (int32_t vbw{args.value_bit_width_start}; vbw <= args.value_bit_width_end;
       ++vbw) {
    result = benchmark_bp<T>(args.decompressor_enum, args.value_count, vbw);
  }
}

void execute_alp(ALPCLIArgs args) {
  using T = float;

  alp::AlpCompressionData<T> *alp_compressed_data =
      data::generate_alp_datastructure<T>(args.value_count,
                                          args.exception_count_start,
                                          args.value_bit_width, 4);

  MicroBenchmarkResult result;
  for (int32_t ec{args.exception_count_start}; ec <= args.exception_count_end;
       ++ec) {
    data::modify_alp_exception_count(args.value_count, ec, alp_compressed_data);
    result = benchmark_alp<T>(args.decompressor_enum, alp_compressed_data);
  }

  delete alp_compressed_data;
}


int main(int argc, char *argv[]) {
  MicroBenchmark micro_benchmark = parse_decompressor(argc, argv);

  if (micro_benchmark == MicroBenchmark::BP) {
    BPCLIArgs args = parse_bp_cli_args(argc, argv);
    execute_bp(args);
  } else if (micro_benchmark == MicroBenchmark::ALP) {
    ALPCLIArgs args = parse_alp_cli_args(argc, argv);
    execute_alp(args);
  }

  return 0;
}
