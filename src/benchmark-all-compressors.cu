#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "benchmark-compressors.cuh"
#include "datageneration.hpp"

struct CLIArgs {
  int32_t n_vecs;
  int32_t vbw_start;
  int32_t vbw_end;
  int32_t ec_start;
  int32_t ec_end;
};

CLIArgs parse_cli_args(int argc, char *argv[]) {
  CLIArgs args;

  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <vector_count> "
                 "<vbw_start> <vbw_end> "
                 "<ec_start> <ec_end>"
              << std::endl;
    exit(1);
  }

  int arg_counter = 0;
  args.n_vecs = std::stoi(argv[++arg_counter]);
  args.vbw_start = std::stoi(argv[++arg_counter]);
  args.vbw_end = std::stoi(argv[++arg_counter]);
  args.ec_start = std::stoi(argv[++arg_counter]);
  args.ec_end = std::stoi(argv[++arg_counter]);

  return args;
}

template <typename T>
void write_array_to_binary_file(T *array, size_t size, std::string out) {
  std::ofstream file(out, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << out << " for writing"
              << std::endl;
    return;
  }

  file.write(reinterpret_cast<const char *>(array), size * sizeof(T));

  if (file.fail()) {
    std::cerr << "Error: Failed to write data to file " << out << std::endl;
  }
}

int main(int argc, char *argv[]) {
  CLIArgs args = parse_cli_args(argc, argv);

  using T = float;

  uint32_t u = 0x4100f9db;
  float value_to_search_for = *reinterpret_cast<float *>(&u);

  const size_t size = static_cast<size_t>(args.n_vecs) * consts::VALUES_PER_VECTOR;

  const alp::AlpCompressionData<T>* alp_datastructure = data::generate_alp_datastructure<T>(
      size, args.ec_start, args.vbw_start, 4);

  T *array = new T[size];
  int_decode(array, alp_datastructure);

  std::string generated_data_name = "generated-vbw-" +
                                    std::to_string(args.vbw_start) + "-ec-" +
                                    std::to_string(args.ec_start);

  for (int i{0}; i < 2; ++i) {
    for (int j{0}; j < 10; ++j) {
      const enums::ComparisonType comparison_type =
          static_cast<enums::ComparisonType>(i);
      const enums::CompressionType compression_type =
          static_cast<enums::CompressionType>(j);
      BenchmarkResult result;

      switch (compression_type) {
      case enums::CompressionType::THRUST:
        if (comparison_type == enums::ComparisonType::DECOMPRESSION_QUERY) {
          result = benchmark_thrust(array, size, value_to_search_for);
        } else {
          continue;
        }
        break;
      case enums::CompressionType::ALP:
      case enums::CompressionType::GALP:
        result = benchmark_alp(comparison_type, compression_type,
                               array, alp_datastructure, value_to_search_for);
        break;
      default:
        result = benchmark_hwc(comparison_type, compression_type, array, size,
                               value_to_search_for);
        break;
      }

      result.log_result(comparison_type, compression_type, generated_data_name);
    }
  }

  // write_array_to_binary_file(array, size, "verify.bin");

  delete[] array;

  return 0;
}
