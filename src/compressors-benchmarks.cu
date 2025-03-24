#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "engine/data.cuh"
#include "engine/enums.cuh"
#include "flsgpu/consts.cuh"
#include "nvcomp/benchmark-compressors.cuh"

std::string extract_filename(const std::string &path) {
  size_t pos = path.find_last_of("/");
  std::string filename =
      (pos == std::string::npos) ? path : path.substr(pos + 1);

  size_t extPos = filename.find_last_of(".");
  return (extPos == std::string::npos) ? filename : filename.substr(0, extPos);
}

struct CLIArgs {
  enums::DataType data_type;
  enums_nvcomp::ComparisonType comparison_type;
  enums_nvcomp::CompressionType decompressor_enum;
  std::string file_path;
  int n_values;
};

CLIArgs parse_cli_args(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << "<data_type> <query_enum:int> <decompressor_enum:int> "
                 "<data_path:string> <vector_count:int>\n";
    throw std::invalid_argument("");
  }

  CLIArgs args;

  int32_t arg_counter = 0;
  args.data_type = enums::string_to_data_type(argv[++arg_counter]);
  args.comparison_type =
      enums_nvcomp::string_to_comparison_type(argv[++arg_counter]);
  args.decompressor_enum =
      enums_nvcomp::string_to_compression_type(argv[++arg_counter]);
  args.file_path = argv[++arg_counter];
  args.n_values = std::atoi(argv[++arg_counter]) * consts::VALUES_PER_VECTOR;

  return args;
}

template <typename T> void execute_benchmark(CLIArgs args) {
  auto [data, count] =
      data::arrays::read_file_as<T>(args.file_path, args.n_values);

  uint32_t u = 0x4100f9db;
  const T value_to_search_for = *reinterpret_cast<T *>(&u);

  BenchmarkResult result;
  for (int i{0}; i < 2; ++i) {
    // One warmup run to load correct modules
    if (args.comparison_type ==
            enums_nvcomp::ComparisonType::DECOMPRESSION_QUERY &&
        args.decompressor_enum == enums_nvcomp::THRUST) {
      result = benchmark_thrust<T>(data, count, value_to_search_for);
    } else if (args.decompressor_enum == enums_nvcomp::ALP ||
               args.decompressor_enum == enums_nvcomp::GALP) {
      result = benchmark_alp<T>(args.comparison_type, args.decompressor_enum,
                                data, count, value_to_search_for);
    } else {
      result = benchmark_hwc<T>(args.comparison_type, args.decompressor_enum,
                                data, count, value_to_search_for);
    }
  }

  result.log_result(args.comparison_type, args.decompressor_enum,
                    count * sizeof(T), extract_filename(args.file_path));

  delete data;
}

int main(int argc, char *argv[]) {
  CLIArgs args = parse_cli_args(argc, argv);

  switch (args.data_type) {
  case enums::DataType::F32:
    execute_benchmark<float>(args);
    break;
  case enums::DataType::F64:
    execute_benchmark<double>(args);
    break;
  default:
    throw std::invalid_argument("This data type is not implemented.");
  }
  return 0;
}
