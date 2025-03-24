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
#include "./benchmark-compressors.cuh"
#include "./datageneration.hpp"
#include "./gpu/alp.cuh"
#include "./gpu/host-alp-utils.cuh"
#include "./gpu/host-utils.cuh"
#include "./kernel.cuh"
#include "./nvcomp-compressors.cuh"

std::string extract_filename(const std::string &path) {
  size_t pos = path.find_last_of("/");
  std::string filename =
      (pos == std::string::npos) ? path : path.substr(pos + 1);

  size_t extPos = filename.find_last_of(".");
  return (extPos == std::string::npos) ? filename : filename.substr(0, extPos);
}

struct CLIArgs {
  enums::ComparisonType comparison_type;
  enums::CompressionType decompressor_enum;
  std::string file_path;
  int n_vecs;
};

CLIArgs parse_cli_args(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << "<query_enum:int> <decompressor_enum:int> "
                 "<data_path:string> <vector_count:int>\n";
    throw std::invalid_argument("");
  }

  CLIArgs args;

  args.comparison_type = static_cast<enums::ComparisonType>(std::atoi(argv[1]));
  args.decompressor_enum =
      static_cast<enums::CompressionType>(std::atoi(argv[2]));
  args.file_path = argv[3];
  args.n_vecs = std::atoi(argv[4]) * consts::VALUES_PER_VECTOR;

  return args;
}

int main(int argc, char *argv[]) {
  CLIArgs args = parse_cli_args(argc, argv);
  auto [data, count] = data::read_file_as<float>(args.n_vecs, args.file_path);

  uint32_t u = 0x4100f9db;
  float value_to_search_for = *reinterpret_cast<float *>(&u);

  BenchmarkResult result;
  if (args.comparison_type == enums::ComparisonType::DECOMPRESSION_QUERY &&
      args.decompressor_enum == enums::THRUST) {
    result = benchmark_thrust(data, count, value_to_search_for);
  } else if (args.decompressor_enum == enums::ALP ||
             args.decompressor_enum == enums::GALP) {
    result = benchmark_alp(args.comparison_type, args.decompressor_enum, data,
                           count, value_to_search_for);
  } else {
    result = benchmark_hwc(args.comparison_type, args.decompressor_enum, data,
                           count, value_to_search_for);
  }

  result.log_result(args.comparison_type, args.decompressor_enum,
                    count * sizeof(float), extract_filename(args.file_path));

  delete data;
  return 0;
}
