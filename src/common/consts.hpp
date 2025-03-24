#include <cstdint>
#include <string>
#include <stdexcept>

#ifndef CONSTS_H
#define CONSTS_H

namespace consts {

constexpr int32_t REGISTER_WIDTH = 1024;
constexpr int32_t VALUES_PER_VECTOR = 1024;
constexpr int32_t THREADS_PER_WARP = 32;

template <class T> struct as {
  static inline constexpr T MAGIC_NUMBER = 0;
};

template <> struct as<float> {
  static inline constexpr float MAGIC_NUMBER = 0.3214f;
};

template <> struct as<double> {
  static inline constexpr double MAGIC_NUMBER = 0.3214;
};

} // namespace consts

namespace enums {

enum ComparisonType {
  DECOMPRESSION,
  DECOMPRESSION_QUERY,
};

template<typename T>
std::string get_name_for_comparison_type(T type) {
  switch (type) {
  case ComparisonType::DECOMPRESSION:
    return "decompression";
  case ComparisonType::DECOMPRESSION_QUERY:
    return "decompression_query";
  default:
    throw std::invalid_argument("Could not parse comparison type");
  }
}

enum CompressionType {
  THRUST,
  ALP,
  GALP,
  BITCOMP,
  BITCOMP_SPARSE,
  LZ4,
  ZSTD,
  DEFLATE,
  GDEFLATE,
  SNAPPY,
};

template<typename T>
std::string get_name_for_compression_type(T type) {
  switch (type) {
  case enums::CompressionType::THRUST:
    return "Thrust";
  case enums::CompressionType::ALP:
    return "ALP";
  case enums::CompressionType::GALP:
    return "GALP";
  case enums::CompressionType::BITCOMP:
    return "Bitcomp";
  case enums::CompressionType::BITCOMP_SPARSE:
    return "BitcompSparse";
  case enums::CompressionType::LZ4:
    return "LZ4";
  case enums::CompressionType::ZSTD:
    return "zstd";
  case enums::CompressionType::DEFLATE:
    return "Deflate";
  case enums::CompressionType::GDEFLATE:
    return "GDeflate";
  case enums::CompressionType::SNAPPY:
    return "Snappy";
  default:
    throw std::invalid_argument("Could not parse decompresor");
  }
}

} // namespace enums

#endif // CONSTS_H
