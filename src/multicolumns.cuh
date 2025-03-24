
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "./alp/alp-bindings.hpp"
#include "./datageneration.hpp"
#include "./gpu/alp.cuh"
#include "./gpu/host-alp-utils.cuh"
#include "./gpu/host-utils.cuh"
#include "./kernel.cuh"
#include "./nvcomp-compressors.cuh"
#include "./benchmark-compressors.cuh"

#ifndef MULTICOLUMNS_CUH
#define MULTICOLUMNS_CUH


template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0, const T value, bool *out) {
constexpr int32_t N_COLS = 1;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 1];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1, const T value, bool *out) {
constexpr int32_t N_COLS = 2;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 2];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2, const T value, bool *out) {
constexpr int32_t N_COLS = 3;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 3];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3, const T value, bool *out) {
constexpr int32_t N_COLS = 4;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 4];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3,const ColumnT column_4, const T value, bool *out) {
constexpr int32_t N_COLS = 5;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 5];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
DecompressorT decompressor_4 = DecompressorT(column_4, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_4.unpack_next_into(registers + 4 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3,const ColumnT column_4,const ColumnT column_5, const T value, bool *out) {
constexpr int32_t N_COLS = 6;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 6];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
DecompressorT decompressor_4 = DecompressorT(column_4, vector_index, lane);
DecompressorT decompressor_5 = DecompressorT(column_5, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_4.unpack_next_into(registers + 4 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_5.unpack_next_into(registers + 5 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3,const ColumnT column_4,const ColumnT column_5,const ColumnT column_6, const T value, bool *out) {
constexpr int32_t N_COLS = 7;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 7];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
DecompressorT decompressor_4 = DecompressorT(column_4, vector_index, lane);
DecompressorT decompressor_5 = DecompressorT(column_5, vector_index, lane);
DecompressorT decompressor_6 = DecompressorT(column_6, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_4.unpack_next_into(registers + 4 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_5.unpack_next_into(registers + 5 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_6.unpack_next_into(registers + 6 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3,const ColumnT column_4,const ColumnT column_5,const ColumnT column_6,const ColumnT column_7, const T value, bool *out) {
constexpr int32_t N_COLS = 8;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 8];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
DecompressorT decompressor_4 = DecompressorT(column_4, vector_index, lane);
DecompressorT decompressor_5 = DecompressorT(column_5, vector_index, lane);
DecompressorT decompressor_6 = DecompressorT(column_6, vector_index, lane);
DecompressorT decompressor_7 = DecompressorT(column_7, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_4.unpack_next_into(registers + 4 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_5.unpack_next_into(registers + 5 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_6.unpack_next_into(registers + 6 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_7.unpack_next_into(registers + 7 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3,const ColumnT column_4,const ColumnT column_5,const ColumnT column_6,const ColumnT column_7,const ColumnT column_8, const T value, bool *out) {
constexpr int32_t N_COLS = 9;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 9];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
DecompressorT decompressor_4 = DecompressorT(column_4, vector_index, lane);
DecompressorT decompressor_5 = DecompressorT(column_5, vector_index, lane);
DecompressorT decompressor_6 = DecompressorT(column_6, vector_index, lane);
DecompressorT decompressor_7 = DecompressorT(column_7, vector_index, lane);
DecompressorT decompressor_8 = DecompressorT(column_8, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_4.unpack_next_into(registers + 4 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_5.unpack_next_into(registers + 5 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_6.unpack_next_into(registers + 6 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_7.unpack_next_into(registers + 7 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_8.unpack_next_into(registers + 8 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT, typename ColumnT,unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__global__ void scan_columns(const ColumnT column_0,const ColumnT column_1,const ColumnT column_2,const ColumnT column_3,const ColumnT column_4,const ColumnT column_5,const ColumnT column_6,const ColumnT column_7,const ColumnT column_8,const ColumnT column_9, const T value, bool *out) {
constexpr int32_t N_COLS = 10;
const auto mapping = VectorToThreadMapping<T, N_VECS_CONCURRENTLY>();
const int32_t vector_index = mapping.get_vector_index();
const lane_t lane = mapping.get_lane();
T registers[UNPACK_N_VALUES * N_VECS_CONCURRENTLY * 10];
bool all_columns_equal = true;
DecompressorT decompressor_0 = DecompressorT(column_0, vector_index, lane);
DecompressorT decompressor_1 = DecompressorT(column_1, vector_index, lane);
DecompressorT decompressor_2 = DecompressorT(column_2, vector_index, lane);
DecompressorT decompressor_3 = DecompressorT(column_3, vector_index, lane);
DecompressorT decompressor_4 = DecompressorT(column_4, vector_index, lane);
DecompressorT decompressor_5 = DecompressorT(column_5, vector_index, lane);
DecompressorT decompressor_6 = DecompressorT(column_6, vector_index, lane);
DecompressorT decompressor_7 = DecompressorT(column_7, vector_index, lane);
DecompressorT decompressor_8 = DecompressorT(column_8, vector_index, lane);
DecompressorT decompressor_9 = DecompressorT(column_9, vector_index, lane);
for (si_t i{0}; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
decompressor_0.unpack_next_into(registers + 0 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_1.unpack_next_into(registers + 1 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_2.unpack_next_into(registers + 2 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_3.unpack_next_into(registers + 3 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_4.unpack_next_into(registers + 4 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_5.unpack_next_into(registers + 5 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_6.unpack_next_into(registers + 6 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_7.unpack_next_into(registers + 7 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_8.unpack_next_into(registers + 8 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
decompressor_9.unpack_next_into(registers + 9 * (UNPACK_N_VALUES * N_VECS_CONCURRENTLY));
#pragma unroll
for (int c{1}; c < N_COLS; ++c) {
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES + c * N_VECS_CONCURRENTLY * UNPACK_N_VALUES] == registers[va + v * UNPACK_N_VALUES + (c-1) * N_VECS_CONCURRENTLY * UNPACK_N_VALUES];
}
}
}
#pragma unroll
for (int va{0}; va < UNPACK_N_VALUES; ++va) {
#pragma unroll
for (int v{0}; v < N_VECS_CONCURRENTLY; ++v) {
all_columns_equal &= registers[va + v * UNPACK_N_VALUES] == value;
}
}
}

if (!all_columns_equal) {
*out = true;
}}

template <typename T, typename DecompressorT,  unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__host__ bool scan_columns_selector(const int32_t n_columns, const alp::AlpCompressionData<T>* alp_data_ptr) {

  constant_memory::load_alp_constants<float>();

  GPUArray<bool> d_out(1);
  T value = 123.456789;

  const ThreadblockMapping<float> mapping(
      utils::get_n_vecs_from_size(alp_data_ptr->size), N_VECS_CONCURRENTLY);
  CudaStopwatch stopwatch = CudaStopwatch();
  float execution_time_ms;

double compression_ratio = alp_data_ptr->get_alp_compression_ratio();
switch (n_columns) {
case 1:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);}break;
case 2:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);}break;
case 3:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);}break;
case 4:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);}break;
case 5:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_4 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);}break;
case 6:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_4 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_5 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);}break;
case 7:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_4 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_5 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_6 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);}break;
case 8:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_4 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_5 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_6 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_7 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,column_7,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);
transfer::destroy_alp_column<T>(column_7);}break;
case 9:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_4 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_5 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_6 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_7 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_8 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,column_7,column_8,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);
transfer::destroy_alp_column<T>(column_7);
transfer::destroy_alp_column<T>(column_8);}break;
case 10:{AlpColumn<T> column_0 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_1 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_2 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_3 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_4 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_5 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_6 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_7 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_8 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);
AlpColumn<T> column_9 = transfer::copy_alp_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,column_7,column_8,column_9,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);
transfer::destroy_alp_column<T>(column_7);
transfer::destroy_alp_column<T>(column_8);
transfer::destroy_alp_column<T>(column_9);}break;
}

bool result;
d_out.copy_to_host(&result);
return result;

}

template <typename T, typename DecompressorT,  unsigned N_VECS_CONCURRENTLY, unsigned UNPACK_N_VALUES>
__host__ bool scan_columns_extended_selector(const int32_t n_columns, const alp::AlpCompressionData<T>* alp_data_ptr) {

  constant_memory::load_alp_constants<float>();

  GPUArray<bool> d_out(1);
  T value = 123.456789;

  const ThreadblockMapping<float> mapping(
      utils::get_n_vecs_from_size(alp_data_ptr->size), N_VECS_CONCURRENTLY);
  CudaStopwatch stopwatch = CudaStopwatch();
  float execution_time_ms;

double compression_ratio = alp_data_ptr->get_galp_compression_ratio();
switch (n_columns) {
case 1:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);}break;
case 2:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);}break;
case 3:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);}break;
case 4:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);}break;
case 5:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_4 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);}break;
case 6:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_4 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_5 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);}break;
case 7:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_4 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_5 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_6 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);}break;
case 8:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_4 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_5 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_6 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_7 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,column_7,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);
transfer::destroy_alp_column<T>(column_7);}break;
case 9:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_4 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_5 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_6 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_7 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_8 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,column_7,column_8,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);
transfer::destroy_alp_column<T>(column_7);
transfer::destroy_alp_column<T>(column_8);}break;
case 10:{AlpExtendedColumn<T> column_0 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_1 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_2 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_3 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_4 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_5 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_6 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_7 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_8 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);
AlpExtendedColumn<T> column_9 = transfer::copy_alp_extended_column_to_gpu<T>(alp_data_ptr);stopwatch.start();scan_columns<T, DecompressorT, AlpExtendedColumn<T>, N_VECS_CONCURRENTLY, UNPACK_N_VALUES><<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column_0,column_1,column_2,column_3,column_4,column_5,column_6,column_7,column_8,column_9,value, d_out.get());execution_time_ms = stopwatch.stop();transfer::destroy_alp_column<T>(column_0);
transfer::destroy_alp_column<T>(column_1);
transfer::destroy_alp_column<T>(column_2);
transfer::destroy_alp_column<T>(column_3);
transfer::destroy_alp_column<T>(column_4);
transfer::destroy_alp_column<T>(column_5);
transfer::destroy_alp_column<T>(column_6);
transfer::destroy_alp_column<T>(column_7);
transfer::destroy_alp_column<T>(column_8);
transfer::destroy_alp_column<T>(column_9);}break;
}

bool result;
d_out.copy_to_host(&result);
return result;

}


#endif // MULTICOLUMNS_CUH
