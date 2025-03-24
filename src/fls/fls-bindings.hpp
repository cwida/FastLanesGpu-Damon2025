#ifndef FFOR_FFOR_HPP
#define FFOR_FFOR_HPP

#include <cstdint>
#include <cstddef>

namespace fls {

void pack(const uint64_t *__restrict in, uint64_t *__restrict out, uint8_t bw);
void pack(const uint32_t *__restrict in, uint32_t *__restrict out, uint8_t bw);
void pack(const uint16_t *__restrict in, uint16_t *__restrict out, uint8_t bw);
void pack(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw);

void unpack(const uint64_t *__restrict in, uint64_t *__restrict out,
            uint8_t bw);
void unpack(const uint32_t *__restrict in, uint32_t *__restrict out,
            uint8_t bw);
void unpack(const uint16_t *__restrict in, uint16_t *__restrict out,
            uint8_t bw);
void unpack(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw);

void ffor(const uint64_t *__restrict in, uint64_t *__restrict out, uint8_t bw,
          const uint64_t *__restrict a_base_p);
void ffor(const uint32_t *__restrict in, uint32_t *__restrict out, uint8_t bw,
          const uint32_t *__restrict a_base_p);
void ffor(const uint16_t *__restrict in, uint16_t *__restrict out, uint8_t bw,
          const uint16_t *__restrict a_base_p);
void ffor(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw,
          const uint8_t *__restrict a_base_p);

void ffor(const int64_t *__restrict in, int64_t *__restrict out, uint8_t bw,
          const int64_t *__restrict a_base_p);
void ffor(const int32_t *__restrict in, int32_t *__restrict out, uint8_t bw,
          const int32_t *__restrict a_base_p);
void ffor(const int16_t *__restrict in, int16_t *__restrict out, uint8_t bw,
          const int16_t *__restrict a_base_p);
void ffor(const int8_t *__restrict in, int8_t *__restrict out, uint8_t bw,
          const int8_t *__restrict a_base_p);

void unffor(const uint64_t *__restrict in, uint64_t *__restrict out, uint8_t bw,
            const uint64_t *__restrict a_base_p);
void unffor(const uint32_t *__restrict in, uint32_t *__restrict out, uint8_t bw,
            const uint32_t *__restrict a_base_p);
void unffor(const uint16_t *__restrict in, uint16_t *__restrict out, uint8_t bw,
            const uint16_t *__restrict a_base_p);
void unffor(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t bw,
            const uint8_t *__restrict a_base_p);

void unffor(const int64_t* __restrict in, int64_t* __restrict out, uint8_t bw, const int64_t* __restrict a_base_p);
void unffor(const int32_t* __restrict in, int32_t* __restrict out, uint8_t bw, const int32_t* __restrict a_base_p);
void unffor(const int16_t* __restrict in, int16_t* __restrict out, uint8_t bw, const int16_t* __restrict a_base_p);
void unffor(const int8_t* __restrict in, int8_t* __restrict out, uint8_t bw, const int8_t* __restrict a_base_p);


} // namespace fls

#endif
