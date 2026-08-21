#ifndef GPU_CPP_NUMERIC_TYPES_HALF_HPP
#define GPU_CPP_NUMERIC_TYPES_HALF_HPP

#include <type_traits>

#if !defined(__FLT16_MANT_DIG__)
#error "gpu.cpp requires compiler support for IEEE 754 _Float16"
#endif

using half = _Float16;

static_assert(sizeof(half) == 2);
static_assert(std::is_trivially_copyable_v<half>);

#endif
