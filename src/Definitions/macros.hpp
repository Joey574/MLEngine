#pragma once

#ifdef DEBUG
#define DEBUG_LOG(m) std::cout << __FILE__ << ":" << __LINE__ << " | " << m << std::endl
#else
#define DEBUG_LOG(m)
#endif

#define AVX512_VALID_PATH() assert(__builtin_cpu_supports("avx512f"))
#define AVX2_VALID_PATH() assert(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && !__builtin_cpu_supports("avx512f"))
#define SCALAR_VALID_PATH() assert(!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) && !__builtin_cpu_supports("avx512f"))
