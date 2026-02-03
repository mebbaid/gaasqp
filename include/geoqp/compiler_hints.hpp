#pragma once

#if defined(__GNUC__) || defined(__clang__)
    #define GEOQP_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define GEOQP_RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
    #define GEOQP_RESTRICT __restrict
#else
    #define GEOQP_RESTRICT
#endif


#if defined(__clang__)
    #define GEOQP_UNROLL_LOOP _Pragma("clang loop unroll(enable)")
    #define GEOQP_UNROLL_LOOP_4 _Pragma("clang loop unroll_count(4)")
#elif defined(__GNUC__) && (__GNUC__ >= 8)
    #define GEOQP_UNROLL_LOOP _Pragma("GCC unroll 4")
    #define GEOQP_UNROLL_LOOP_4 _Pragma("GCC unroll 4")
#elif defined(_MSC_VER)
    #define GEOQP_UNROLL_LOOP __pragma(loop(hint_parallel(4)))
    #define GEOQP_UNROLL_LOOP_4 __pragma(loop(hint_parallel(4)))
#else
    #define GEOQP_UNROLL_LOOP
    #define GEOQP_UNROLL_LOOP_4
#endif


#if defined(__GNUC__) || defined(__clang__)
    #define GEOQP_ALWAYS_INLINE inline __attribute__((always_inline))
    #define GEOQP_NEVER_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define GEOQP_ALWAYS_INLINE __forceinline
    #define GEOQP_NEVER_INLINE __declspec(noinline)
#else
    #define GEOQP_ALWAYS_INLINE inline
    #define GEOQP_NEVER_INLINE
#endif


#if defined(__GNUC__) || defined(__clang__)
    #define GEOQP_LIKELY(x) __builtin_expect(!!(x), 1)
    #define GEOQP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define GEOQP_LIKELY(x) (x)
    #define GEOQP_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define GEOQP_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
    #define GEOQP_ALIGN(n) __declspec(align(n))
#else
    #define GEOQP_ALIGN(n)
#endif


#if defined(__GNUC__) || defined(__clang__)
    #define GEOQP_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
    #define GEOQP_PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#else
    #define GEOQP_PREFETCH_READ(addr) ((void)0)
    #define GEOQP_PREFETCH_WRITE(addr) ((void)0)
#endif


#if defined(__OPTIMIZE__) || defined(NDEBUG) || defined(_NDEBUG)
    #define GEOQP_OPTIMIZED 1
#else
    #define GEOQP_OPTIMIZED 0
#endif


#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    #define GEOQP_ARCH_X86_64 1
    #define GEOQP_CACHE_LINE_SIZE 64
#elif defined(__i386__) || defined(_M_IX86)
    #define GEOQP_ARCH_X86_32 1
    #define GEOQP_CACHE_LINE_SIZE 64
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define GEOQP_ARCH_ARM64 1
    #define GEOQP_CACHE_LINE_SIZE 64
#elif defined(__arm__) || defined(_M_ARM)
    #define GEOQP_ARCH_ARM32 1
    #define GEOQP_CACHE_LINE_SIZE 64
#elif defined(__riscv)
    #define GEOQP_ARCH_RISCV 1
    #define GEOQP_CACHE_LINE_SIZE 64
#else
    #define GEOQP_ARCH_UNKNOWN 1
    #define GEOQP_CACHE_LINE_SIZE 64
#endif


#if defined(__clang__)
    #define GEOQP_COMPILER_CLANG 1
    #define GEOQP_COMPILER_NAME "Clang"
#elif defined(__GNUC__)
    #define GEOQP_COMPILER_GCC 1
    #define GEOQP_COMPILER_NAME "GCC"
#elif defined(_MSC_VER)
    #define GEOQP_COMPILER_MSVC 1
    #define GEOQP_COMPILER_NAME "MSVC"
#elif defined(__INTEL_COMPILER)
    #define GEOQP_COMPILER_INTEL 1
    #define GEOQP_COMPILER_NAME "Intel ICC"
#else
    #define GEOQP_COMPILER_UNKNOWN 1
    #define GEOQP_COMPILER_NAME "Unknown"
#endif

