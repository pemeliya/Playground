
#ifndef COMMON_FUNCS_CU_H
#define COMMON_FUNCS_CU_H 1

#include <math.h>

__device__ __forceinline__ float divApprox(float a, float b) {
    float res;
    asm volatile(R"( {
        div.full.f32 %0, %1, %2;
    })" : "=f"(res) : "f"(a), "f"(b));
    return res;
}

__device__ __forceinline__ double divApprox(double a, double b) {
    double res;
    asm volatile(R"( {
        div.rn.f64 %0, %1, %2;
    })" : "=d"(res) : "d"(a), "d"(b));
    return res;
}

// extracts bitfield from src of length 'width' starting at startIdx
__device__ __forceinline__ uint32_t bfe(uint32_t src, uint32_t startIdx, uint32_t width)
{
    uint32_t bit;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"(src), "r"(startIdx), "r"(width));
    return bit;
}


enum ShuffleType {
    stSync,
    stUp,
    stDown,
    stXor
};

template < ShuffleType Type, class NT >
__device__ __forceinline__  NT shflType(NT val, uint32_t idx,
                                   uint32_t allmsk = 0xffffffffu)
{
    constexpr uint32_t SZ = (sizeof(NT) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    union S {
        NT v;
        uint32_t d[SZ];
    };
    S in{ val }, res;

    #pragma unroll
    for(uint32_t i = 0; i < SZ; i++) {
#if COMPILE_FOR_ROCM
#else
        if(Type == stSync)
            res.d[i] = __shfl_sync(allmsk, in.d[i], idx);
        else if(Type == stUp)
            res.d[i] = __shfl_up_sync(allmsk, in.d[i], idx);
        else if(Type == stDown)
            res.d[i] = __shfl_down_sync(allmsk, in.d[i], idx);
        else if(Type == stXor)
            res.d[i] = __shfl_xor_sync(allmsk, in.d[i], idx);
#endif            
    }
    return res.v;
}

template < class NT >
__device__ __forceinline__  NT shflUpPred(NT val, uint32_t ofs, int32_t& pred,
                                   uint32_t allmsk = 0xffffffffu, uint32_t shfl_c = 31)
{
    constexpr uint32_t SZ = (sizeof(NT) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    union {
        NT v;
        uint32_t d[SZ];
    } in{val}, res;

     asm(R"({
        .reg .pred p;
        .reg .u32 res, pred;
        shfl.sync.up.b32 res|p, %2, %3, %4, %5;
        selp.u32 pred, 1, 0, p;
        mov.u32 %0, res;
        mov.u32 %1, pred;
        })" : "=r"(res.d[0]), "=r"(pred) : "r"(in.d[0]), "r"(ofs), "r"(shfl_c), "r"(allmsk));

    #pragma unroll
    for(uint32_t i = 1; i < SZ; i++) {
        res.d[i] = __shfl_up_sync(allmsk, in.d[i], ofs);
    }
    return res.v;
}

template < class NT >
__device__ __forceinline__  NT shflDownPred(NT val, uint32_t ofs, int32_t& pred,
                                   uint32_t allmsk = 0xffffffffu, uint32_t shfl_c = 31)
{
    constexpr uint32_t SZ = (sizeof(NT) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    union {
        NT v;
        uint32_t d[SZ];
    } in{val}, res;

     asm(R"({
        .reg .pred p;
        .reg .u32 res, pred;
        shfl.sync.down.b32 res|p, %2, %3, %4, %5;
        selp.u32 pred, 1, 0, p;
        mov.u32 %0, res;
        mov.u32 %1, pred;
        })" : "=r"(res.d[0]), "=r"(pred) : "r"(in.d[0]), "r"(ofs), "r"(shfl_c), "r"(allmsk));

    #pragma unroll
    for(uint32_t i = 1; i < SZ; i++) {
        res.d[i] = __shfl_down_sync(allmsk, in.d[i], ofs);
    }
    return res.v;
}

#endif // COMMON_FUNCS_CU_H
