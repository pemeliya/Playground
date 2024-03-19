
#ifndef BUFFER_ADDRESSING_HPP
#define BUFFER_ADDRESSING_HPP 1

#include <cstdint>

using index_t = int32_t;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));

//typename vector_type<int32_t, 4>::type;

__device__ void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

enum struct BufCoherence
{
    Def = 0, // default value
    GLC              = 1,
    SLC              = 2,
    GLC_SLC          = 3,
};

#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000

template <typename T>
__device__ int32x4_t make_buffer_resource(T* p_wave, index_t space_size)
{
    union {
        int32x4_t content;
        struct {
            void *address;
            int32_t range;
            int32_t config;
        };
    } S = {
        .address = (void *)p_wave,
        .range = space_size,
        .config = CK_BUFFER_RESOURCE_3RD_DWORD,
    };
    return S.content;
}

template <BufCoherence coherence = BufCoherence::Def>
__device__ int32x4_t
amd_buffer_load(const int32x4_t& src_buffer_res, index_t thread_ofs,
        index_t wave_ofs)
{
    return llvm_amdgcn_raw_buffer_load_i32x4(src_buffer_res, thread_ofs,
                        wave_ofs, static_cast<index_t>(coherence));
}

template <BufCoherence coherence = BufCoherence::Def>
__device__ void
amd_buffer_store(const int32x4_t& dst_buffer_res, 
        index_t thread_ofs, index_t wave_ofs, const int32x4_t& data)
{
    llvm_amdgcn_raw_buffer_store_i32x4(data, dst_buffer_res, 
                thread_ofs, wave_ofs, static_cast<index_t>(coherence));
}

#endif // BUFFER_ADDRESSING_HPP
