#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "litemla_traits.cuh"

namespace efficientvit::litemla {

using namespace cute;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}

template <typename LiteMLATraits>
__global__ void litemla(
          void* Optr,
    const void* Qptr,
    const void* Kptr,
    const void* Vptr,
    const int bs,
    const int head,
    const float eps
) {
    using T = typename LiteMLATraits::T;
    using TiledMMA = typename LiteMLATraits::MMA;
    using G2SCopy = typename LiteMLATraits::G2SCopy;
    using S2RCopyAtom_N = typename LiteMLATraits::S2RCopyAtom_N;
    using SmemLayoutK = typename LiteMLATraits::SmemLayoutK;
    using SmemLayoutV = typename LiteMLATraits::SmemLayoutV;
    using R2SCopyAtom = typename LiteMLATraits::R2SCopyAtom;
    using SmemLayoutKV = typename LiteMLATraits::SmemLayoutKV;

    using SmemLayoutQ = typename LiteMLATraits::SmemLayoutQ;
    using SmemLayoutQtransposed = typename LiteMLATraits::SmemLayoutQtransposed;
    using SmemLayoutQtransposedNoSwizzle = typename LiteMLATraits::SmemLayoutQtransposedNoSwizzle;
    using S2RCopyAtom_T = typename LiteMLATraits::S2RCopyAtom_T;

    using R2GCopyAtom = typename LiteMLATraits::R2GCopyAtom;

    // m32n64k384 -> m32n64k32 -> m32n64k16
    constexpr int seq_len = LiteMLATraits::seq_len;
    constexpr int head_dim = LiteMLATraits::head_dim;
    constexpr int stages = LiteMLATraits::stages;

    constexpr int tile_m = LiteMLATraits::tile_m;
    constexpr int tile_n = LiteMLATraits::tile_n;
    constexpr int tile_k = LiteMLATraits::tile_k;

    extern __shared__ T smem_data[];
    T *smem_k = smem_data;
    T *smem_v = smem_data + cosize(SmemLayoutK{});

    int global_offset = head_dim * seq_len * blockIdx.x;

    Tensor Q = make_tensor(
        make_gmem_ptr((T*)Qptr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, _1{}));
    Tensor K = make_tensor(
        make_gmem_ptr((T*)Kptr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, _1{}));
    Tensor V = make_tensor(
        make_gmem_ptr((T*)Vptr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, _1{}));

    Tensor cK = make_identity_tensor(shape(K));
    Tensor cV = make_identity_tensor(shape(V));

    Tensor gQ = local_tile(Q, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));
    Tensor gK = local_tile(K, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _)); 
    Tensor gV = local_tile(V, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));

    Tensor cta_cK = local_tile(cK, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));
    Tensor cta_cV = local_tile(cV, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));

    auto sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{});

    auto sQ = make_tensor(make_smem_ptr(smem_k), SmemLayoutQ{});
    auto sQt = make_tensor(sQ.data(), SmemLayoutQtransposed{});
    auto sQtns = make_tensor(sQ.data().get(), SmemLayoutQtransposedNoSwizzle{});

    // v @ k ([64, 384] @ [384, 32])
    TiledMMA mma;
    auto mma_thr = mma.get_slice(threadIdx.x);
    auto tKVrV = mma_thr.partition_fragment_A(sV(_, _, 0));
    auto tKVrK = mma_thr.partition_fragment_B(sK(_, _, 0));
    auto tKVrKV = partition_fragment_C(mma, Shape<Int<tile_m>, Int<tile_n>>{});
    clear(tKVrKV);

    G2SCopy g2s_copy;
    auto g2s_copy_thr = g2s_copy.get_slice(threadIdx.x);
    auto tVgV_g2s = g2s_copy_thr.partition_S(gV); // (cpy, cpy_m, cpy_k, k_repeats)
    auto tKgK_g2s = g2s_copy_thr.partition_S(gK);
    auto tVsV_g2s = g2s_copy_thr.partition_D(sV); // (cpy, cpy_m, cpy_k, stage)
    auto tKsK_g2s = g2s_copy_thr.partition_D(sK);

    auto tVcV = g2s_copy_thr.partition_S(cta_cV);
    auto tKcK = g2s_copy_thr.partition_S(cta_cK);

    Tensor tVpV = cute::lazy::transform(tVcV, [&](auto c) { return elem_less(c, shape(V)); });
    Tensor tKpK = cute::lazy::transform(tKcK, [&](auto c) { return elem_less(c, shape(K)); });

    auto s2r_copy_v = make_tiled_copy_A(S2RCopyAtom_N{}, mma);
    auto s2r_copy_k = make_tiled_copy_B(S2RCopyAtom_N{}, mma);
    auto s2r_copy_v_thr = s2r_copy_v.get_slice(threadIdx.x);
    auto s2r_copy_k_thr = s2r_copy_k.get_slice(threadIdx.x);
    auto tKsK_s2r = s2r_copy_k_thr.partition_S(sK);
    auto tVsV_s2r = s2r_copy_v_thr.partition_S(sV); // (cpy, cpy_m, cpy_k, stage)
    auto tKVrK_s2r = s2r_copy_k_thr.retile_D(tKVrK);
    auto tKVrV_s2r = s2r_copy_v_thr.retile_D(tKVrV);

    int gmem_read = 0;
    int smem_read = 0;
    int smem_write = 0;

#pragma unroll
    for (int istage = 0; istage < stages - 1; ++istage) {
        copy_if(g2s_copy, tVpV(_, _, _, istage), tVgV_g2s(_, _, _, istage), tVsV_g2s(_, _, _, istage));
        copy_if(g2s_copy, tKpK(_, _, _, istage), tKgK_g2s(_, _, _, istage), tKsK_g2s(_, _, _, istage));
        cp_async_fence();
        ++gmem_read;
        ++smem_write;
    }

    cp_async_wait<stages - 2>();
    __syncthreads();

    int outer_tile = size<2>(gV);
    int inner_tile = size<2>(tKVrV);
    copy(s2r_copy_v, tVsV_s2r(make_coord(_, _0{}), _, _0{}, smem_read), tKVrV_s2r(make_coord(_, _0{}), _, _0{}));
    copy(s2r_copy_k, tKsK_s2r(_, _, _0{}, smem_read), tKVrK_s2r(_, _, _0{}));

    Tensor tK = group_modes<1, rank(tKVrK)>(tKVrK);
    const T _zero = T(0.0f);
    const T _one = T(1.0f);

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tK(make_coord(_0{}, _0{}), i) = tK(make_coord(_0{}, _0{}), i) > _zero ? tK(make_coord(_0{}, _0{}), i) : _zero;
        tK(make_coord(_1{}, _0{}), i) = tK(make_coord(_1{}, _0{}), i) > _zero ? tK(make_coord(_1{}, _0{}), i) : _zero;
        tK(make_coord(_0{}, _1{}), i) = tK(make_coord(_0{}, _1{}), i) > _zero ? tK(make_coord(_0{}, _1{}), i) : _zero;
        tK(make_coord(_1{}, _1{}), i) = tK(make_coord(_1{}, _1{}), i) > _zero ? tK(make_coord(_1{}, _1{}), i) : _zero;
    }
    int offset = (((threadIdx.x >> 6) << 3) + ((threadIdx.x & 3) << 1));
    bool flag = ((threadIdx.x & 63) < 4);
    if (flag) {
        tKVrV(make_coord(_0{}, _0{}, _0{}), _1{}, _0{}) = _one;
        tKVrV(make_coord(_1{}, _0{}, _0{}), _1{}, _0{}) = _one;
        tKVrV(make_coord(_0{}, _0{}, _1{}), _1{}, _0{}) = _one;
        tKVrV(make_coord(_1{}, _0{}, _1{}), _1{}, _0{}) = _one;
        tKVrV(make_coord(_0{}, _0{}, _0{}), _1{}, _1{}) = _one;
        tKVrV(make_coord(_1{}, _0{}, _0{}), _1{}, _1{}) = _one;
        tKVrV(make_coord(_0{}, _0{}, _1{}), _1{}, _1{}) = _one;
        tKVrV(make_coord(_1{}, _0{}, _1{}), _1{}, _1{}) = _one;
    }
    for (int outer_iter = 0; outer_iter < outer_tile; ++outer_iter) {
#pragma unroll
        for (int inner_iter = 0; inner_iter < inner_tile; ++inner_iter) {
            int next = (inner_iter + 1) % inner_tile;
            if (inner_iter == inner_tile - 1) {
                cp_async_wait<stages - 2>();
                __syncthreads();
                smem_read = (smem_read + 1) % stages;
            }
            copy(s2r_copy_v, tVsV_s2r(make_coord(_, _0{}), _, next, smem_read), tKVrV_s2r(make_coord(_, _0{}), _, next));
            copy(s2r_copy_k, tKsK_s2r(_, _, next, smem_read), tKVrK_s2r(_, _, next));
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int next_ = next * 2 + i;
                tK(make_coord(_0{}, _0{}), next_) = tK(make_coord(_0{}, _0{}), next_) > _zero ? tK(make_coord(_0{}, _0{}), next_) : _zero;
                tK(make_coord(_1{}, _0{}), next_) = tK(make_coord(_1{}, _0{}), next_) > _zero ? tK(make_coord(_1{}, _0{}), next_) : _zero;
                tK(make_coord(_0{}, _1{}), next_) = tK(make_coord(_0{}, _1{}), next_) > _zero ? tK(make_coord(_0{}, _1{}), next_) : _zero;
                tK(make_coord(_1{}, _1{}), next_) = tK(make_coord(_1{}, _1{}), next_) > _zero ? tK(make_coord(_1{}, _1{}), next_) : _zero;
            }
            if (inner_iter == 0) {
                if (gmem_read < outer_tile) {
                    copy_if(g2s_copy, tVpV(_, _, _, gmem_read), tVgV_g2s(_, _, _, gmem_read), tVsV_g2s(_, _, _, smem_write));
                    copy_if(g2s_copy, tKpK(_, _, _, gmem_read), tKgK_g2s(_, _, _, gmem_read), tKsK_g2s(_, _, _, smem_write));
                    ++gmem_read;
                    smem_write = (smem_write + 1) % stages;
                }
                cp_async_fence();
            }
            gemm(mma, tKVrKV, tKVrV(_, _, inner_iter), tKVrK(_, _, inner_iter), tKVrKV);
        }
    }

    auto sKV = make_tensor(sV(_, _, smem_read).data(), SmemLayoutKV{});
    auto r2s_copy = make_tiled_copy_C(R2SCopyAtom{}, mma);
    auto r2s_copy_thr = r2s_copy.get_slice(threadIdx.x);
    auto tKVrKV_r2s = r2s_copy_thr.retile_S(convert_type<T>(tKVrKV));
    auto tKVsKV_r2s = r2s_copy_thr.partition_D(sKV);
    __syncthreads();
    copy(r2s_copy, tKVrKV_r2s, tKVsKV_r2s);

    auto tOrKV = tKVrV;
    auto tOrQ = tKVrK;
    auto tOrO = tKVrKV;
    clear(tOrO);

    auto tQgQ_g2s = g2s_copy_thr.partition_S(gQ); // (cpy, cpy_m, cpy_k, k_repeats)
    auto tQsQ_g2s = g2s_copy_thr.partition_D(sQ); // (cpy, cpy_m, cpy_k, stage)
    auto tQpQ = tKpK;

    auto s2r_copy_kv = make_tiled_copy_A(S2RCopyAtom_N{}, mma);
    auto s2r_copy_q = make_tiled_copy_B(S2RCopyAtom_T{}, mma);
    auto s2r_copy_kv_thr = s2r_copy_kv.get_slice(threadIdx.x);
    auto s2r_copy_q_thr = s2r_copy_q.get_slice(threadIdx.x);
    auto tKVsKV_s2r = s2r_copy_kv_thr.partition_S(sKV);
    auto tQsQt_s2r = s2r_copy_q_thr.partition_S(sQt);
    auto tOrKV_s2r = s2r_copy_kv_thr.retile_D(tOrKV);
    auto tOrQ_s2r = s2r_copy_q_thr.retile_D(tOrQ);

    Tensor O = make_tensor(
        make_gmem_ptr((T*)Optr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, Int<1>{})
    );
    Tensor cO = make_identity_tensor(shape(O));
    Tensor gO = local_tile(O, make_tile(Int<head_dim>{}, Int<tile_n>{}), make_coord(_0{}, _));
    Tensor cta_cO = local_tile(cO, make_tile(Int<head_dim>{}, Int<tile_n>{}), make_coord(_0{}, _));

    auto r2g_copy_o = make_tiled_copy_C(R2GCopyAtom{}, mma);
    auto r2g_copy_thr = r2g_copy_o.get_slice(threadIdx.x);
    auto tOgO_r2g = r2g_copy_thr.partition_D(gO);
    auto tOcO = r2g_copy_thr.partition_D(cta_cO);
    auto tOpO = cute::lazy::transform(tOcO, [&](auto c) { return elem_less(c, shape(O)); });

    float *s_proj = reinterpret_cast<float*>(smem_data + cosize(SmemLayoutK{}) + cosize(SmemLayoutV{}));

    gmem_read = 0;
    smem_read = 0;
    smem_write = 0;
#pragma unroll
    for (int istage = 0; istage < stages - 1; ++istage) {
        copy_if(g2s_copy, tQpQ(_, _, _, istage), tQgQ_g2s(_, _, _, istage), tQsQ_g2s(_, _, _, istage));
        cp_async_fence();
        ++gmem_read;
        ++smem_write;
    }

    cp_async_wait<stages - 2>();
    __syncthreads();

    outer_tile = size<2>(gQ);
    inner_tile = size<2>(tOrKV);
    copy(s2r_copy_kv, tKVsKV_s2r(_, _, _0{}), tOrKV_s2r(_, _, _0{}));
    copy(s2r_copy_kv, tKVsKV_s2r(_, _, _1{}), tOrKV_s2r(_, _, _1{}));
    copy(s2r_copy_q, tQsQt_s2r(_, _, _0{}, smem_read), tOrQ_s2r(_, _, _0{}));
    Tensor tQ = group_modes<1, rank(tOrQ)>(tOrQ);

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tQ(make_coord(_0{}, _0{}), i) = tQ(make_coord(_0{}, _0{}), i) > _zero ? tQ(make_coord(_0{}, _0{}), i) : _zero;
        tQ(make_coord(_1{}, _0{}), i) = tQ(make_coord(_1{}, _0{}), i) > _zero ? tQ(make_coord(_1{}, _0{}), i) : _zero;
        tQ(make_coord(_0{}, _1{}), i) = tQ(make_coord(_0{}, _1{}), i) > _zero ? tQ(make_coord(_0{}, _1{}), i) : _zero;
        tQ(make_coord(_1{}, _1{}), i) = tQ(make_coord(_1{}, _1{}), i) > _zero ? tQ(make_coord(_1{}, _1{}), i) : _zero;
    }

    Tensor tO = group_modes<1, rank(tOrO)>(tOrO);
    for (int outer_iter = 0; outer_iter < outer_tile; ++outer_iter) {
#pragma unroll
        for (int inner_iter = 0; inner_iter < inner_tile; ++inner_iter) {
            int next = (inner_iter + 1) % inner_tile;
            if (inner_iter == inner_tile - 1) {
                cp_async_wait<stages - 2>();
                __syncthreads();

                smem_read = (smem_read + 1) % stages;
            }
            copy(s2r_copy_q, tQsQt_s2r(_, _, next, smem_read), tOrQ_s2r(_, _, next));
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int next_ = next * 2 + i;
                tQ(make_coord(_0{}, _0{}), next_) = tQ(make_coord(_0{}, _0{}), next_) > _zero ? tQ(make_coord(_0{}, _0{}), next_) : _zero;
                tQ(make_coord(_1{}, _0{}), next_) = tQ(make_coord(_1{}, _0{}), next_) > _zero ? tQ(make_coord(_1{}, _0{}), next_) : _zero;
                tQ(make_coord(_0{}, _1{}), next_) = tQ(make_coord(_0{}, _1{}), next_) > _zero ? tQ(make_coord(_0{}, _1{}), next_) : _zero;
                tQ(make_coord(_1{}, _1{}), next_) = tQ(make_coord(_1{}, _1{}), next_) > _zero ? tQ(make_coord(_1{}, _1{}), next_) : _zero;
            }
            if (inner_iter == 0) {
                if (gmem_read < outer_tile) {
                    copy_if(g2s_copy, tQpQ(_, _, _, gmem_read), tQgQ_g2s(_, _, _, gmem_read), tQsQ_g2s(_, _, _, smem_write));
                    ++gmem_read;
                    smem_write = (smem_write + 1) % stages;
                }
                cp_async_fence();
            }
            gemm(mma, tOrO, tOrKV(_, _, inner_iter), tOrQ(_, _, inner_iter), tOrO);
        }

        if (flag) {
            s_proj[offset] = tO(make_coord(_0{}, _0{}), _1{});
            s_proj[offset + 1] = tO(make_coord(_1{}, _0{}), _1{});
            s_proj[offset + 16] = tO(make_coord(_0{}, _0{}), _3{});
            s_proj[offset + 17] = tO(make_coord(_1{}, _0{}), _3{});
        }
        __syncthreads();

        float4 proj;
        proj.x = s_proj[offset] + eps;
        proj.y = s_proj[offset + 1] + eps;
        proj.z = s_proj[offset + 16] + eps;
        proj.w = s_proj[offset + 17] + eps;

        tO(make_coord(_0{}, _0{}), _0{}) = __fdividef(tO(make_coord(_0{}, _0{}), _0{}), proj.x);
        tO(make_coord(_1{}, _0{}), _0{}) = __fdividef(tO(make_coord(_1{}, _0{}), _0{}), proj.y);
        tO(make_coord(_0{}, _1{}), _0{}) = __fdividef(tO(make_coord(_0{}, _1{}), _0{}), proj.x);
        tO(make_coord(_1{}, _1{}), _0{}) = __fdividef(tO(make_coord(_1{}, _1{}), _0{}), proj.y);
        tO(make_coord(_0{}, _0{}), _2{}) = __fdividef(tO(make_coord(_0{}, _0{}), _2{}), proj.z);
        tO(make_coord(_1{}, _0{}), _2{}) = __fdividef(tO(make_coord(_1{}, _0{}), _2{}), proj.w);
        tO(make_coord(_0{}, _1{}), _2{}) = __fdividef(tO(make_coord(_0{}, _1{}), _2{}), proj.z);
        tO(make_coord(_1{}, _1{}), _2{}) = __fdividef(tO(make_coord(_1{}, _1{}), _2{}), proj.w);

        auto tOrO_r2g = r2g_copy_thr.retile_S(convert_type<T>(tOrO));
        copy_if(r2g_copy_o, tOpO(_, _, _, outer_iter), tOrO_r2g, tOgO_r2g(_, _, _, outer_iter));

        clear(tOrO);
    }
}

template <typename LiteMLATraits_C_16>
__global__ void litemla_C_16(
          void* Optr,
    const void* Qptr,
    const void* Kptr,
    const void* Vptr,
    const int bs,
    const int head,
    const float eps
) {
    using T = typename LiteMLATraits_C_16::T;
    using TiledMMA = typename LiteMLATraits_C_16::MMA;
    using G2SCopy = typename LiteMLATraits_C_16::G2SCopy;
    using S2RCopyAtom_N = typename LiteMLATraits_C_16::S2RCopyAtom_N;
    using S2RCopyAtom_x2_N = typename LiteMLATraits_C_16::S2RCopyAtom_x2_N;
    using SmemLayoutK = typename LiteMLATraits_C_16::SmemLayoutK;
    using SmemLayoutV = typename LiteMLATraits_C_16::SmemLayoutV;
    using R2SCopyAtom = typename LiteMLATraits_C_16::R2SCopyAtom;
    using SmemLayoutKV = typename LiteMLATraits_C_16::SmemLayoutKV;

    using SmemLayoutQ = typename LiteMLATraits_C_16::SmemLayoutQ;
    using SmemLayoutQtransposed = typename LiteMLATraits_C_16::SmemLayoutQtransposed;
    using SmemLayoutQtransposedNoSwizzle = typename LiteMLATraits_C_16::SmemLayoutQtransposedNoSwizzle;
    using S2RCopyAtom_T = typename LiteMLATraits_C_16::S2RCopyAtom_T;
    using S2RCopyAtom_x2_T = typename LiteMLATraits_C_16::S2RCopyAtom_x2_T;

    // using R2GCopyAtom = typename LiteMLATraits_C_16::R2GCopyAtom;
    using SmemLayoutO = typename LiteMLATraits_C_16::SmemLayoutO;
    using S2GCopy = typename LiteMLATraits_C_16::S2GCopy;

    // m32n64k384 -> m32n64k32 -> m32n64k16
    constexpr int seq_len = LiteMLATraits_C_16::seq_len;
    constexpr int head_dim = LiteMLATraits_C_16::head_dim;
    constexpr int stages = LiteMLATraits_C_16::stages;

    constexpr int tile_m = LiteMLATraits_C_16::tile_m;
    constexpr int tile_n = LiteMLATraits_C_16::tile_n;
    constexpr int tile_k = LiteMLATraits_C_16::tile_k;

    constexpr int out_n = LiteMLATraits_C_16::out_n;

    extern __shared__ T smem_data[];
    T *smem_k = smem_data;
    T *smem_v = smem_data + cosize(SmemLayoutK{});

    int global_offset = head_dim * seq_len * blockIdx.x;

    Tensor Q = make_tensor(
        make_gmem_ptr((T*)Qptr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, _1{}));
    Tensor K = make_tensor(
        make_gmem_ptr((T*)Kptr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, _1{}));
    Tensor V = make_tensor(
        make_gmem_ptr((T*)Vptr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, _1{}));

    Tensor cK = make_identity_tensor(shape(K));
    Tensor cV = make_identity_tensor(shape(V));

    Tensor gQ = local_tile(Q, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));
    Tensor gK = local_tile(K, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _)); 
    Tensor gV = local_tile(V, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));

    Tensor cta_cK = local_tile(cK, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));
    Tensor cta_cV = local_tile(cV, make_tile(Int<head_dim>{}, Int<tile_k>{}), make_coord(_0{}, _));

    auto sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{});

    auto sQ = make_tensor(make_smem_ptr(smem_k), SmemLayoutQ{});
    auto sQt = make_tensor(sQ.data(), SmemLayoutQtransposed{});
    auto sQtns = make_tensor(sQ.data().get(), SmemLayoutQtransposedNoSwizzle{});

    // v @ k ([32, 384] @ [384, 16])
    TiledMMA mma;
    // if (thread0()) print_latex(mma);
    auto mma_thr = mma.get_slice(threadIdx.x);
    auto tKVrV = mma_thr.partition_fragment_A(sV(_, _, 0));
    auto tKVrK = mma_thr.partition_fragment_B(sK(_, _, 0));
    auto tKVrKV = partition_fragment_C(mma, Shape<Int<tile_m>, Int<tile_n>>{});
    clear(tKVrKV);

    G2SCopy g2s_copy;
    auto g2s_copy_thr = g2s_copy.get_slice(threadIdx.x);
    auto tVgV_g2s = g2s_copy_thr.partition_S(gV); // (cpy, cpy_m, cpy_k, k_repeats)
    auto tKgK_g2s = g2s_copy_thr.partition_S(gK);
    auto tVsV_g2s = g2s_copy_thr.partition_D(sV); // (cpy, cpy_m, cpy_k, stage)
    auto tKsK_g2s = g2s_copy_thr.partition_D(sK);

    auto tVcV = g2s_copy_thr.partition_S(cta_cV);
    auto tKcK = g2s_copy_thr.partition_S(cta_cK);

    Tensor tVpV = cute::lazy::transform(tVcV, [&](auto c) { return elem_less(c, shape(V)); });
    Tensor tKpK = cute::lazy::transform(tKcK, [&](auto c) { return elem_less(c, shape(K)); });

    auto s2r_copy_v = make_tiled_copy_A(S2RCopyAtom_N{}, mma);
    auto s2r_copy_k = make_tiled_copy_B(S2RCopyAtom_x2_N{}, mma);
    auto s2r_copy_v_thr = s2r_copy_v.get_slice(threadIdx.x);
    auto s2r_copy_k_thr = s2r_copy_k.get_slice(threadIdx.x);
    auto tKsK_s2r = s2r_copy_k_thr.partition_S(sK);
    auto tVsV_s2r = s2r_copy_v_thr.partition_S(sV); // (cpy, cpy_m, cpy_k, stage)
    auto tKVrK_s2r = s2r_copy_k_thr.retile_D(tKVrK);
    auto tKVrV_s2r = s2r_copy_v_thr.retile_D(tKVrV);

    int gmem_read = 0;
    int smem_read = 0;
    int smem_write = 0;

#pragma unroll
    for (int istage = 0; istage < stages - 1; ++istage) {
        copy_if(g2s_copy, tVpV(_, _, _, istage), tVgV_g2s(_, _, _, istage), tVsV_g2s(_, _, _, istage));
        copy_if(g2s_copy, tKpK(_, _, _, istage), tKgK_g2s(_, _, _, istage), tKsK_g2s(_, _, _, istage));
        cp_async_fence();
        ++gmem_read;
        ++smem_write;
    }

    cp_async_wait<stages - 2>();
    __syncthreads();

    int outer_tile = size<2>(gV);
    int inner_tile = size<2>(tKVrV);
    const int warp = __shfl_sync(0xffffffff, threadIdx.x >> 5, 0);

    int offset = (((threadIdx.x >> 6) << 3) + ((threadIdx.x & 3) << 1));
    bool flag = ((warp == 1 || warp == 3) && ((threadIdx.x & 31) < 4));
    Tensor tV = group_modes<1, rank(tKVrV)>(tKVrV);
    const T _zero = T(0.0f);
    const T _one = T(1.0f);

    if (warp == 0 || warp == 2) copy(s2r_copy_v, tVsV_s2r(_, _, _0{}, smem_read), tKVrV_s2r(_, _, _0{}));
    else {
        if ((threadIdx.x & 31) < 4) {
            for (uint i = 0; i < inner_tile; ++i) {
                tV(make_coord(_0{}, _0{}, _0{}), i) = _one;
                tV(make_coord(_1{}, _0{}, _0{}), i) = _one;
                tV(make_coord(_0{}, _0{}, _1{}), i) = _one;
                tV(make_coord(_1{}, _0{}, _1{}), i) = _one;
                tV(make_coord(_0{}, _0{}, _0{}), i) = _one;
                tV(make_coord(_1{}, _0{}, _0{}), i) = _one;
                tV(make_coord(_0{}, _0{}, _1{}), i) = _one;
                tV(make_coord(_1{}, _0{}, _1{}), i) = _one;
            }
        }
    }
    copy(s2r_copy_k, tKsK_s2r(_, _, _0{}, smem_read), tKVrK_s2r(_, _, _0{}));

    Tensor tK = group_modes<1, rank(tKVrK)>(tKVrK);

    tK(make_coord(_0{}, _0{}), 0) = tK(make_coord(_0{}, _0{}), 0) > _zero ? tK(make_coord(_0{}, _0{}), 0) : _zero;
    tK(make_coord(_1{}, _0{}), 0) = tK(make_coord(_1{}, _0{}), 0) > _zero ? tK(make_coord(_1{}, _0{}), 0) : _zero;
    tK(make_coord(_0{}, _1{}), 0) = tK(make_coord(_0{}, _1{}), 0) > _zero ? tK(make_coord(_0{}, _1{}), 0) : _zero;
    tK(make_coord(_1{}, _1{}), 0) = tK(make_coord(_1{}, _1{}), 0) > _zero ? tK(make_coord(_1{}, _1{}), 0) : _zero;

    for (int outer_iter = 0; outer_iter < outer_tile; ++outer_iter) {
#pragma unroll
        for (int inner_iter = 0; inner_iter < inner_tile; ++inner_iter) {
            int next = (inner_iter + 1) % inner_tile;
            if (inner_iter == inner_tile - 1) {
                cp_async_wait<stages - 2>();
                __syncthreads();
                smem_read = (smem_read + 1) % stages;
            }
            if (warp == 0 || warp == 2) copy(s2r_copy_v, tVsV_s2r(_, _, next, smem_read), tKVrV_s2r(_, _, next));
            copy(s2r_copy_k, tKsK_s2r(_, _, next, smem_read), tKVrK_s2r(_, _, next));

            tK(make_coord(_0{}, _0{}), next) = tK(make_coord(_0{}, _0{}), next) > _zero ? tK(make_coord(_0{}, _0{}), next) : _zero;
            tK(make_coord(_1{}, _0{}), next) = tK(make_coord(_1{}, _0{}), next) > _zero ? tK(make_coord(_1{}, _0{}), next) : _zero;
            tK(make_coord(_0{}, _1{}), next) = tK(make_coord(_0{}, _1{}), next) > _zero ? tK(make_coord(_0{}, _1{}), next) : _zero;
            tK(make_coord(_1{}, _1{}), next) = tK(make_coord(_1{}, _1{}), next) > _zero ? tK(make_coord(_1{}, _1{}), next) : _zero;
            if (inner_iter == 0) {
                if (gmem_read < outer_tile) {
                    copy_if(g2s_copy, tVpV(_, _, _, gmem_read), tVgV_g2s(_, _, _, gmem_read), tVsV_g2s(_, _, _, smem_write));
                    copy_if(g2s_copy, tKpK(_, _, _, gmem_read), tKgK_g2s(_, _, _, gmem_read), tKsK_g2s(_, _, _, smem_write));
                    ++gmem_read;
                    smem_write = (smem_write + 1) % stages;
                }
                cp_async_fence();
            }
            gemm(mma, tKVrKV, tKVrV(_, _, inner_iter), tKVrK(_, _, inner_iter), tKVrKV);
        }
    }

    auto sKV = make_tensor(sV(_, _, smem_read).data(), SmemLayoutKV{});
    auto r2s_copy_kv = make_tiled_copy_C(R2SCopyAtom{}, mma);
    auto r2s_copy_thr_kv = r2s_copy_kv.get_slice(threadIdx.x);
    auto tKVrKV_r2s = r2s_copy_thr_kv.retile_S(convert_type<T>(tKVrKV));
    auto tKVsKV_r2s = r2s_copy_thr_kv.partition_D(sKV);
    __syncthreads();
    copy(r2s_copy_kv, tKVrKV_r2s, tKVsKV_r2s);

    auto tOrKV = tKVrV;
    auto tOrQ = tKVrK;
    auto tOrO = tKVrKV;
    clear(tOrO);

    auto tQgQ_g2s = g2s_copy_thr.partition_S(gQ); // (cpy, cpy_m, cpy_k, k_repeats)
    auto tQsQ_g2s = g2s_copy_thr.partition_D(sQ); // (cpy, cpy_m, cpy_k, stage)
    auto tQpQ = tKpK;

    auto s2r_copy_kv = make_tiled_copy_A(S2RCopyAtom_N{}, mma);
    auto s2r_copy_q = make_tiled_copy_B(S2RCopyAtom_x2_T{}, mma);
    auto s2r_copy_kv_thr = s2r_copy_kv.get_slice(threadIdx.x);
    auto s2r_copy_q_thr = s2r_copy_q.get_slice(threadIdx.x);
    auto tKVsKV_s2r = s2r_copy_kv_thr.partition_S(sKV);
    auto tQsQt_s2r = s2r_copy_q_thr.partition_S(sQt);
    auto tOrKV_s2r = s2r_copy_kv_thr.retile_D(tOrKV);
    auto tOrQ_s2r = s2r_copy_q_thr.retile_D(tOrQ);

    Tensor O = make_tensor(
        make_gmem_ptr((T*)Optr + global_offset),
        make_shape(Int<head_dim>{}, Int<seq_len>{}),
        make_stride(Int<seq_len>{}, Int<1>{})
    );
    Tensor cO = make_identity_tensor(shape(O));
    Tensor gO = local_tile(O, make_tile(Int<head_dim>{}, Int<out_n>{}), make_coord(_0{}, _));
    Tensor cta_cO = local_tile(cO, make_tile(Int<head_dim>{}, Int<out_n>{}), make_coord(_0{}, _));

    auto sO = make_tensor(sV(_, _, _0{}).data(), SmemLayoutO{});

    Tensor tO = group_modes<1, rank(tOrO)>(tOrO);
    Tensor rO = make_tensor_like<T>(tOrO);

    auto r2s_copy_o = make_tiled_copy_C(R2SCopyAtom{}, mma);
    auto r2s_copy_thr_o = r2s_copy_o.get_slice(threadIdx.x);
    auto tOrO_r2s = r2s_copy_thr_o.retile_S(rO);
    auto tOsO_r2s = r2s_copy_thr_o.partition_D(sO);


    S2GCopy s2g_copy;
    auto s2g_copy_thr = s2g_copy.get_slice(threadIdx.x);
    auto tOsO_s2g = s2g_copy_thr.partition_S(sO);
    auto tOgO_s2g = s2g_copy_thr.partition_D(gO);
    auto tOcO = s2g_copy_thr.partition_D(cta_cO);
    auto tOpO = cute::lazy::transform(tOcO, [&](auto c) { return elem_less(c, shape(O)); });

    float *s_proj = reinterpret_cast<float*>(smem_data + cosize(SmemLayoutK{}) + cosize(SmemLayoutV{}));

    gmem_read = 0;
    smem_read = 0;
    smem_write = 0;
#pragma unroll
    for (int istage = 0; istage < stages - 1; ++istage) {
        copy_if(g2s_copy, tQpQ(_, _, _, istage), tQgQ_g2s(_, _, _, istage), tQsQ_g2s(_, _, _, istage));
        cp_async_fence();
        ++gmem_read;
        ++smem_write;
    }

    cp_async_wait<stages - 2>();
    __syncthreads();

    outer_tile = size<2>(gQ);
    inner_tile = size<2>(tOrKV);
    copy(s2r_copy_kv, tKVsKV_s2r(_, _, _0{}), tOrKV_s2r(_, _, _0{}));
    copy(s2r_copy_q, tQsQt_s2r(_, _0{}, _, smem_read), tOrQ_s2r(_, _, _0{}));

    Tensor tQ = group_modes<1, rank(tOrQ)>(tOrQ);

    tQ(make_coord(_0{}, _0{}), 0) = tQ(make_coord(_0{}, _0{}), 0) > _zero ? tQ(make_coord(_0{}, _0{}), 0) : _zero;
    tQ(make_coord(_1{}, _0{}), 0) = tQ(make_coord(_1{}, _0{}), 0) > _zero ? tQ(make_coord(_1{}, _0{}), 0) : _zero;
    tQ(make_coord(_0{}, _1{}), 0) = tQ(make_coord(_0{}, _1{}), 0) > _zero ? tQ(make_coord(_0{}, _1{}), 0) : _zero;
    tQ(make_coord(_1{}, _1{}), 0) = tQ(make_coord(_1{}, _1{}), 0) > _zero ? tQ(make_coord(_1{}, _1{}), 0) : _zero;

    for (int outer_iter = 0; outer_iter < outer_tile; ++outer_iter) {
#pragma unroll
        for (int inner_iter = 0; inner_iter < inner_tile; ++inner_iter) {
            int next = (inner_iter + 1) % inner_tile;
            if (inner_iter == inner_tile - 1) {
                cp_async_wait<stages - 2>();
                __syncthreads();

                smem_read = (smem_read + 1) % stages;
            }
            copy(s2r_copy_q, tQsQt_s2r(_, next, _, smem_read), tOrQ_s2r(_, _, next));
            tQ(make_coord(_0{}, _0{}), next) = tQ(make_coord(_0{}, _0{}), next) > _zero ? tQ(make_coord(_0{}, _0{}), next) : _zero;
            tQ(make_coord(_1{}, _0{}), next) = tQ(make_coord(_1{}, _0{}), next) > _zero ? tQ(make_coord(_1{}, _0{}), next) : _zero;
            tQ(make_coord(_0{}, _1{}), next) = tQ(make_coord(_0{}, _1{}), next) > _zero ? tQ(make_coord(_0{}, _1{}), next) : _zero;
            tQ(make_coord(_1{}, _1{}), next) = tQ(make_coord(_1{}, _1{}), next) > _zero ? tQ(make_coord(_1{}, _1{}), next) : _zero;
            if (inner_iter == 0) {
                if (gmem_read < outer_tile) {
                    copy_if(g2s_copy, tQpQ(_, _, _, gmem_read), tQgQ_g2s(_, _, _, gmem_read), tQsQ_g2s(_, _, _, smem_write));
                    ++gmem_read;
                    smem_write = (smem_write + 1) % stages;
                }
                cp_async_fence();
            }
            gemm(mma, tOrO, tOrKV(_, _, _0{}), tOrQ(_, _, inner_iter), tOrO);

            if (flag) {
                s_proj[offset] = tO(make_coord(_0{}, _0{}), _0{});
                s_proj[offset + 1] = tO(make_coord(_1{}, _0{}), _0{});
            }
            __syncthreads();

            float2 proj;
            proj.x = s_proj[offset] + eps;
            proj.y = s_proj[offset + 1] + eps;

            tO(make_coord(_0{}, _0{}), _0{}) = __fdividef(tO(make_coord(_0{}, _0{}), _0{}), proj.x);
            tO(make_coord(_1{}, _0{}), _0{}) = __fdividef(tO(make_coord(_1{}, _0{}), _0{}), proj.y);
            tO(make_coord(_0{}, _1{}), _0{}) = __fdividef(tO(make_coord(_0{}, _1{}), _0{}), proj.x);
            tO(make_coord(_1{}, _1{}), _0{}) = __fdividef(tO(make_coord(_1{}, _1{}), _0{}), proj.y);

            convert_type_out(tOrO, rO);
            if (warp == 0 || warp == 2) copy(r2s_copy_o, tOrO_r2s, tOsO_r2s(_, _, inner_iter));

            clear(tOrO);
        }
        copy_if(s2g_copy, tOpO(_, _, _, outer_iter), tOsO_s2g, tOgO_s2g(_, _, _, outer_iter));
    }
}

} // namespace efficientvit::litemla