#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace efficientvit::litemla {

using namespace cute;

// [32, 368] @ [368, 33] => [32, 384] @ [384, 64] => [32, 96] @ [96, 64] => [32, 64]
// [368, 32] @ [32, 33] => [384, 32] @ [32, 64]
// q [384, 32] @ k [32, 384] @ v [384, 64] => [384,64]
template <typename T_, int seq_len_, int head_dim_, int stages_>
struct LiteMLATraits {
    using T = T_;
    static constexpr int seq_len = seq_len_;
    // static_assert(seq_len == 368, "seq_len must be equal to 368");
    static constexpr int head_dim = head_dim_;
    // static_assert(head_dim == 32, "head_dim must be equal to 32");
    static constexpr int stages = stages_;

// v @ k_trans = [64, 384] @ [384, 32]
    static constexpr int tile_m = 64;
    static constexpr int tile_n = 32;
    static constexpr int tile_k = 32;

    // mma
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int thr_m_repeats = 2;
    static constexpr int thr_n_repeats = 2;
    static constexpr int thr_k_repeats = 1;
    static constexpr int thr = 32 * thr_m_repeats * thr_n_repeats;
    using MMAThrLayout = decltype(make_layout(make_shape(
        Int<thr_m_repeats>{}, Int<thr_n_repeats>{}, Int<thr_k_repeats>{}
    )));

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int mma_m = 2 * thr_m_repeats * get<0>(mma_atom_shape{}); // 64
    static constexpr int mma_n = 2 * thr_n_repeats * get<1>(mma_atom_shape{}); // 32
    static constexpr int mma_k = 1 * thr_k_repeats * get<2>(mma_atom_shape{}); // 16
    using MMAPermLayout = Tile<Int<mma_m>, Int<mma_n>, Int<mma_k>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMAThrLayout{}, MMAPermLayout{}));

    // smem
    static constexpr int swizzle_b = 3;
    static constexpr int swizzle_m = 3;
    static constexpr int swizzle_s = 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<swizzle_b, swizzle_m, swizzle_s>{},
        make_layout(make_shape(Int<8>{}, Int<tile_k>{}), make_stride(Int<tile_k>{}, Int<1>{}))
    ));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<head_dim>{}, Int<tile_k>{}, Int<stages>{})
    ));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<head_dim>{}, Int<tile_k>{}, Int<stages>{})
    ));

    // gmem to smem
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopy = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // smem to register
    using s2r_copy_n_op = SM75_U32x4_LDSM_N;
    using s2r_copy_n_traits = Copy_Traits<s2r_copy_n_op>;
    using S2RCopyAtom_N = Copy_Atom<s2r_copy_n_traits, T>;

    using R2SCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>;

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<tile_m>{}, Int<tile_n>{})
    ));

// vk @ q = [64, 32] @ [32, 384]
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<head_dim>{}, Int<tile_k>{}, Int<stages>{})
    ));
    using SmemLayoutAtomQtransposed = decltype(
        composition(SmemLayoutQ{}, make_layout(Shape<Int<tile_k>, Int<head_dim>>{}, GenRowMajor{}))
    );
    using SmemLayoutQtransposed = decltype(tile_to_shape(
        SmemLayoutAtomQtransposed{},
        make_shape(Int<tile_k>{}, Int<head_dim>{}, Int<stages>{})
    ));
    using SmemLayoutQtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQtransposed{}));

    using s2r_copy_t_op = SM75_U16x8_LDSM_T;
    using s2r_copy_t_traits = Copy_Traits<s2r_copy_t_op>;
    using S2RCopyAtom_T = Copy_Atom<s2r_copy_t_traits, T>;

    using R2GCopyAtom = Copy_Atom<UniversalCopy<int>, T>;

    static constexpr int smem_size = (cosize(SmemLayoutK{}) + cosize(SmemLayoutV{})) * sizeof(T) + Int<head_dim>{} * sizeof(float);
};

template <typename T_, int seq_len_, int head_dim_, int stages_>
struct LiteMLATraits_C_16 {
    using T = T_;
    static constexpr int seq_len = seq_len_;
    // static_assert(seq_len == 368, "seq_len must be equal to 368");
    static constexpr int head_dim = head_dim_;
    // static_assert(head_dim == 32, "head_dim must be equal to 32");
    static constexpr int stages = stages_;

// v @ k_trans = [32, 384] @ [384, 16]
    static constexpr int tile_m = 32;
    static constexpr int tile_n = 16;
    static constexpr int tile_k = 64;
    // static constexpr int q_tile_k = 16;

    // mma
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int thr_m_repeats = 2;
    static constexpr int thr_n_repeats = 2;
    static constexpr int thr_k_repeats = 1;
    static constexpr int thr = 32 * thr_m_repeats * thr_n_repeats;
    using MMAThrLayout = decltype(make_layout(make_shape(
        Int<thr_m_repeats>{}, Int<thr_n_repeats>{}, Int<thr_k_repeats>{}
    )));

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int mma_m = 1 * thr_m_repeats * get<0>(mma_atom_shape{}); // 32
    static constexpr int mma_n = 1 * thr_n_repeats * get<1>(mma_atom_shape{}); // 16
    static constexpr int mma_k = 1 * thr_k_repeats * get<2>(mma_atom_shape{}); // 16
    using MMAPermLayout = Tile<Int<mma_m>, Int<mma_n>, Int<mma_k>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMAThrLayout{}, MMAPermLayout{}));

    // smem
    static constexpr int swizzle_b = 3;
    static constexpr int swizzle_m = 3;
    static constexpr int swizzle_s = 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<swizzle_b, swizzle_m, swizzle_s>{},
        make_layout(make_shape(Int<8>{}, Int<tile_k>{}), make_stride(Int<tile_k>{}, Int<1>{}))
    ));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<head_dim>{}, Int<tile_k>{}, Int<stages>{})
    ));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<head_dim>{}, Int<tile_k>{}, Int<stages>{})
    ));

    // gmem to smem
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopy = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // smem to register
    using s2r_copy_n_op = SM75_U32x4_LDSM_N;
    using s2r_copy_n_traits = Copy_Traits<s2r_copy_n_op>;
    using S2RCopyAtom_N = Copy_Atom<s2r_copy_n_traits, T>;
    using S2RCopyAtom_x2_N = Copy_Atom<Copy_Traits<SM75_U32x2_LDSM_N>, T>;

    using R2SCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, T>;
    using SmemLayoutAtomKV = decltype(composition(
        Swizzle<Int<3>{}, Int<3>{}, Int<3>{}>{},
        make_layout(make_shape(Int<8>{}, Int<tile_n>{}), make_stride(Int<tile_n>{}, Int<1>{}))
    ));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        make_shape(Int<tile_m>{}, Int<tile_n>{})
    ));

// vk @ q = [32, 16] @ [16, 384]
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<head_dim>{}, Int<tile_k>{}, Int<stages>{})
    ));
    using SmemLayoutAtomQtransposed = decltype(
        composition(SmemLayoutQ{}, make_layout(Shape<Int<tile_k>, Int<head_dim>>{}, GenRowMajor{}))
    );
    using SmemLayoutQtransposed = decltype(tile_to_shape(
        SmemLayoutAtomQtransposed{},
        make_shape(Int<tile_k>{}, Int<head_dim>{}, Int<stages>{})
    ));
    using SmemLayoutQtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQtransposed{}));

    using s2r_copy_t_op = SM75_U16x8_LDSM_T;
    using s2r_copy_t_traits = Copy_Traits<s2r_copy_t_op>;
    using S2RCopyAtom_T = Copy_Atom<s2r_copy_t_traits, T>;
    using S2RCopyAtom_x2_T = Copy_Atom<Copy_Traits<SM75_U16x4_LDSM_T>, T>;

    // using R2GCopyAtom = Copy_Atom<UniversalCopy<int>, T>;
    using SmemLayoutAtomO = decltype(composition(
        Swizzle<Int<3>{}, Int<3>{}, Int<3>{}>{},
        make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}))));
    static constexpr int out_n = 64;
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        // make_shape(Int<head_dim>{}, Int<tile_n>{}, Int<out_n / tile_n>{})
        make_shape(Int<head_dim>{}, Int<out_n>{})
    ));
    using S2GCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopy = decltype(make_tiled_copy(
        S2GCopyAtom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    static constexpr int smem_size = (max(cosize(SmemLayoutK{}), cosize(SmemLayoutQ{})) + max(cosize(SmemLayoutV{}), cosize(SmemLayoutO{}))) * sizeof(T) + Int<head_dim>{} * sizeof(float);
};

} // namespace efficientvit::litemla
