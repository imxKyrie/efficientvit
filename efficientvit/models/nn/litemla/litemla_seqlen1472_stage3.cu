#include "litemla_kernel.cuh"

namespace efficientvit::litemla {

template __global__ void litemla<LiteMLATraits<cute::half_t, /*seqlen=*/1472, /*headdim=*/32, /*stages=*/3>>( \
    void*, const void*, const void*, const void*, const int, const int, const float);
template __global__ void litemla_C_16<LiteMLATraits_C_16<cute::half_t, /*seqlen=*/1472, /*headdim=*/16, /*stages=*/3>>( \
    void*, const void*, const void*, const void*, const int, const int, const float);

template __global__ void litemla<LiteMLATraits<cute::half_t, /*seqlen=*/1536, /*headdim=*/32, /*stages=*/3>>( \
    void*, const void*, const void*, const void*, const int, const int, const float);
template __global__ void litemla_C_16<LiteMLATraits_C_16<cute::half_t, /*seqlen=*/1536, /*headdim=*/16, /*stages=*/3>>( \
    void*, const void*, const void*, const void*, const int, const int, const float);

} // namespace efficientvit::litemla
