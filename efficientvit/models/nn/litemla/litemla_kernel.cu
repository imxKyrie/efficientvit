#include "litemla_traits.cuh"
#include "litemla_api.h"

namespace efficientvit::litemla {

template<typename Traits>
__global__ void litemla(void* Optr, const void* Qptr, const void* Kptr, const void* Vptr, const int bs, const int head, const float eps);

template<typename Traits_C_16>
__global__ void litemla_C_16(void* Optr, const void* Qptr, const void* Kptr, const void* Vptr, const int bs, const int head, const float eps);

int litemla_cuda(
          void* Optr,
    const void* Qptr,
    const void* Kptr,
    const void* Vptr,
    const int bs,
    const int head,
    const int seq_len,
    const int head_dim,
    const int stages,
    const float eps,
    cudaStream_t stream = 0
) {
    using T = cute::half_t;

    switch (seq_len) {
        case 368:
            switch (stages) {
                case 3: {
                    switch (head_dim) {
                        case 16: {
                            LiteMLATraits_C_16<T, 368, 16, 3> traits;
                            litemla_C_16<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        case 32: {
                            LiteMLATraits<T, 368, 32, 3> traits;
                            litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        default:
                            printf("Unsupported head_dim: %d\n", head_dim);
                            return -1;
                    }
                    break;
                }
                // case 4: {
                //     LiteMLATraits<T, 368, 32, 4> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // case 5: {
                //     LiteMLATraits<T, 368, 32, 5> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // case 6: {
                //     LiteMLATraits<T, 368, 32, 6> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // default:
                //     printf("Unsupported stages: %d\n", stages);
                //     return -1;
            }
            break;
        case 384:
            switch (stages) {
                case 3: {
                    switch (head_dim) {
                        case 16: {
                            LiteMLATraits_C_16<T, 384, 16, 3> traits;
                            litemla_C_16<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        case 32: {
                            LiteMLATraits<T, 384, 32, 3> traits;
                            litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        default:
                            printf("Unsupported head_dim: %d\n", head_dim);
                            return -1;
                    }
                    break;
                }
                // case 4: {
                //     LiteMLATraits<T, 368, 32, 4> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // case 5: {
                //     LiteMLATraits<T, 368, 32, 5> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // case 6: {
                //     LiteMLATraits<T, 368, 32, 6> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // default:
                //     printf("Unsupported stages: %d\n", stages);
                //     return -1;
            }
            break;
        case 1472:
            switch (stages) {
                case 3: {
                    switch (head_dim) {
                        case 16: {
                            LiteMLATraits_C_16<T, 1472, 16, 3> traits;
                            litemla_C_16<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        case 32: {
                            LiteMLATraits<T, 1472, 32, 3> traits;
                            litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        default:
                            printf("Unsupported head_dim: %d\n", head_dim);
                            return -1;
                    }
                    break;
                }
                // case 4: {
                //     LiteMLATraits<T, 1472, 32, 4> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // case 5: {
                //     LiteMLATraits<T, 1472, 32, 5> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                // case 6: {
                //     LiteMLATraits<T, 1472, 32, 6> traits;
                //     litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                //         (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                //     break;
                // }
                default:
                    printf("Unsupported stages: %d\n", stages);
                    return -1;
            }
            break;
        case 1536:
            switch (stages) {
                case 3: {
                    switch (head_dim) {
                        case 16: {
                            LiteMLATraits_C_16<T, 1536, 16, 3> traits;
                            litemla_C_16<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        case 32: {
                            LiteMLATraits<T, 1536, 32, 3> traits;
                            litemla<decltype(traits)><<<dim3(bs * head), traits.thr, traits.smem_size, stream>>>
                                (Optr, Qptr, Kptr, Vptr, bs, head, eps);
                            break;
                        }
                        default:
                            printf("Unsupported head_dim: %d\n", head_dim);
                            return -1;
                    }
                    break;
                }
                default:
                    printf("Unsupported stages: %d\n", stages);
                    return -1;
            }
            break;
        default:
            printf("Unsupported sequence length: %d\n", seq_len);
            return -1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // cudaDeviceSynchronize();
    return 1;
}

} // namespace efficientvit::litemla
