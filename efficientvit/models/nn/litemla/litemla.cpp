#include <torch/library.h>
#include <ATen/Tensor.h>          // for at::empty_like
#include <ATen/ops/empty_like.h>  // for at::empty_like
#include <c10/cuda/CUDAStream.h>  // for c10::cuda::getCurrentCUDAStream
#include <cuda_runtime.h>         // for cudaStream_t
#include "litemla_api.h"          // for litemla_cuda

at::Tensor attn(
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    int64_t stages,
    double  eps
) {
    TORCH_CHECK(Q.is_cuda(), "Query tensor must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "Key tensor must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "Value tensor must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Query tensor must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "Key tensor must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "Value tensor must be contiguous");
    int64_t bs = Q.size(0);
    int64_t heads = Q.size(1);
    int64_t head_dim = Q.size(2);
    int64_t seq_len = Q.size(3);
    int dev = Q.get_device();
    at::Tensor O = at::empty_like(Q);
    int err = efficientvit::litemla::litemla_cuda(
        O.data_ptr(),
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        bs,
        heads,
        seq_len,
        head_dim,
        stages,
        eps,
        at::cuda::getCurrentCUDAStream(dev)
    );
    if (err == -1) {
        AT_ERROR("LiteMLA GEMM failed: CUDA error");
    }
    return O;
}

TORCH_LIBRARY_FRAGMENT(efficientvit, m) {
  // clang-format off
  m.def("litemla_attn(Tensor Q, Tensor K, Tensor V, int stages=3, float eps=1e-5) -> Tensor");
  // clang-format on
}

TORCH_LIBRARY_IMPL(efficientvit, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("litemla_attn"), TORCH_FN(attn));
}