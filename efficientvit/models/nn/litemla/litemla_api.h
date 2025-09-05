#include <cuda_runtime.h> // for cudaStream_t

namespace efficientvit::litemla {

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
    cudaStream_t stream
);

} // namespace efficientvit::litemla