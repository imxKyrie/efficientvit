#include <cutlass/layout/tensor.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cute/numeric/numeric_types.hpp>
#include <nvbench/nvbench.cuh>

#include "litemla_api.h"

template<typename ElementType>
void fused_litemla(nvbench::state& state, nvbench::type_list<ElementType>) {
    using HostTensor = cutlass::HostTensor<ElementType, cutlass::layout::TensorNHWC>;
    const int B = state.get_int64("B");
    const int H = state.get_int64("H");
    const int L = state.get_int64("L");
    const int D = state.get_int64("D");
    const int stages = state.get_int64("stages");
    HostTensor Q{{B, H, D, L}}, K{{B, H, D, L}}, V{{B, H, D + 1, L}}, O{{B, H, D, L}};
    cutlass::reference::host::TensorFillRandomUniform(Q.host_view(), /*seed=*/3407);
    cutlass::reference::host::TensorFillRandomUniform(K.host_view(), /*seed=*/3407);
    cutlass::reference::host::TensorFillRandomUniform(V.host_view(), /*seed=*/3407);
    Q.sync_device();
    K.sync_device();
    V.sync_device();
    O.sync_device();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        efficientvit::litemla::litemla_cuda(
            /*Optr=*/O.device_data(),
            /*Qptr=*/Q.device_data(),
            /*Kptr=*/K.device_data(),
            /*Vptr=*/V.device_data(),
            /*bs=*/B,
            /*head=*/H,
            /*seq_len=*/L,
            /*head_dim=*/D,
            /*stages=*/stages,
            /*eps=*/1e-5,
            /*stream=*/launch.get_stream()
        );
    });
}

using cts_types = nvbench::type_list<cute::half_t>;

NVBENCH_BENCH_TYPES(fused_litemla, NVBENCH_TYPE_AXES(cts_types))
    .add_int64_axis("B", {3})
    .add_int64_axis("H", {12, 24})
    .add_int64_axis("L", {368, 1472})
    .add_int64_axis("D", {32})
    .add_int64_axis("stages", {3, 4, 5, 6});