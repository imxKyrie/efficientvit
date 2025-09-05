add_library(cutlass INTERFACE)

target_link_libraries(cutlass
    INTERFACE
        nvidia::cutlass::cutlass
        nvidia::cutlass::tools::util
)
target_compile_options(cutlass
    INTERFACE
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>"
)

target_compile_definitions(cutlass INTERFACE
    CUTLASS_ENABLE_TENSOR_CORE_MMA=1
)

target_compile_features(cutlass INTERFACE
    cuda_std_17
)

add_library(efficientvit-litemla-cuda
    SHARED
        litemla_kernel.cu
        litemla_seqlen368_stage3.cu
        litemla_seqlen368_stage4.cu
        litemla_seqlen368_stage5.cu
        litemla_seqlen368_stage6.cu
        litemla_seqlen1472_stage3.cu
        litemla_seqlen1472_stage4.cu
        litemla_seqlen1472_stage5.cu
        litemla_seqlen1472_stage6.cu
)

target_link_libraries(efficientvit-litemla-cuda
    PRIVATE
        cutlass
)

add_executable(benchmark-litemla benchmark.cu)

target_link_libraries(benchmark-litemla
    PRIVATE
        efficientvit-litemla-cuda
        nvbench::main
        cutlass
)
