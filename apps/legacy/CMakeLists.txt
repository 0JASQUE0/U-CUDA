add_executable(U-CUDA-legacy main_NonLinAnal.cu)
target_link_libraries(U-CUDA-legacy PRIVATE ucuda::cuda ucuda::platform CUDA::cudart)
ucuda_runtime(U-CUDA-legacy)
