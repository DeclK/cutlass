# 确保找到CUDA 
find_package(CUDAToolkit REQUIRED)

cutlass_example_add_executable(
  00_cublas
  cublas_baseline.cu
)

# 链接cublas库
target_link_libraries(00_cublas 
  PRIVATE 
  CUDA::cublas  # 现代CMake的cublas目标
  # 或者直接指定库路径（旧式写法）
  # ${CUDAToolkit_LIBRARY_DIR}/libcublas.so 
)