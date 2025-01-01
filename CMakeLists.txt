#使用cmakelist方式编译时，使用本文件通过以下命令执行：
#mkdir build
#cd build
#cmake ..
#make

#编写该文件需要注意的点是依赖库的匹配、编译过程及软连接的建立。
#编译完成后，会在build目录下生成一个libvadd.so，通过如下方式在python端调用：
#import torch
#torch.ops.load_library("build/libvadd.so")
#torch.ops.vadd.torch_launch_add2(c, a, b, n)


cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(vadd LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# modify to your own python path, or delete it if ok
include_directories(/usr/include/python3.10)
include_directories(include)

set(SRCS vadd_ops.cpp vadd_kernel.cu)
add_library(vadd SHARED ${SRCS})

target_link_libraries(vadd "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")




