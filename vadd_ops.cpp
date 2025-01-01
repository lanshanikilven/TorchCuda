/*
cuda kernel can't be invoked by pytorch directly, actually we need to
provide an interface which includes three steps below:

1. 先编写CUDA算子和对应的调用函数。
2. 然后编写xx_ops.cpp建立PyTorch和CUDA之间的联系，用pybind11封装。
3. 最后用PyTorch的cpp扩展库进行编译和调用。
*/


#include <torch/extension.h>
#include "vadd.h"

__global__ void MatAdd(float* c, const float* a, const float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j*n + i;
    if (i < n && j < n)
        c[idx] = a[idx] + b[idx];
}

void launch_add2(float* c, const float* a, const float* b, int n) {
    dim3 block(16, 16);
    dim3 grid(n/block.x, n/block.y);
    MatAdd<<<grid, block>>>(c, a, b, n);
}

//
void torch_launch_add2(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b, int64_t n) {
    launch_add2((float *)c.data_ptr(), (const float *)a.data_ptr(), (const float *)b.data_ptr(), n);
}

//用 pybind11 来对torch_launch_add2函数进行封装，然后用cmake编译就可以产生python可以调用的.so库。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2", &torch_launch_add2, "vadd kernel warpper");
}

TORCH_LIBRARY(vadd, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}