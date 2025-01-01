 #使用jit方式编译时，使用本文件通过以下命令执行：
#python3 run.py --compiler jit

from torch.utils.cpp_extension import load
cuda_module = load(name="vadd",
                           extra_include_paths=["include"],
                           sources=["vadd_ops.cpp", "vadd_kernel.cu"],
                           verbose=True)

cuda_module.torch_launch_add2(cuda_c, a, b, n)
