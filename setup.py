#使用setup方式编译时，使用本文件通过以下命令执行：
#python3 setup.py install
'''
编译过程的核心是进行了以下操作：
1. nvcc -c vadd_kernel.cu -o vadd_kernel.o
2. g++ -c vadd_ops.cpp -o vadd.o
3. x86_64-linux-gnu-g++ -shared vadd.o vadd_kernel.o -o vadd.cpython-310m-x86_64-linux-gnu.so
这样就能生成动态链接库，同时将vadd添加为python的模块了，可以直接import vadd来调用
最后调用运行即可：
import torch
import vadd
vadd.torch_launch_add2(c, a, b, n)

'''




from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vadd",
    include_dirs=["include"], #指定相关头文件目录
    ext_modules=[  #指定算子和封装函数
        CUDAExtension( #调用Torch的CUDAExtension模块，将算子注册为vadd， 
            "vadd",
            ["vadd_ops.cpp", "vadd_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)


