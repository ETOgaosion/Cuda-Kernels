import cupy as cp
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time

# 定义 CUDA 核心函数
mod = SourceModule("""
__global__ void kernel_func(float *a) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    a[idx] = sinf(a[idx]);
}
""")

# CUDA 核心函数
kernel_func = mod.get_function("kernel_func")


def occupy_memory(ratio):
    """占用一定比例的显存"""
    total_mem = cp.cuda.Device(0).mem_info[1]  # 获取总显存
    mem_to_use = int(total_mem * ratio)        # 按比例计算需要占用的显存
    
    print(f"Total Memory: {total_mem / (1024**2)} MB")
    print(f"Memory to use: {mem_to_use / (1024**2)} MB")
    
    # 分配显存数组，占用显存
    _ = cp.zeros((mem_to_use // cp.float32().itemsize,), dtype=cp.float32)
    print("Memory allocated.")

def occupy_gpu_compute(ratio, array_size=1024*1024):
    """
    持续占用 GPU 一定比例的算力，通过减少并行线程数量。
    :param ratio: 期望占用的 GPU 算力比例 (0-1)
    :param array_size: 数组大小，控制计算任务的规模
    """
    # 分配 GPU 显存并初始化数据
    a = np.random.rand(array_size).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    # 计算使用的线程数和块数
    block_size = 64  # 每个线程块的线程数
    num_blocks = 32 * 80 * ratio  # 线程块数

    print(f"Running kernel with {num_blocks} blocks of {block_size} threads each.")

    # 持续运行计算任务
    while True:
        kernel_func(a_gpu, block=(block_size, 1, 1), grid=(num_blocks, 1))
        cuda.Context.synchronize()  # 同步 GPU 以确保完成


if __name__ == "__main__":
    # 获取用户输入的显存和算力占用比例
    memory_ratio = 0.2
    compute_ratio = 0.2
    # 占用指定比例的显存
    occupy_memory(memory_ratio)
    
    # 占用指定比例的 GPU 算力
    occupy_gpu_compute(compute_ratio)
