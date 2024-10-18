#include <iostream>
#include <cuda_runtime.h>

__global__ void dummyKernel(float *a) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    a[idx] = sinf(a[idx]);
}

void occupyGPU(int device, float gpuRatio, float memRatio) {
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t totalMem = prop.totalGlobalMem;
    size_t memToUse = static_cast<size_t>(totalMem * memRatio);

    // 分配显存
    float* d_mem;
    cudaMalloc((void**)&d_mem, memToUse);

    // 启动 CUDA 核心函数
    const int blockSize = 64;
    const int SMCount = prop.multiProcessorCount;
    const int numBlocks = 2;
    // const int numBlocks = 32 * SMCount * gpuRatio;

    while (true) {
        dummyKernel<<<numBlocks, blockSize>>>(d_mem);
        cudaDeviceSynchronize();
    }

    cudaFree(d_mem);
}

int main() {
    float gpuRatio = 0.4, memRatio = 0.4;
    int deviceIndex = 0;

    occupyGPU(deviceIndex, gpuRatio, memRatio);
    return 0;
}
