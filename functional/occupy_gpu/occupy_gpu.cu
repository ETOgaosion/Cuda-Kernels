#include <iostream>
#include <string>
#include <cuda_runtime.h>

__global__ void dummyKernel(float *a, int max_iter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (true) {
        a[idx] = sinf(a[idx]);
    }
}

void occupyGPU(int device, float gpuRatio, float memRatio, int max_iter) {
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
    const int total_threads = 2048 * SMCount * gpuRatio;
    int calculatedBlocks = total_threads / blockSize;
    if (calculatedBlocks < 4) {
        calculatedBlocks = 4;
    }
    const int numBlocks = (calculatedBlocks - (calculatedBlocks % 4));
    std::cout << "SM count: " << SMCount << ", numBlocks: " << numBlocks << std::endl;

    while (true) {
        dummyKernel<<<numBlocks, blockSize>>>(d_mem, max_iter);
        cudaDeviceSynchronize();
    }

    cudaFree(d_mem);
}

int main(int argc, char **argv) {
    float gpuRatio = 0.4, memRatio = 0.4;
    if (argc >= 3) {
        std::cout << "argc: " << argc << ", argv[1]: " << argv[1] << std::endl;
        gpuRatio = (float)std::stoi(argv[1]) / 100.0;
        memRatio = (float)std::stoi(argv[2]) / 100.0;
    }
    std::cout << "GPU ratio: " << gpuRatio << ", memory ratio: " << memRatio << std::endl;

    int deviceIndex = 0;
    if (argc >= 4) {
        deviceIndex = std::stoi(argv[3]);
    }

    int max_iter = 10;
    if (argc >= 5) {
        max_iter = std::stoi(argv[4]);
    }

    occupyGPU(deviceIndex, gpuRatio, memRatio, max_iter);
    return 0;
}
