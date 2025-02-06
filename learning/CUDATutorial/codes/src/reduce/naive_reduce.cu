#include <stdio.h>

#include "reduce.cuh"

template <int BLOCK_SIZE>
__global__ void reduce_naive_kernel(float *arr, float *out, int len) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int i = bid * bdim  + tid;
    if (i < len) {
        sdata[tid] = arr[i];
    }
    __syncthreads();
    for (int s = 1; s < bdim; s *= 2) {
        if (tid % (2 * s) == 0 && i + s < len) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[bid] = sdata[0];
    }
}

int main() {
    int len = 1024;
    float *arr = new float[len];
    float *out = new float[len];
    randomize_matrix(arr, 1, len);
    float ref_sum = 0.0f;
    for (int i = 0; i < len; i++) {
        ref_sum += arr[i];
    }
    float *d_arr, *d_out;
    cudaMalloc(&d_arr, len * sizeof(float));
    cudaMalloc(&d_out, len * sizeof(float));
    cudaMemcpy(d_arr, arr, len * sizeof(float), cudaMemcpyHostToDevice);

    const int block_size = 32;
    const int grid_size = (len + block_size - 1) / block_size;
    reduce_naive_kernel<block_size><<<grid_size, block_size>>>(d_arr, d_out, len);

    cudaMemcpy(out, d_out, len * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < grid_size; i++) {
        sum += out[i];
    }
    printf("Sum: %f, RefSum: %f\n", sum, ref_sum);
    delete[] arr;
    delete[] out;
    cudaFree(d_arr);
    cudaFree(d_out);
    return 0;
}