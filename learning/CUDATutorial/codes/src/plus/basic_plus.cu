#include <stdio.h>

__global__ void add_kernel(float *x, float *y, float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        out[tid] = x[tid] + y[tid];
    }
}

int main() {
    int n = 1 << 20;
    int nBytes = n * sizeof(float);

    float *x, *y, *out;
    float *d_x, *d_y, *d_out;

    x = (float *)malloc(nBytes);
    y = (float *)malloc(nBytes);
    out = (float *)malloc(nBytes);

    cudaMalloc(&d_x, nBytes);
    cudaMalloc(&d_y, nBytes);
    cudaMalloc(&d_out, nBytes);

    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, nBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    add_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_out, n);

    cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", x[i], y[i], out[i]);
    }

    free(x);
    free(y);
    free(out);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    return 0;
}