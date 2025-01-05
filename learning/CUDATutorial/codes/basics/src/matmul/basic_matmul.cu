#include <stdio.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

/**
 * @brief matrix multiplication kernel
 * 
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @param M A's column size
 * @param N B's row size
 * @param K A's row size or B's column size
 */
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = (1 << 10);
    int N = (1 << 10);
    int K = (1 << 10);
    int nBytes_A = M * K * sizeof(float);
    int nBytes_B = K * N * sizeof(float);
    int nBytes_C = M * N * sizeof(float);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float *)malloc(nBytes_A);
    b = (float *)malloc(nBytes_B);
    c = (float *)malloc(nBytes_C);

    cudaMalloc(&d_a, nBytes_A);
    cudaMalloc(&d_b, nBytes_B);
    cudaMalloc(&d_c, nBytes_C);

    for (int i = 0; i < M * K; i++) {
        a[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = 1.0f;
    }
    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }

    cudaMemcpy(d_a, a, nBytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes_B, cudaMemcpyHostToDevice);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // 32x32 = 1024 threads per block
    dim3 blockDim(32, 32, 1);

    matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(c, d_c, nBytes_C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f * %f = %f\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}