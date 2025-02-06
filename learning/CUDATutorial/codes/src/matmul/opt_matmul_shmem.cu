#include <stdio.h>
#include "matmul.cuh"

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

void run_matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

/**
 * @brief optimized matrix multiplication kernel with share memory
 *        which means that instead of use all of A, B, C in global memory,
 *        we use sharded matrix to store A, B, C
 * 
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @param M A's column size
 * @param N B's row size
 * @param K A's row size or B's column size
 */
template <const uint BLOCK_SIZE>
__global__ void matmul_shmem_kernel(float *A, float *B, float *C, int M, int N, int K) {
    /**
     * @brief In this design, we use BLOCK_SIZE * BLOCK_SIZE threads in a block,
     * and each thead calculate a element in C, which means that each thread need to traverse whole row of A and column of B,
     * luckily we have BLOCK_SIZE * BLOCK_SIZE share memory, thus can reduce global memory access times to K / BLOCK_SIZE,
     * to balance the memory access cost, each threads load one element from A and B to share memory,
     * the outer loop switch Share Memory to traverse A's row and B's column
     * So, we use (M / BLOCK_SIZE) * (N / BLOCK_SIZE) blocks totally as grid
     * 
     */
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    A += cRow * BLOCK_SIZE * K;
    B += cCol * BLOCK_SIZE;
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];

    const uint threadRow = threadIdx.x / BLOCK_SIZE;
    const uint threadCol = threadIdx.x % BLOCK_SIZE;

    float tmp = 0.0f;
    // traverse blocks in A's row and B's column
    for (int blkIdx = 0; blkIdx < K; blkIdx += BLOCK_SIZE) {
        sA[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
        sB[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++) {
            tmp += sA[threadRow * BLOCK_SIZE + dotIdx] * sB[dotIdx * BLOCK_SIZE + threadCol];
        }
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }

    C[threadRow * N + threadCol] = tmp;
}

void run_matmul_shmem_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const uint BLOCK_SIZE = 32;
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE, 1, 1);

    matmul_shmem_kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

int main() {
    int M = (1 << 10);
    int N = (1 << 6);
    int K = (1 << 8);
    int nBytes_A = M * K * sizeof(float);
    int nBytes_B = K * N * sizeof(float);
    int nBytes_C = M * N * sizeof(float);

    float *a, *b, *c_ref, *c_shmem;
    float *d_a, *d_b, *d_c_ref, *d_c_shmem;

    a = (float *)malloc(nBytes_A);
    b = (float *)malloc(nBytes_B);
    c_ref = (float *)malloc(nBytes_C);
    c_shmem = (float *)malloc(nBytes_C);

    cudaMalloc(&d_a, nBytes_A);
    cudaMalloc(&d_b, nBytes_B);
    cudaMalloc(&d_c_ref, nBytes_C);
    cudaMalloc(&d_c_shmem, nBytes_C);

    randomize_matrix(a, M, K);
    randomize_matrix(b, K, N);

    cudaMemcpy(d_a, a, nBytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes_B, cudaMemcpyHostToDevice);

    run_matmul_kernel(d_a, d_b, d_c_ref, M, N, K);
    cudaMemcpy(c_ref, d_c_ref, nBytes_C, cudaMemcpyDeviceToHost);

    run_matmul_shmem_kernel(d_a, d_b, d_c_shmem, M, N, K);
    cudaMemcpy(c_shmem, d_c_shmem, nBytes_C, cudaMemcpyDeviceToHost);

    diff_matrix(c_ref, c_shmem, M, N);

    free(a);
    free(b);
    free(c_ref);
    free(c_shmem);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_ref);
    cudaFree(d_c_shmem);

    return 0;
}