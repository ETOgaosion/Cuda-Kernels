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
template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void matmul_shmem_thtile_1d_kernel(float *A, float *B, float *C, int M, int N, int K) {
    /**
     * @brief In this design, we use (BM * BN) / TM threads in a block,
     * and each thead calculate TM elements in C's column, which means that each thread need to traverse in TM rows of A and 1 column of B,
     * so, blockDim can be (BM / TM, BN)
     * we use (BM * BN) / TM share memory, and to balance the memory access cost, each threads load one element from A and B to share memory,
     * the outer loop switch Share Memory to traverse A's row and B's column
     * the middle loop calculate BK elements in share memory block
     * the inner loop calculate TM elements in C colomn
     * So, we use (M / BM) * (N / BN) blocks totally as grid
     * 
     */
    // for alignment, we require to use all threads to load data to share memory, there are BM * BK and BN * BK elements in A and B's share memory
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    // This indexes are only used for share memory loading
    const uint A_shmem_row = threadIdx.x / BK;
    const uint A_shmem_col = threadIdx.x % BK;
    const uint B_shmem_row = threadIdx.x / BN;
    const uint B_shmem_col = threadIdx.x % BN;

    float thread_res[TM] = {0.0f};
    // traverse blocks in A's row and B's column
    for (int blkIdx = 0; blkIdx < K; blkIdx += BK) {
        // Here we use all thread to load BM * BK of A and BK * BN of B to share memory
        sA[A_shmem_row * BK + A_shmem_col] = A[A_shmem_row * K + A_shmem_col];
        sB[B_shmem_row * BN + B_shmem_col] = B[B_shmem_row * N + B_shmem_col];
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float Btmp = sB[dotIdx * BN + threadCol];
            for (int tm = 0; tm < TM; tm++) {
                thread_res[tm] += sA[(threadRow * TM + tm) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for (int tm = 0; tm < TM; tm++) {
        C[(threadRow * TM + tm) * N + threadCol] = thread_res[tm];
    }
}

void run_matmul_shmem_thtile_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // requirements: BM * BK == blockDim.x, BN * BK == blockDim.x
    // because blockDim.x = BM * BN / TM
    // BN == BK * TM, BM == BK * TM
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1);
    dim3 blockDim(BM * BN / TM, 1, 1);

    matmul_shmem_thtile_1d_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

int main() {
    int M = (1 << 10);
    int N = (1 << 6);
    int K = (1 << 8);
    int nBytes_A = M * K * sizeof(float);
    int nBytes_B = K * N * sizeof(float);
    int nBytes_C = M * N * sizeof(float);

    float *a, *b, *c_ref, *c_shmem_thtile;
    float *d_a, *d_b, *d_c_ref, *d_c_shmem_thtile;

    a = (float *)malloc(nBytes_A);
    b = (float *)malloc(nBytes_B);
    c_ref = (float *)malloc(nBytes_C);
    c_shmem_thtile = (float *)malloc(nBytes_C);

    cudaMalloc(&d_a, nBytes_A);
    cudaMalloc(&d_b, nBytes_B);
    cudaMalloc(&d_c_ref, nBytes_C);
    cudaMalloc(&d_c_shmem_thtile, nBytes_C);

    randomize_matrix(a, M, K);
    randomize_matrix(b, K, N);

    cudaMemcpy(d_a, a, nBytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes_B, cudaMemcpyHostToDevice);

    run_matmul_kernel(d_a, d_b, d_c_ref, M, N, K);
    cudaMemcpy(c_ref, d_c_ref, nBytes_C, cudaMemcpyDeviceToHost);

    run_matmul_shmem_thtile_kernel(d_a, d_b, d_c_shmem_thtile, M, N, K);
    cudaMemcpy(c_shmem_thtile, d_c_shmem_thtile, nBytes_C, cudaMemcpyDeviceToHost);

    diff_matrix(c_ref, c_shmem_thtile, M, N);

    free(a);
    free(b);
    free(c_ref);
    free(c_shmem_thtile);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_ref);
    cudaFree(d_c_shmem_thtile);

    return 0;
}