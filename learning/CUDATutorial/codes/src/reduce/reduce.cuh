#pragma once

#define FLOAT_SIZE sizeof(float)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void randomize_matrix(float *mat, int M, int N)
{
    for (int i = 0; i < M * N; i++)
    {
        mat[i] = rand() % 100;
    }
}

void diff_matrix(float *mat1, float *mat2, int M, int N)
{
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(mat1[i] - mat2[i]) > 1e-5)
        {
            printf("Error: mat1[%d] = %f, mat2[%d] = %f\n", i, mat1[i], i, mat2[i]);
            return;
        }
    }
    printf("Matrices are equal\n");
}