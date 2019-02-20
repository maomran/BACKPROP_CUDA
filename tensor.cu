#include "tensor.h"

__global__ void tensorAdd(const float *A, const float *B, float *C, int m, int n)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n)
    {
        C[row*m + col] = A[row*m + col] + B[row*m + col];
    }
}

__global__ void tensorSub(const float *A, const float *B, float *C, int m, int n)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n)
    {
        C[row*m + col] = A[row*m + col] - B[row*m + col];
    }
}


__global__
void kScale(float *a, float factor, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] *= factor;
    }
}

__global__
void kMultiply(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
               float* A, int aX, int aY,
               float* B, int bX, int bY,
               float* C){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    float sum = 0;
    if( col < k && row < m) 
    {
        for( i = 0; i < n; i++) 
        {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }

}
