#include <cuda_runtime.h>
#include <helper_cuda.h>
#ifndef KERNELS_H
__global__ void kAdd(float *A, float *B, int m, int n)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < m )
    {
        for (int i = 0; i < n; ++i)
        {
        A[i*m + row] += B[row];
            /* code */
        }
    }
}

__global__ void kSub(float *A, float *B, int m, int n)
{
   int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col & n) {
        A[m*col + row] -= B[m*col + row];
    }
}


__global__ void kScale(float *A, float factor, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n){
        A[col*m + row] *= factor;
    }
}

__global__
void kMul(float *A, float *B, float *C, int m, int n, int k){
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

#define TILE_WIDTH 32
__global__ void kMulShared(float * A, float * B, float * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
        tx = threadIdx.x, ty = threadIdx.y,
        Row = by * TILE_WIDTH + ty,
        Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

__global__
void kGradMul(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                           float* A, int aX, int aY,
                           float* B, int bX, int bY,
                           float* C)
{
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(bX, blockStartX + fieldsPerBlockX);
    int blockEndY = min(aX, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < bY; i++) {
                sum += A[i*aX + y] * B[i*bX + x];
            }
            C[y*bX + x] = sum;
        }
    }
}

__global__
void kGradMulShared(float * A, float * B, float * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
        tx = threadIdx.x, ty = threadIdx.y,
        Row = by * TILE_WIDTH + ty,
        Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

__global__
void kBackMul(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                              float* A, int m, int aY,
                              float* B, int n, int bY,
                              float* C)
{
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(bY, blockStartX + fieldsPerBlockX);
    int blockEndY = min(aY, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int col = startY; col < endY; col++) {
        for (int row = startX; row < endX; row++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) {
                sum += A[col*m + i] * B[row*n + i];
            }
            C[col*bY + row] = sum;
        }
    }
}

__global__
void kAvg(float* A,float* B, int m, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < m) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[i*m + col];
        }
        B[col] = sum / n;
    }
}

#else

__global__ void kSoftMax(float *output, int m, int n, float* labels, float* y) {
        float sum = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j){
            sum += exp(output[i*n+j]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j){
            y[i*n + j] = (exp(output[i*n + j]) / sum) - labels[i*n + j];
            }
        }
}

__global__ void kSoftMaxAccuracy(float *output, int m, int n, float* labels, float* accuracy) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        int labelsloc = 1;
        int outputloc = 1;
        float *maxlabel = labels;
        float *maxoutput = output;
        for (int i = 1; i < m; i++) {
            if(labels[row*m + i] > *maxlabel){
                *maxlabel = labels[row*m + i];
                labelsloc = i+1;
            }
            if(output[row*m + i] > *maxoutput){
                *maxoutput = output[row*m + i];
                outputloc = i+1;
            }
        }
            if (labelsloc == outputloc)
                    atomicAdd(accuracy, 1);
    }
}
#endif
