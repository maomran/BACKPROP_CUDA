#include "tensor.h"
#include "utils.h"

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
void tensorScale(float *A, float scale, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n){
        A[col*m + row] *= scale;
    }
}

__global__
void tensorMul(float *A, float *B, float *C, int m, int n, int k){
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

tensor::tensor(int row, int col) {
    this->row = row;
    this->col = col;
    if (this->row && this->col) {
        cudaMalloc((void **)&(this->d_data), this->row*this->col*sizeof(float));
    } else {
        this->d_data = NULL;
    }
}

tensor::tensor(int row, int col, float** h_data) {
    this->row = row;
    this->col = col;
    if (this->row && this->col) {
        cudaMalloc((void **)&(this->d_data), this->row*this->col*sizeof(float));
        cudaMemcpy(this->d_data, *h_data, 
            this->row*this->col*sizeof(float), cudaMemcpyHostToDevice);
    } 
    else {
        this->d_data = NULL;
    }
}

tensor::tensor(int row, int col, float* d_data) {
    this->row = row;
    this->col = col;
    this->d_data = d_data;
}

tensor::~tensor() {
    cudaFree(this->d_data);
}

// int tensor::getSize(tensorAxis axis) {
//     if (axis == X) {
//         return this->row;
//     } else if (axis == Y) {
//         return this->col;
//     }
//     return -1;
// }

float* tensor::Host2Dev() {
    return this->d_data;
}

float** tensor::Dev2Host() {
    float** h_data = new float*[this->col];
    *h_data = new float[this->col * this->row];
    for (int i = 1; i < this->col; i++) 
        h_data[i] = h_data[i-1] + this->row;
    cudaMemcpy(*h_data, this->d_data, this->row*this->col*sizeof(float), cudaMemcpyDeviceToHost);
    return h_data;
}


tensor* tensor::add(tensor* tensor_t, tensor* output) {
    if (this->row != tensor_t->row || this->col != tensor_t->col) {
        printf("ERROR! Cannot add matrix with size %dx%d to matrix %dx%d.\n",
               tensor_t->row, tensor_t->row, this->row, this->col);
        exit(1);
    }

    dim3 dimBlock(TIDX, TIDY);
    dim3 dimGrid((this->row + dimBlock.x)/dimBlock.x,
                   (this->col + dimBlock.y)/dimBlock.y);
    tensorAdd<<<dimGrid, dimBlock>>>(this->Host2Dev(), tensor_t->Host2Dev(),output->Host2Dev, this->row, this->col);
    return output;
}

tensor* tensor::subtract(tensor* tensor_t) {
    if (this->row != tensor_t->row || this->col != tensor_t->col) {
        printf("ERROR! Cannot sub matrix with size %dx%d to matrix %dx%d.\n",
               tensor_t->row, ten sor_t->row, this->row, this->col);
        exit(1);
    }
    dim3 dimBlock(TIDX, TIDY);
    dim3 dimGrid((this->row + dimBlock.x)/dimBlock.x,
                   (this->col + dimBlock.y)/dimBlock.y);
    tensorSub<<<dimGrid, dimBlock>>>(this->Host2Dev(), tensor_t->Host2Dev(),output->Host2Dev, this->row, this->col);
    return output;
}


void tensor::scale(float factor) {
    dim3 dimBlock(TIDX, TIDY);
    dim3 dimGrid((this->row + dimBlock.x)/dimBlock.x,
                   (this->col + dimBlock.y)/dimBlock.y);
    tensorScale<<<dimGrid, dimBlock>>>(this->Host2Dev(), factor, this->row, this->col);
}


tensor* tensor::multiply(tensor* tensor_t, tensor* output) {
    if (this->row != tensor_t->col) {
        printf("ERROR! Cannot multiply matrices with shape %dx%d and %dx%d.\n",
               this->row, this->col, tensor_t->row, tensor_t->col);
        exit(1);
    }


        // Prepare configuration for CUDA kernel
        int threadsX = TIDX;
        int threadsY = TIDX;
        int blocksX = Configuration::tensorMultiplyBlockNumber == -1
                       ? (tensor_t->row + threadsX) / threadsX
                       : Configuration::tensorMultiplyBlockNumber;
        int blocksY = Configuration::tensorMultiplyBlockNumber == -1
                       ? (this->col + threadsY) / threadsY
                       : Configuration::tensorMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor_t->col + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->col + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 dimBlock(threadsX, threadsY);
        dim3 dimGrid(blocksX, blocksY);

        // Defer calculations on GPU
        kMultiply<<<dimGrid, dimBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->Host2Dev(), this->row, this->col,
            tensor_t->Host2Dev(), tensor_t->row, tensor_t->col,
            output->Host2Dev()
        );
    return output;
}

// tensor* tensor::multiplyByTransposition(tensor* tensor, tensor* output) {
//     // Check sizes and exit program in case of invalid multiplication
//     if (this->row != tensor_t->row) {
//         printf("ERROR! Cannot multiply matrix with shape %dx%d by transposition of matrix %dx%d.\n",
//                this->row, this->col, tensor_t->row, tensor_t->col);
//         exit(1);
//     }

//     // In case of using shared memory, we've got to use dynamic amount of blocks
//     if (Configuration::tensorMultiplySharedMemory == 1) {
//         // Prepare configuration for CUDA kernel
//         dim3 dimBlock(TIDX, TIDX);
//         dim3 dimGrid((tensor_t->col + dimBlock.x)/dimBlock.x,
//                        (this->col + dimBlock.y)/dimBlock.y);
//         int sharedMemorySize = 2 * dimBlock.y * dimBlock.x * sizeof(float);

//         // Defer calculations on GPU
//         kMultiplyByTranspositionWithSharedMemory<<<dimGrid, dimBlock, sharedMemorySize>>>(
//             this->Host2Dev(), this->row, this->col,
//             tensor_t->Host2Dev(), tensor_t->row, tensor_t->col,
//             output->Host2Dev()
//         );
//     } else {
//         // Prepare configuration for CUDA kernel
//         int threadsX = TIDX;
//         int threadsY = TIDX;
//         int blocksX = Configuration::tensorMultiplyBlockNumber == -1
//                        ? (tensor_t->col + threadsX) / threadsX
//                        : Configuration::tensorMultiplyBlockNumber;
//         int blocksY = Configuration::tensorMultiplyBlockNumber == -1
//                        ? (this->col + threadsY) / threadsY
//                        : Configuration::tensorMultiplyBlockNumber;
//         int fieldsPerBlockX = max(1, (tensor_t->col + blocksX) / blocksX);
//         int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
//         int fieldsPerBlockY = max(1, (this->col + blocksY) / blocksY);
//         int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
//         dim3 dimBlock(threadsX, threadsY);
//         dim3 dimGrid(blocksX, blocksY);

//         // Defer calculations on GPU
//         kMultiplyByTransposition<<<dimGrid, dimBlock>>>(
//             fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
//             this->Host2Dev(), this->row, this->col,
//             tensor_t->Host2Dev(), tensor_t->row, tensor_t->col,
//             output->Host2Dev()
//         );
//     }
//     return output;
// }

// tensor* tensor::transposeAndMultiply(tensor* tensor_t, tensor* output) {
//     // Check sizes and exit program in case of invalid multiplication
//     if (this->col != tensor_t->col) {
//         printf("ERROR! Cannot multiply transposition of matrix with shape %dx%d by matrix %dx%d.\n",
//                this->row, this->col, tensor_t->row, tensor_t->col);
//         exit(1);
//     }

//     // In case of using shared memory, we've got to use dynamic amount of blocks
//     if (Configuration::tensorMultiplySharedMemory == 1) {
//         // Prepare configuration for CUDA kernel
//         dim3 dimBlock(TIDX, TIDX);
//         dim3 dimGrid((tensor_t->row + dimBlock.x)/dimBlock.x,
//                        (this->row + dimBlock.y)/dimBlock.y);
//         int sharedMemorySize = 2 * dimBlock.y * dimBlock.x * sizeof(float);

//         // Defer calculations on GPU
//         kTransposeAndMultiplyWithSharedMemory<<<dimGrid, dimBlock, sharedMemorySize>>>(
//             this->Host2Dev(), this->row, this->col,
//             tensor_t->Host2Dev(), tensor_t->row, tensor_t->col,
//             output->Host2Dev()
//         );
//     } else {
//         // Prepare configuration for CUDA kernel
//         int threadsX = TIDX;
//         int threadsY = TIDX;
//         int blocksX = Configuration::tensorMultiplyBlockNumber == -1
//                        ? (tensor_t->row + threadsX) / threadsX
//                        : Configuration::tensorMultiplyBlockNumber;
//         int blocksY = Configuration::tensorMultiplyBlockNumber == -1
//                        ? (this->row + threadsY) / threadsY
//                        : Configuration::tensorMultiplyBlockNumber;
//         int fieldsPerBlockX = max(1, (tensor_t->row + blocksX) / blocksX);
//         int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
//         int fieldsPerBlockY = max(1, (this->row + blocksY) / blocksY);
//         int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
//         dim3 dimBlock(threadsX, threadsY);
//         dim3 dimGrid(blocksX, blocksY);

//         // Defer calculations on GPU
//         kTransposeAndMultiply<<<dimGrid, dimBlock>>>(
//             fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
//             this->Host2Dev(), this->row, this->col,
//             tensor_t->Host2Dev(), tensor_t->row, tensor_t->col,
//             output->Host2Dev()
//         );
//     }
//     return output;
// }

Tensor1D* tensor::avg(Tensor1D* output) {
    int dimBlock = TIDX;
    int dimGrid = (this->row + dimBlock)/dimBlock;
    kMeanX<<<dimGrid, dimBlock>>>(this->Host2Dev(), this->row, this->col, output->Host2Dev());
    return output;
}

void tensor::toString() {
    float** values = this->Dev2Host();
    for (int y = 0; y < this->col; y++) {
        for (int x = 0; x < this->row; x++) {
            printf("%8.5f; ", values[y][x]);
        }
        printf("\n");
    }
    delete[] values;
}

