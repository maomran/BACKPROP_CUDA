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

void tensor::add(tensor* tensor_t) {
    if (this->row != tensor->getSize()) {
        printf("ERROR! Cannot add vector with size %d to matrix %dx%d.\n",
               tensor_t->getSize(), this->row, this->col);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensorAddBlockSize, Configuration::tensorAddBlockSize);
    dim3 numBlocks((this->row + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->col + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd1D<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor_t->getDeviceData(), this->row, this->col);
}

void tensor::add(tensor* tensor_t) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->row != tensor_t->getSize(X) || this->col != tensor_t->getSize(Y)) {
        printf("ERROR! Cannot add matrix with size %dx%d to matrix %dx%d.\n",
               tensor_t->getSize(X), tensor_t->getSize(Y), this->row, this->col);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensorAddBlockSize, Configuration::tensorAddBlockSize);
    dim3 numBlocks((this->row + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->col + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd2D<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor_t->getDeviceData(), this->row, this->col);
}

void tensor::subtract(tensor* tensor_t) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->row != tensor_t->getSize(X) || this->col != tensor_t->getSize(Y)) {
        printf("ERROR! Cannot subtract matrix with size %dx%d to matrix %dx%d.\n",
               tensor_t->getSize(X), tensor_t->getSize(Y), this->row, this->col);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensorSubtractBlockSize, Configuration::tensorSubtractBlockSize);
    dim3 numBlocks((this->row + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->col + threadsPerBlock.y)/threadsPerBlock.y);
    kSubtract<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor_t->getDeviceData(), this->row, this->col);
}

void tensor::scale(float factor) {
    dim3 threadsPerBlock(Configuration::tensorScaleBlockSize, Configuration::tensorScaleBlockSize);
    dim3 numBlocks((this->row + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->col + threadsPerBlock.y)/threadsPerBlock.y);
    kScale<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), factor, this->row, this->col);
}

tensor* tensor::multiply(tensor* tensor_t, tensor* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->row != tensor_t->getSize(Y)) {
        printf("ERROR! Cannot multiply matrices with shape %dx%d and %dx%d.\n",
               this->row, this->col, tensor_t->getSize(X), tensor_t->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensorMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensorMultiplyBlockSize, Configuration::tensorMultiplyBlockSize);
        dim3 numBlocks((tensor_t->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->col + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kMultiplyWithSharedMemory<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(
            this->getDeviceData(), this->row, this->col,
            tensor_t->getDeviceData(), tensor_t->getSize(X), tensor_t->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensorMultiplyBlockSize;
        int threadsY = Configuration::tensorMultiplyBlockSize;
        int blocksX = Configuration::tensorMultiplyBlockNumber == -1
                       ? (tensor_t->getSize(X) + threadsX) / threadsX
                       : Configuration::tensorMultiplyBlockNumber;
        int blocksY = Configuration::tensorMultiplyBlockNumber == -1
                       ? (this->col + threadsY) / threadsY
                       : Configuration::tensorMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor_t->getSize(Y) + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->getSize(Y) + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
        kMultiply<<<numBlocks, threadsPerBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->row, this->col,
            tensor_t->getDeviceData(), tensor_t->getSize(X), tensor_t->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

tensor* tensor::multiplyByTransposition(tensor* tensor, tensor* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->row != tensor_t->getSize(X)) {
        printf("ERROR! Cannot multiply matrix with shape %dx%d by transposition of matrix %dx%d.\n",
               this->row, this->col, tensor_t->getSize(X), tensor_t->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensorMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensorMultiplyBlockSize, Configuration::tensorMultiplyBlockSize);
        dim3 numBlocks((tensor_t->getSize(Y) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->col + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kMultiplyByTranspositionWithSharedMemory<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(
            this->getDeviceData(), this->row, this->col,
            tensor_t->getDeviceData(), tensor_t->getSize(X), tensor_t->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensorMultiplyBlockSize;
        int threadsY = Configuration::tensorMultiplyBlockSize;
        int blocksX = Configuration::tensorMultiplyBlockNumber == -1
                       ? (tensor_t->getSize(Y) + threadsX) / threadsX
                       : Configuration::tensorMultiplyBlockNumber;
        int blocksY = Configuration::tensorMultiplyBlockNumber == -1
                       ? (this->col + threadsY) / threadsY
                       : Configuration::tensorMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor_t->getSize(Y) + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->getSize(Y) + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
        kMultiplyByTransposition<<<numBlocks, threadsPerBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->row, this->col,
            tensor_t->getDeviceData(), tensor_t->getSize(X), tensor_t->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

tensor* tensor::transposeAndMultiply(tensor* tensor_t, tensor* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->col != tensor_t->getSize(Y)) {
        printf("ERROR! Cannot multiply transposition of matrix with shape %dx%d by matrix %dx%d.\n",
               this->row, this->col, tensor_t->getSize(X), tensor_t->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensorMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensorMultiplyBlockSize, Configuration::tensorMultiplyBlockSize);
        dim3 numBlocks((tensor_t->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->row + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kTransposeAndMultiplyWithSharedMemory<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(
            this->getDeviceData(), this->row, this->col,
            tensor_t->getDeviceData(), tensor_t->getSize(X), tensor_t->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensorMultiplyBlockSize;
        int threadsY = Configuration::tensorMultiplyBlockSize;
        int blocksX = Configuration::tensorMultiplyBlockNumber == -1
                       ? (tensor_t->getSize(X) + threadsX) / threadsX
                       : Configuration::tensorMultiplyBlockNumber;
        int blocksY = Configuration::tensorMultiplyBlockNumber == -1
                       ? (this->getSize(X) + threadsY) / threadsY
                       : Configuration::tensorMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor_t->getSize(X) + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->getSize(X) + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
        kTransposeAndMultiply<<<numBlocks, threadsPerBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->row, this->col,
            tensor_t->getDeviceData(), tensor_t->getSize(X), tensor_t->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor1D* tensor::meanX(Tensor1D* output) {
    int threadsPerBlock = Configuration::tensorMeanBlockSize;
    int numBlocks = (this->row + threadsPerBlock)/threadsPerBlock;
    kMeanX<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->row, this->col, output->getDeviceData());
    return output;
}

void tensor::debugPrint() {
    float** values = this->fetchDataFromDevice();
    for (int y = 0; y < this->col; y++) {
        for (int x = 0; x < this->row; x++) {
            printf("%8.5f; ", values[y][x]);
        }
        printf("\n");
    }
    delete[] values;
}

