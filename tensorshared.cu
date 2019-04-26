#include "tensor.h"
#include "kernels.h"
#include <iostream>
#include <cstdlib>
cudaError_t err = cudaSuccess;

tensor::tensor(int row, int col) {
    this->row = row;
    this->col = col;
    err = cudaMalloc((void **)&(this->d_data), this->row*this->col*sizeof(float));
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector at line 131(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

tensor::tensor(int row, int col, float** h_data) {
    this->row = row;
    this->col = col;
    if (this->row && this->col) {
        err = cudaMalloc((void **)&(this->d_data), this->row*this->col*sizeof(float));
        if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector A at line 149(error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

        err = cudaMemcpy(this->d_data, *h_data, 
            this->row*this->col*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device at line 329(error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    } 
    else {
        this->d_data = NULL;
    }
}

tensor::tensor(int row, int col, float* h_data) {
    this->row = row;
    this->col = col;
        cudaMalloc((void **)&(this->d_data), this->row*sizeof(float));
        cudaMemcpy(this->d_data, h_data, this->row*sizeof(float), cudaMemcpyHostToDevice);
}

tensor::~tensor(){
    cudaFree(this->d_data);

}

float** tensor::Dev2Host() {
    float** h_data = new float*[this->col];
    *h_data = new float[this->col * this->row];
    for (int i = 1; i < this->col; i++) 
        h_data[i] = h_data[i-1] + this->row;
    err = cudaMemcpy(*h_data, this->d_data, this->row*this->col*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host at line 121(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return h_data;
}


void tensor::Add(tensor* input) {
    if (this->row != input->row ) {
        printf("ERROR! Cannot Add matrix with size %dx%d to matrix %dx%d.\n",
               input->row, input->row, this->row, this->col);
        exit(1);
    }

    dim3 dimBlock(32,1);
    dim3 dimGrid((this->row + dimBlock.x)/dimBlock.x,1);
    kAdd<<<dimGrid, dimBlock>>>(this->d_data, input->d_data, this->row, this->col);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch Add kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void tensor::Subtract(tensor* input) {
    if (this->row != input->row && this->col != input->col ) {
        printf("ERROR! Cannot sub matrix with size %dx%d to matrix %dx%d.\n",
               input->row, input->row, this->row, this->col);
        exit(1);
    }
    dim3 dimBlock(32,32);
    dim3 dimGrid((this->row + dimBlock.x)/dimBlock.x,(this->col + dimBlock.y)/dimBlock.y);
    kSub<<<dimGrid, dimBlock>>>(this->d_data, input->d_data, this->row, this->col);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch sub kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}


void tensor::Scale(float factor) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((this->row + dimBlock.x)/dimBlock.x,
                   (this->col + dimBlock.y)/dimBlock.y);
    kScale<<<dimGrid, dimBlock>>>(this->d_data, factor, this->row, this->col);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch Scale kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


tensor* tensor::MatMul(tensor* input, tensor* output) {
    if (this->col != input->row) {
        printf("ERROR! Cannot MatMul matrices with shape %dx%d and %dx%d.\n",
               this->row, this->col, input->row, input->col);
        exit(1);
    }

    dim3 dimBlock(32, 32);
    dim3 dimGrid((output->col + dimBlock.x)/dimBlock.x,
                   (output->row + dimBlock.y)/dimBlock.y);
        kMulShared<<<dimGrid, dimBlock>>>(
            this->d_data, input->d_data,output->d_data,
            this->row, this->col, input->row,input->col
            , output->row,output->col
        );
    err = cudaGetLastError();
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch MatMul kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    return output;
}

tensor* tensor::GradientMul(tensor* input, tensor* output) {
    if (this->col != input->col) {
        printf("ERROR! Cannot MatMul matrices with shape %dx%d and %dx%d.\n",
               this->row, this->col, input->row, input->col);
        exit(1);
    }
    int threadsX = 32;
    int threadsY = 32;

    int blocksX =  (input->row + threadsX) / threadsX;
    int blocksY =  (this->row + threadsY) / threadsY;
    int fieldsPerBlockX =  (input->row + blocksX) / blocksX;
    int fieldsPerThreadX =  (fieldsPerBlockX + threadsX) / threadsX;
    int fieldsPerBlockY =  (this->row + blocksY) / blocksY;
    int fieldsPerThreadY =  (fieldsPerBlockY + threadsY) / threadsY;
        dim3 dimBlock(32, 32);
        dim3 dimGrid(blocksX, blocksY);

        kGradMul<<<dimGrid, dimBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->d_data, this->row, this->col,
            input->d_data, input->row, input->col,
            output->d_data);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch GradientMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);

    }

    return output;
}
tensor* tensor::BackwardMul(tensor* input, tensor* output) {
    if (this->row != input->row) {
        printf("ERROR! Cannot MatMul matrices with shape %dx%d and %dx%d.\n",
               this->row, this->col, input->row, input->col);
        exit(1);
    }
 
        int threadsX = 32;
        int threadsY = 32;
        int blocksX =  (input->col + threadsX) / threadsX;
        int blocksY =  (this->col + threadsY) / threadsY;
        int fieldsPerBlockX =  (input->col + blocksX) / blocksX;
        int fieldsPerThreadX =  (fieldsPerBlockX + threadsX) / threadsX;
        int fieldsPerBlockY =  (this->col + blocksY) / blocksY;
        int fieldsPerThreadY =  (fieldsPerBlockY + threadsY) / threadsY;
        dim3 dimBlock(threadsX, threadsY);
        dim3 dimGrid(blocksX, blocksY);
        kBackMul<<<dimGrid, dimBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->d_data, this->row, this->col,
            input->d_data, input->row, input->col,
            output->d_data);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch BackwardMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return output;
}

tensor* tensor::GradAvg(tensor* output) {
    int dimBlock = 32;
    int dimGrid = (this->row + dimBlock)/dimBlock;
    kAvg<<<dimGrid, dimBlock>>>(this->d_data, output->d_data, this->row, this->col);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch GradAvg kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return output;
}

void tensor::toString() {
    float** values = this->Dev2Host();
    for (int y = 0; y < this->col; y++) {
        for (int x = 0; x < this->row; x++) {
            printf("%f; ", values[y][x]);
        }
        printf("\n");
    }
    delete[] values;
}

