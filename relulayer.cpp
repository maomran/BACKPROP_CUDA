#include "relu.h"

__global__
void kReLu(float *A, int m, int n, float* B) {
    int row = blockIdx.row * blockDim.row + threadIdx.row;
    int col = blockIdx.col * blockDim.col + threadIdx.col;
    if (row < m && col < n) {
        if (A[col*m + row] < 0.0) {
            B[col*m + row] = 0.0;
        } else {
            B[col*m + row] = A[col*m + row];
        }
    }
}

ReluLayer::ReluLayer(int inputOutput) {
    this->input = this->output = inputOutput;
    this->weights = NULL;
    this->bias = NULL;
    this->deltaWeights = NULL;
    this->deltaBias = NULL;

    // Prepare output for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;
}

tensor* ReluLayer::forward(tensor* data) {
    this->inputData = data;

    if (!this->outputForward) {
        this->outputForward = new tensor(data->row, data->col);
    }

    dim3 dimBlock(TIDX, TIDY);
    dim3 dimGrid((data->row + dimBlock.x)/dimBlock.x,
                   (data->col + dimBlock.y)/dimBlock.y);
    kReLu<<<dimGrid, dimBlock>>>(
        data->data, data->row, data->col,
        this->outputForward->data
    );
    return this->outputForward;
}
 
tensor* ReluLayer::backward(tensor* gradients) {
    dim3 dimBlock(TIDX, TIDY);
    dim3 dimGrid((gradients->row + dimBlock.x)/dimBlock.x,
                   (gradients->col + dimBlock.y)/dimBlock.y);
    kReLu<<<dimGrid, dimBlock>>>(
        gradients->data, gradients->row, gradients->col,
        gradients->data
    );
    return new tensor(gradients->row, gradients->col, gradients->data);
}
