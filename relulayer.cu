#include "relulayer.h"

__global__ void kReLu(float *A, int m, int n, float* B) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
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
    this->w = NULL;
    this->bias = NULL;
    this->w_gradient = NULL;
    this->b_gradient = NULL;

    // Prepare output for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;
}

tensor* ReluLayer::forward(tensor* data) {
    this->inputData = data;

    if (!this->outputForward) {
        this->outputForward = new tensor(data->row, data->col);
    }

    dim3 dimBlock(32, 32);
    dim3 dimGrid((data->row + dimBlock.x)/dimBlock.x,
                   (data->col + dimBlock.y)/dimBlock.y);
    kReLu<<<dimGrid, dimBlock>>>(data->DevData(), data->row, data->col,
        this->outputForward->DevData());
    return this->outputForward;
}
 
tensor* ReluLayer::backward(tensor* gradients) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((gradients->row + dimBlock.x)/dimBlock.x,
                   (gradients->col + dimBlock.y)/dimBlock.y);
    kReLu<<<dimGrid, dimBlock>>>(
        gradients->DevData(), gradients->row, gradients->col,
        gradients->DevData());
    return gradients;
}
