#include "sigmoidlayer.h"

__global__ void kSig(float *A, int m, int n, float* B) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
            B[col*m + row] = 1/ (1+ exp(-A[col*m + row]));
    }
}

SigmoidLayer::SigmoidLayer(int inputOutput) {
    this->input = this->output = inputOutput;
    this->w = NULL;
    this->bias = NULL;
    this->w_gradient = NULL;
    this->b_gradient = NULL;

    // Prepare output for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;
}

tensor* SigmoidLayer::forward(tensor* data) {
    this->inputData = data;

    if (!this->outputForward) {
        this->outputForward = new tensor(data->row, data->col);
    }
    // printf("reluin\n");
    // data->toString();

    dim3 dimBlock(32, 32);
    dim3 dimGrid((data->row + dimBlock.x)/dimBlock.x,
                   (data->col + dimBlock.y)/dimBlock.y);
    kSig<<<dimGrid, dimBlock>>>(data->d_data, data->row, data->col,
        this->outputForward->d_data);

    // printf("outputrelu\n");
    // outputForward->toString();

    // VERBOSE_PRINT("=== Layer %d ===\n", this);
    // VERBOSE_PRINT("Input Data = X: %d Y: %d\n", this->inputData->row, this->inputData->col);
    // VERBOSE_PRINT("Output = X: %d Y: %d\n", this->outputForward->row, this->outputForward->col);

    return this->outputForward;
}
 
tensor* SigmoidLayer::backward(tensor* gradients) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((gradients->row + dimBlock.x)/dimBlock.x,
                   (gradients->col + dimBlock.y)/dimBlock.y);
    kSig<<<dimGrid, dimBlock>>>(
        gradients->d_data, gradients->row, gradients->col,
        gradients->d_data);
    return gradients;
}
