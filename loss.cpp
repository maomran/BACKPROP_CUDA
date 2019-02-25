#include "loss.h"

#define VERY_SMALL_NUMBER 1e-10

__global__ void kSoftMaxCrossEntropy(float *output, int oX, int oY, float* labels, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < VERY_SMALL_NUMBER) {
            sum = VERY_SMALL_NUMBER;
        }

        // Softmax = exp(value) / sum(exp(allValues))
        // Subtract truth (which is one hot)
        for (int i = 0; i < oX; i++) {
            y[row*oX + i] = (exp(output[row*oX + i]) / sum) - labels[row*oX + i];
        }
    }
}

__global__ void kSoftMaxCrossEntropyLoss(float *output, int oX, int oY, float* labels, float* error) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < VERY_SMALL_NUMBER) {
            sum = VERY_SMALL_NUMBER;
        }

        // Error = target * log(softmaxOutput) + (1 - target) * log(1 - softmaxOutput)
        float tmpError = 0.0;
        for (int i = 0; i < oX; i++) {
            float softmaxOutput = exp(output[row*oX + i]) / sum;
            tmpError -= labels[row*oX + i] * log(softmaxOutput) + 
                        (1 - labels[row*oX + i]) * log(1 - softmaxOutput);
        }
        atomicAdd(error, tmpError);
    }
}

__global__ void kSoftMaxCrossEntropyAccuracy(float *output, int oX, int oY, float* labels, float* accuracy) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        int maxIdx = 0;
        float maxValue = output[row*oX];
        for (int x = 1; x < oX; x++) {
            if (output[row*oX + x] > maxValue) {
                maxIdx = x;
                maxValue = output[row*oX + x];
            }
        }
        if (output[row*oX + maxIdx] > 1.0 - VERY_SMALL_NUMBER) {
            atomicAdd(accuracy, 1);
        }
    }
}

CrossEntropyLoss::CrossEntropyLoss() {}

float CrossEntropyLoss::getLoss(tensor* networkOutput, tensor* labels) {
    float error = 0.0;
    float* dError;
    cudaMalloc((void**)&dError, sizeof(float));
    cudaMemcpy(dError, &error, sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock = Configuration::crossEntropyGetMetricBlockSize;
    dim3 dimGrid((networkOutput->col + dimBlock.x)/dimBlock.x);
    kSoftMaxCrossEntropyLoss<<<dimGrid, dimBlock>>>(networkOutput->data, networkOutput->row, networkOutput->col, labels->data, dError);
    cudaMemcpy(&error, dError, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dError);
    return error / networkOutput->col;
}

float CrossEntropyLoss::getAccuracy(tensor* networkOutput, tensor* labels) {
    float accuracy = 0.0;
    float* dAccuracy;
    cudaMalloc((void**)&dAccuracy, sizeof(float));
    cudaMemcpy(dAccuracy, &accuracy, sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock = Configuration::crossEntropyGetMetricBlockSize;
    dim3 dimGrid((networkOutput->col + dimBlock.x)/dimBlock.x);
    kSoftMaxCrossEntropyAccuracy<<<dimGrid, dimBlock>>>(networkOutput->data, networkOutput->row, networkOutput->col, labels->data, dAccuracy);
    cudaMemcpy(&accuracy, dAccuracy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dAccuracy);
    return 100.0 * accuracy / networkOutput->col;
}

tensor* CrossEntropyLoss::calculate(tensor* networkOutput, tensor* labels, tensor* output) {
    dim3 dimBlock = Configuration::crossEntropyCalculateBlockSize;
    dim3 dimGrid((networkOutput->col + dimBlock.x)/dimBlock.x);
    kSoftMaxCrossEntropy<<<dimGrid, dimBlock>>>(networkOutput->data, networkOutput->row, networkOutput->col, labels->data, output->data);
    return output;
}
