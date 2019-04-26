#define KERNELS_H
#include "funobj.h"
#include "kernels.h"

LossFunction::LossFunction() {}


float LossFunction::TrainingAccuracy(tensor* networkOutput, tensor* labels) {
    float* TrainAccuracy;
    float* dAccuracy;

    TrainAccuracy = (float*) calloc(0,sizeof(float));

    cudaMalloc((void**)&dAccuracy, sizeof(float));
    cudaMemcpy(dAccuracy, TrainAccuracy, sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32);
    dim3 dimGrid((networkOutput->col + dimBlock.x)/dimBlock.x);
    kSoftMaxAccuracy<<<dimGrid, dimBlock>>>(networkOutput->d_data, networkOutput->row, networkOutput->col, labels->d_data, dAccuracy);
    cudaMemcpy(TrainAccuracy, dAccuracy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dAccuracy);
    return *TrainAccuracy / networkOutput->col;
}

tensor* LossFunction::calculate(tensor* networkOutput, tensor* labels, tensor* output) {
    dim3 dimBlock(1,1);
    dim3 dimGrid(1,1);
    kSoftMax<<<dimGrid, dimBlock>>>(networkOutput->d_data, networkOutput->row, networkOutput->col, labels->d_data, output->d_data);
 
    return output;
}

float LossFunction::TestAccuracy(tensor* OutputVector, tensor* labels) {
    float TestAccuracy = 0;
    float* dev;
    cudaMalloc((void**)&dev, sizeof(float));
    cudaMemcpy(dev, &TestAccuracy, sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimBlock(32);
    dim3 dimGrid((OutputVector->col + dimBlock.x)/dimBlock.x);
    kSoftMaxAccuracy<<<dimGrid, dimBlock>>>(OutputVector->d_data, OutputVector->row, OutputVector->col, labels->d_data, dev);
    cudaMemcpy(&TestAccuracy, dev, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    return 100.0 * TestAccuracy / OutputVector->col;
}
