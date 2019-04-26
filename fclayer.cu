#include "fclayer.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <iostream>

std::default_random_engine generator;
std::normal_distribution<float> distribution(0,2);


FCLayer::FCLayer(int inSize, int outSize){
    this->inSize = inSize;
    this->outSize = outSize;

    float** initialWeigths = new float*[outSize];
    *initialWeigths = new float[inSize * outSize];
    for (int i = 1; i < outSize; i++) 
        initialWeigths[i] = initialWeigths[i-1] + inSize;

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < inSize; j++) {
            initialWeigths[i][j] = distribution(generator);
        }
    }

    this->w = new tensor(outSize, inSize,initialWeigths);
    this->w_gradient = NULL;
    this->bias = NULL;
    this->b_gradient = NULL;
    this->outputForward = NULL;
    this->outputBackward = NULL;

    float *initialBias = new float[outSize*1];

    for (int i = 0; i < outSize; i++) 
        initialBias[i] = 0;
    this->bias = new tensor(this->w->row,1,initialBias);
}

tensor* FCLayer::forward(tensor* data) {
    this->inputData = data;
    if (!this->outputForward) {
        this->outputForward = new tensor(this->w->row, this->inputData->col);
    }

    this->w->MatMul(this->inputData, this->outputForward);

    this->outputForward->Add(this->bias);
    return this->outputForward;
}

tensor* FCLayer::backward(tensor* gradient) {
    if (!this->w_gradient) {
        this->w_gradient = new tensor(gradient->row, this->inputData->row);
    }
    if (!this->b_gradient) {
        this->b_gradient = new tensor(this->w->row,1);
    }
    this->inputData->GradientMul(gradient, this->w_gradient);
    if (!this->outputBackward) {
        this->outputBackward = new tensor(this->w->col, gradient->col);
    }
    gradient->BackwardMul(this->w, this->outputBackward);
    return this->outputBackward;
}
