#ifndef DENSE_H
#define DENSE_H

#include <cstdio>
#include <cmath>

#include "tensor.cu"
#include "utils.h"


class FCLayer{

public:
    int inSize;
    int outSize;
	tensor* w;
    tensor* bias;
    tensor* w_gradient;
    tensor* b_gradient;

    tensor* inputData;
    tensor* outputForward;
    tensor* outputBackward;

    FCLayer(int inSize, int outSize);

    tensor* forward(tensor* data);
    tensor* backward(tensor* gradients);
};

#endif  
