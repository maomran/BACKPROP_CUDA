#ifndef DENSE_H
#define DENSE_H

#include <cstdio>
#include <cmath>

#include "tensor.cu"
#include "utils.h"


class Dense{

public:
    int inSize;
    int outSize;
	tensor* w;
    tensor* b;
    tensor* w_gradients;
    tensor* b_gradients;

    tensor* inputData;
    tensor* outputForward;
    tensor* outputBackward;

    Dense(int inSize, int outSize);

    tensor* forward(tensor* data);
    tensor* backward(tensor* gradients);
};

#endif  
