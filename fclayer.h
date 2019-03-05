#ifndef DENSE_H
#define DENSE_H

#include <cstdio>
#include <cmath>

#include "tensor.h"
// #include "utils.h"
#include "layer.h"

class FCLayer: public Layer{

public:
    int inSize;
    int outSize;
    tensor* inputData;
    tensor* outputForward;
    tensor* outputBackward;

    FCLayer(int inSize, int outSize);

    tensor* forward(tensor* data);
    tensor* backward(tensor* gradients);
};

#endif  
