#ifndef RELU_H
#define RELU_H

#include <cstdio>
#include <cmath>
#include "tensor.h"
// #include "utils.h"
#include "layer.h"

class ReluLayer: public Layer{

public:
    int input;
    int output;
    
    tensor* inputData;
    tensor* outputForward;
    tensor* outputBackward;
    ReluLayer(int inputOutput);

    tensor* forward(tensor* data);
    tensor* backward(tensor* gradients);
};

#endif  /* RELU_H */
