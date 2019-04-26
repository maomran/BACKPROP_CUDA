#ifndef SIGMOD_H
#define SIGMOD_H

#include <cstdio>
#include <cmath>
#include "tensor.h"
// #include "utils.h"
#include "layer.h"

class SigmoidLayer: public Layer{

public:
    int input;
    int output;
    
    tensor* inputData;
    tensor* outputForward;
    tensor* outputBackward;
    SigmoidLayer(int inputOutput);

    tensor* forward(tensor* data);
    tensor* backward(tensor* gradients);
};

#endif  /* SIGMOD_H */
