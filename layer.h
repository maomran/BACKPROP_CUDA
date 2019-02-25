#ifndef LAYER_H
#define LAYER_H

#include <cstdio>
#include <cmath>

#include "tensor.h"

class Layer {

public:
    tensor* w;
    tensor* bias;
    tensor* w_gradient;
    tensor* b_gradient;

    virtual tensor* forward(tensor* data) = 0;
    virtual tensor* backward(tensor* gradients) = 0;
};

#endif  /* LAYER_H */
