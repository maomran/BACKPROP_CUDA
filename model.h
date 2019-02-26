#ifndef SEQUENTIAL_MODEL_H
#define SEQUENTIAL_MODEL_H

#include <cstdio>
#include <cmath>
#include <vector>

#include "layer.h"
#include "loss.h"
#include "sgd.h"
#include "tensor.h"
#include "utils.h"


class Model {
public:
    SGD* optimizer;
    CrossEntropyLoss* funobj;
    std::vector<Layer*> layers;
    tensor* gradients;

    Model(SGD* optimizer, CrossEntropyLoss* funobj);

    void addLayer(Layer* layer);
    tensor* forward(tensor* input);
    void backward(tensor* output, tensor* layers);
};

#endif  /* SEQUENTIAL_MODEL_H */
